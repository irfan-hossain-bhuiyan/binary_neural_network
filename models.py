"""Multi-layer differentiable binary neural network model."""

import copy
from pathlib import Path
from typing import Any, Callable, cast

import torch
import torch.nn as nn

from layers import OrNorGateLayer


class MultiLayerLogicGateNet(nn.Module):
    """Expectation-based multi-layer gate network with configurable depth and temperature sharing."""

    def __init__(
        self,
        input_dim: int = 64,
        layer_dims: list[int] | tuple[int, ...] = (256, 128, 64, 32),
        init_temperature: float = 1.0,
        shared_temperature: bool = False,
        learnable_tau: bool = False,
        use_softmax: bool = False,
        even_initialization: Callable[..., Any] = lambda x: nn.init.normal_(x, mean=1.0),
        odd_initialization: Callable[..., Any] = lambda x: nn.init.normal_(x, mean=0.0),
        bias_initialization: Callable[..., Any] = lambda x: nn.init.normal_(x, mean=1.0),
        grad_scalar: bool = False,
        load_file: str | Path | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.layer_dims = list(layer_dims)
        self.use_softmax = use_softmax
        self.shared_temperature = shared_temperature
        self.learnable_tau = learnable_tau
        self.expectation_layers: nn.ModuleList = nn.ModuleList()

        if shared_temperature:
            if learnable_tau:
                shared_temp = nn.Parameter(torch.tensor(float(init_temperature)))
            else:
                shared_temp = torch.tensor(float(init_temperature))
            self.temperatures: nn.Parameter | torch.Tensor | list[nn.Parameter | torch.Tensor] = shared_temp
        else:
            self.temperatures = []

        in_dim = input_dim
        for i, out_dim in enumerate(self.layer_dims):
            initialization = even_initialization if i % 2 == 0 else odd_initialization
            temperature = self.temperatures if shared_temperature else (
                nn.Parameter(torch.tensor(float(init_temperature)))
                if learnable_tau
                else torch.tensor(float(init_temperature))
            )
            if not shared_temperature:
                self.temperatures.append(temperature)
            layer = OrNorGateLayer(
                in_features=in_dim,
                out_features=out_dim,
                temperature=temperature,
                learnable_tau=learnable_tau,
                use_softmax=use_softmax,
                weight_initialization=initialization,
                bias_initialization=bias_initialization,
                grad_scalar=grad_scalar,
            )
            layer.enable_input_capture()
            self.expectation_layers.append(layer)
            in_dim = out_dim

        self.sequential = nn.Sequential(*self.expectation_layers)

        if load_file is not None:
            load_path = Path(load_file) if isinstance(load_file, str) else load_file
            if load_path.exists():
                self.load_state_dict(torch.load(load_path, map_location="cpu", weights_only=True))
            else:
                load_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.state_dict(), load_path)

    def clone(self) -> "MultiLayerLogicGateNet":
        """Return a deep copy of this network."""
        return copy.deepcopy(self)

    @staticmethod
    def noise_injector_factory(std_dev: float) -> Callable[["MultiLayerLogicGateNet"], None]:
        """Return a function that injects Gaussian noise into all layer weights."""

        def inject_noise(module: "MultiLayerLogicGateNet") -> None:
            print("random noise added.")
            with torch.no_grad():
                for layer in module.expectation_layers:
                    layer.weight.add_(torch.randn_like(cast(torch.Tensor, layer.weight)) * std_dev)

        return inject_noise

    def constraint(module: Any) -> None:
        """Clamp raw weights, biases, and temperatures to sensible ranges."""
        for layer in module.expectation_layers:
            layer = cast(OrNorGateLayer, layer)
            layer.weight.clamp_(-20.0, 20.0)
            layer.bias.clamp_(-20.0, 20.0)
            layer.temperature.clamp_(1e-3, 10.0)

    def set_temperature(self, temperature: float) -> None:
        """Set the softmax temperature for all layers in place.

        A very small temperature produces a sharp, near-discrete softmax.
        """
        temperatures = self.temperatures
        if isinstance(temperatures, (nn.Parameter, torch.Tensor)):
            with torch.no_grad():
                temperatures.fill_(temperature)
        else:
            for temp in temperatures:
                with torch.no_grad():
                    temp.fill_(temperature)

    def set_use_softmax(self, value: bool) -> None:
        """Switch all layers between softmax and hard-max mode."""
        for layer in self.expectation_layers:
            layer = cast(OrNorGateLayer, layer)
            layer.use_softmax = value

    @staticmethod
    def linear_temperature_anneal_factory(
        start_temperature: float = 1.0,
        end_temperature: float = 0.01,
        start_epoch: int = 1,
        end_epoch: int | None = None,
    ) -> Callable[[Any], None]:
        """Return a constraint that linearly anneals temperature each epoch.

        ``end_epoch=None`` uses the total number of epochs from the trainer's
        ``stop_on`` condition when it is available; otherwise it falls back to
        ``start_epoch`` so the temperature stays at ``end_temperature``.
        """

        def anneal(module: Any, epoch: int = start_epoch, **kwargs: Any) -> None:
            model = cast(MultiLayerLogicGateNet, module)
            total = end_epoch if end_epoch is not None else epoch
            progress = min(1.0, max(0.0, (epoch - start_epoch) / max(1, total - start_epoch)))
            temperature = start_temperature + (end_temperature - start_temperature) * progress
            temperature = max(temperature, end_temperature)
            temperatures = model.temperatures
            if isinstance(temperatures, (nn.Parameter, torch.Tensor)):
                with torch.no_grad():
                    temperatures.fill_(temperature)
            else:
                for temp in temperatures:
                    with torch.no_grad():
                        temp.fill_(temperature)

        return anneal

    def peek(self) -> dict[str, Any]:
        """Return a snapshot of current temperatures for logging."""
        result: dict[str, Any] = {}
        with torch.no_grad():
            if self.shared_temperature:
                result["shared_temperature"] = (
                    self.temperatures.item()
                    if isinstance(self.temperatures, (nn.Parameter, torch.Tensor))
                    else self.temperatures[0].item()
                )
            else:
                for i, temp in enumerate(self.temperatures):
                    result[f"temperature_{i}"] = temp.item()
        return result

    @property
    def temperature(self) -> torch.Tensor | list[torch.Tensor]:
        """Return the temperature value(s) of the network."""
        return self.temperatures

    @property
    def tau(self) -> torch.Tensor | list[torch.Tensor]:
        """Return the inverse temperature (tau) value(s) of the network."""
        if self.shared_temperature:
            temps = [self.temperatures]
        else:
            temps = self.temperatures
        taus = [1.0 / temp.clamp_min(1e-6) for temp in temps]
        return taus[0] if self.shared_temperature else taus

    @staticmethod
    def discretize(module: Any, threshold: float = 0.5) -> None:
        """Discretize every layer in ``module`` to 0/1 based on ``threshold``."""
        for layer in module.expectation_layers:
            layer = cast(OrNorGateLayer, layer)
            layer.discretize(threshold)

    def to_discrete(self, threshold: float = 0.5) -> Any:
        """Convert this continuous network into a ``DiscreteMultiLayerLogicGateNet``."""
        from discrete_logic_net import DiscreteMultiLayerLogicGateNet

        discrete_net = DiscreteMultiLayerLogicGateNet(
            input_dim=self.input_dim,
            layer_dims=self.layer_dims,
        )

        for cont_layer, disc_layer in zip(self.expectation_layers, discrete_net.expectation_layers):
            cont_layer = cast(Any, cont_layer)
            disc_layer.weight.copy_(cont_layer.actual_weight() >= threshold)
            disc_layer.bias.copy_(cont_layer.actual_bias() >= threshold)

        return discrete_net

    @staticmethod
    def batch_variance_regularization(module: Any) -> torch.Tensor:
        """Maximize batch-wise activation variance across all expectation layers.

        The negative mean variance is returned; minimizing this loss maximizes
        variance.  Inputs are captured automatically by forward-pre hooks.
        """
        cost = torch.tensor(0.0)
        for layer in module.expectation_layers:
            layer = cast(OrNorGateLayer, layer)
            x = getattr(layer, "_last_input", None)
            if not isinstance(x, torch.Tensor):
                continue
            cost = cost + layer.batch_variance_cost(x)
        return cost

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequential(x)
