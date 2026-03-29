from math import log
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Any, Callable, cast
from torch.optim import Adam
from prelude import DEVICE, EarlyStopping, leaky_clamp, plot_training_loss, Trainer, split_dataset
from data_utils import save_xor_dataset, load_xor_dataset
from prelude import plot_weight_distribution

def pass_invert(x: torch.Tensor) -> torch.Tensor:
    """Concatenate inputs with their inverted values (1 - x)."""
    inverted = 1.0 - x
    return torch.cat([x, inverted], dim=-1)

class OrGateLayer(nn.Module):
    """Expectation layer that can operate in soft (softmax) or hard (argmax) mode.

    Args:
        in_features: input feature dimension
        out_features: output feature dimension
        shared_tau_unconstrained: if a float is provided (default), a new nn.Parameter is created
            and owned by this layer (not shared). If an nn.Parameter is provided, it
            will be used directly, enabling temperature sharing across layers.
        use_softmax: if True, uses temperature-scaled softmax expectation; otherwise
            uses hard max selection.
        max_threshold: Used when softmax is used,It make a upper floor for the softmax function,Say if 
        in or gate rest of the input are 0 and one is 1,(output should be 1),but softmax doesn't make that,
        This parameter sets what is the minimum truth value will be in that scenerio.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_threshold: float = 0.9,
        tau: float|nn.Parameter = 0.0,
        use_softmax: bool = False,
        initialization: Callable[..., Any] = nn.init.normal_,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_softmax = use_softmax
        # Compute gradient scale based on square root of the input dimension

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        initialization(self.weight)
        if isinstance(tau,nn.Parameter):
            self.tau_adder=tau
        else:
            self.tau_adder=nn.Parameter(torch.tensor(tau))
        self.tau_floor=log(in_features-1)+log(max_threshold)-log(1-max_threshold)
    @property
    def tau(self) -> torch.Tensor:
        return self.tau_floor + F.leaky_relu(self.tau_adder,negative_slope=0.05)

    def actual_weight(self) -> torch.Tensor:
        return cast(torch.Tensor, leaky_clamp(self.weight, 0, 1, 0.1))

    def discretize(self, threshold: float) -> None:
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        with torch.no_grad():
            discrete_w = (self.weight >= threshold).float()
            self.weight.copy_(discrete_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        actual_weight = self.actual_weight()
        # z: (batch_size, out_features, in_features)
        z = x.unsqueeze(1) * actual_weight.unsqueeze(0)
        if self.use_softmax:
            z_scaled = self.tau * z
            p = F.softmax(z_scaled, dim=-1)
            s = (p * z).sum(dim=-1)
        else:
            s = z.max(dim=-1).values

        return s
    def to_hardmax(self):
        self.use_softmax=False

class MultiLayerLogicGateNet(nn.Module):
    """Expectation-based multi-layer gate network with configurable depth and tau sharing."""

    def __init__(
        self,
        input_dim: int = 64,
        layer_dims: list[int] | tuple[int, ...] = (256, 128, 64, 32),
        init_tau_param:nn.Parameter |float = 0.0,
        max_threshold:float =0.95,
        # Parameter is the tau is shared.
        # float if all of them are isolated
        # None for default prefered value
        use_softmax: bool = False,
        only_inverter=True,
        initialization: Callable[..., Any] = nn.init.normal_,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.layer_dims = list(layer_dims)
        self.use_softmax = use_softmax
        self.is_shared_tau = isinstance(init_tau_param,nn.Parameter)
        self.only_inverter=only_inverter
        self.expectation_layers: nn.ModuleList = nn.ModuleList()

        in_dim = input_dim *2 # As the first one passes to an inverter.
        for out_dim in self.layer_dims:
            layer = OrGateLayer(
                in_features=in_dim,
                out_features=out_dim,
                tau=init_tau_param,
                use_softmax=use_softmax,
                max_threshold=max_threshold,
                initialization=initialization,
            )
            self.expectation_layers.append(layer)
            in_dim = out_dim * (1 if only_inverter else 2) # inverter doubles the features
 
    def clone(self):
        return copy.deepcopy(self)

    def weight_constraint(self):
        for layer in self.expectation_layers:
            layer = cast(OrGateLayer, layer)
            layer.weight.clamp_(-3.0, 3.0)
    
    def regularization(self, l1_lambda=1e-1, disc_lambda=1e-1, tau_lambda=1e-1):
        reg = torch.tensor(0.0, device=DEVICE)
        for layer in self.expectation_layers:
            layer = cast(OrGateLayer, layer)
            w = layer.weight
            l1_error = w.relu().mean()
            disc_error = (0.5-(w-0.5).abs()).relu().mean()
            tau_err = torch.exp(-layer.tau)
            reg += (l1_lambda * l1_error) + (disc_lambda * disc_error) + (tau_lambda * tau_err)
            # Encourage tau to grow larger (L1 regularization, negative sign)
        return reg
    def set_use_softmax(self,value:bool):
        for layer in self.expectation_layers:
            layer = cast(OrGateLayer,layer)
            layer.use_softmax=value

    def peek(self) -> dict[str, Any]:
        result = {}
        with torch.no_grad():
            if self.is_shared_tau:
                first_layer = cast(OrGateLayer, self.expectation_layers[0])
                result["shared_tau"] = first_layer.tau.item()
            else:
                for i, layer in enumerate(self.expectation_layers):
                    layer = cast(OrGateLayer, layer)
                    result[f"tau_{i}"] = layer.tau.item()
        return result

    @property
    def tau(self) -> torch.Tensor | list[torch.Tensor]:
        if self.is_shared_tau:
            first_layer = cast(OrGateLayer, self.expectation_layers[0])
            return first_layer.tau
        return [cast(OrGateLayer, layer).tau for layer in self.expectation_layers]

    def discretize(self, threshold: float) -> None:
        for layer in self.expectation_layers:
            layer = cast(OrGateLayer, layer)
            layer.discretize(threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = pass_invert(x)
        for idx, layer in enumerate(self.expectation_layers):
            x = layer(x)
            if idx < len(self.expectation_layers) - 1:
                x = (1-x) if self.only_inverter else pass_invert(x)
            else:
                x = (1-x)
        return x

def main():
    print("checking if new code get updated")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = Path("artifacts/xor_dataset.pt")
    if not dataset_path.exists():
        save_xor_dataset(dataset_path, num_samples=100000)

    x_all, y_all = load_xor_dataset(dataset_path, device=device)
    x_train, y_train, _, _ = split_dataset(x_all, y_all, train_ratio=0.8, shuffle=True)

    net = MultiLayerLogicGateNet(
        input_dim=64,
        layer_dims=(256, 128, 64, 32),
        use_softmax=True,
    )
    
    
    # We define a custom preview function that both gives string metrics to Console
    # and logs to TensorBoard for plotting.
    
    trainer = Trainer(
        dataset=(x_train, y_train),
        training_type=EarlyStopping(70),
        batch_size=128,
        model=net,
        loss_fn=nn.L1Loss(),
        optimizer_cls= Adam,
        optimizer_kwargs= {"betas":(0.5,0.5),"lr":0.1},
        regularization_fn=net.regularization,
        lr_schedular=None, #CosineAnnealingWarmRestarts,
        lr_schedular_kargs={"T_0": 200,"T_mult":1,"eta_min":1e-3},
        constraint=net.weight_constraint,
        checkpoint_path=Path("artifacts/binary_transformer_checkpoint.pt"),
        device=device,
        check_grad=True,
        peek=net.peek,
    )
    checkpoint = trainer.train()
    #trainer.export_for_burn(Path("artifacts/burn_export"))
    plot_training_loss(checkpoint.get_avg_losses())
    plot_weight_distribution(checkpoint.model)
    
    # Cleanup TensorBoard
    return None


if __name__ == "__main__":
    main()

