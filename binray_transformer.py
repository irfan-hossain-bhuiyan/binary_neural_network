import torch
from torch._prims_common import Tensor
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, cast
from rich.console import Console
from rich.table import Table
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from prelude import DEVICE, leaky_clamp, split_dataset, train_model, Checkpoint, HistoryEntry, TrainConfig, save_training_checkpoint, load_training_checkpoint
from data_utils import generate_xor_dataset, save_xor_dataset, load_xor_dataset
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
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        shared_tau_unconstrained: float | nn.Parameter = 0.0,
        use_softmax: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_softmax = use_softmax

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_normal_(self.weight)

        if isinstance(shared_tau_unconstrained, nn.Parameter):
            self.tau_unconstrained = shared_tau_unconstrained
            self._owns_tau = False
        else:
            self.tau_unconstrained = nn.Parameter(torch.tensor(float(shared_tau_unconstrained),device=DEVICE))
            self._owns_tau = True

    @property
    def tau(self) -> torch.Tensor:
        return 1.0 + F.softplus(self.tau_unconstrained)

    def actual_weight(self) -> torch.Tensor:
        return cast(torch.Tensor, leaky_clamp(self.weight, 0, 1, 0.1))

    def discretize(self, threshold: float) -> None:
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        with torch.no_grad():
            discrete_w = (self.actual_weight() >= threshold).float()
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


class MultiLayerLogicGateNet(nn.Module):
    """Expectation-based multi-layer gate network with configurable depth and tau sharing."""

    def __init__(
        self,
        input_dim: int = 64,
        layer_dims: list[int] | tuple[int, ...] = (256, 128, 64, 32),
        is_shared_tau: bool = True,
        init_log_tau: float = 0.0,
        use_softmax: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.layer_dims = list(layer_dims)
        self.is_shared_tau = is_shared_tau
        self.use_softmax = use_softmax

        shared_tau_unconstrained: float | nn.Parameter
        if is_shared_tau:
            shared_tau_unconstrained = nn.Parameter(torch.tensor(float(init_log_tau),device=DEVICE))
        else:
            shared_tau_unconstrained = float(init_log_tau)
        self.expectation_layers: nn.ModuleList = nn.ModuleList()

        current_dim = input_dim
        for out_dim in self.layer_dims:
            in_dim = current_dim * 2  # inverter doubles the features
            layer = OrGateLayer(
                in_features=in_dim,
                out_features=out_dim,
                shared_tau_unconstrained=shared_tau_unconstrained,
                use_softmax=use_softmax,
            )
            self.expectation_layers.append(layer)
            current_dim = out_dim
 
    def weight_constraint(self):
        for layer in self.expectation_layers:
            layer.weight.clamp_(-2.0, 2.0)
    def regularization(self, l1_lambda=1e-4, disc_lambda=1e-4):
        reg = torch.tensor(0.0, device=self.expectation_layers[0].weight.device)
        for layer in self.expectation_layers:
            w = layer.weight
            l1_error = w.abs().mean()
            disc_error = (w.abs() + (w - 1.0).abs()).mean()
            reg += (l1_lambda * l1_error) + (disc_lambda * disc_error)
        return reg


    @property
    def tau(self) -> torch.Tensor | list[torch.Tensor]:
        if self.is_shared_tau:
            return self.expectation_layers[0].tau
        return [layer.tau for layer in self.expectation_layers]

    def discretize(self, threshold: float) -> None:
        for layer in self.expectation_layers:
            layer.discretize(threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = pass_invert(x)
        for idx, layer in enumerate(self.expectation_layers):
            x = layer(x)
            if idx < len(self.expectation_layers) - 1:
                x = pass_invert(x)
        return x

def plot_training_loss(loss_history: list[float]):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(loss_history) + 1), loss_history, color="tomato", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def evaluate_bit_accuracy(
    model: nn.Module,
    threshold: float = 0.5,
    num_samples: int = 2000,
    device: torch.device | None = None,
    x_test: torch.Tensor | None = None,
    y_test: torch.Tensor | None = None,
) -> float:
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        if x_test is None or y_test is None:
            X_test, Y_test = generate_xor_dataset(num_samples, device=device)
        else:
            X_test = x_test.to(device)
            Y_test = y_test.to(device)
        logits = model(X_test)
        preds = (logits >= threshold).float()
        correct_bits = (preds == Y_test).float().mean().item()
    model.train()
    return correct_bits

def main(epochs: int = 50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = Path("artifacts/xor_dataset.pt")
    if not dataset_path.exists():
        save_xor_dataset(dataset_path, num_samples=100000)

    x_all, y_all = load_xor_dataset(dataset_path, device=device)
    x_train, y_train, x_test, y_test = split_dataset(x_all, y_all, train_ratio=0.8, shuffle=True)

    net = MultiLayerLogicGateNet(
        input_dim=64,
        layer_dims=(256, 128, 64, 32),
        is_shared_tau=False,
        init_log_tau=0.0,
        use_softmax=True,
    )
    checkpoint = train_model(
        dataset=(x_all, y_all),
        num_epochs=epochs,
        batch_size=128,
        model=net,
        loss_fn=nn.MSELoss(),
        lr=0.25,
        optimizer_cls= Adam,
        optimizer_kwargs= {"betas":(0.5,0.5)},
        regularization_fn=net.regularization,
        lr_schedular=CosineAnnealingWarmRestarts,
        lr_schedular_kargs={"T_0": 200,"T_mult":1,"eta_min":1e-3},
        constraint=net.weight_constraint,
        checkpoint_path=Path("artifacts/binary_transformer_checkpoint.pt"),
        device=device,
        check_grad=True,
    )
    plot_training_loss(checkpoint.get_avg_losses())
    plot_weight_distribution(checkpoint.model)
    acc = evaluate_bit_accuracy(checkpoint.model, device=device, x_test=x_test, y_test=y_test)
    print(f"Bitwise accuracy on XOR test set: {acc * 100:.2f}%")
    return checkpoint


if __name__ == "__main__":
    main(2000)

