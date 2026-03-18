import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Any, cast
from rich.console import Console
from rich.table import Table
from prelude import leaky_clamp
from data_utils import generate_xor_dataset


def pass_invert(x: torch.Tensor) -> torch.Tensor:
    """Concatenate inputs with their inverted values (1 - x)."""
    inverted = 1.0 - x
    return torch.cat([x, inverted], dim=-1)

class OrGateLayer(nn.Module):
    """Expectation layer that can operate in soft (softmax) or hard (argmax) mode.

    Args:
        in_features: input feature dimension
        out_features: output feature dimension
        shared_log_tau: if a float is provided (default), a new nn.Parameter is created
            and owned by this layer (not shared). If an nn.Parameter is provided, it
            will be used directly, enabling temperature sharing across layers.
        use_softmax: if True, uses temperature-scaled softmax expectation; otherwise
            uses hard max selection.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        shared_log_tau: float | nn.Parameter = 0.0,
        use_softmax: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_softmax = use_softmax

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_normal_(self.weight)

        if isinstance(shared_log_tau, nn.Parameter):
            self.log_tau = shared_log_tau
            self._owns_tau = False
        else:
            self.log_tau = nn.Parameter(torch.tensor(float(shared_log_tau)))
            self._owns_tau = True

    @property
    def tau(self) -> torch.Tensor:
        return torch.exp(self.log_tau)

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

def _collect_grad_norms(model: nn.Module) -> dict[str, float]:
    stats: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        stats[name] = param.grad.detach().norm().item()
    return stats

def _format_grad_stats(stats: dict[str, float], max_items: int = 6) -> Table:
    table = Table(title="Gradient Norms", show_header=True, header_style="bold cyan")
    table.add_column("param", style="white", overflow="fold")
    table.add_column("norm", justify="right", style="magenta")

    if not stats:
        table.add_row("(no gradients)", "-")
        return table

    items = sorted(stats.items(), key=lambda kv: kv[0])[:max_items]
    for k, v in items:
        table.add_row(k, f"{v:.3e}")
    return table

def _detect_vanishing_grads(stats: dict[str, float], threshold: float = 1e-8) -> bool:
    if not stats:
        return True
    return max(stats.values()) < threshold

def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = resolve_device()
CONSOLE = Console()

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

        shared_log_tau: float | nn.Parameter
        if is_shared_tau:
            shared_log_tau = nn.Parameter(torch.tensor(float(init_log_tau)))
        else:
            shared_log_tau = float(init_log_tau)
        self.expectation_layers: nn.ModuleList[OrGateLayer] = nn.ModuleList()

        current_dim = input_dim
        for out_dim in self.layer_dims:
            in_dim = current_dim * 2  # inverter doubles the features
            layer = OrGateLayer(
                in_features=in_dim,
                out_features=out_dim,
                shared_log_tau=shared_log_tau,
                use_softmax=use_softmax,
            )
            self.expectation_layers.append(layer)
            current_dim = out_dim

    @property
    def tau(self) -> torch.Tensor | list[torch.Tensor]:
        if self.is_shared_tau:
            return self.expectation_layers[0].tau
        return [layer.tau for layer in self.expectation_layers]

    def discretize(self, threshold: float) -> None:
        for layer in self.expectation_layers:
            if hasattr(layer, "discretize"):
                layer.discretize(threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = pass_invert(x)
        for idx, layer in enumerate(self.expectation_layers):
            x = layer(x)
            if idx < len(self.expectation_layers) - 1:
                x = pass_invert(x)
        return x

def train_model(
    num_epochs: int = 20,
    batch_size: int = 128,
    train_samples: int = 10000,
    tau_reg_weight: float = 1e-3,
    weight_l1_reg: float = 0.0,
    binary_weight_reg: float = 0.0,
    log_gradients_every: int | None = None,
    input_dim: int = 64,
    is_shared_tau: bool = True,
    init_log_tau: float = 0.0,
    use_softmax: bool = False,
    model: MultiLayerLogicGateNet | None = None,
    lr:float =0.1,
    optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
    optimizer_kwargs: dict[str, Any] | None = None,
    loss_fn: nn.modules.loss._Loss | None = None,
):
    if model is None:
        model = MultiLayerLogicGateNet(
            input_dim=input_dim,
            layer_dims=(256, 128, 64, 32),
            is_shared_tau=is_shared_tau,
            init_log_tau=init_log_tau,
            use_softmax=use_softmax,
        )
    model = model.to(DEVICE)

    if optimizer_kwargs is None:
        optimizer_kwargs = {"lr":lr,}
    optimizer = optimizer_cls(model.parameters(),**optimizer_kwargs)
    loss_fn = nn.BCELoss() if loss_fn is None else loss_fn

    X, Y = generate_xor_dataset(train_samples, device=DEVICE)
    loss_history: list[float] = []

    for epoch in range(1, num_epochs + 1):
        perm = torch.randperm(train_samples, device=DEVICE)
        X_epoch = X[perm]
        Y_epoch = Y[perm]

        epoch_loss = 0.0
        num_batches = 0
        vanishing_warned = False

        grad_msg = None
        epoch_vanishing = False
        for i in range(0, train_samples, batch_size):
            xb = X_epoch[i : i + batch_size]
            yb = Y_epoch[i : i + batch_size]

            optimizer.zero_grad()
            logits = model(xb)

            if tau_reg_weight == 0.0:
                reg_loss = 0.0
            else:
                if model.is_shared_tau:
                    tau_tensor: torch.Tensor = cast(torch.Tensor, model.tau)
                    reg_loss = tau_reg_weight / tau_tensor
                else:
                    taus: list[torch.Tensor] = cast(list[torch.Tensor], model.tau)
                    reg_loss = tau_reg_weight * sum(1.0 / t for t in taus) / float(len(taus))

            weight_penalty = 0.0
            if weight_l1_reg > 0.0 or binary_weight_reg > 0.0:
                for layer in model.expectation_layers:
                    actual_w = layer.actual_weight()
                    if weight_l1_reg > 0.0:
                        weight_penalty = weight_penalty + weight_l1_reg * actual_w.mean()
                    if binary_weight_reg > 0.0:
                        weight_penalty = weight_penalty + binary_weight_reg * (actual_w * (actual_w - 1.0)).mean()

            loss = loss_fn(logits, yb) + reg_loss + weight_penalty
            loss.backward()

            if log_gradients_every is not None and epoch % log_gradients_every == 0 and i == 0:
                grad_stats = _collect_grad_norms(model)
                grad_msg = _format_grad_stats(grad_stats)
            grad_stats = _collect_grad_norms(model)
            is_vanishing = _detect_vanishing_grads(grad_stats)
            epoch_vanishing = epoch_vanishing or is_vanishing
            if not vanishing_warned and is_vanishing:
                CONSOLE.print(
                    f"[bold red][warn][/bold red] epoch {epoch:03d} batch {i//batch_size:03d}: "
                    "gradients near zero (max_norm < 1e-8)"
                )
                vanishing_warned = True

            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        with torch.no_grad():
            for layer in model.expectation_layers:
                layer.weight.clamp_(-2.0, 2.0)

        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)

        if model.is_shared_tau:
            tau_val: float | list[float] = cast(torch.Tensor, model.tau).item()
        else:
            taus = cast(list[torch.Tensor], model.tau)
            tau_val = [t.item() for t in taus]

        vanish_status = "[bold red]VANISHING[/bold red]" if epoch_vanishing else "[green]OK[/green]"
        CONSOLE.print(f"Epoch {epoch:03d} | loss = {avg_loss:.6f} | tau = {tau_val} | grads = {vanish_status}")
        if grad_msg:
            CONSOLE.print(grad_msg)

    return model, loss_history

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
) -> float:
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        X_test, Y_test = generate_xor_dataset(num_samples, device=device)
        logits = model(X_test)
        preds = (logits >= threshold).float()
        correct_bits = (preds == Y_test).float().mean().item()
    model.train()
    return correct_bits

def plot_weight_distribution(model: nn.Module, bins: int = 50):
    with torch.no_grad():
        weights = []
        for layer in model.expectation_layers:
            weights.append(layer.actual_weight().detach().cpu().flatten())
    all_weights = torch.cat(weights)

    plt.figure(figsize=(6, 4))
    plt.hist(all_weights.numpy(), bins=bins, range=(0.0, 1.0), color="steelblue", edgecolor="white")
    plt.xlabel("Weight value (sigmoid)")
    plt.ylabel("Frequency")
    plt.title(f"Sigmoid(weight) distribution | mean={all_weights.mean():.3f}, std={all_weights.std():.3f}")
    plt.tight_layout()
    plt.show()

def main(epochs: int = 1000):
    model, loss_history = train_model(
        num_epochs=epochs,
        tau_reg_weight=0.0,
        binary_weight_reg=0.0,
        log_gradients_every=1,
        use_softmax= True,
        is_shared_tau=False,
    )
    plot_training_loss(loss_history)
    plot_weight_distribution(model)
    acc = evaluate_bit_accuracy(model, device=DEVICE)
    print(f"Bitwise accuracy on XOR test set: {acc * 100:.2f}%")
    return model


if __name__ == "__main__":
    main(2000)

