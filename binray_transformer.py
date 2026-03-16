import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from prelude import leaky_clamp
from data_utils import generate_xor_dataset


class ExpectationSoftmaxLayer(nn.Module):
    """Custom layer with expectation-based softmax aggregation.
    For each output neuron j:
        z_j = w_j ⊙ x                      (elementwise product)
        z_scaled_j = tau * z_j             (tau is a learnable scalar temperature)
        p_j = softmax(z_scaled_j)          (over input dimension)
        s_j = Σ_i p_j[i] * z_j[i]          (expectation w.r.t. original z_j)

    This mimics a soft-argmax / soft-max operator, where tau controls how
    close the behavior is to a hard max. No bias term is used.

    This mimics or gate, and weight=1 for the the node is connected to or gate =0 if not.
    Or gate because or(A)=max(A) if A has element 0 to 1 append.
    And weight =0 makes weights*x =0 making or not having input
    weights=1 makes the or gate to connect to previous node,
    Then I invert it,to get nor gate.
    """

    def __init__(self, in_features: int, out_features: int,  shared_log_tau: torch.nn.Parameter):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight shape: (out_features, in_features)
        # These are unconstrained parameters; actual weights are sigmoid(weight)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_normal_(self.weight)
        # Learnable temperature scalar; higher -> closer to hard max
        # If shared_log_tau is provided, all layers will use the same temperature
        self.log_tau = shared_log_tau


    @property
    def tau(self) -> torch.Tensor:
        # TODO: Need some experiment,I think I will remove this and implement a simple scalar.
        return torch.exp(self.log_tau)

    def actual_weight(self) -> torch.Tensor:
        return leaky_clamp(self.weight,0,1,0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch_size, in_features)

        Returns:
            Tensor of shape (batch_size, out_features)
        """
        # Constrain weights to [0, 1]
        actual_weight = self.actual_weight()

        # z: (batch_size, out_features, in_features)
        z = x.unsqueeze(1) * actual_weight.unsqueeze(0)

        # Scale before softmax to control sharpness
        z_scaled = self.tau * z

        # Softmax over input dimension (last dim)
        p = F.softmax(z_scaled, dim=-1)

        # Expectation with respect to original z
        # s: (batch_size, out_features)
        s = (p * z).sum(dim=-1)

        return s


def _collect_grad_norms(model: nn.Module) -> dict[str, float]:
    stats: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        stats[name] = param.grad.detach().norm().item()
    return stats


def _format_grad_stats(stats: dict[str, float], max_items: int = 6) -> str:
    items = list(stats.items())[:max_items]
    return "; ".join(f"{k}: {v:.3e}" for k, v in items)


class XorExpectationNet(nn.Module):
    """Network for 32-bit XOR using expectation-softmax layers.

    Architecture:
        64 -> 128 -> 64 -> 32
    First two layers use an additional nonlinearity (1 - x) after the
    expectation-softmax layer. 
    """

    def __init__(self):
        super().__init__()
        # Shared temperature parameter for all layers
        self.shared_log_tau = nn.Parameter(torch.zeros(1))

        self.layer1 = ExpectationSoftmaxLayer(in_features=64, out_features=256, shared_log_tau=self.shared_log_tau)
        self.layer2 = ExpectationSoftmaxLayer(in_features=256, out_features=64, shared_log_tau=self.shared_log_tau)
        self.layer3 = ExpectationSoftmaxLayer(in_features=64, out_features=32, shared_log_tau=self.shared_log_tau)

    @property
    def tau(self) -> torch.Tensor:
        return torch.exp(self.shared_log_tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, 64)
        x = self.layer1(x)
        x = 1.0 - x  # inverter activation on hidden layer

        x = self.layer2(x)
        x = 1.0 - x  # inverter activation on hidden layer

        # Final layer: no inverter, outputs are logits
        x = self.layer3(x)
        return x

def train_model(
    num_epochs: int = 20,
    batch_size: int = 128,
    train_samples: int = 10000,
    lr: float = 1e-3,
    tau_reg_weight: float = 1e-3,
    weight_l1_reg: float = 0.0,
    binary_weight_reg: float = 0.0,
    log_gradients_every: int | None = None,
    device: torch.device | None = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = XorExpectationNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    X, Y = generate_xor_dataset(train_samples, device=device)

    for epoch in range(1, num_epochs + 1):
        perm = torch.randperm(train_samples, device=device)
        X_epoch = X[perm]
        Y_epoch = Y[perm]

        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, train_samples, batch_size):
            xb = X_epoch[i : i + batch_size]
            yb = Y_epoch[i : i + batch_size]

            optimizer.zero_grad()
            logits = model(xb)
            reg_loss = tau_reg_weight / model.tau

            weight_penalty = 0.0
            if weight_l1_reg > 0.0 or binary_weight_reg > 0.0:
                for layer in (model.layer1, model.layer2, model.layer3):
                    actual_w = layer.actual_weight()
                    if weight_l1_reg > 0.0:
                        weight_penalty = weight_penalty + weight_l1_reg * actual_w.mean()
                    if binary_weight_reg > 0.0:
                        weight_penalty = weight_penalty + binary_weight_reg * (actual_w * (actual_w - 1.0)).mean()

            loss = loss_fn(logits, yb) + reg_loss + weight_penalty
            loss.backward()

            grad_msg = None
            if log_gradients_every is not None and epoch % log_gradients_every == 0 and i == 0:
                grad_stats = _collect_grad_norms(model)
                grad_msg = _format_grad_stats(grad_stats)

            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        # Simple progress print; you can adapt to your logging style
        if grad_msg:
            print(f"Epoch {epoch:03d} | loss = {avg_loss:.6f} | tau = {model.tau.item():.3f} | grads: {grad_msg}")
        else:
            print(f"Epoch {epoch:03d} | loss = {avg_loss:.6f} | tau = {model.tau.item():.3f}")

    return model


def evaluate_bit_accuracy(model: nn.Module, num_samples: int = 2000, device: torch.device | None = None) -> float:
    """Evaluate bitwise accuracy on a fresh XOR dataset."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        X_test, Y_test = generate_xor_dataset(num_samples, device=device)
        logits = model(X_test)
        probs = logits
        preds = (probs >= 0.5).float()
        correct_bits = (preds == Y_test).float().mean().item()
    model.train()
    return correct_bits


def plot_weight_distribution(model: nn.Module, bins: int = 50):
    with torch.no_grad():
        weights = []
        for layer in (model.layer1, model.layer2, model.layer3):
            weights.append(layer.actual_weight().detach().cpu().flatten())
    all_weights = torch.cat(weights)

    plt.figure(figsize=(6, 4))
    plt.hist(all_weights.numpy(), bins=bins, range=(0.0, 1.0), color="steelblue", edgecolor="white")
    plt.xlabel("Weight value (sigmoid)")
    plt.ylabel("Frequency")
    plt.title(f"Sigmoid(weight) distribution | mean={all_weights.mean():.3f}, std={all_weights.std():.3f}")
    plt.tight_layout()
    plt.show()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_model(num_epochs=500,device=device,tau_reg_weight=0.0,binary_weight_reg=0.001)
    plot_weight_distribution(model)
    acc = evaluate_bit_accuracy(model, device=device)
    print(f"Bitwise accuracy on XOR test set: {acc * 100:.2f}%")


if __name__ == "__main__":
    main()
