import torch

from binray_transformer import train_model, evaluate_bit_accuracy


EXPERIMENT_CONFIGS = [
    {"name": "baseline", "num_epochs": 75, "tau_reg_weight": 1e-3, "weight_l1_reg": 0.0, "binary_weight_reg": 0.0},
    {"name": "tau_push", "num_epochs": 75, "tau_reg_weight": 5e-4, "weight_l1_reg": 0.0, "binary_weight_reg": 0.0},
    {"name": "l1_sparse", "num_epochs": 75, "tau_reg_weight": 5e-4, "weight_l1_reg": 1e-3, "binary_weight_reg": 0.0},
    {"name": "binary_pull", "num_epochs": 75, "tau_reg_weight": 5e-4, "weight_l1_reg": 0.0, "binary_weight_reg": 1e-3},
    {"name": "combined", "num_epochs": 75, "tau_reg_weight": 1e-4, "weight_l1_reg": 5e-4, "binary_weight_reg": 5e-4},
]


def gather_weight_stats(model: torch.nn.Module) -> dict[str, float]:
    with torch.no_grad():
        weights = torch.cat(
            [layer.actual_weight().detach().flatten() for layer in (model.layer1, model.layer2, model.layer3)]
        )
    midpoint_distance = (weights * (1.0 - weights)).mean().item()
    binary_penalty = (weights * (weights - 1.0)).abs().mean().item()
    return {
        "mean": weights.mean().item(),
        "std": weights.std().item(),
        "midpoint": midpoint_distance,
        "binary_abs": binary_penalty,
    }


def run_experiments(device: torch.device | None = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results: list[dict[str, float | str]] = []
    for cfg in EXPERIMENT_CONFIGS:
        print(f"\n=== Running experiment: {cfg['name']} ===")
        model = train_model(
            num_epochs=cfg["num_epochs"],
            tau_reg_weight=cfg["tau_reg_weight"],
            weight_l1_reg=cfg["weight_l1_reg"],
            binary_weight_reg=cfg["binary_weight_reg"],
            device=device,
        )
        acc = evaluate_bit_accuracy(model, device=device)
        stats = gather_weight_stats(model)
        results.append(
            {
                "name": cfg["name"],
                "acc": acc,
                "tau": model.tau.item(),
                "mean": stats["mean"],
                "std": stats["std"],
                "midpoint": stats["midpoint"],
                "binary_abs": stats["binary_abs"],
            }
        )
        print(
            f"Accuracy: {acc * 100:.2f}% | tau: {model.tau.item():.3f} | "
            f"mean={stats['mean']:.3f} std={stats['std']:.3f} midpoint={stats['midpoint']:.4f} "
            f"binary_abs={stats['binary_abs']:.4f}"
        )

    print("\nSummary:\n")
    header = "{:<12} {:>8} {:>8} {:>8} {:>8} {:>10}".format(
        "name", "acc%", "tau", "mean", "std", "binary_abs"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            "{:<12} {:>8.2f} {:>8.3f} {:>8.3f} {:>8.3f} {:>10.4f}".format(
                row["name"],
                row["acc"] * 100,
                row["tau"],
                row["mean"],
                row["std"],
                row["binary_abs"],
            )
        )


if __name__ == "__main__":
    run_experiments()
