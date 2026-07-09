
"""Train XOR with frozen temperature, linear temperature annealing, and variance regularizer."""

from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from data_utils import load_xor_dataset, save_xor_dataset
from initializers import NormalInitWrapper
from models import MultiLayerLogicGateNet
from plot_utils import plot_training_loss, plot_weight_distribution
from prelude import Trainer, split_dataset, stop_on_epoch
from stopping_utils import call_fn_on_plateau


def train_xor_anneal(
    epoch: int = 50,
    device_id: int = 0,
    print_terminal: bool = True,
    check_grad: bool = False,
    num_bits: int = 32,
    start_temperature: float = 1.0,
    end_temperature: float = 0.01,
    variance_weight: float = 1e-3,
):
    """Train with annealed temperature and batch-variance regularization."""
    if torch.cuda.is_available() and torch.cuda.device_count() > device_id:
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path = Path(f"artifacts/xor{num_bits}_dataset.pt")
    if not dataset_path.exists():
        save_xor_dataset(dataset_path, num_samples=100000, num_bits=num_bits)

    x_all, y_all = load_xor_dataset(dataset_path, device=device)
    x_train, y_train, x_test, y_test = split_dataset(x_all, y_all, train_ratio=0.8, shuffle=True)

    net = MultiLayerLogicGateNet(
        input_dim=2 * num_bits,
        layer_dims=(64, 32, 64, num_bits),
        use_softmax=True,
        grad_scalar=True,
        odd_initialization=NormalInitWrapper(0.5),
        even_initialization=NormalInitWrapper(0.5),
        bias_initialization=NormalInitWrapper(1.0),
    ).to(device)

    def variance_regularizer(module ):
        return variance_weight * MultiLayerLogicGateNet.batch_variance_regularization(module)

    trainer = Trainer(
        dataset=(x_train, y_train),
        stop_on=stop_on_epoch(epoch),
        batch_size=256,
        model=net,
        loss_fn=nn.MSELoss(),
        optimizer_cls=Adam,
        optimizer_kwargs={"lr": 0.05},
        regularization_fn=variance_regularizer,
        lr_scheduler_factory=None,
        constraints=[
            MultiLayerLogicGateNet.constraint,
            MultiLayerLogicGateNet.linear_temperature_anneal_factory(
                start_temperature=start_temperature,
                end_temperature=end_temperature,
                end_epoch=epoch,
            ),
           # call_fn_on_plateau(
           #     MultiLayerLogicGateNet.noise_injector_factory(0.3),
           #     patience=15,
           #     min_delta=0.01,
           # ),
        ],
        checkpoint_path=None,
        device=device,
        check_grad=check_grad,
        peek=net.peek,
    )

    ckpt = trainer.train(print_terminal=print_terminal)

    # ----- discrete test evaluation -----
    unwrapped_model = getattr(ckpt.model, "module", ckpt.model)
    MultiLayerLogicGateNet.discretize(unwrapped_model, threshold=0.5)
    discrete_model = unwrapped_model.to_discrete(threshold=0.5)
    discrete_model = discrete_model.to(device)
    discrete_model.eval()

    test_correct = 0
    test_total = x_test.shape[0]
    with torch.no_grad():
        for i in range(0, test_total, 256):
            xb = x_test[i : i + 256].to(device).to(torch.bool)
            yb = y_test[i : i + 256].to(device).to(torch.bool)
            preds = discrete_model(xb)
            test_correct += (preds == yb).all(dim=-1).sum().item()

    exact_match_acc = test_correct / test_total
    print(
        f"Discrete test exact-match accuracy: {test_correct}/{test_total} "
        f"= {exact_match_acc:.6%}"
    )
    # -------------------------------------

    plot_training_loss(ckpt.avg_errors(), header="Errors")
    plot_weight_distribution(ckpt.model)
    return ckpt


if __name__ == "__main__":
    train_xor_anneal(epoch=40, num_bits=16,check_grad=True)
