"""Training entry point for the XOR reconstruction task."""

from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from data_utils import load_xor_dataset, save_xor_dataset
from initializers import NormalInitWrapper
from models import MultiLayerLogicGateNet
from plot_utils import plot_training_loss, plot_weight_distribution
from prelude import Trainer, split_dataset, stop_on_epoch
from regularizers import regularization_factory2
from stopping_utils import call_fn_on_plateau


def train_xor_main(
    epoch: int = 40,
    device_id: int = 0,
    print_terminal: bool = True,
    check_grad: bool = False,
    num_bits: int = 16,
):
    """Train a ``MultiLayerLogicGateNet`` on the n-bit XOR dataset.

    Args:
        epoch: number of training epochs.
        device_id: CUDA device index, or falls back to CPU.
        print_terminal: whether to print per-epoch logs.
        check_grad: whether to collect and print gradient statistics.
        num_bits: bit-width of the XOR operands (input is 2*num_bits, output is num_bits).
    """
    if torch.cuda.is_available() and torch.cuda.device_count() > device_id:
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path = Path(f"artifacts/xor{num_bits}_dataset.pt")
    if not dataset_path.exists():
        save_xor_dataset(dataset_path, num_samples=100000, num_bits=num_bits)

    x_all, y_all = load_xor_dataset(dataset_path, device=device)
    x_train, y_train, _, _ = split_dataset(x_all, y_all, train_ratio=0.8, shuffle=True)

    net = MultiLayerLogicGateNet(
        input_dim=2 * num_bits,
        layer_dims=(64, 32, 16),
        init_temperature=1.0,
        shared_temperature=True,
        learnable_tau=True,
        use_softmax=True,
        grad_scalar=True,
        odd_initialization=NormalInitWrapper(0.0),
        even_initialization=NormalInitWrapper(1.0),
        bias_initialization=NormalInitWrapper(1.0),
    ).to(device)

    trainer = Trainer(
        dataset=(x_train, y_train),
        stop_on=stop_on_epoch(epoch),
        batch_size=256,
        model=net,
        loss_fn=nn.MSELoss(),
        optimizer_cls=Adam,
        optimizer_kwargs={"lr": 0.01},
        regularization_fn=regularization_factory2(
            0.5, tau_lambda=0.3, isolate_on_plateau=True
        ),
        lr_scheduler_factory=None,
        constraints=[
            MultiLayerLogicGateNet.constraint,
            call_fn_on_plateau(
                MultiLayerLogicGateNet.noise_injector_factory(0.3),
                patience=15,
                min_delta=0.01,
            ),
        ],
        checkpoint_path=None,
        device=device,
        check_grad=check_grad,
        peek=net.peek,
    )

    ckpt = trainer.train(print_terminal=print_terminal)
    plot_training_loss(ckpt.avg_errors(), header="Errors")
    plot_weight_distribution(ckpt.model)
    return ckpt


def main():
    return train_xor_main(300, check_grad=True)


if __name__ == "__main__":
    main()
