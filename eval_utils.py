import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
from device_utils import DEVICE

def evaluate_accuracy(
    model:      nn.Module,
    dataset:    Tuple[Tensor, Tensor],
    threshold:  float        = 0.5,
    batch_size: int          = 200,
    device:     torch.device = DEVICE,
    sample_wise_comparison: bool = False,
) -> float:
    """
    Bit-accuracy for binary / multi-label classification.
    Returns the fraction of bits predicted correctly across the full dataset.
    If sample_wise_comparison is True, computes exact match accuracy per sample instead.
    """
    x_test, y_test = dataset
    model.eval()
    correct = 0.0
    total   = x_test.size(0)

    with torch.no_grad():
        for i in range(0, total, batch_size):
            xb    = x_test[i : i + batch_size].to(device)
            yb    = y_test[i : i + batch_size].to(device)
            preds = (model(xb) >= threshold).float()
            
            if sample_wise_comparison:
                # Compare per sample (row-wise match)
                if yb.dim() > 1:
                    correct += (preds == yb).all(dim=-1).float().sum().item()
                else:
                    correct += (preds == yb).float().sum().item()
            else:
                # Compare every individual bit
                correct += (preds == yb).float().sum().item()

    if sample_wise_comparison:
        total_items = total
    else:
        total_items = total * (y_test.shape[-1] if y_test.dim() > 1 else 1)
        
    model.train()
    return correct / total_items
