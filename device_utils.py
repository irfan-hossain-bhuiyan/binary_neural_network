import torch

def resolve_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE: torch.device = resolve_device()
