import torch
from pathlib import Path
def load_xor_dataset(filepath: str | Path, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a previously saved XOR dataset."""
    payload = torch.load(Path(filepath), map_location="cpu")
    x = payload["X"].float()
    y = payload["Y"].float()
    if device is not None:
        x = x.to(device)
        y = y.to(device)
    return x, y

def int_to_bits(x: torch.Tensor, num_bits: int = 32) -> torch.Tensor:
    """Convert non-negative integers to binary bit vectors."""
    x = x.unsqueeze(-1)
    device = x.device
    powers = 2 ** torch.arange(num_bits - 1, -1, -1, device=device).view(1, -1)
    bits = (x & powers) != 0
    return bits.float()


def generate_xor_dataset(num_samples: int, device: torch.device = torch.device("cpu")):
    """Generate dataset for 32-bit XOR."""
    a = torch.randint(low=0, high=2**32, size=(num_samples,), device=device, dtype=torch.long)
    b = torch.randint(low=0, high=2**32, size=(num_samples,), device=device, dtype=torch.long)

    a_bits = int_to_bits(a, num_bits=32)
    b_bits = int_to_bits(b, num_bits=32)
    x = torch.cat([a_bits, b_bits], dim=1)

    xor_val = a ^ b
    y = int_to_bits(xor_val, num_bits=32)

    return x, y


def save_xor_dataset(
    filepath: str | Path,
    num_samples: int,
    device: torch.device = torch.device("cpu"),
) -> Path:
    """Generate XOR dataset once and save to disk for reuse."""
    x, y = generate_xor_dataset(num_samples=num_samples, device=device)
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"X": x.cpu(), "Y": y.cpu(), "num_samples": num_samples}, path)
    return path

if __name__ == "__main__":
    dataset_path = Path("artifacts/xor_dataset.pt")
    if not dataset_path.exists():
        save_xor_dataset(dataset_path, num_samples=100000)
        print(f"Generated and saved dataset to {dataset_path}")
    else:
        print(f"Dataset already exists at {dataset_path}")

