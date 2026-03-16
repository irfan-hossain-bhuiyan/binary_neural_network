import torch


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
