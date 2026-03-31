import torch
from pathlib import Path
from prelude import DEVICE

def load_xor_dataset(filepath: str | Path, device: torch.device | None = DEVICE) -> tuple[torch.Tensor, torch.Tensor]:
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

def load_mnist(filepath: str | Path, device: torch.device | None = DEVICE, input_flatten: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a previously saved MNIST (binary 0/1) dataset. Optionally flatten and binarize input."""
    payload = torch.load(Path(filepath), map_location="cpu")
    x = payload["X"].float()
    y = payload["Y"].float()
    # Binarize input: 0 for 0-127, 1 for 128-255
    x = (x >= 128).float()
    if input_flatten:
        x = x.view(x.size(0), -1)
    if device is not None:
        x = x.to(device)
        y = y.to(device)
    return x, y

def save_mnist(filepath: str | Path) -> Path:
    """Download MNIST using kagglehub, concatenate train/test, and save raw data (binary only)."""
    import numpy as np
    import torch
    import kagglehub
    from pathlib import Path
    import struct
    from array import array
    import os

    dataset_dir = kagglehub.dataset_download("hojjatk/mnist-dataset")
    def resolve_file(path):
        # If path is a directory, find the file inside it
        if os.path.isdir(path):
            files = os.listdir(path)
            if len(files) == 1:
                return os.path.join(path, files[0])
            else:
                raise FileNotFoundError(f"Multiple or no files found in directory: {path}")
        return path

    train_images = resolve_file(os.path.join(dataset_dir, "train-images-idx3-ubyte"))
    train_labels = resolve_file(os.path.join(dataset_dir, "train-labels-idx1-ubyte"))
    test_images = resolve_file(os.path.join(dataset_dir, "t10k-images-idx3-ubyte"))
    test_labels = resolve_file(os.path.join(dataset_dir, "t10k-labels-idx1-ubyte"))

    def read_images_labels(images_filepath, labels_filepath):
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            assert magic == 2049
            labels = array("B", file.read())
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            assert magic == 2051
            image_data = array("B", file.read())
        images = np.frombuffer(image_data, dtype=np.uint8).reshape(size, rows, cols)
        labels = np.frombuffer(labels, dtype=np.uint8)
        return images, labels

    x_train, y_train = read_images_labels(train_images, train_labels)
    x_test, y_test = read_images_labels(test_images, test_labels)
    x = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    # Convert labels to one-hot vectors
    y_onehot = torch.zeros((y.size(0), 10), dtype=torch.float32)
    y_onehot.scatter_(1, y.unsqueeze(1), 1.0)
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"X": x, "Y": y_onehot, "num_samples": x.size(0)}, path)
    return path

if __name__ == "__main__":
    xor_dataset_path = Path("artifacts/xor_dataset.pt")
    if not xor_dataset_path.exists():
        save_xor_dataset(xor_dataset_path, num_samples=100000)
        print(f"Generated and saved XOR dataset to {xor_dataset_path}")
    else:
        print(f"XOR dataset already exists at {xor_dataset_path}")

    mnist_dataset_path = Path("artifacts/mnist_binary.pt")
    if not mnist_dataset_path.exists():
        save_mnist(mnist_dataset_path)
        print(f"Generated and saved MNIST binary dataset to {mnist_dataset_path}")
    else:
        print(f"MNIST binary dataset already exists at {mnist_dataset_path}")

