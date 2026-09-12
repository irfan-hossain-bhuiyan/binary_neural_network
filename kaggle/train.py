import json
import platform
from pathlib import Path

import torch


result = {
    "status": "success",
    "python_version": platform.python_version(),
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
}

if torch.cuda.is_available():
    result["gpu_name"] = torch.cuda.get_device_name(0)

    x = torch.randn(2048, 2048, device="cuda")
    y = x @ x

    result["gpu_test_value"] = float(y[0, 0].item())

else:
    result["gpu_name"] = None


print(json.dumps(result, indent=2))

Path("/kaggle/working/result.json").write_text(
    json.dumps(result, indent=2)
)
