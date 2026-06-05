import torch
from typing import Union

def get_device(device_input: Union[str, torch.device] = "auto") -> torch.device:
    if isinstance(device_input, torch.device):
        return device_input
    if device_input != "auto":
        return torch.device(device_input)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
