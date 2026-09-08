import torch

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def get_dtype(device):
    return torch.float16 if device.type == "cuda" else torch.float32