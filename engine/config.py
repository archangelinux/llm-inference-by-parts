import torch
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"