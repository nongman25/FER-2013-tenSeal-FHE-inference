"""FHE-friendly CNN architecture and helpers."""
from __future__ import annotations

from typing import Dict, List

import torch
from torch import Tensor, nn


class Square(nn.Module):
    """Square activation (x^2). HE-friendly and non-linear."""

    def forward(self, x: Tensor) -> Tensor:
        return x * x


class FHEEmotionCNN(nn.Module):
    """
    CNN architecture compatible with TenSEAL Tutorial 4.
    Adapted for FER2013 (48x48 input, 7 classes).
    """

    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        # Conv1: 1 -> 4 channels, kernel 7x7, stride 3
        # Input: 48x48
        # Output: (48 - 7) // 3 + 1 = 14. Shape: 4 x 14 x 14
        self.conv1 = nn.Conv2d(1, 4, kernel_size=7, stride=3, padding=0)
        self.act1 = Square()
        
        # Flatten size: 4 * 14 * 14 = 784
        self.fc1 = nn.Linear(4 * 14 * 14, 64)
        self.act2 = Square()
        
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.act1(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.act2(x)
        x = self.fc2(x)
        return x


def extract_fhe_parameters(model: nn.Module) -> Dict[str, List[Dict[str, Tensor]]]:
    """Extract weights and biases for FHE inference."""
    params = {"conv": [], "linear": []}
    
    # Conv1
    params["conv"].append({
        "weight": model.conv1.weight.detach(),  # Shape: (4, 1, 7, 7)
        "bias": model.conv1.bias.detach()       # Shape: (4,)
    })
    
    # FC1
    params["linear"].append({
        "weight": model.fc1.weight.detach(),    # Shape: (64, 784)
        "bias": model.fc1.bias.detach()         # Shape: (64,)
    })
    
    # FC2
    params["linear"].append({
        "weight": model.fc2.weight.detach(),    # Shape: (7, 64)
        "bias": model.fc2.bias.detach()         # Shape: (7,)
    })
    
    return params


__all__ = ["FHEEmotionCNN", "Square", "extract_fhe_parameters"]
