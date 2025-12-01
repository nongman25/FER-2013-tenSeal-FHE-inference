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
    Wide 1-Conv FHE-friendly CNN without Batch Normalization.
    Optimized for TenSEAL Packed Inference (im2col) - Fast inference variant.
    Structure: Conv(16ch, k12, s12) -> Square -> FC(128) -> Square -> FC
    """

    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        # Input: 48x48
        
        # Conv1: 1 -> 16 channels, kernel 12x12, stride 12
        # Output: (48 - 12) // 12 + 1 = 4. Shape: 16 x 4 x 4
        # Total slots needed: 16 * 16 = 256 (Very efficient! Fits easily in 16384 slots)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=12, stride=12, padding=0)
        self.act1 = Square()
        
        # Flatten size: 16 * 4 * 4 = 256
        self.fc1 = nn.Linear(16 * 4 * 4, 128)
        self.act2 = Square()
        
        self.fc2 = nn.Linear(128, num_classes)

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
        "weight": model.conv1.weight.detach(),
        "bias": model.conv1.bias.detach()
    })
    
    # FC1
    params["linear"].append({
        "weight": model.fc1.weight.detach(),
        "bias": model.fc1.bias.detach()
    })
    
    # FC2
    params["linear"].append({
        "weight": model.fc2.weight.detach(),
        "bias": model.fc2.bias.detach()
    })
    
    return params


__all__ = ["FHEEmotionCNN", "Square", "extract_fhe_parameters"]
