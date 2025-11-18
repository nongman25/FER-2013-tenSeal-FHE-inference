"""FHE-friendly CNN architecture and helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import Tensor, nn


class PolyAct(nn.Module):
    """Low-degree polynomial activation for homomorphic evaluation."""

    def __init__(self, a: float = 0.8, b: float = 0.2) -> None:
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        """Apply `a * x + b * x^3`."""
        cubic = x * x * x
        return self.a * x + self.b * cubic


class FHEEmotionCNN(nn.Module):
    """Shallow CNN that only uses HE-friendly primitives."""

    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            PolyAct(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            PolyAct(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 12 * 12, 128),
            PolyAct(),
            nn.Linear(128, num_classes),
        )

        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, a=0.2)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=0.2)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        logits = self.classifier(x)
        return logits

    def fhe_parameters(self) -> Dict[str, List[Dict[str, Tensor]]]:
        """Return weights in a TenSEAL-friendly structure."""
        return extract_fhe_parameters(self)


@dataclass
class LayerParameters:
    """Simple container for serialized layer parameters."""

    weight: Tensor
    bias: Tensor
    layer_type: str

    def as_dict(self) -> Dict[str, Tensor]:
        return {"type": self.layer_type, "weight": self.weight.detach().clone(), "bias": self.bias.detach().clone()}


def extract_fhe_parameters(model: nn.Module) -> Dict[str, List[Dict[str, Tensor]]]:
    """Create a structured parameter list for HE evaluation."""
    params: Dict[str, List[Dict[str, Tensor]]] = {"conv": [], "linear": []}
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            params["conv"].append(LayerParameters(module.weight, module.bias, "conv").as_dict())
        elif isinstance(module, nn.Linear):
            params["linear"].append(LayerParameters(module.weight, module.bias, "linear").as_dict())
    return params


__all__ = ["PolyAct", "FHEEmotionCNN", "extract_fhe_parameters"]
