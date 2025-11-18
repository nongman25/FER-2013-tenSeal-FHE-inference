"""End-to-end TenSEAL inference demo for the FHE-friendly CNN."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import tenseal as ts

from he.tenseal_context import create_context
from models.fhe_cnn import FHEEmotionCNN, extract_fhe_parameters

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "fhe_cnn_fer2013.pt"
NORM_STATS_PATH = PROJECT_ROOT / "models" / "normalization_stats.json"

EncryptedScalar = ts.CKKSVector
FeatureMap = List[List[List[EncryptedScalar]]]  # channel -> row -> col


@dataclass
class NormalizationStats:
    mean: float = 0.0
    std: float = 1.0

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / max(self.std, 1e-6)


class EncryptedOps:
    """Helper factory for frequently used encrypted primitives."""

    def __init__(self, context: ts.Context) -> None:
        self.context = context

    def zero(self) -> EncryptedScalar:
        return ts.ckks_vector(self.context, [0.0])

    def encrypt_scalar(self, value: float) -> EncryptedScalar:
        return ts.ckks_vector(self.context, [float(value)])

    def poly_act(self, cipher: EncryptedScalar, a: float, b: float) -> EncryptedScalar:
        cubic = cipher * cipher * cipher
        return cipher * a + cubic * b


class EncryptedCNNRunner:
    """Mirror the PyTorch CNN using scalar CKKS ciphertexts."""

    def __init__(self, context: ts.Context, params: Dict[str, List[Dict[str, torch.Tensor]]], poly_a: float = 0.8, poly_b: float = 0.2) -> None:
        self.context = context
        self.ops = EncryptedOps(context)
        self.conv_params = params["conv"]
        self.linear_params = params["linear"]
        self.poly_a = poly_a
        self.poly_b = poly_b

    def encrypt_image(self, tensor: torch.Tensor) -> FeatureMap:
        assert tensor.ndim == 3 and tensor.shape[0] == 1, "Expected (1, H, W) tensor"
        _, height, width = tensor.shape
        channel = []
        for y in range(height):
            row: List[EncryptedScalar] = []
            for x in range(width):
                row.append(self.ops.encrypt_scalar(float(tensor[0, y, x].item())))
            channel.append(row)
        return [channel]

    def conv2d(self, feature_map: FeatureMap, weight: torch.Tensor, bias: torch.Tensor | None, padding: int = 1) -> FeatureMap:
        in_channels = len(feature_map)
        height = len(feature_map[0])
        width = len(feature_map[0][0])
        kernel_size = weight.shape[-1]
        padded = self._pad_feature_map(feature_map, padding)
        outputs: FeatureMap = []
        for out_idx in range(weight.shape[0]):
            channel_rows: List[List[EncryptedScalar]] = []
            for y in range(height):
                row: List[EncryptedScalar] = []
                for x in range(width):
                    acc = self.ops.zero()
                    for in_idx in range(in_channels):
                        for ky in range(kernel_size):
                            for kx in range(kernel_size):
                                coeff = float(weight[out_idx, in_idx, ky, kx].item())
                                if abs(coeff) < 1e-9:
                                    continue
                                pixel = padded[in_idx][y + ky][x + kx]
                                acc = acc + (pixel * coeff)
                    if bias is not None:
                        acc = acc + float(bias[out_idx].item())
                    row.append(acc)
                channel_rows.append(row)
            outputs.append(channel_rows)
        return outputs

    def avg_pool2d(self, feature_map: FeatureMap, kernel_size: int = 2, stride: int = 2) -> FeatureMap:
        outputs: FeatureMap = []
        factor = 1.0 / float(kernel_size * kernel_size)
        for channel in feature_map:
            pooled_rows: List[List[EncryptedScalar]] = []
            height = len(channel)
            width = len(channel[0])
            for y in range(0, height - kernel_size + 1, stride):
                row: List[EncryptedScalar] = []
                for x in range(0, width - kernel_size + 1, stride):
                    acc = self.ops.zero()
                    for ky in range(kernel_size):
                        for kx in range(kernel_size):
                            acc = acc + channel[y + ky][x + kx]
                    row.append(acc * factor)
                pooled_rows.append(row)
            outputs.append(pooled_rows)
        return outputs

    def poly_act_map(self, feature_map: FeatureMap) -> FeatureMap:
        for channel in feature_map:
            for y in range(len(channel)):
                for x in range(len(channel[y])):
                    channel[y][x] = self.ops.poly_act(channel[y][x], self.poly_a, self.poly_b)
        return feature_map

    def flatten(self, feature_map: FeatureMap) -> List[EncryptedScalar]:
        flat: List[EncryptedScalar] = []
        for channel in feature_map:
            for row in channel:
                flat.extend(row)
        return flat

    def linear(self, inputs: Sequence[EncryptedScalar], weight: torch.Tensor, bias: torch.Tensor | None) -> List[EncryptedScalar]:
        outputs: List[EncryptedScalar] = []
        for out_idx in range(weight.shape[0]):
            acc = self.ops.zero()
            for in_idx, enc_value in enumerate(inputs):
                coeff = float(weight[out_idx, in_idx].item())
                if abs(coeff) < 1e-9:
                    continue
                acc = acc + (enc_value * coeff)
            if bias is not None:
                acc = acc + float(bias[out_idx].item())
            outputs.append(acc)
        return outputs

    def poly_act_vector(self, values: List[EncryptedScalar]) -> List[EncryptedScalar]:
        return [self.ops.poly_act(v, self.poly_a, self.poly_b) for v in values]

    def forward(self, tensor: torch.Tensor) -> List[EncryptedScalar]:
        fmap = self.encrypt_image(tensor)
        LOGGER.info("Encrypt -> Conv1")
        fmap = self.conv2d(fmap, self.conv_params[0]["weight"], self.conv_params[0]["bias"], padding=1)
        fmap = self.poly_act_map(fmap)
        fmap = self.avg_pool2d(fmap, kernel_size=2, stride=2)
        LOGGER.info("Conv1 done -> Conv2")
        fmap = self.conv2d(fmap, self.conv_params[1]["weight"], self.conv_params[1]["bias"], padding=1)
        fmap = self.poly_act_map(fmap)
        fmap = self.avg_pool2d(fmap, kernel_size=2, stride=2)
        flat = self.flatten(fmap)
        LOGGER.info("Flatten -> FC1")
        vec = self.linear(flat, self.linear_params[0]["weight"], self.linear_params[0]["bias"])
        vec = self.poly_act_vector(vec)
        LOGGER.info("FC1 -> FC2")
        logits = self.linear(vec, self.linear_params[1]["weight"], self.linear_params[1]["bias"])
        return logits

    def _pad_feature_map(self, feature_map: FeatureMap, padding: int) -> FeatureMap:
        if padding == 0:
            return feature_map
        padded: FeatureMap = []
        for channel in feature_map:
            height = len(channel)
            width = len(channel[0])
            full_width = width + padding * 2
            new_rows: List[List[EncryptedScalar]] = []
            for _ in range(padding):
                new_rows.append([self.ops.zero() for _ in range(full_width)])
            for row in channel:
                new_row = [self.ops.zero() for _ in range(padding)] + row + [self.ops.zero() for _ in range(padding)]
                new_rows.append(new_row)
            for _ in range(padding):
                new_rows.append([self.ops.zero() for _ in range(full_width)])
            padded.append(new_rows)
        return padded


def load_plain_model(device: torch.device | None = None) -> Tuple[FHEEmotionCNN, NormalizationStats]:
    device = device or torch.device("cpu")
    model = FHEEmotionCNN()
    model.to(device)
    if MODEL_PATH.exists():
        LOGGER.info("Loading model weights from %s", MODEL_PATH)
        state = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state)
    else:
        LOGGER.warning("Model weights not found at %s; using randomly initialized model", MODEL_PATH)
    model.eval()
    stats = NormalizationStats()
    if NORM_STATS_PATH.exists():
        stats = NormalizationStats(**json.loads(NORM_STATS_PATH.read_text()))
    else:
        LOGGER.warning("Normalization stats not found at %s; defaulting to mean=0,std=1", NORM_STATS_PATH)
    return model, stats


def _load_split_tensors(split: str) -> Tuple[torch.Tensor, torch.Tensor]:
    images_path = DATA_DIR / f"{split}_images.pt"
    labels_path = DATA_DIR / f"{split}_labels.pt"
    if not images_path.exists():
        raise FileNotFoundError(f"Missing tensor at {images_path}. Run the data prep notebook first.")
    images = torch.load(images_path)
    labels = torch.load(labels_path)
    return images, labels


def decrypt_logits(logits: Sequence[EncryptedScalar]) -> np.ndarray:
    values = []
    for idx, cipher in enumerate(logits):
        plaintext = cipher.decrypt()
        values.append(float(plaintext[0]))
        LOGGER.debug("Logit %d decrypted value: %.4f", idx, values[-1])
    return np.asarray(values)


def encrypted_inference_demo(context: ts.Context | None = None, sample_index: int = 0) -> Dict[str, np.ndarray]:
    model, stats = load_plain_model()
    params = extract_fhe_parameters(model)
    context = context or create_context()
    runner = EncryptedCNNRunner(context, params)

    images, labels = _load_split_tensors("test")
    total_samples = images.shape[0]
    sample_index = int(np.clip(sample_index, 0, total_samples - 1))

    image = images[sample_index].float()
    label = int(labels[sample_index].item())
    normalized = stats.normalize(image)

    with torch.no_grad():
        plain_logits = model(normalized.unsqueeze(0)).squeeze(0)
        plain_probs = F.softmax(plain_logits, dim=-1)
    LOGGER.info("Plain inference complete")

    encrypted_logits = runner.forward(normalized)
    decrypted_logits = decrypt_logits(encrypted_logits)
    encrypted_probs = F.softmax(torch.from_numpy(decrypted_logits), dim=0).numpy()

    plain_pred = int(torch.argmax(plain_probs).item())
    enc_pred = int(np.argmax(encrypted_probs))

    LOGGER.info("Sample %d | True label: %d", sample_index, label)
    LOGGER.info("Plain logits: %s", np.array2string(plain_logits.numpy(), precision=3))
    LOGGER.info("Encrypted logits (decrypted): %s", np.array2string(decrypted_logits, precision=3))
    LOGGER.info("Plain probs: %s", np.array2string(plain_probs.numpy(), precision=3))
    LOGGER.info("Encrypted probs: %s", np.array2string(encrypted_probs, precision=3))
    LOGGER.info("Plain pred: %d | Encrypted pred: %d", plain_pred, enc_pred)

    return {
        "plain_logits": plain_logits.numpy(),
        "plain_probs": plain_probs.numpy(),
        "encrypted_logits": decrypted_logits,
        "encrypted_probs": encrypted_probs,
        "plain_pred": plain_pred,
        "encrypted_pred": enc_pred,
        "true_label": label,
    }


def main() -> None:
    encrypted_inference_demo()


if __name__ == "__main__":
    main()
