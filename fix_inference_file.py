
from pathlib import Path

content = r'''"""End-to-end TenSEAL inference demo for the FHE-friendly CNN."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
import tenseal as ts

from he.tenseal_context import create_context, DEFAULT_GLOBAL_SCALE
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

    def square(self, cipher: EncryptedScalar) -> EncryptedScalar:
        return cipher.square()


class EncryptedCNNRunner:
    """
    Scalar-based Encrypted CNN Runner.
    Slow but easy to understand. Uses one ciphertext per pixel.
    """

    def __init__(self, context: ts.Context, params: Dict[str, List[Dict[str, torch.Tensor]]]) -> None:
        self.context = context
        self.ops = EncryptedOps(context)
        self.conv_params = params["conv"]
        self.linear_params = params["linear"]

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

    def conv2d(self, feature_map: FeatureMap, weight: torch.Tensor, bias: torch.Tensor | None, stride: int = 1) -> FeatureMap:
        in_channels = len(feature_map)
        height = len(feature_map[0])
        width = len(feature_map[0][0])
        out_channels = weight.shape[0]
        kernel_size = weight.shape[-1]
        
        # Output dimensions
        out_height = (height - kernel_size) // stride + 1
        out_width = (width - kernel_size) // stride + 1
        
        outputs: FeatureMap = []
        for out_idx in range(out_channels):
            channel_rows: List[List[EncryptedScalar]] = []
            for y in range(out_height):
                row: List[EncryptedScalar] = []
                for x in range(out_width):
                    # Calculate receptive field top-left
                    in_y = y * stride
                    in_x = x * stride
                    
                    acc = self.ops.zero()
                    for in_idx in range(in_channels):
                        for ky in range(kernel_size):
                            for kx in range(kernel_size):
                                coeff = float(weight[out_idx, in_idx, ky, kx].item())
                                if abs(coeff) < 1e-9:
                                    continue
                                pixel = feature_map[in_idx][in_y + ky][in_x + kx]
                                acc = acc + (pixel * coeff)
                    if bias is not None:
                        acc = acc + float(bias[out_idx].item())
                    row.append(acc)
                channel_rows.append(row)
            outputs.append(channel_rows)
        return outputs

    def square_map(self, feature_map: FeatureMap) -> FeatureMap:
        for channel in feature_map:
            for y in range(len(channel)):
                for x in range(len(channel[y])):
                    channel[y][x] = self.ops.square(channel[y][x])
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

    def square_vector(self, values: List[EncryptedScalar]) -> List[EncryptedScalar]:
        return [self.ops.square(v) for v in values]

    def forward(self, tensor: torch.Tensor) -> List[EncryptedScalar]:
        # Conv1: kernel=7, stride=3
        fmap = self.encrypt_image(tensor)
        LOGGER.info("Encrypt -> Conv1")
        fmap = self.conv2d(fmap, self.conv_params[0]["weight"], self.conv_params[0]["bias"], stride=3)
        fmap = self.square_map(fmap)
        
        flat = self.flatten(fmap)
        LOGGER.info("Flatten -> FC1")
        vec = self.linear(flat, self.linear_params[0]["weight"], self.linear_params[0]["bias"])
        vec = self.square_vector(vec)
        
        LOGGER.info("FC1 -> FC2")
        logits = self.linear(vec, self.linear_params[1]["weight"], self.linear_params[1]["bias"])
        return logits


class PackedEncryptedCNNRunner:
    """
    Packed Encrypted CNN Runner using TenSEAL's im2col and pack_vectors.
    Follows Tutorial 4 structure.
    """

    def __init__(
        self,
        context: ts.Context,
        params: Dict[str, List[Dict[str, torch.Tensor]]],
        *,
        log_steps: bool = True,
    ) -> None:
        self.context = context
        self.conv1_weight = params["conv"][0]["weight"].tolist() # (out, in, k, k)
        self.conv1_bias = params["conv"][0]["bias"].tolist()
        
        # Linear weights need to be transposed for .mm() (input_size, output_size)
        self.fc1_weight = params["linear"][0]["weight"].T.tolist()
        self.fc1_bias = params["linear"][0]["bias"].tolist()
        
        self.fc2_weight = params["linear"][1]["weight"].T.tolist()
        self.fc2_bias = params["linear"][1]["bias"].tolist()
        
        self._log_steps = log_steps

    def forward(self, tensor: torch.Tensor) -> ts.CKKSVector:
        # 1. im2col encoding
        # tensor shape: (1, 48, 48)
        # Conv1: kernel=7, stride=3
        kernel_shape = (7, 7)
        stride = 3
        
        image_list = tensor.view(48, 48).tolist()
        
        if self._log_steps: LOGGER.info("▶ im2col Encoding")
        enc_x, windows_nb = ts.im2col_encoding(
            self.context, image_list, kernel_shape[0], kernel_shape[1], stride
        )
        
        # 2. Conv1
        if self._log_steps: LOGGER.info("▶ Packed Conv1")
        enc_channels = []
        # self.conv1_weight is list of kernels [out_channel][in_channel][k][k]
        # Since in_channel is 1, we iterate over out_channels
        for kernel, bias in zip(self.conv1_weight, self.conv1_bias):
            # kernel is [1, 7, 7] list. conv2d_im2col expects [7, 7] if single channel?
            # Actually conv2d_im2col expects a single kernel window.
            # Since input is 1 channel, kernel[0] is the 7x7 matrix.
            k_flat = kernel[0] 
            y = enc_x.conv2d_im2col(k_flat, windows_nb) + bias
            enc_channels.append(y)
            
        # 3. Pack (Flatten)
        if self._log_steps: LOGGER.info("▶ Packing Channels")
        enc_x = ts.CKKSVector.pack_vectors(enc_channels)
        
        # 4. Square
        if self._log_steps: LOGGER.info("▶ Square Activation 1")
        enc_x.square_()
        
        # 5. FC1
        if self._log_steps: LOGGER.info("▶ FC1")
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias
        
        # 6. Square
        if self._log_steps: LOGGER.info("▶ Square Activation 2")
        enc_x.square_()
        
        # 7. FC2
        if self._log_steps: LOGGER.info("▶ FC2")
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias
        
        return enc_x


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


def decrypt_logits(logits: Sequence[EncryptedScalar] | ts.CKKSVector) -> np.ndarray:
    if isinstance(logits, ts.CKKSVector):
        return np.asarray(logits.decrypt())
    
    values = []
    for idx, cipher in enumerate(logits):
        plaintext = cipher.decrypt()
        values.append(float(plaintext[0]))
    return np.asarray(values)


def encrypted_inference_demo(context: ts.Context | None = None, sample_index: int = 0, use_packed: bool = True) -> Dict[str, Any]:
    model, stats = load_plain_model()
    params = extract_fhe_parameters(model)
    context = context or create_context()
    
    if use_packed:
        LOGGER.info("Using packed TenSEAL runner for inference")
        runner = PackedEncryptedCNNRunner(context, params)
    else:
        LOGGER.info("Using scalar TenSEAL runner for inference (fallback)")
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
    
    # If packed, decrypted_logits might be larger than num_classes due to packing slots?
    # No, FC2 output size is num_classes. But CKKSVector might have more slots.
    # We should slice it to num_classes.
    num_classes = plain_logits.shape[0]
    decrypted_logits = decrypted_logits[:num_classes]
    
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
'''

target_path = Path("/Users/hyunwookkim/Documents/study/25y2s/보안프로젝트설계/프로젝트/프로토타입/learning_test_01/fhe_emotion/he/fhe_inference.py")
target_path.write_text(content, encoding="utf-8")
print(f"Successfully overwrote {target_path}")
