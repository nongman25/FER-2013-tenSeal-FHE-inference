"""End-to-end TenSEAL inference demo for the FHE-friendly CNN."""
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

from __future__ import annotations

import json
import logging
import time
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
class PackedFeatureMap:
    """한 개의 CKKS 벡터에 채널별 특징 지도를 평탄화해 저장한다."""

    ciphers: List[ts.CKKSVector]
    height: int
    width: int

    @property
    def channels(self) -> int:
        return len(self.ciphers)

    @property
    def slots_per_channel(self) -> int:
        return self.height * self.width


@dataclass
class PackedVector:
    """완전히 평탄화 된 벡터 표현 (FC 계층 입력/출력에 사용)."""

    segments: List[ts.CKKSVector]
    segment_length: int

    def total_length(self) -> int:
        return len(self.segments) * self.segment_length


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
    """Mirror the PyTorch CNN using scalar CKKS ciphertexts."""

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

    def poly_act_vector(self, values: List[EncryptedScalar]) -> List[EncryptedScalar]:
        return [self.ops.square(v) for v in values]

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


class PackedEncryptedCNNRunner:
    """채널 단위로 패킹한 CKKS 벡터를 이용해 회전 기반 합성곱을 수행한다."""

    def __init__(
        self,
        context: ts.Context,
        params: Dict[str, List[Dict[str, torch.Tensor]]],
        *,
        log_steps: bool = True,
        warn_over_seconds: float | None = 180.0,
    ) -> None:
        self.context = context
        self.conv_params = params["conv"]
        self.linear_params = params["linear"]
        self._shift_cache: Dict[Tuple[int, int, int, int], ts.PlainTensor] = {}
        self._pool_matrix_cache: Dict[Tuple[int, int, int], ts.PlainTensor] = {}
        self._log_steps = log_steps
        self._warn_over_seconds = warn_over_seconds

    def encrypt_image_packed(self, tensor: torch.Tensor) -> PackedFeatureMap:
        assert tensor.ndim == 3 and tensor.shape[0] == 1, "Expected (1, H, W) tensor"
        _, height, width = tensor.shape
        flat = tensor.reshape(-1).tolist()
        cipher = ts.ckks_vector(self.context, flat, scale=self.context.global_scale)
        return PackedFeatureMap([cipher], height, width)

    def conv2d(self, fmap: PackedFeatureMap, weight: torch.Tensor, bias: torch.Tensor | None, padding: int = 1) -> PackedFeatureMap:
        outputs: List[ts.CKKSVector] = []
        height, width = fmap.height, fmap.width
        for out_idx in range(weight.shape[0]):
            if self._log_steps:
                LOGGER.info("    • 출력 채널 %d/%d 누적 시작", out_idx + 1, weight.shape[0])
            acc = self._zero_vector(fmap.slots_per_channel)
            for in_idx, cipher in enumerate(fmap.ciphers):
                kernel = weight[out_idx, in_idx]
                if self._log_steps:
                    LOGGER.info("        - 입력 채널 %d/%d 처리", in_idx + 1, len(fmap.ciphers))
                acc = self._accumulate_conv(acc, cipher, kernel, height, width, padding)
            if bias is not None:
                acc = acc + float(bias[out_idx].item())
            outputs.append(acc)
        return PackedFeatureMap(outputs, height, width)

    def avg_pool2d(self, fmap: PackedFeatureMap, kernel_size: int = 2, stride: int = 2) -> PackedFeatureMap:
        new_height = fmap.height // stride
        new_width = fmap.width // stride
        matrix = self._get_pool_matrix(fmap.height, fmap.width, stride)
        pooled = [cipher.mm(matrix) for cipher in fmap.ciphers]
        return PackedFeatureMap(pooled, new_height, new_width)

    def poly_act_map(self, fmap: PackedFeatureMap) -> PackedFeatureMap:
        activated = [self._poly_act(cipher) for cipher in fmap.ciphers]
        return PackedFeatureMap(activated, fmap.height, fmap.width)

    def flatten(self, fmap: PackedFeatureMap) -> PackedVector:
        return PackedVector(fmap.ciphers, fmap.slots_per_channel)

    def linear(self, vector: PackedVector, weight: torch.Tensor, bias: torch.Tensor | None) -> PackedVector:
        outputs: List[ts.CKKSVector] = []
        seg_len = vector.segment_length
        total_inputs = vector.total_length()
        assert weight.shape[1] == total_inputs, "Linear input dimension mismatch"
        for row_idx in range(weight.shape[0]):
            row = weight[row_idx]
            acc = self._zero_scalar()
            offset = 0
            for segment in vector.segments:
                coeffs = row[offset : offset + seg_len].detach().cpu().numpy()
                contrib = segment.dot(coeffs)
                acc = acc + contrib
                offset += seg_len
            if bias is not None:
                acc = acc + float(bias[row_idx].item())
            outputs.append(acc)
        return PackedVector(outputs, 1)

    def poly_act_vector(self, vector: PackedVector) -> PackedVector:
        activated = [self._poly_act(segment) for segment in vector.segments]
        return PackedVector(activated, vector.segment_length)

    def forward(self, tensor: torch.Tensor) -> List[ts.CKKSVector]:
        fmap = self.encrypt_image_packed(tensor)
        fmap = self._timed("Packed Conv1", lambda: self.conv2d(fmap, self.conv_params[0]["weight"], self.conv_params[0]["bias"], padding=1))
        fmap = self._timed("Packed PolyAct1", lambda: self.poly_act_map(fmap))
        fmap = self._timed("Packed AvgPool1", lambda: self.avg_pool2d(fmap, kernel_size=2, stride=2))
        fmap = self._timed("Packed Conv2", lambda: self.conv2d(fmap, self.conv_params[1]["weight"], self.conv_params[1]["bias"], padding=1))
        fmap = self._timed("Packed PolyAct2", lambda: self.poly_act_map(fmap))
        fmap = self._timed("Packed AvgPool2", lambda: self.avg_pool2d(fmap, kernel_size=2, stride=2))
        vector = self._timed("Flatten", lambda: self.flatten(fmap))
        vector = self._timed("FC1", lambda: self.linear(vector, self.linear_params[0]["weight"], self.linear_params[0]["bias"]))
        vector = self._timed("PolyAct FC1", lambda: self.poly_act_vector(vector))
        vector = self._timed("FC2", lambda: self.linear(vector, self.linear_params[1]["weight"], self.linear_params[1]["bias"]))
        return vector.segments

    def _accumulate_conv(self, acc: ts.CKKSVector, cipher: ts.CKKSVector, kernel: torch.Tensor, height: int, width: int, padding: int) -> ts.CKKSVector:
        kernel_size = kernel.shape[0]
        for ky in range(kernel_size):
            if self._log_steps:
                LOGGER.debug("            ky=%d/%d", ky + 1, kernel_size)
            for kx in range(kernel_size):
                coeff = float(kernel[ky, kx].item())
                if abs(coeff) < 1e-9:
                    continue
                dy = ky - padding
                dx = kx - padding
                shifted = cipher.mm(self._get_shift_matrix(height, width, dy, dx))
                acc = acc + (shifted * coeff)
        return acc

    def _poly_act(self, cipher: ts.CKKSVector) -> ts.CKKSVector:
        return cipher.square()

    def _zero_vector(self, length: int) -> ts.CKKSVector:
        return ts.ckks_vector(self.context, [0.0] * length, scale=self.context.global_scale)

    def _zero_scalar(self) -> ts.CKKSVector:
        return ts.ckks_vector(self.context, [0.0], scale=self.context.global_scale)

    def _get_shift_matrix(self, height: int, width: int, dy: int, dx: int) -> ts.PlainTensor:
        key = (height, width, dy, dx)
        if key not in self._shift_cache:
            size = height * width
            # TenSEAL mm expects (input_size, output_size) - transpose needed
            matrix = np.zeros((size, size), dtype=np.float64)
            for y in range(height):
                for x in range(width):
                    src_y = y - dy  # source position (reverse the shift)
                    src_x = x - dx
                    src_idx = y * width + x
                    # Only copy if source is within bounds
                    if 0 <= src_y < height and 0 <= src_x < width:
                        dest_idx = src_y * width + src_x
                        matrix[dest_idx, src_idx] = 1.0
            self._shift_cache[key] = ts.plain_tensor(matrix)
        return self._shift_cache[key]

    def _get_pool_matrix(self, height: int, width: int, stride: int) -> ts.PlainTensor:
        key = (height, width, stride)
        if key not in self._pool_matrix_cache:
            new_height = height // stride
            new_width = width // stride
            matrix = np.zeros((height * width, new_height * new_width), dtype=np.float64)
            scale = 1.0 / (stride * stride)
            for y in range(new_height):
                for x in range(new_width):
                    col = y * new_width + x
                    for ky in range(stride):
                        for kx in range(stride):
                            src_y = y * stride + ky
                            src_x = x * stride + kx
                            row = src_y * width + src_x
                            matrix[row, col] = scale
            self._pool_matrix_cache[key] = ts.plain_tensor(matrix)
        return self._pool_matrix_cache[key]

    def _timed(self, name: str, fn):
        if not self._log_steps:
            return fn()
        LOGGER.info("▶ %s 시작", name)
        start = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - start
        LOGGER.info("✓ %s 완료 (%.2f초)", name, elapsed)
        if self._warn_over_seconds and elapsed > self._warn_over_seconds:
            LOGGER.warning("%s 단계가 %.2f초 이상 소요되었습니다. 매개변수(tensor 크기/컨텍스트)를 줄이거나 use_packed=False로 비교해 보세요.", name, elapsed)
        return result


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


def encrypted_inference_demo(context: ts.Context | None = None, sample_index: int = 0, use_packed: bool = True) -> Dict[str, np.ndarray]:
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
