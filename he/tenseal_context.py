"""Utilities for building and managing TenSEAL CKKS contexts."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import tenseal as ts


# TenSEAL Tutorial 4 parameters
# Security level 128-bits with poly_modulus_degree=8192
# Scale = 2^26
# Updated to 32768 to accommodate FER2013 input size (48x48) with im2col encoding
DEFAULT_POLY_MODULUS_DEGREE = 32768
# [31, 26, 26, 26, 26, 26, 26, 31]
DEFAULT_COEFF_MOD_BIT_SIZES = (31, 26, 26, 26, 26, 26, 26, 31)
DEFAULT_GLOBAL_SCALE = 2**26


def create_context(
    poly_modulus_degree: int = DEFAULT_POLY_MODULUS_DEGREE,
    coeff_mod_bit_sizes: Sequence[int] = DEFAULT_COEFF_MOD_BIT_SIZES,
    global_scale: float = DEFAULT_GLOBAL_SCALE,
) -> ts.Context:
    """Instantiate a CKKS context with keys for rotations and relinearization."""
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=list(coeff_mod_bit_sizes),
    )
    context.global_scale = global_scale
    context.generate_galois_keys()
    context.generate_relin_keys()
    return context


def encrypt_vector(context: ts.Context, values: Iterable[float] | np.ndarray) -> ts.CKKSVector:
    """Encrypt a flat vector of floats using CKKS."""
    if isinstance(values, np.ndarray):
        values = values.flatten().tolist()
    return ts.ckks_vector(context, list(values))


def decrypt_vector(context: ts.Context, ciphertext: ts.CKKSVector) -> np.ndarray:
    """Decrypt a ciphertext back into a NumPy array."""
    return np.asarray(ciphertext.decrypt())


def save_context(context: ts.Context, path: Path, *, include_secret_key: bool = True) -> None:
    """Persist the context to disk for reuse."""
    path = Path(path)
    if include_secret_key:
        path.write_bytes(context.serialize())
    else:
        path.write_bytes(context.serialize(save_public_key=True, save_secret_key=False, save_galois_keys=True, save_relin_keys=True))


def load_context(path: Path) -> ts.Context:
    """Load a previously serialized TenSEAL context."""
    data = Path(path).read_bytes()
    return ts.context_from(data)


__all__ = [
    "create_context",
    "encrypt_vector",
    "decrypt_vector",
    "save_context",
    "load_context",
    "DEFAULT_POLY_MODULUS_DEGREE",
    "DEFAULT_COEFF_MOD_BIT_SIZES",
    "DEFAULT_GLOBAL_SCALE",
]
