"""Utilities for building and managing TenSEAL CKKS contexts."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import tenseal as ts


# TenSEAL 튜토리얼에서 권장하는 4-레벨 체인(60,40,40,60 비트)을 기본값으로 사용한다.
DEFAULT_POLY_MODULUS_DEGREE = 8192
DEFAULT_COEFF_MOD_BIT_SIZES = (60, 40, 40, 60)
DEFAULT_GLOBAL_SCALE = 2**40
ALT_CHAINS = [
    (40, 21, 21, 40),
    (60, 30, 30, 30, 60),
]


def create_context(
    poly_modulus_degree: int = DEFAULT_POLY_MODULUS_DEGREE,
    coeff_mod_bit_sizes: Sequence[int] = DEFAULT_COEFF_MOD_BIT_SIZES,
    global_scale: float = DEFAULT_GLOBAL_SCALE,
) -> ts.Context:
    """Instantiate a CKKS context with keys for rotations and relinearization."""
    attempted = []
    chosen_chain = list(coeff_mod_bit_sizes)
    tried = []
    for chain in [chosen_chain, *ALT_CHAINS]:
        try:
            context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=poly_modulus_degree,
                coeff_mod_bit_sizes=list(chain),
            )
            chosen_chain = list(chain)
            break
        except ValueError as exc:
            tried.append((chain, exc))
    else:
        msg = "Failed to create CKKS context. Tried chains:\n"
        for chain, exc in tried:
            msg += f"  - {chain}: {exc}\n"
        raise ValueError(msg)
    context.global_scale = global_scale
    context.auto_rescale = True
    context.auto_mod_switch = True
    if hasattr(context, "auto_relin"):
        context.auto_relin = True
    context.generate_galois_keys()
    context.generate_relin_keys()
    return context


def encrypt_vector(context: ts.Context, values: Iterable[float]) -> ts.CKKSVector:
    """Encrypt a flat vector of floats using CKKS."""
    arr = np.asarray(list(values), dtype=np.float64)
    return ts.ckks_vector(context, arr)


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
