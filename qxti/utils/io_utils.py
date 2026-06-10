from __future__ import annotations

from pathlib import Path

import numpy as np


def normalize_complex_storage_dtype(value: str | np.dtype | type[np.complex64] | type[np.complex128]) -> np.dtype:
    dtype = np.dtype(value)
    if dtype not in {np.dtype(np.complex64), np.dtype(np.complex128)}:
        raise ValueError("rho_storage_dtype must be either complex64 or complex128.")
    return dtype


def open_array_npy(
    output_path: str | Path,
    *,
    shape: tuple[int, ...],
    dtype: str | np.dtype | type[np.complex64] | type[np.complex128] | type[np.float32] | type[np.float64],
    mode: str = "w+",
) -> np.memmap:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode=mode, dtype=np.dtype(dtype), shape=shape)


def save_array_npy(
    output_path: str | Path,
    array: np.ndarray,
    *,
    dtype: str | np.dtype | None = None,
) -> Path:
    values = np.asarray(array)
    target_dtype = values.dtype if dtype is None else np.dtype(dtype)
    writer = open_array_npy(output_path, shape=values.shape, dtype=target_dtype, mode="w+")
    writer[...] = values
    writer.flush()
    return Path(output_path)
