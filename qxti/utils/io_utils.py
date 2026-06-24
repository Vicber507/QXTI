from __future__ import annotations

from pathlib import Path

import numpy as np


# Sentinel dtype name used when the user sets rho_storage_dtype = "float16_complex".
# We store real and imaginary parts as two float16 values, giving 4 bytes/element
# instead of 8 (complex64) or 16 (complex128).  The solver tolerance is ~1e-3
# and float16 has machine epsilon ~1e-3, so no physically meaningful precision
# is lost.  The on-disk shape is (*original_shape, 2) float16 where [..., 0] is
# the real part and [..., 1] is the imaginary part.
_FLOAT16_COMPLEX_SENTINEL = "float16_complex"


def normalize_complex_storage_dtype(
    value: str | np.dtype | type[np.complex64] | type[np.complex128],
) -> np.dtype:
    """Return the numpy dtype for on-disk rho storage.

    Accepts ``"complex64"``, ``"complex128"``, or the special string
    ``"float16_complex"`` (4 bytes per complex element, half the size of
    complex64).  The float16_complex format stores real and imaginary parts as
    two float16 values per matrix element; on-disk shape is ``(*shape, 2)``.
    """
    if isinstance(value, str) and value.strip().lower() == _FLOAT16_COMPLEX_SENTINEL:
        return np.dtype("float16")  # sentinel; caller uses is_float16_complex_dtype()
    try:
        dtype = np.dtype(value)
    except TypeError:
        dtype = np.dtype(type(value))
    if dtype not in {np.dtype(np.complex64), np.dtype(np.complex128)}:
        raise ValueError(
            "rho_storage_dtype must be 'complex64', 'complex128', or 'float16_complex'."
        )
    return dtype


def is_float16_complex_dtype(dtype: np.dtype) -> bool:
    """Return True when the dtype was requested as ``float16_complex``."""
    return np.dtype(dtype) == np.dtype("float16")


def is_float16_complex_layout(array: np.ndarray) -> bool:
    """Return True when *array* uses the ``float16_complex`` on-disk layout."""
    return (
        isinstance(array, np.ndarray)
        and array.ndim >= 1
        and array.shape[-1] == 2
        and np.dtype(array.dtype) == np.dtype("float16")
    )


def open_array_npy(
    output_path: str | Path,
    *,
    shape: tuple[int, ...],
    dtype: str | np.dtype | type[np.complex64] | type[np.complex128] | type[np.float32] | type[np.float64],
    mode: str = "w+",
) -> np.memmap:
    """Open (or create) a numpy memory-mapped array at *output_path*.

    When *dtype* is ``float16`` (i.e. the caller passed
    ``rho_storage_dtype = "float16_complex"``), the on-disk shape is
    ``(*shape, 2)`` float16 where the last axis encodes (real, imag).
    The returned memmap has that extended shape and dtype — the caller is
    responsible for writing ``arr[ik, ..., 0] = real`` and
    ``arr[ik, ..., 1] = imag``.

    All other dtypes use the standard ``(*shape,)`` layout.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    disk_dtype = np.dtype(dtype)
    disk_shape = (*shape, 2) if is_float16_complex_dtype(disk_dtype) else shape
    return np.lib.format.open_memmap(path, mode=mode, dtype=disk_dtype, shape=disk_shape)


def write_complex_to_float16_memmap(
    dest: np.memmap,
    ik: int,
    data: np.ndarray,
) -> None:
    """Write one complex128 slice to a float16_complex memmap at row *ik*."""
    dest[ik, ..., 0] = np.real(data).astype(np.float16)
    dest[ik, ..., 1] = np.imag(data).astype(np.float16)


def read_float16_memmap_as_complex(arr: np.ndarray) -> np.ndarray:
    """Convert a float16_complex memmap/array to complex64.

    *arr* must have shape ``(*original_shape, 2)`` float16.
    Returns a complex64 array of shape ``original_shape``.
    """
    return (arr[..., 0].astype(np.float32) + 1j * arr[..., 1].astype(np.float32)).astype(np.complex64)


def read_complex_slice(arr: np.ndarray, key) -> np.ndarray:
    """Return one complex slice from either native-complex or float16_complex storage."""
    values = arr[key]
    if is_float16_complex_layout(values):
        return read_float16_memmap_as_complex(values)
    return np.asarray(values)


def save_array_npy(
    output_path: str | Path,
    array: np.ndarray,
    *,
    dtype: str | np.dtype | None = None,
) -> Path:
    values = np.asarray(array)
    target_dtype = values.dtype if dtype is None else np.dtype(dtype)
    writer = open_array_npy(output_path, shape=values.shape, dtype=target_dtype, mode="w+")
    if is_float16_complex_dtype(target_dtype):
        # array is (Nk, Nt, Nb, Nb) complex; writer is (Nk, Nt, Nb, Nb, 2) float16
        writer[..., 0] = np.real(values).astype(np.float16)
        writer[..., 1] = np.imag(values).astype(np.float16)
    else:
        writer[...] = values
    writer.flush()
    return Path(output_path)


def expand_rho_tensor_time_axis(
    tensor: np.ndarray,
    *,
    nt: int,
) -> np.ndarray:
    """Expand a compressed rho tensor to full ``(Nk, Nt, Nb, Nb)`` shape.

    Handles three on-disk layouts:

    * ``(Nk, Nt, Nb, Nb)`` complex — returned as-is.
    * ``(Nk, 1, Nb, Nb)`` complex — broadcast to ``(Nk, Nt, Nb, Nb)``
      (equilibrium constant-time representation).
    * ``(Nk, Nt, Nb, Nb, 2)`` float16 — converted to complex64 and returned
      (``float16_complex`` format).
    * ``(Nk, 1, Nb, Nb, 2)`` float16 — converted then broadcast.
    """
    values = tensor

    # Detect float16_complex layout: last axis has size 2 and dtype is float16
    is_f16 = is_float16_complex_layout(values)

    if is_f16:
        # Convert to complex64 first, then treat as normal (Nk, Nt_or_1, Nb, Nb)
        complex_values = read_float16_memmap_as_complex(values)
        # Now complex_values has shape (Nk, Nt_or_1, Nb, Nb)
        if complex_values.shape[1] == nt:
            return complex_values
        if complex_values.shape[1] == 1:
            return np.broadcast_to(
                complex_values,
                (complex_values.shape[0], nt, complex_values.shape[2], complex_values.shape[3]),
            )
        raise ValueError(
            f"float16_complex rho tensor has unexpected time axis size "
            f"{complex_values.shape[1]} (expected 1 or {nt})."
        )

    if values.ndim == 4:
        if values.shape[1] == nt:
            return values
        if values.shape[1] == 1:
            return np.broadcast_to(values, (values.shape[0], nt, values.shape[2], values.shape[3]))
    if values.ndim == 3:
        return np.broadcast_to(
            values[:, np.newaxis, :, :],
            (values.shape[0], nt, values.shape[1], values.shape[2]),
        )
    raise ValueError(
        f"rho tensor must have shape (Nk, Nt, Nb, Nb) or compact "
        f"(Nk, Nb, Nb)/(Nk, 1, Nb, Nb); got {values.shape}."
    )
