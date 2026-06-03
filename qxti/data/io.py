from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import numpy as np


def save_dataset_npz(output_path: str | Path, data: dict[str, Any]) -> Path:
    """Save one mixed metadata/array dataset into a compressed ``.npz`` file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            arrays[key] = np.asarray(value)
        else:
            metadata[key] = _jsonify(value)

    arrays["__meta_json__"] = np.asarray(json.dumps(metadata), dtype=str)
    np.savez_compressed(path, **arrays)
    return path


def load_dataset_npz(input_path: str | Path) -> dict[str, Any]:
    """Load one dataset previously saved with :func:`save_dataset_npz`."""

    path = Path(input_path)
    with np.load(path, allow_pickle=False) as archive:
        metadata: dict[str, Any] = {}
        if "__meta_json__" in archive.files:
            metadata = json.loads(str(archive["__meta_json__"].item()))

        result = dict(metadata)
        for key in archive.files:
            if key == "__meta_json__":
                continue
            result[key] = np.asarray(archive[key])
    return result


def load_rho_orders_from_npy(
    output_dir: str | Path,
    *,
    mmap_mode: str | None = None,
) -> dict[int, np.ndarray]:
    """Load every available ``rho_order_*.npy`` tensor from one output directory.

    Pass ``mmap_mode="r"`` to keep large tensors on disk and page them in only
    as needed.
    """

    directory = Path(output_dir)
    pattern = re.compile(r"rho_order_(\d+)\.npy$")
    rho_orders: dict[int, np.ndarray] = {}
    for path in sorted(directory.glob("rho_order_*.npy")):
        match = pattern.match(path.name)
        if match is None:
            continue
        tensor = np.load(path, mmap_mode=mmap_mode)
        if tensor.dtype == np.complex128:
            rho_orders[int(match.group(1))] = tensor
        else:
            rho_orders[int(match.group(1))] = np.asarray(tensor, dtype=np.complex128)
    return rho_orders


def load_rho_orders_from_dat(output_dir: str | Path) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """Load ``rho_order_*.dat`` tensors plus the saved k-points and time axis."""

    directory = Path(output_dir)
    pattern = re.compile(r"rho_order_(\d+)\.dat$")
    rho_orders: dict[int, np.ndarray] = {}
    reference_k_points: np.ndarray | None = None
    reference_time_axis: np.ndarray | None = None

    for path in sorted(directory.glob("rho_order_*.dat")):
        match = pattern.match(path.name)
        if match is None:
            continue
        order = int(match.group(1))
        tensor, k_points, time_axis = load_rho_order_dat(path)
        rho_orders[order] = tensor
        if reference_k_points is None:
            reference_k_points = k_points
            reference_time_axis = time_axis

    if reference_k_points is None or reference_time_axis is None:
        return {}, np.empty((0, 3), dtype=float), np.empty(0, dtype=float)
    return rho_orders, reference_k_points, reference_time_axis


def load_rho_order_dat(input_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse one flat ``rho_order_*.dat`` file into a tensor plus axes."""

    path = Path(input_path)
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[np.newaxis, :]

    ik = data[:, 0].astype(int)
    it = data[:, 1].astype(int)
    row = data[:, 6].astype(int)
    col = data[:, 7].astype(int)
    real_part = data[:, 8]
    imag_part = data[:, 9]

    nk = int(np.max(ik)) + 1
    nt = int(np.max(it)) + 1
    nb = max(int(np.max(row)), int(np.max(col))) + 1

    tensor = np.zeros((nk, nt, nb, nb), dtype=np.complex128)
    tensor[ik, it, row, col] = real_part + 1.0j * imag_part

    k_points = np.zeros((nk, 3), dtype=float)
    time_axis = np.zeros(nt, dtype=float)
    for k_index in range(nk):
        first = np.flatnonzero(ik == k_index)[0]
        k_points[k_index] = data[first, 2:5]
    for t_index in range(nt):
        first = np.flatnonzero(it == t_index)[0]
        time_axis[t_index] = float(data[first, 5])

    return tensor, k_points, time_axis


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonify(item) for item in value]
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    return value
