from __future__ import annotations

import json
from pathlib import Path
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
