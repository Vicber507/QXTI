from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qxti.graphics import plot_hamiltonian_diagnostics
from qxti.physics import CustomHamiltonian

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


PREVIEW_LINE_PARAMS = {
    "k_extent": 0.18,
    "line_points": 401,
    "map_points": 121,
}
PREVIEW_DRIVEN_PARAMS = {
    "driven": False,
    "phi": 0.31,
}
PREVIEW_DRIVEN_SNAPSHOT = {
    "driven": True,
    "phi": 0.31,
    "t": 1.75,
}
PREVIEW_TIME_PARAMS = {
    "cycles": 1.5,
    "points": 260,
}


def build_surface_model(**kwargs) -> CustomHamiltonian:
    return CustomHamiltonian(source_file="bi2se3_surface.py", **kwargs)


def build_preview_time(
    hamiltonian: CustomHamiltonian,
    cycles: float = 1.5,
    points: int = 260,
) -> np.ndarray:
    period = 2.0 * np.pi / float(hamiltonian.params["wlaser"])
    return np.linspace(0.0, cycles * period, points)


def sample_time_energies(
    hamiltonian: CustomHamiltonian,
    time_values: np.ndarray,
    *,
    kx: float,
    ky: float,
    kz: float,
    extra_params: dict[str, object] | None = None,
) -> np.ndarray:
    defaults = hamiltonian.default_params()
    params = dict(defaults)
    params.update(hamiltonian.params)
    if extra_params:
        params.update(extra_params)

    energies = np.empty((len(time_values), hamiltonian.basis_size), dtype=float)
    for index, time in enumerate(time_values):
        params["t"] = float(time)
        hamiltonian.set_params(params)
        energies[index] = hamiltonian.eigenvalues(kx, ky, kz)
    return energies


def test_custom_hamiltonian_loads_model_metadata_and_defaults() -> None:
    hamiltonian = build_surface_model()
    matrix = hamiltonian.H(0.08, -0.11, 0.0)

    assert hamiltonian.model_name == "bi2se3-surface"
    assert hamiltonian.basis_size == 2
    assert hamiltonian.dimension == 2
    assert hamiltonian.basis_type == "spin"
    assert "a0" in hamiltonian.params
    assert "wlaser" in hamiltonian.params
    assert matrix.shape == (2, 2)
    assert hamiltonian.validate_hermiticity(0.08, -0.11, 0.0)


def test_custom_hamiltonian_accepts_source_file_without_py_suffix() -> None:
    hamiltonian = CustomHamiltonian(source_file="bi2se3_surface")

    assert hamiltonian.model_name == "bi2se3-surface"
    assert hamiltonian.H(0.0, 0.0, 0.0).shape == (2, 2)


def test_custom_hamiltonian_merges_user_params_with_model_defaults() -> None:
    hamiltonian = build_surface_model(params={"driven": False, "phi": 0.3})

    assert hamiltonian.params["driven"] is False
    assert np.isclose(hamiltonian.params["phi"], 0.3)
    assert "A14" in hamiltonian.params


def test_custom_hamiltonian_external_model_responds_to_parameter_changes() -> None:
    equilibrium = build_surface_model(params={"driven": False})
    driven = build_surface_model(params={"driven": True, "t": 1.75, "phi": 0.31})

    matrix_equilibrium = equilibrium.H(0.04, 0.11, 0.0)
    matrix_driven = driven.H(0.04, 0.11, 0.0)

    assert not np.allclose(matrix_equilibrium, matrix_driven)
    np.testing.assert_allclose(matrix_driven, matrix_driven.conj().T, atol=1.0e-12)


def test_custom_hamiltonian_rejects_missing_function() -> None:
    with pytest.raises(AttributeError, match="missing"):
        CustomHamiltonian(source_file="bi2se3_surface.py", function_name="missing")


def test_custom_hamiltonian_rejects_missing_source_file() -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        CustomHamiltonian(source_file="does_not_exist.py")


def test_custom_hamiltonian_preview_generation(tmp_path: Path) -> None:
    if not HAS_MATPLOTLIB:
        return

    equilibrium = build_surface_model(params={"driven": False})
    driven = build_surface_model(params=PREVIEW_DRIVEN_SNAPSHOT)
    driven_scan = build_surface_model(params=PREVIEW_DRIVEN_PARAMS)
    time_values = build_preview_time(driven_scan, **PREVIEW_TIME_PARAMS)
    time_energies = sample_time_energies(
        driven_scan,
        time_values,
        kx=0.05,
        ky=0.02,
        kz=0.0,
    )
    output_path = tmp_path / "custom_hamiltonian_preview.png"

    plot_hamiltonian_diagnostics(
        equilibrium,
        output_path,
        comparison_hamiltonian=driven,
        comparison_label="driven",
        time_values=time_values,
        time_energies=time_energies,
        time_label="t",
        **PREVIEW_LINE_PARAMS,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


if __name__ == "__main__":
    test_custom_hamiltonian_loads_model_metadata_and_defaults()
    test_custom_hamiltonian_accepts_source_file_without_py_suffix()
    test_custom_hamiltonian_merges_user_params_with_model_defaults()
    test_custom_hamiltonian_external_model_responds_to_parameter_changes()
    test_custom_hamiltonian_rejects_missing_function()
    test_custom_hamiltonian_rejects_missing_source_file()
    print("CustomHamiltonian checks passed.")

    if HAS_MATPLOTLIB:
        preview_path = Path(__file__).with_name("custom_hamiltonian_preview.png")
        equilibrium_model = build_surface_model(params={"driven": False})
        driven_model = build_surface_model(params=PREVIEW_DRIVEN_SNAPSHOT)
        driven_scan_model = build_surface_model(params=PREVIEW_DRIVEN_PARAMS)
        preview_time = build_preview_time(driven_scan_model, **PREVIEW_TIME_PARAMS)
        preview_energies = sample_time_energies(
            driven_scan_model,
            preview_time,
            kx=0.05,
            ky=0.02,
            kz=0.0,
        )
        plot_hamiltonian_diagnostics(
            equilibrium_model,
            preview_path,
            comparison_hamiltonian=driven_model,
            comparison_label="driven",
            time_values=preview_time,
            time_energies=preview_energies,
            time_label="t",
            **PREVIEW_LINE_PARAMS,
        )
        print(f"Saved Hamiltonian preview to {preview_path}")
    else:
        print("matplotlib not found: skipped preview generation.")

    test_custom_hamiltonian_preview_generation(Path(tempfile.mkdtemp()))
