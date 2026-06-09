from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qxti.physics import CustomHamiltonian


def build_bilayer_model(**kwargs) -> CustomHamiltonian:
    return CustomHamiltonian(source_file="graphene_bilayer.py", **kwargs)


def load_bilayer_module():
    module_path = PROJECT_ROOT / "models" / "graphene_bilayer.py"
    spec = importlib.util.spec_from_file_location("qxti_test_graphene_bilayer", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not build import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_graphene_bilayer_model_loads_metadata_and_is_hermitian() -> None:
    module = load_bilayer_module()
    hamiltonian = build_bilayer_model()
    matrix = hamiltonian.H(0.07, -0.11, 0.0)
    direct = module.H(0.07, -0.11, 0.0, hamiltonian.params)

    assert hamiltonian.model_name == "graphene-bilayer-ab"
    assert hamiltonian.basis_size == 4
    assert hamiltonian.dimension == 2
    assert hamiltonian.basis_type == "layer-sublattice"
    assert hamiltonian.is_periodic is True
    assert hamiltonian.lattice["stacking"] == "AB/Bernal"
    assert hamiltonian.lattice["basis_order"] == ["A1", "B1", "A2", "B2"]
    assert set(hamiltonian.params) == {
        "a0",
        "gamma0",
        "gamma1",
        "gamma3",
        "gamma4",
        "delta_prime",
        "u",
    }
    assert matrix.shape == (4, 4)
    assert np.iscomplexobj(matrix)
    assert hamiltonian.validate_hermiticity(0.07, -0.11, 0.0)
    np.testing.assert_allclose(matrix, matrix.conj().T, atol=1.0e-12)
    np.testing.assert_allclose(matrix, direct, atol=1.0e-12)


def test_graphene_bilayer_model_is_independent_of_kz() -> None:
    hamiltonian = build_bilayer_model()

    np.testing.assert_allclose(
        hamiltonian.H(0.03, 0.08, 0.0),
        hamiltonian.H(0.03, 0.08, 7.0),
        atol=1.0e-12,
    )


def test_graphene_bilayer_reduces_to_two_decoupled_graphene_layers_when_interlayer_terms_vanish() -> None:
    hamiltonian = build_bilayer_model(
        params={
            "gamma1": 0.0,
            "gamma3": 0.0,
            "gamma4": 0.0,
            "delta_prime": 0.0,
            "u": 0.0,
        }
    )

    kx, ky = 0.12, -0.09
    values = np.sort(hamiltonian.eigenvalues(kx, ky, 0.0))

    a0 = float(hamiltonian.params["a0"])
    gamma0 = float(hamiltonian.params["gamma0"])
    deltas = np.array(
        [
            [0.0, a0],
            [-np.sqrt(3.0) * 0.5 * a0, -0.5 * a0],
            [np.sqrt(3.0) * 0.5 * a0, -0.5 * a0],
        ],
        dtype=float,
    )
    f = np.exp(1.0j * (deltas @ np.array([kx, ky], dtype=float))).sum()
    monolayer_energy = gamma0 * abs(f)
    expected = np.array(
        [-monolayer_energy, -monolayer_energy, monolayer_energy, monolayer_energy],
        dtype=float,
    )

    np.testing.assert_allclose(values, expected, atol=1.0e-10)


def test_graphene_bilayer_has_two_low_energy_bands_touching_at_k_when_unbiased() -> None:
    hamiltonian = build_bilayer_model(
        params={
            "gamma3": 0.0,
            "gamma4": 0.0,
            "delta_prime": 0.0,
            "u": 0.0,
        }
    )

    a0 = float(hamiltonian.params["a0"])
    kx = 4.0 * np.pi / (3.0 * np.sqrt(3.0) * a0)
    ky = 0.0
    values = np.sort(hamiltonian.eigenvalues(kx, ky, 0.0))
    gamma1 = float(hamiltonian.params["gamma1"])

    np.testing.assert_allclose(
        values,
        np.array([-gamma1, 0.0, 0.0, gamma1], dtype=float),
        atol=1.0e-10,
    )
