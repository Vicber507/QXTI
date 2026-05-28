from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qxti.physics import Hamiltonian


class TwoBandHamiltonian(Hamiltonian):
    DEFAULT_LATTICE = {
        "lattice_constants": {"ax": 1.0, "ay": 1.0},
        "real_space_vectors": {"a1": [1.0, 0.0], "a2": [0.0, 1.0]},
    }

    def default_params(self) -> dict[str, float]:
        return {"mass": 1.0, "coupling": 2.0}

    def H(self, kx: float, ky: float, kz: float) -> np.ndarray:
        del kz
        mass = float(self.params["mass"])
        coupling = float(self.params["coupling"])
        return np.array(
            [
                [mass + kx**2, coupling * (kx - 1j * ky)],
                [coupling * (kx + 1j * ky), -mass + ky**2],
            ],
            dtype=complex,
        )


class NonHermitianHamiltonian(Hamiltonian):
    def H(self, kx: float, ky: float, kz: float) -> np.ndarray:
        del kx, ky, kz
        return np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)


def build_two_band_hamiltonian(**kwargs) -> TwoBandHamiltonian:
    options = {
        "model_name": "two-band",
        "basis_size": 2,
        "dimension": 2,
        "dk_derivative": 1.0e-5,
    }
    options.update(kwargs)
    return TwoBandHamiltonian(**options)


def print_two_band_report() -> None:
    hamiltonian = build_two_band_hamiltonian()
    kx, ky, kz = 0.3, 0.2, 0.0

    np.set_printoptions(precision=6, suppress=True)
    matrix = hamiltonian.H(kx, ky, kz)
    values, vectors = hamiltonian.diagonalize(kx, ky, kz)
    velocity_x = hamiltonian.velocity_operator(kx, ky, kz, "x")
    velocity_y = hamiltonian.velocity_operator(kx, ky, kz, "y")
    inverse_mass_xx = hamiltonian.inverse_mass_operator(kx, ky, kz, "x", "x")
    gap_value = hamiltonian.gap(kx, ky, kz)
    projector_0 = hamiltonian.band_projector(kx, ky, kz, 0)
    occupied = hamiltonian.occupied_projector(kx, ky, kz, fermi_level=0.0)

    print("Hamiltonian report")
    print(f"  model_name: {hamiltonian.model_name}")
    print(f"  summary: {hamiltonian.summary()}")
    print(f"  k-point: ({kx:.3f}, {ky:.3f}, {kz:.3f})")
    print(f"  hermitian: {hamiltonian.validate_hermiticity(kx, ky, kz)}")
    print("  H(k):")
    print(matrix)
    print("  eigenvalues:")
    print(values)
    print("  eigenvectors:")
    print(vectors)
    print("  velocity_operator x:")
    print(velocity_x)
    print("  velocity_operator y:")
    print(velocity_y)
    print("  inverse_mass_operator xx:")
    print(inverse_mass_xx)
    print(f"  direct gap: {gap_value:.12f}")
    print(f"  band-0 projector trace: {np.trace(projector_0):.12f}")
    print(f"  occupied projector trace: {np.trace(occupied):.12f}")


def test_hamiltonian_construction_and_summary() -> None:
    hamiltonian = build_two_band_hamiltonian()
    summary = hamiltonian.summary()

    assert hamiltonian.params == {"mass": 1.0, "coupling": 2.0}
    assert hamiltonian.lattice["lattice_constants"] == {"ax": 1.0, "ay": 1.0}
    assert hamiltonian.H(0.0, 0.0, 0.0).shape == (2, 2)
    assert summary["model_name"] == "two-band"
    assert summary["lattice"]["lattice_constants"] == {"ax": 1.0, "ay": 1.0}
    assert summary["basis_size"] == 2
    assert summary["dimension"] == 2
    assert summary["basis_type"] == "orbital"
    assert summary["is_periodic"] is True
    assert np.isclose(summary["dk_derivative"], 1.0e-5)


def test_hamiltonian_infers_real_space_lengths_and_reciprocal_box() -> None:
    hamiltonian = build_two_band_hamiltonian()

    lengths = hamiltonian.real_space_axis_lengths()
    reciprocal_bounds = np.asarray(hamiltonian.reciprocal_box_bounds(), dtype=float)

    assert lengths == (1.0, 1.0)
    np.testing.assert_allclose(
        reciprocal_bounds,
        np.array([[-np.pi, np.pi], [-np.pi, np.pi]], dtype=float),
    )


def test_hamiltonian_constructor_validates_core_inputs() -> None:
    with pytest.raises(ValueError, match="basis_size"):
        build_two_band_hamiltonian(basis_size=0)
    with pytest.raises(ValueError, match="dimension"):
        build_two_band_hamiltonian(dimension=4)
    with pytest.raises(ValueError, match="dk_derivative"):
        build_two_band_hamiltonian(dk_derivative=0.0)


def test_set_params_preserves_defaults() -> None:
    hamiltonian = build_two_band_hamiltonian()

    hamiltonian.set_params({"mass": 3.5})

    assert hamiltonian.params == {"mass": 3.5, "coupling": 2.0}


def test_set_lattice_preserves_defaults() -> None:
    hamiltonian = build_two_band_hamiltonian()

    hamiltonian.set_lattice({"notes": "square test lattice"})

    assert hamiltonian.lattice["lattice_constants"] == {"ax": 1.0, "ay": 1.0}
    assert hamiltonian.lattice["notes"] == "square test lattice"


def test_validate_matrix_accepts_complex_square_shape() -> None:
    hamiltonian = build_two_band_hamiltonian()
    matrix = hamiltonian.validate_matrix([[1.0, 1.0j], [-1.0j, 2.0]])

    assert matrix.dtype == np.complex128
    assert matrix.shape == (2, 2)


def test_validate_matrix_rejects_non_square_or_wrong_shape() -> None:
    hamiltonian = build_two_band_hamiltonian()

    with pytest.raises(ValueError, match="2D"):
        hamiltonian.validate_matrix(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="square"):
        hamiltonian.validate_matrix(np.ones((2, 3)))
    with pytest.raises(ValueError, match="shape"):
        hamiltonian.validate_matrix(np.eye(3))


def test_validate_hermiticity_and_require_hermitian() -> None:
    hermitian = build_two_band_hamiltonian()
    non_hermitian = NonHermitianHamiltonian(model_name="bad", basis_size=2)

    assert hermitian.validate_hermiticity(0.2, -0.1, 0.0)
    assert not non_hermitian.validate_hermiticity(0.0, 0.0, 0.0)

    with pytest.raises(ValueError, match="not Hermitian"):
        non_hermitian.require_hermitian(0.0, 0.0, 0.0)


def test_diagonalize_uses_hermitian_solver_and_returns_real_eigenvalues() -> None:
    hamiltonian = build_two_band_hamiltonian()
    values, vectors = hamiltonian.diagonalize(0.3, 0.2, 0.0)

    assert values.shape == (2,)
    assert vectors.shape == (2, 2)
    assert np.all(np.isreal(values))
    assert np.allclose(vectors.conj().T @ vectors, np.eye(2), atol=1.0e-12)


def test_diagonalize_rejects_non_hermitian_matrices() -> None:
    hamiltonian = NonHermitianHamiltonian(model_name="bad", basis_size=2)

    with pytest.raises(ValueError, match="not Hermitian"):
        hamiltonian.diagonalize(0.0, 0.0, 0.0)


def test_hamiltonian_derivatives_and_operators() -> None:
    hamiltonian = build_two_band_hamiltonian()

    velocity_x = hamiltonian.velocity_operator(0.3, 0.2, 0.0, "x")
    velocity_y = hamiltonian.velocity_operator(0.3, 0.2, 0.0, "y")
    inverse_mass_xx = hamiltonian.inverse_mass_operator(0.3, 0.2, 0.0, "x", "x")
    inverse_mass_xy = hamiltonian.inverse_mass_operator(0.3, 0.2, 0.0, "x", "y")

    np.testing.assert_allclose(
        velocity_x,
        np.array([[0.6, 2.0], [2.0, 0.0]], dtype=complex),
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        velocity_y,
        np.array([[0.0, -2.0j], [2.0j, 0.4]], dtype=complex),
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        inverse_mass_xx,
        np.array([[2.0, 0.0], [0.0, 0.0]], dtype=complex),
        atol=1.0e-5,
    )
    np.testing.assert_allclose(
        inverse_mass_xy,
        np.zeros((2, 2), dtype=complex),
        atol=1.0e-5,
    )


def test_transform_to_band_basis_matches_unitary_similarity() -> None:
    hamiltonian = build_two_band_hamiltonian()
    operator = hamiltonian.velocity_operator(0.3, 0.2, 0.0, "y")
    vectors = hamiltonian.eigenvectors(0.3, 0.2, 0.0)

    transformed = hamiltonian.transform_to_band_basis(operator, 0.3, 0.2, 0.0)

    np.testing.assert_allclose(transformed, vectors.conj().T @ operator @ vectors)


def test_gap_and_projectors_are_consistent() -> None:
    hamiltonian = build_two_band_hamiltonian()
    values, _ = hamiltonian.diagonalize(0.2, 0.1, 0.0)

    direct_gap = hamiltonian.gap(0.2, 0.1, 0.0)
    explicit_gap = hamiltonian.gap(0.2, 0.1, 0.0, occupied_bands=1)
    projector = hamiltonian.band_projector(0.2, 0.1, 0.0, 0)
    occupied = hamiltonian.occupied_projector(0.2, 0.1, 0.0, fermi_level=0.0)

    assert np.isclose(direct_gap, values[1] - values[0])
    assert np.isclose(explicit_gap, direct_gap)
    assert projector.shape == (2, 2)
    assert occupied.shape == (2, 2)
    assert np.allclose(projector, projector.conj().T, atol=1.0e-12)
    assert np.allclose(projector @ projector, projector, atol=1.0e-12)
    assert np.allclose(occupied, occupied.conj().T, atol=1.0e-12)


def test_gap_rejects_invalid_occupied_band_count() -> None:
    hamiltonian = build_two_band_hamiltonian()

    with pytest.raises(ValueError, match="Gap requires"):
        hamiltonian.gap(0.0, 0.0, 0.0, occupied_bands=0)


def test_band_projector_rejects_invalid_band_index() -> None:
    hamiltonian = build_two_band_hamiltonian()

    with pytest.raises(ValueError, match="band_index"):
        hamiltonian.band_projector(0.0, 0.0, 0.0, 5)


def test_hamiltonian_rejects_invalid_direction_inputs() -> None:
    hamiltonian = build_two_band_hamiltonian()

    with pytest.raises(ValueError, match="one of 'x', 'y', or 'z'"):
        hamiltonian.dH_dk(0.0, 0.0, 0.0, "u")
    with pytest.raises(ValueError, match="outside dimension"):
        hamiltonian.dH_dk(0.0, 0.0, 0.0, "z")


if __name__ == "__main__":
    test_hamiltonian_construction_and_summary()
    test_hamiltonian_constructor_validates_core_inputs()
    test_set_params_preserves_defaults()
    test_set_lattice_preserves_defaults()
    test_validate_matrix_accepts_complex_square_shape()
    test_validate_matrix_rejects_non_square_or_wrong_shape()
    test_validate_hermiticity_and_require_hermitian()
    test_diagonalize_uses_hermitian_solver_and_returns_real_eigenvalues()
    test_diagonalize_rejects_non_hermitian_matrices()
    test_hamiltonian_derivatives_and_operators()
    test_transform_to_band_basis_matches_unitary_similarity()
    test_gap_and_projectors_are_consistent()
    test_gap_rejects_invalid_occupied_band_count()
    test_band_projector_rejects_invalid_band_index()
    test_hamiltonian_rejects_invalid_direction_inputs()
    print("Hamiltonian checks passed.")
    print_two_band_report()
