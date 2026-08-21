from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from textwrap import dedent

import numpy as np
import pytest

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

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


def build_surface_model(**kwargs) -> CustomHamiltonian:
    return CustomHamiltonian(source_file="bi2se3_surface.py", **kwargs)


def load_surface_module():
    module_path = PROJECT_ROOT / "models" / "bi2se3_surface.py"
    spec = importlib.util.spec_from_file_location("qxti_test_bi2se3_surface", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not build import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_external_model(tmp_path: Path, filename: str, contents: str) -> Path:
    file_path = tmp_path / filename
    file_path.write_text(dedent(contents))
    return file_path


def print_surface_model_report() -> None:
    hamiltonian = build_surface_model()
    module = load_surface_module()
    kx, ky, kz = 0.04, 0.11, 0.0

    np.set_printoptions(precision=6, suppress=True)
    matrix = hamiltonian.H(kx, ky, kz)
    values, vectors = hamiltonian.diagonalize(kx, ky, kz)
    velocity_x = hamiltonian.velocity_operator(kx, ky, kz, "x")
    velocity_y = hamiltonian.velocity_operator(kx, ky, kz, "y")
    inverse_mass_xx = hamiltonian.inverse_mass_operator(kx, ky, kz, "x", "x")
    inverse_mass_xy = hamiltonian.inverse_mass_operator(kx, ky, kz, "x", "y")
    projector_0 = hamiltonian.band_projector(kx, ky, kz, 0)
    occupied = hamiltonian.occupied_projector(kx, ky, kz, fermi_level=0.0)
    direct_gap = hamiltonian.gap(kx, ky, kz)
    direct_matrix = module.H(kx, ky, kz, hamiltonian.params)

    print("CustomHamiltonian report")
    print(f"  source_file: {hamiltonian.source_file}")
    print(f"  resolved_source_file: {hamiltonian.summary()['resolved_source_file']}")
    print(f"  model_name: {hamiltonian.model_name}")
    print(f"  summary: {hamiltonian.summary()}")
    print(f"  k-point: ({kx:.3f}, {ky:.3f}, {kz:.3f})")
    print(f"  hermitian: {hamiltonian.validate_hermiticity(kx, ky, kz)}")
    print(f"  module/direct match: {np.allclose(matrix, direct_matrix)}")
    print("  params:")
    print(hamiltonian.params)
    print("  H(k):")
    print(matrix)
    print("  eigenvalues:")
    print(values)
    print("  eigenvectors:")
    print(vectors)
    print("  dH/dkx:")
    print(velocity_x)
    print("  dH/dky:")
    print(velocity_y)
    print("  d2H/dkx2:")
    print(inverse_mass_xx)
    print("  d2H/dkx dky:")
    print(inverse_mass_xy)
    print(f"  direct gap: {direct_gap:.12f}")
    print(f"  band-0 projector trace: {np.trace(projector_0):.12f}")
    print(f"  occupied projector trace: {np.trace(occupied):.12f}")


def test_surface_model_file_is_configured_consistently_with_custom_hamiltonian() -> None:
    module = load_surface_module()
    hamiltonian = build_surface_model()
    params = hamiltonian.default_params()
    direct_matrix = module.H(0.08, -0.11, 0.0, params)
    loaded_matrix = hamiltonian.H(0.08, -0.11, 0.0)

    assert module.MODEL_NAME == hamiltonian.model_name
    assert module.BASIS_SIZE == hamiltonian.basis_size
    assert module.DIMENSION == hamiltonian.dimension
    assert module.BASIS_TYPE == hamiltonian.basis_type
    assert module.IS_PERIODIC is hamiltonian.is_periodic
    assert module.default_params() == params
    assert module.DEFAULT_LATTICE == hamiltonian.lattice
    np.testing.assert_allclose(direct_matrix, loaded_matrix, atol=1.0e-12)


def test_custom_hamiltonian_loads_model_metadata_defaults_and_summary() -> None:
    hamiltonian = build_surface_model()
    matrix = hamiltonian.H(0.08, -0.11, 0.0)
    summary = hamiltonian.summary()

    assert hamiltonian.model_name == "bi2se3-surface"
    assert hamiltonian.basis_size == 2
    assert hamiltonian.dimension == 2
    assert hamiltonian.basis_type == "spin"
    assert "a0" in hamiltonian.params
    assert "B11" in hamiltonian.params
    assert "lattice_constants" in hamiltonian.lattice
    assert "real_space_vectors" in hamiltonian.lattice
    assert "BZaxis" in hamiltonian.lattice
    assert "BZorigin" in hamiltonian.lattice
    assert matrix.shape == (2, 2)
    assert np.iscomplexobj(matrix)
    assert hamiltonian.validate_hermiticity(0.08, -0.11, 0.0)
    assert summary["source_file"] == "bi2se3_surface.py"
    assert summary["function_name"] == "H"
    assert summary["basis_size"] == 2
    assert np.isclose(summary["lattice"]["lattice_constants"]["a0"], hamiltonian.params["a0"])
    assert "resolved_source_file" in summary


def test_custom_hamiltonian_load_from_file_returns_callable() -> None:
    hamiltonian = build_surface_model()
    loaded_function = hamiltonian.load_from_file()

    assert callable(loaded_function)
    assert loaded_function(0.0, 0.0, 0.0, hamiltonian.params) is not None


def test_custom_hamiltonian_default_params_match_external_model_defaults() -> None:
    hamiltonian = build_surface_model()
    defaults = hamiltonian.default_params()

    assert defaults == hamiltonian.params
    assert set(defaults) == {"a0", "A0", "B0", "A11", "A12", "A14", "B11", "B14"}


def test_custom_hamiltonian_default_lattice_matches_external_model_defaults() -> None:
    hamiltonian = build_surface_model()
    defaults = hamiltonian.default_lattice()

    assert defaults == hamiltonian.lattice
    assert set(defaults) == {"lattice_type", "lattice_constants", "real_space_vectors", "BZorigin", "BZaxis"}
    assert np.isclose(defaults["lattice_constants"]["a0"], hamiltonian.params["a0"])


def test_parameter_dependent_lattice_provider_takes_precedence() -> None:
    hamiltonian = CustomHamiltonian(
        source_file="wsm_two_weyl.py",
        params={"a0": 6.0, "a1": 7.0, "a2": 12.0},
    )

    np.testing.assert_allclose(
        hamiltonian.real_space_axis_lengths(),
        np.array([6.0, 7.0, 12.0]),
    )
    np.testing.assert_allclose(
        hamiltonian.reciprocal_box_bounds(),
        np.array(
            [
                [-np.pi / 6.0, np.pi / 6.0],
                [-np.pi / 7.0, np.pi / 7.0],
                [-np.pi / 12.0, np.pi / 12.0],
            ]
        ),
    )


def test_custom_hamiltonian_uses_explicit_brillouin_zone_box_from_model() -> None:
    hamiltonian = build_surface_model()
    bounds = np.asarray(hamiltonian.reciprocal_box_bounds(), dtype=float)

    np.testing.assert_allclose(
        bounds,
        np.array(
            [
                [-np.pi / hamiltonian.params["a0"], np.pi / hamiltonian.params["a0"]],
                [
                    -2.0 * np.pi / (np.sqrt(3.0) * hamiltonian.params["a0"]),
                    2.0 * np.pi / (np.sqrt(3.0) * hamiltonian.params["a0"]),
                ],
            ],
            dtype=float,
        ),
    )


def test_custom_hamiltonian_accepts_source_file_without_py_suffix() -> None:
    hamiltonian = CustomHamiltonian(source_file="bi2se3_surface")

    assert hamiltonian.model_name == "bi2se3-surface"
    assert hamiltonian.H(0.0, 0.0, 0.0).shape == (2, 2)


def test_custom_hamiltonian_set_params_preserves_external_defaults() -> None:
    hamiltonian = build_surface_model(params={"A14": 0.003})

    hamiltonian.set_params({"A0": -0.0011})

    assert np.isclose(hamiltonian.params["A14"], 0.003)
    assert np.isclose(hamiltonian.params["A0"], -0.0011)
    assert "B11" in hamiltonian.params


def test_custom_hamiltonian_set_lattice_preserves_external_defaults() -> None:
    hamiltonian = build_surface_model()

    hamiltonian.set_lattice({"notes": "surface cell"})

    assert np.isclose(hamiltonian.lattice["lattice_constants"]["a0"], hamiltonian.params["a0"])
    assert hamiltonian.lattice["notes"] == "surface cell"


def test_custom_hamiltonian_exercises_all_surface_model_methods() -> None:
    hamiltonian = build_surface_model()
    matrix = hamiltonian.H(0.04, 0.11, 0.0)
    validated = hamiltonian.validate_matrix(matrix)
    assert validated.shape == (2, 2)
    assert hamiltonian.validate_hermiticity(0.04, 0.11, 0.0)
    hamiltonian.require_hermitian(0.04, 0.11, 0.0)

    dH_dx = hamiltonian.dH_dk(0.04, 0.11, 0.0, "x")
    dH_dy = hamiltonian.dH_dk(0.04, 0.11, 0.0, "y")
    d2H_xx = hamiltonian.d2H_dk2(0.04, 0.11, 0.0, "x", "x")
    d2H_xy = hamiltonian.d2H_dk2(0.04, 0.11, 0.0, "x", "y")
    velocity = hamiltonian.velocity_operator(0.04, 0.11, 0.0, "x")
    inverse_mass = hamiltonian.inverse_mass_operator(0.04, 0.11, 0.0, "x", "x")
    values, vectors = hamiltonian.diagonalize(0.04, 0.11, 0.0)
    eigenvalues = hamiltonian.eigenvalues(0.04, 0.11, 0.0)
    eigenvectors = hamiltonian.eigenvectors(0.04, 0.11, 0.0)
    transformed_identity = hamiltonian.transform_to_band_basis(np.eye(2), 0.04, 0.11, 0.0)
    projector = hamiltonian.band_projector(0.04, 0.11, 0.0, 0)
    occupied = hamiltonian.occupied_projector(0.04, 0.11, 0.0, fermi_level=0.0)
    direct_gap = hamiltonian.gap(0.04, 0.11, 0.0)
    explicit_gap = hamiltonian.gap(0.04, 0.11, 0.0, occupied_bands=1)
    summary = hamiltonian.summary()

    assert dH_dx.shape == (2, 2)
    assert dH_dy.shape == (2, 2)
    assert d2H_xx.shape == (2, 2)
    assert d2H_xy.shape == (2, 2)
    assert values.shape == (2,)
    assert vectors.shape == (2, 2)
    assert eigenvalues.shape == (2,)
    assert eigenvectors.shape == (2, 2)
    assert velocity.shape == (2, 2)
    assert inverse_mass.shape == (2, 2)
    assert projector.shape == (2, 2)
    assert occupied.shape == (2, 2)
    assert np.isclose(direct_gap, values[1] - values[0])
    assert np.isclose(explicit_gap, direct_gap)
    np.testing.assert_allclose(values, eigenvalues, atol=1.0e-12)
    np.testing.assert_allclose(vectors, eigenvectors, atol=1.0e-12)
    np.testing.assert_allclose(velocity, dH_dx, atol=1.0e-12)
    np.testing.assert_allclose(inverse_mass, d2H_xx, atol=1.0e-12)
    np.testing.assert_allclose(transformed_identity, np.eye(2), atol=1.0e-12)
    np.testing.assert_allclose(projector, projector.conj().T, atol=1.0e-12)
    np.testing.assert_allclose(projector @ projector, projector, atol=1.0e-10)
    np.testing.assert_allclose(occupied, occupied.conj().T, atol=1.0e-12)
    assert summary["model_name"] == "bi2se3-surface"


def test_custom_hamiltonian_surface_model_is_independent_of_kz() -> None:
    hamiltonian = build_surface_model()

    np.testing.assert_allclose(
        hamiltonian.H(0.07, -0.05, 0.0),
        hamiltonian.H(0.07, -0.05, 4.2),
        atol=1.0e-12,
    )


def test_custom_hamiltonian_rejects_missing_source_file() -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        CustomHamiltonian(source_file="does_not_exist.py")


def test_custom_hamiltonian_rejects_missing_function() -> None:
    with pytest.raises(AttributeError, match="was not found"):
        CustomHamiltonian(source_file="bi2se3_surface.py", function_name="missing")


def test_custom_hamiltonian_rejects_non_callable_function(tmp_path: Path) -> None:
    model_file = write_external_model(
        tmp_path,
        "not_callable_model.py",
        """
        H = 3.14
        """,
    )

    with pytest.raises(TypeError, match="not callable"):
        CustomHamiltonian(source_file=str(model_file), basis_size=2)


def test_custom_hamiltonian_rejects_bad_function_signature(tmp_path: Path) -> None:
    model_file = write_external_model(
        tmp_path,
        "bad_signature_model.py",
        """
        def H(kx, ky):
            return [[0.0, 0.0], [0.0, 0.0]]
        """,
    )

    with pytest.raises(TypeError, match="signature"):
        CustomHamiltonian(source_file=str(model_file), basis_size=2)


def test_custom_hamiltonian_rejects_wrong_matrix_shape_from_external_function(
    tmp_path: Path,
) -> None:
    model_file = write_external_model(
        tmp_path,
        "wrong_shape_model.py",
        """
        def H(kx, ky, kz, params):
            return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        """,
    )
    hamiltonian = CustomHamiltonian(source_file=str(model_file), basis_size=2)

    with pytest.raises(ValueError, match="shape"):
        hamiltonian.H(0.0, 0.0, 0.0)


def test_custom_hamiltonian_rejects_non_hermitian_external_model_on_diagonalize(
    tmp_path: Path,
) -> None:
    model_file = write_external_model(
        tmp_path,
        "non_hermitian_model.py",
        """
        def H(kx, ky, kz, params):
            return [[0.0, 1.0], [0.0, 0.0]]
        """,
    )
    hamiltonian = CustomHamiltonian(source_file=str(model_file), basis_size=2)

    assert not hamiltonian.validate_hermiticity(0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="not Hermitian"):
        hamiltonian.diagonalize(0.0, 0.0, 0.0)


def test_custom_hamiltonian_supports_user_written_external_file(tmp_path: Path) -> None:
    model_file = write_external_model(
        tmp_path,
        "simple_model.py",
        """
        MODEL_NAME = "simple-model"
        BASIS_SIZE = 2
        DIMENSION = 3
        BASIS_TYPE = "site"
        IS_PERIODIC = False
        DEFAULT_PARAMS = {"delta": 0.5}
        DEFAULT_LATTICE = {
            "lattice_constants": {"a": 1.0, "b": 1.0, "c": 1.0},
            "real_space_vectors": {"a1": [1.0, 0.0, 0.0]},
        }

        def H(kx, ky, kz, params):
            delta = params["delta"]
            return [
                [delta + kx, ky - 1j * kz],
                [ky + 1j * kz, -delta - kx],
            ]
        """,
    )
    hamiltonian = CustomHamiltonian(source_file=str(model_file))
    matrix = hamiltonian.H(0.1, 0.2, 0.3)

    assert hamiltonian.model_name == "simple-model"
    assert hamiltonian.basis_size == 2
    assert hamiltonian.dimension == 3
    assert hamiltonian.basis_type == "site"
    assert hamiltonian.is_periodic is False
    assert hamiltonian.lattice["lattice_constants"]["a"] == 1.0
    assert matrix.shape == (2, 2)
    assert np.iscomplexobj(matrix)


def test_custom_hamiltonian_detects_invalid_direction_from_base_class() -> None:
    hamiltonian = build_surface_model()

    with pytest.raises(ValueError, match="one of 'x', 'y', or 'z'"):
        hamiltonian.dH_dk(0.0, 0.0, 0.0, "u")
    with pytest.raises(ValueError, match="outside dimension"):
        hamiltonian.dH_dk(0.0, 0.0, 0.0, "z")


def test_custom_hamiltonian_rejects_invalid_matrix_shape_and_band_index() -> None:
    hamiltonian = build_surface_model()

    with pytest.raises(ValueError, match="2D"):
        hamiltonian.validate_matrix(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="shape"):
        hamiltonian.validate_matrix(np.eye(3))
    with pytest.raises(ValueError, match="band_index"):
        hamiltonian.band_projector(0.0, 0.0, 0.0, 5)


def test_custom_hamiltonian_gap_rejects_invalid_occupied_band_count() -> None:
    hamiltonian = build_surface_model()

    with pytest.raises(ValueError, match="Gap requires"):
        hamiltonian.gap(0.0, 0.0, 0.0, occupied_bands=0)


def test_custom_hamiltonian_preview_generation(tmp_path: Path) -> None:
    if not HAS_MATPLOTLIB:
        return

    model = build_surface_model()
    output_path = tmp_path / "custom_hamiltonian_preview.png"

    plot_hamiltonian_diagnostics(
        model,
        output_path,
        **PREVIEW_LINE_PARAMS,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


if __name__ == "__main__":
    test_surface_model_file_is_configured_consistently_with_custom_hamiltonian()
    test_custom_hamiltonian_loads_model_metadata_defaults_and_summary()
    test_custom_hamiltonian_load_from_file_returns_callable()
    test_custom_hamiltonian_default_params_match_external_model_defaults()
    test_custom_hamiltonian_default_lattice_matches_external_model_defaults()
    test_custom_hamiltonian_accepts_source_file_without_py_suffix()
    test_custom_hamiltonian_set_params_preserves_external_defaults()
    test_custom_hamiltonian_set_lattice_preserves_external_defaults()
    test_custom_hamiltonian_exercises_all_surface_model_methods()
    test_custom_hamiltonian_surface_model_is_independent_of_kz()
    test_custom_hamiltonian_detects_invalid_direction_from_base_class()
    test_custom_hamiltonian_rejects_invalid_matrix_shape_and_band_index()
    test_custom_hamiltonian_gap_rejects_invalid_occupied_band_count()
    test_custom_hamiltonian_rejects_missing_source_file()
    test_custom_hamiltonian_rejects_missing_function()
    temp_dir = Path(tempfile.mkdtemp())
    test_custom_hamiltonian_rejects_non_callable_function(temp_dir)
    test_custom_hamiltonian_rejects_bad_function_signature(temp_dir)
    test_custom_hamiltonian_rejects_wrong_matrix_shape_from_external_function(temp_dir)
    test_custom_hamiltonian_rejects_non_hermitian_external_model_on_diagonalize(temp_dir)
    test_custom_hamiltonian_supports_user_written_external_file(temp_dir)
    print("CustomHamiltonian checks passed.")
    print_surface_model_report()

    if HAS_MATPLOTLIB:
        preview_path = Path(__file__).with_name("custom_hamiltonian_preview.png")
        model = build_surface_model()
        plot_hamiltonian_diagnostics(model, preview_path, **PREVIEW_LINE_PARAMS)
        print(f"Saved Hamiltonian preview to {preview_path}")
    else:
        print("matplotlib not found: skipped preview generation.")

    test_custom_hamiltonian_preview_generation(Path(tempfile.mkdtemp()))
