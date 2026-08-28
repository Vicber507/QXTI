from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qxti.core import QXTIConfig, QXTISimulation
from qxti.physics import CustomHamiltonian


def _load_model(filename: str):
    path = PROJECT_ROOT / "models" / filename
    spec = importlib.util.spec_from_file_location(f"qxti_test_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _finite_difference(module, kpoint, direction, params, step=1.0e-6):
    axis = {"x": 0, "y": 1, "z": 2}[direction]
    plus = np.asarray(kpoint, dtype=float).copy()
    minus = np.asarray(kpoint, dtype=float).copy()
    plus[axis] += step
    minus[axis] -= step
    return (
        module.H(*plus, params) - module.H(*minus, params)
    ) / (2.0 * step)


def test_bkagome_flux_matches_the_cpp_matrix_elements() -> None:
    module = _load_model("bkagome_flux.py")
    params = {
        "a0": 9.3,
        "ta": 0.017,
        "tb": -0.004,
        "phi_a": 0.61,
        "phi_b": -0.37,
    }
    kx, ky = 0.14, -0.09
    a0 = params["a0"]
    vectors = np.array(
        [
            [0.25 * a0, np.sqrt(3.0) * 0.25 * a0],
            [-0.25 * a0, np.sqrt(3.0) * 0.25 * a0],
            [-0.5 * a0, 0.0],
        ]
    )
    dots = vectors @ np.array([kx, ky])
    h12 = (
        params["ta"] * np.exp(-1j * params["phi_a"] / 3.0) * np.exp(-1j * dots[2])
        + params["tb"] * np.exp(-1j * params["phi_b"] / 3.0) * np.exp(1j * dots[2])
    )
    h13 = (
        params["ta"] * np.exp(1j * params["phi_a"] / 3.0) * np.exp(-1j * dots[1])
        + params["tb"] * np.exp(1j * params["phi_b"] / 3.0) * np.exp(1j * dots[1])
    )
    h23 = (
        params["ta"] * np.exp(-1j * params["phi_a"] / 3.0) * np.exp(-1j * dots[0])
        + params["tb"] * np.exp(-1j * params["phi_b"] / 3.0) * np.exp(1j * dots[0])
    )
    expected = np.array(
        [
            [0.0, h12, h13],
            [np.conjugate(h12), 0.0, h23],
            [np.conjugate(h13), np.conjugate(h23), 0.0],
        ],
        dtype=complex,
    )

    np.testing.assert_allclose(module.H(kx, ky, 4.0, params), expected, atol=1.0e-15)


def test_bkagome_flux_loads_in_qxti_with_atomic_units_and_lattice_metadata() -> None:
    hamiltonian = CustomHamiltonian(source_file="bkagome_flux.py")

    assert hamiltonian.model_name == "bkagome-flux"
    assert hamiltonian.basis_size == 3
    assert hamiltonian.dimension == 2
    assert hamiltonian.basis_type == "sublattice"
    assert np.isclose(hamiltonian.params["a0"], 7.0 / 0.529177210903)
    assert hamiltonian.lattice["basis_order"] == ["A", "B", "C"]
    assert hamiltonian.validate_hermiticity(0.12, -0.08, 0.0)


@pytest.mark.parametrize("filename,size", [("bkagome_flux.py", 3), ("bkagome_flux_2l.py", 6)])
def test_bkagome_flux_batch_matches_scalar_hamiltonian(filename: str, size: int) -> None:
    module = _load_model(filename)
    params = module.default_params()
    params.update({"phi_a": np.pi / 2.0, "phi_b": -0.4})
    if size == 6:
        params["rot"] = 0.31
    kpoints = np.array(
        [[0.0, 0.0, 0.0], [0.13, -0.17, 0.0], [-0.21, 0.04, 2.0]],
        dtype=float,
    )
    expected = np.stack([module.H(*point, params) for point in kpoints])
    batched = module.H_batch(kpoints, params)

    assert batched.shape == (len(kpoints), size, size)
    np.testing.assert_allclose(batched, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("filename,size", [("bkagome_flux.py", 3), ("bkagome_flux_2l.py", 6)])
@pytest.mark.parametrize("direction", ["x", "y", "z"])
def test_bkagome_flux_analytic_derivatives_match_finite_differences(
    filename: str,
    size: int,
    direction: str,
) -> None:
    module = _load_model(filename)
    params = module.default_params()
    params.update({"phi_a": 0.7, "phi_b": -0.2})
    if size == 6:
        params["rot"] = -0.43
    kpoint = np.array([0.071, -0.116, 0.0])

    analytic = module.dH_dk(*kpoint, direction, params)
    numeric = _finite_difference(module, kpoint, direction, params)

    assert analytic.shape == (size, size)
    np.testing.assert_allclose(analytic, numeric, rtol=2.0e-9, atol=3.0e-11)


def test_bkagome_flux_2l_is_the_two_rotated_decoupled_cpp_blocks() -> None:
    monolayer = _load_model("bkagome_flux.py")
    bilayer = _load_model("bkagome_flux_2l.py")
    params = {
        "a0": 11.0,
        "ta": 0.013,
        "tb": 0.007,
        "phi_a": 0.8,
        "phi_b": -0.5,
        "rot": 0.37,
        "FB": -1,
    }
    monolayer_params = {key: params[key] for key in ("a0", "ta", "tb", "phi_a", "phi_b")}
    k = np.array([0.12, -0.19])
    cosine, sine = np.cos(params["rot"]), np.sin(params["rot"])
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    rotated_k = rotation.T @ k
    matrix = bilayer.H(k[0], k[1], 0.0, params)

    np.testing.assert_allclose(
        matrix[:3, :3],
        monolayer.H(k[0], k[1], 0.0, monolayer_params),
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        matrix[3:, 3:],
        monolayer.H(rotated_k[0], rotated_k[1], 0.0, monolayer_params),
        atol=1.0e-15,
    )
    np.testing.assert_array_equal(matrix[:3, 3:], np.zeros((3, 3)))
    np.testing.assert_array_equal(matrix[3:, :3], np.zeros((3, 3)))


def test_bkagome_flux_2l_preserves_fb_filling_semantics_and_rejects_invalid_fb() -> None:
    module = _load_model("bkagome_flux_2l.py")

    assert module.occupied_bands_from_fb({"FB": 1}) == 2
    assert module.occupied_bands_from_fb({"FB": -1}) == 4
    np.testing.assert_array_equal(
        module.H(0.08, -0.13, 0.0, {"FB": 1}),
        module.H(0.08, -0.13, 0.0, {"FB": -1}),
    )
    with pytest.raises(ValueError, match="FB"):
        module.H(0.0, 0.0, 0.0, {"FB": 0})


@pytest.mark.parametrize(
    "config_name,model_name,basis_size",
    [
        ("inputParams.bkagome_flux.cfg", "bkagome-flux", 3),
        ("inputParams.bkagome_flux_2l.cfg", "bkagome-flux-2l", 6),
    ],
)
def test_bkagome_flux_example_configs_build(
    config_name: str,
    model_name: str,
    basis_size: int,
) -> None:
    simulation = QXTISimulation.from_file(PROJECT_ROOT / "inputs" / config_name)
    hamiltonian = simulation.build_hamiltonian()
    kgrid = simulation.build_kgrid(hamiltonian)

    assert hamiltonian.model_name == model_name
    assert hamiltonian.basis_size == basis_size
    assert hamiltonian.validate_hermiticity(0.03, -0.04, 0.0)
    assert kgrid.dimension == 2


def test_phi_0_25_pfddm_config_preserves_the_antelope_physical_parameters() -> None:
    config_path = PROJECT_ROOT / "inputs" / "inputParams.bkagome_flux_phi_0.25_pfddm.cfg"
    config = QXTIConfig.from_file(config_path)
    simulation = QXTISimulation(config)
    hamiltonian = simulation.build_hamiltonian()
    kgrid = simulation.build_kgrid(hamiltonian)
    laser_system = simulation.build_laser_system()
    timegrid = simulation.build_timegrid(laser_system)

    assert config.cmd.response_method == "pfddm"
    assert config.cmd.max_order == 3
    assert config.cmd.population_time == -1.0
    assert config.cmd.coherence_time == 700.0
    assert config.cmd.distribution == "valence_occupation"
    assert kgrid.shape == (200, 200, 1)
    assert not kgrid.shifted
    assert timegrid.Nt == 232349
    assert timegrid.dt == 0.25
    assert config.laser.omega == 3.039168943926668e-3
    assert config.laser.E0 == 1.688032357869117e-5
    assert config.laser.ncycles == 7.0
    assert config.laser.blaser == config.laser.alaser == 100.0
    assert hamiltonian.params == {
        "a0": 13.22808287238039,
        "ta": 0.00752,
        "tb": 0.00376,
        "phi_a": 0.785398163397,
        "phi_b": 0.785398163397,
    }
