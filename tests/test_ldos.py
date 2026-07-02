from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from textwrap import dedent

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qxti.core import LDOSRunner, QXTIConfig
from qxti.data import load_dataset_npz
from qxti.graphics.plot_dos import _AU_K_TO_ANG_INV, _spectral_map_x_axis


def _write_model(tmp_path: Path) -> Path:
    """A simple 2-band gapped model: eigenvalues +-sqrt(m^2 + k^2)."""
    model_path = tmp_path / "ldos_toy_model.py"
    model_path.write_text(
        dedent(
            """
            from __future__ import annotations
            import numpy as np

            MODEL_NAME = "ldos-toy"
            BASIS_SIZE = 2
            DIMENSION = 2
            BASIS_TYPE = "spin"
            IS_PERIODIC = True
            DEFAULT_PARAMS = {"mass": 0.5}
            DEFAULT_LATTICE = {
                "lattice_constants": {"a": 1.0, "b": 1.0},
                "real_space_vectors": {"a1": [1.0, 0.0], "a2": [0.0, 1.0]},
            }

            def H(kx, ky, kz, params):
                del kz
                m = params["mass"]
                return np.array(
                    [[m, kx - 1j * ky], [kx + 1j * ky, -m]],
                    dtype=complex,
                )
            """
        )
    )
    return model_path


def _write_config(
    tmp_path: Path,
    model_path: Path,
    *,
    broadening: str = "lorentzian",
    spectral: bool = False,
    plane: bool = False,
) -> Path:
    config_path = tmp_path / "inputParams.cfg"
    config_path.write_text(
        dedent(
            f"""
            [hamiltonian]
            source_file = {model_path}

            [kgrid]
            dimension = 2
            k_points = [40, 40]
            shifted = true

            [ldos]
            enabled = true
            output_dir = {tmp_path / "ldos"}
            broadening = {broadening}
            num_energies = 400
            projected = true
            spectral_enabled = {str(spectral).lower()}
            spectral_num_k = 60
            spectral_plane_enabled = {str(plane).lower()}
            spectral_plane_nkx = 24
            spectral_plane_nky = 24
            """
        )
    )
    return config_path


def test_ldos_runner_saves_dataset_and_satisfies_sum_rule(tmp_path: Path) -> None:
    model_path = _write_model(tmp_path)
    config_path = _write_config(tmp_path, model_path)

    outputs = LDOSRunner.from_file(config_path).run()
    dataset_path = outputs["ldos_data"]
    assert dataset_path.exists()
    assert dataset_path == tmp_path / "ldos" / "data" / "ldos.npz"

    data = load_dataset_npz(dataset_path)
    energies = np.asarray(data["energies"], dtype=float)
    dos = np.asarray(data["dos"], dtype=float)

    # DOS is non-negative everywhere.
    assert np.all(dos >= 0.0)
    # Sum rule: integral of g(E) equals the number of bands within a few percent
    # (the small deficit is broadening leaking past the finite energy window).
    trapezoid = getattr(np, "trapezoid", np.trapz)
    integral = float(trapezoid(dos, energies))
    assert integral == pytest.approx(2.0, rel=0.05)
    # Cumulative N(E) is monotonically non-decreasing.
    cumulative = np.asarray(data["cumulative"], dtype=float)
    assert np.all(np.diff(cumulative) >= -1e-12)


def test_ldos_pdos_sums_to_total_dos(tmp_path: Path) -> None:
    model_path = _write_model(tmp_path)
    config_path = _write_config(tmp_path, model_path)

    outputs = LDOSRunner.from_file(config_path).run()
    data = load_dataset_npz(outputs["ldos_data"])

    dos = np.asarray(data["dos"], dtype=float)
    pdos = np.asarray(data["pdos"], dtype=float)
    # sum_alpha |U_{alpha n}|^2 = 1  =>  sum_alpha g_alpha(E) = g(E) exactly.
    assert np.max(np.abs(pdos.sum(axis=0) - dos)) < 1e-10


def test_ldos_gaussian_keeps_gap_clean(tmp_path: Path) -> None:
    model_path = _write_model(tmp_path)
    config_path = _write_config(tmp_path, model_path, broadening="gaussian")

    outputs = LDOSRunner.from_file(config_path).run()
    data = load_dataset_npz(outputs["ldos_data"])
    energies = np.asarray(data["energies"], dtype=float)
    dos = np.asarray(data["dos"], dtype=float)

    # The model is gapped (|E| >= mass = 0.5). With Gaussian broadening the DOS
    # at the gap center must be essentially zero.
    gap_region = np.abs(energies) < 0.3
    assert dos[gap_region].max() < 1e-3


def test_ldos_spectral_map_has_expected_shape(tmp_path: Path) -> None:
    model_path = _write_model(tmp_path)
    config_path = _write_config(tmp_path, model_path, spectral=True)

    outputs = LDOSRunner.from_file(config_path).run()
    data = load_dataset_npz(outputs["ldos_data"])

    spectral = np.asarray(data["spectral"], dtype=float)
    energies = np.asarray(data["energies"], dtype=float)
    assert spectral.shape == (60, energies.size)
    assert np.all(spectral >= 0.0)


def test_ldos_single_axis_path_can_be_defined_from_axis_and_endpoints(tmp_path: Path) -> None:
    model_path = _write_model(tmp_path)
    config_path = tmp_path / "inputParams_axis.cfg"
    config_path.write_text(
        dedent(
            f"""
            [hamiltonian]
            source_file = {model_path}

            [kgrid]
            dimension = 2
            k_points = [20, 20]
            shifted = true

            [ldos]
            enabled = true
            output_dir = {tmp_path / "ldos_axis"}
            num_energies = 200
            spectral_enabled = true
            spectral_num_k = 75
            spectral_axis = ky
            spectral_axis_start = -0.4
            spectral_axis_end = 0.6
            spectral_axis_origin = 0.15, 0.0, 0.0
            """
        )
    )

    outputs = LDOSRunner.from_file(config_path).run()
    data = load_dataset_npz(outputs["ldos_data"])
    path_k = np.asarray(data["spectral_path_k"], dtype=float)
    spectral = np.asarray(data["spectral"], dtype=float)
    axis_meta = _spectral_map_x_axis(data, spectral)

    assert spectral.shape[0] == path_k.shape[0]
    assert path_k[:, 0].min() == pytest.approx(0.15)
    assert path_k[:, 0].max() == pytest.approx(0.15)
    assert path_k[:, 1].min() == pytest.approx(-0.4)
    assert path_k[:, 1].max() == pytest.approx(0.6)
    assert axis_meta["bottom_label"] == r"$k_y\;(\mathrm{\AA}^{-1})$"
    assert bool(axis_meta["use_symmetry_ticks"]) is False


def test_ldos_constant_energy_plane_shape_and_metadata(tmp_path: Path) -> None:
    model_path = _write_model(tmp_path)
    config_path = _write_config(tmp_path, model_path, plane=True)

    outputs = LDOSRunner.from_file(config_path).run()
    data = load_dataset_npz(outputs["ldos_data"])

    plane = np.asarray(data["spectral_plane"], dtype=float)
    kx = np.asarray(data["spectral_plane_kx"], dtype=float)
    ky = np.asarray(data["spectral_plane_ky"], dtype=float)
    assert plane.shape == (ky.size, kx.size) == (24, 24)
    assert np.all(plane >= 0.0)
    # Default cut energy is the Fermi level (0 here); the example's style metadata
    # travels with the dataset so graphics reproduces the original look.
    assert float(data["spectral_plane_energy"]) == pytest.approx(0.0)
    assert str(data["spectral_cmap"]) == "Blues"
    assert bool(data["spectral_log_scale"]) is True


def test_ldos_multisegment_path_ticks_and_labels(tmp_path: Path) -> None:
    model_path = _write_model(tmp_path)
    config_path = tmp_path / "inputParams.cfg"
    config_path.write_text(
        dedent(
            f"""
            [hamiltonian]
            source_file = {model_path}

            [kgrid]
            dimension = 2
            k_points = [20, 20]
            shifted = true

            [ldos]
            enabled = true
            output_dir = {tmp_path / "ldos"}
            num_energies = 200
            spectral_enabled = true
            spectral_num_k = 90
            spectral_path = -0.5,0,0 ; 0,0,0 ; 0.5,0,0
            spectral_path_labels = K', \\Gamma, K
            """
        )
    )
    outputs = LDOSRunner.from_file(config_path).run()
    data = load_dataset_npz(outputs["ldos_data"])

    ticks = np.asarray(data["spectral_path_ticks"], dtype=float)
    labels = [str(x) for x in data["spectral_path_tick_labels"]]
    # 3 vertices -> 3 ticks/labels; monotically increasing arclength.
    assert ticks.size == 3
    assert labels == ["$K'$", "$\\Gamma$", "$K$"]
    assert np.all(np.diff(ticks) > 0)
    # The path runs through the BZ corners (not clipped to the integration box).
    path_k = np.asarray(data["spectral_path_k"], dtype=float)
    assert path_k[:, 0].min() == pytest.approx(-0.5)
    assert path_k[:, 0].max() == pytest.approx(0.5)


def test_ldos_single_axis_line_aliases_build_requested_path(tmp_path: Path) -> None:
    model_path = _write_model(tmp_path)
    config_path = tmp_path / "inputParams_line_alias.cfg"
    config_path.write_text(
        dedent(
            f"""
            [hamiltonian]
            source_file = {model_path}

            [kgrid]
            dimension = 2
            k_points = [20, 20]
            shifted = true

            [ldos]
            enabled = true
            output_dir = {tmp_path / "ldos_line_alias"}
            num_energies = 120
            spectral_enabled = true
            spectral_num_k = 40
            spectral_line_axis = kx
            spectral_line_start = -0.4
            spectral_line_end = 0.25
            spectral_line_origin = 0.0, 0.15, 0.0
            spectral_plane_enabled = false
            """
        )
    )

    outputs = LDOSRunner.from_file(config_path).run()
    data = load_dataset_npz(outputs["ldos_data"])
    path_k = np.asarray(data["spectral_path_k"], dtype=float)

    assert np.allclose(path_k[:, 1], 0.15, atol=1e-12)
    assert np.allclose(path_k[:, 2], 0.0, atol=1e-12)
    assert path_k[0, 0] == pytest.approx(-0.4)
    assert path_k[-1, 0] == pytest.approx(0.25)


def test_ldos_multisegment_plot_axis_exposes_inverse_angstrom_scale() -> None:
    spectral = np.zeros((5, 8), dtype=float)
    data = {
        "spectral_path_s": np.array([0.0, 0.25, 0.50, 0.80, 1.10], dtype=float),
        "spectral_path_k": np.array(
            [
                [-0.5, 0.0, 0.0],
                [-0.25, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.20, 0.18, 0.0],
                [0.40, 0.36, 0.0],
            ],
            dtype=float,
        ),
        "spectral_path_ticks": np.array([0.0, 0.5, 1.1], dtype=float),
        "spectral_path_tick_labels": ["$K'$", "$\\Gamma$", "$M$"],
    }

    axis_meta = _spectral_map_x_axis(data, spectral)

    assert bool(axis_meta["use_symmetry_ticks"]) is True
    assert axis_meta["bottom_label"] == ""
    assert axis_meta["top_label"] == r"$s_{\mathbf{k}}\;(\mathrm{\AA}^{-1})$"
    np.testing.assert_allclose(
        np.asarray(axis_meta["x"], dtype=float),
        data["spectral_path_s"] * _AU_K_TO_ANG_INV,
    )
    np.testing.assert_allclose(
        np.asarray(axis_meta["ticks"], dtype=float),
        data["spectral_path_ticks"] * _AU_K_TO_ANG_INV,
    )


def _write_haldane_surface_config(tmp_path: Path, m0: float) -> Path:
    config_path = tmp_path / f"haldane_surface_{m0}.cfg"
    config_path.write_text(
        dedent(
            f"""
            [hamiltonian]
            source_file = haldane.py
            t1 = 0.075
            t2 = 0.025
            phi0 = 1.16
            M0 = {m0}
            a0 = 1.8897268777743552

            [kgrid]
            dimension = 2
            k_points = [21, 21]
            shifted = true

            [ldos]
            enabled = true
            output_dir = {tmp_path / f"ldos_{m0}"}
            method = surface
            surface_normal = y
            surface_kn_points = 80
            surface_ldos_enabled = false
            num_energies = 160
            e_min = -0.12
            e_max =  0.12
            spectral_enabled = true
            spectral_path = -0.96,0,0 ; 0,0,0 ; 0.96,0,0
            spectral_num_k = 80
            spectral_plane_enabled = false
            """
        )
    )
    return config_path


def _write_haldane_surface_both_graphics_config(tmp_path: Path, m0: float) -> Path:
    config_path = tmp_path / f"haldane_surface_both_{m0}.cfg"
    config_path.write_text(
        dedent(
            f"""
            [hamiltonian]
            source_file = haldane.py
            t1 = 0.075
            t2 = 0.025
            phi0 = 1.16
            M0 = {m0}
            a0 = 1.8897268777743552

            [kgrid]
            dimension = 2
            k_points = [17, 17]
            shifted = true

            [ldos]
            enabled = true
            output_dir = {tmp_path / f"ldos_surface_both_{m0}"}
            method = surface
            surface_normal = y
            surface_side = both
            surface_kn_points = 60
            surface_ldos_enabled = true
            projected = true
            num_energies = 120
            e_min = -0.12
            e_max =  0.12
            spectral_enabled = true
            spectral_path = -0.96,0,0 ; 0,0,0 ; 0.96,0,0
            spectral_num_k = 60
            spectral_plane_enabled = false
            """
        )
    )
    return config_path


def _write_haldane_finite_config(
    tmp_path: Path,
    m0: float,
    *,
    nx: int = 7,
    ny: int = 7,
) -> Path:
    config_path = tmp_path / f"haldane_finite_{m0}.cfg"
    config_path.write_text(
        dedent(
            f"""
            [hamiltonian]
            source_file = haldane.py
            t1 = 0.075
            t2 = 0.025
            phi0 = 1.16
            M0 = {m0}
            a0 = 1.8897268777743552

            [ldos]
            enabled = true
            output_dir = {tmp_path / f"ldos_finite_{m0}"}
            method = finite
            finite_nx = {nx}
            finite_ny = {ny}
            finite_edge_cells = 1
            finite_ldos_energy = 0.0
            num_energies = 240
            e_min = -0.25
            e_max =  0.25
            eta = 0.008
            projected = true
            spectral_enabled = true
            spectral_plane_enabled = true
            """
        )
    )
    return config_path


def test_surface_method_resolves_bulk_boundary_correspondence(tmp_path: Path) -> None:
    """Topological Haldane shows an in-gap edge mode; the trivial phase does not."""
    import numpy as np

    def in_gap_weight(m0: float) -> float:
        cfg = _write_haldane_surface_config(tmp_path, m0)
        data = load_dataset_npz(LDOSRunner.from_file(cfg).run()["ldos_data"])
        energies = np.asarray(data["energies"], dtype=float)
        spectral = np.asarray(data["spectral"], dtype=float)   # (num_k, num_e)
        # Total surface spectral weight in a narrow window around the Fermi level
        # (mid-gap): finite only when an edge state crosses the gap.
        mask = np.abs(energies) < 0.02
        return float(spectral[:, mask].sum())

    topological = in_gap_weight(0.0635)   # |M0| < critical -> Chern phase
    trivial = in_gap_weight(0.175)        # |M0| > critical -> trivial phase
    # The chiral edge mode puts substantially more spectral weight in the gap.
    assert topological > 2.0 * trivial


def test_surface_method_dataset_shape(tmp_path: Path) -> None:
    import numpy as np

    cfg = _write_haldane_surface_config(tmp_path, 0.0635)
    data = load_dataset_npz(LDOSRunner.from_file(cfg).run()["ldos_data"])
    assert str(data["method"]) == "surface"
    assert str(data["surface_normal"]) == "y"
    spectral = np.asarray(data["spectral"], dtype=float)
    dos_bulk = np.asarray(data["dos"], dtype=float)
    dos_surface = np.asarray(data["surface_dos"], dtype=float)
    assert spectral.shape[1] == np.asarray(data["energies"]).size
    assert np.all(spectral >= 0.0)
    assert dos_bulk.shape == np.asarray(data["energies"]).shape
    assert dos_surface.shape == np.asarray(data["energies"]).shape
    assert np.all(dos_bulk >= 0.0)
    assert np.all(dos_surface >= 0.0)


def test_finite_method_dataset_shape_and_metadata(tmp_path: Path) -> None:
    import numpy as np

    cfg = _write_haldane_finite_config(tmp_path, 0.0635, nx=6, ny=5)
    data = load_dataset_npz(LDOSRunner.from_file(cfg).run()["ldos_data"])

    assert str(data["method"]) == "finite"
    assert data["finite_num_cells"] == [6, 5]
    assert int(data["finite_num_sites"]) == 60
    positions = np.asarray(data["finite_positions"], dtype=float)
    site_ldos = np.asarray(data["finite_site_ldos"], dtype=float)
    edge_weight = np.asarray(data["finite_edge_weight"], dtype=float)
    pdos = np.asarray(data["pdos"], dtype=float)
    assert positions.shape == (60, 2)
    assert site_ldos.shape == (60,)
    assert edge_weight.shape == (60,)
    assert pdos.shape[0] == 2
    assert np.all(site_ldos >= 0.0)
    assert float(data["finite_ldos_energy"]) == pytest.approx(0.0)
    # k-space spectral plots do not exist in finite mode.
    assert "spectral" not in data
    assert "spectral_plane" not in data


def test_finite_method_distinguishes_topological_and_trivial_edges(tmp_path: Path) -> None:
    topological = load_dataset_npz(LDOSRunner.from_file(_write_haldane_finite_config(tmp_path, 0.0635)).run()["ldos_data"])
    trivial = load_dataset_npz(LDOSRunner.from_file(_write_haldane_finite_config(tmp_path, 0.175)).run()["ldos_data"])

    topo_min_abs = float(np.min(np.abs(np.asarray(topological["finite_eigenvalues"], dtype=float))))
    triv_min_abs = float(np.min(np.abs(np.asarray(trivial["finite_eigenvalues"], dtype=float))))
    topo_edge_fraction = float(topological["finite_edge_fraction"])
    triv_edge_fraction = float(trivial["finite_edge_fraction"])
    topo_ldos_sum = float(np.asarray(topological["finite_site_ldos"], dtype=float).sum())
    triv_ldos_sum = float(np.asarray(trivial["finite_site_ldos"], dtype=float).sum())

    # In the topological phase the edge branch crosses the gap, so the state
    # closest to E_F sits much nearer to zero and the LDOS at E_F is both
    # larger and more edge-localized than in the trivial phase.
    assert topo_min_abs < 0.5 * triv_min_abs
    assert topo_edge_fraction > triv_edge_fraction
    assert topo_ldos_sum > 3.0 * triv_ldos_sum


def test_ldos_graphics_generate_from_saved_data(tmp_path: Path) -> None:
    if importlib.util.find_spec("matplotlib") is None:
        pytest.skip("matplotlib is not available in this environment.")

    from qxti.graphics.graphics import plot_ldos_graphics_from_saved_data

    model_path = _write_model(tmp_path)
    config_path = _write_config(tmp_path, model_path, spectral=True, plane=True)
    LDOSRunner.from_file(config_path).run()

    outputs = plot_ldos_graphics_from_saved_data(config_path)
    assert "dos_total" in outputs
    assert "dos_projected" in outputs
    assert "spectral_map" in outputs
    assert "spectral_plane" in outputs
    for path in outputs.values():
        assert Path(path).exists()


def test_finite_ldos_graphics_generate_from_saved_data(tmp_path: Path) -> None:
    if importlib.util.find_spec("matplotlib") is None:
        pytest.skip("matplotlib is not available in this environment.")

    from qxti.graphics.graphics import plot_ldos_graphics_from_saved_data

    config_path = _write_haldane_finite_config(tmp_path, 0.0635)
    LDOSRunner.from_file(config_path).run()

    outputs = plot_ldos_graphics_from_saved_data(config_path)
    assert "dos_total" in outputs
    assert "dos_projected" in outputs
    assert "finite_spectrum" in outputs
    assert "finite_ldos_map" in outputs
    for path in outputs.values():
        assert Path(path).exists()


def test_surface_ldos_graphics_generate_from_saved_data(tmp_path: Path) -> None:
    if importlib.util.find_spec("matplotlib") is None:
        pytest.skip("matplotlib is not available in this environment.")

    from qxti.graphics.graphics import plot_ldos_graphics_from_saved_data

    config_path = _write_haldane_surface_both_graphics_config(tmp_path, 0.0635)
    LDOSRunner.from_file(config_path).run()

    outputs = plot_ldos_graphics_from_saved_data(config_path)
    assert "dos_total" in outputs
    assert "dos_projected" in outputs
    assert "dos_projected_surface" in outputs
    assert "spectral_map" in outputs
    assert "dos_surface_sides" not in outputs
    assert "dos_projected_bottom" not in outputs
    assert "dos_projected_top" not in outputs
    assert "spectral_map_bottom" not in outputs
    assert "spectral_map_top" not in outputs
    for path in outputs.values():
        assert Path(path).exists()
