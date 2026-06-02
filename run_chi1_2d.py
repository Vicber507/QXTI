from __future__ import annotations

from pathlib import Path

import numpy as np

from qxti.core import QXTIConfig, QXTISimulation
from qxti.grids import FrequencyGrid
from qxti.physics import OperatorFactory
from qxti.response import SusceptibilityTensorCalculator, XTP


def build_frequencygrid_from_timegrid(timegrid) -> FrequencyGrid:
    omega_axis = np.asarray(timegrid.frequency_axis(), dtype=float)
    return FrequencyGrid(
        omega_min=float(np.min(omega_axis)),
        omega_max=float(np.max(omega_axis)),
        Nomega=len(omega_axis),
    )


def build_xtp_from_config(config_path: str | Path, *, rho_path: str | Path) -> XTP:
    config = QXTIConfig.from_file(config_path)
    simulation = QXTISimulation(config)

    hamiltonian = simulation.build_hamiltonian()
    laser_system = simulation.build_laser_system()
    kgrid = simulation.build_kgrid(hamiltonian)
    timegrid = simulation.build_timegrid(laser_system)
    frequencygrid = build_frequencygrid_from_timegrid(timegrid)

    operator_factory = OperatorFactory(
        hamiltonian=hamiltonian,
        basis=config.cmd.basis,
    )

    rho_order_1 = np.load(rho_path)

    return XTP(
        hamiltonian=hamiltonian,
        rho_orders={1: rho_order_1},
        kgrid=kgrid,
        timegrid=timegrid,
        frequencygrid=frequencygrid,
        operator_factory=operator_factory,
        directions=["x", "y"],
        orders=[1],
        laser_system=laser_system,
    )


def save_chi1_dat(
    path: str | Path,
    omega_axis: np.ndarray,
    chi1: np.ndarray,
) -> None:
    data = np.column_stack(
        [
            omega_axis,
            np.real(chi1[:, 0, 0]),
            np.imag(chi1[:, 0, 0]),
            np.real(chi1[:, 0, 1]),
            np.imag(chi1[:, 0, 1]),
            np.real(chi1[:, 1, 0]),
            np.imag(chi1[:, 1, 0]),
            np.real(chi1[:, 1, 1]),
            np.imag(chi1[:, 1, 1]),
        ]
    )

    header = (
        "omega "
        "Re_chi_xx Im_chi_xx "
        "Re_chi_xy Im_chi_xy "
        "Re_chi_yx Im_chi_yx "
        "Re_chi_yy Im_chi_yy"
    )

    np.savetxt(path, data, header=header, comments="", fmt="%.12e")


def main() -> int:
    xtp_x = build_xtp_from_config(
        "input_run_chi1_2d_x.cfg",
        rho_path="outputs/cmd_x/rho_order_1.npy",
    )
    xtp_y = build_xtp_from_config(
        "input_run_chi1_2d_y.cfg",
        rho_path="outputs/cmd_y/rho_order_1.npy",
    )

    calculator = SusceptibilityTensorCalculator(
        {
            "x": xtp_x,
            "y": xtp_y,
        }
    )

    omega_axis, chi1 = calculator.chi1()

    output_dir = Path("outputs/chi1")
    output_dir.mkdir(parents=True, exist_ok=True)

    omega_path = output_dir / "omega_axis.npy"
    chi1_path = output_dir / "chi1_2d.npy"
    dat_path = output_dir / "chi1_2d.dat"

    np.save(omega_path, omega_axis)
    np.save(chi1_path, chi1)
    save_chi1_dat(dat_path, omega_axis, chi1)

    print("Saved linear susceptibility tensor:")
    print(f"  omega_axis: {omega_path}")
    print(f"  chi1 npy:   {chi1_path}")
    print(f"  chi1 dat:   {dat_path}")
    print(f"  chi1 shape: {chi1.shape}")
    print()
    print("DAT columns:")
    print("  omega")
    print("  Re_chi_xx Im_chi_xx")
    print("  Re_chi_xy Im_chi_xy")
    print("  Re_chi_yx Im_chi_yx")
    print("  Re_chi_yy Im_chi_yy")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())