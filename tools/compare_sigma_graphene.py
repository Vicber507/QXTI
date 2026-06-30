"""Compare analytical graphene conductivity vs QXTI numerical tensors.

Run after a dedicated susceptibility sweep with orders ``[1, 2]``:

    python main.py inputs/inputParams.graphene.cfg
    python tools/compare_sigma_graphene.py

Outputs:
  - ``compare_sigma1_vs_analytic.png``: linear conductivity ``sigma^(1)_xx``
  - ``compare_sigma2_vs_analytic.png``: second-harmonic conductivity
    ``sigma^(2)_{xxx}`` and ``sigma^(2)_{yxx}``
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import numpy as np

from qxti.analytics.hipolito2018 import analytical_sigma1_fast, load_model
from qxti.analytics.rho_analytic import sigma_analytic
from qxti.data.io import load_dataset_npz


# -- configuration -----------------------------------------------------------
OUTPUT_DIR = Path("outputs/susceptibility_graphene")
QXTI_DATA = OUTPUT_DIR / "data" / "xtp_susceptibility.npz"
MODEL_FILE = "models/graphene.py"
FUNCTION_NAME = "H"

GAMMA_AU = 0.001      # 10 meV dephasing in a.u. (≈ 0.001 Ha)
MU_AU = 0.0           # undoped (μ = 0)
T_K = 10              # temperature in Kelvin (matches paper's T=10 K default)
SPIN_DEG = 2          # spin degeneracy (paper formulas usually include g=2)
KPOINTS = (101, 101)  # grid for analytical sum
E_FIELD_ORDER2 = np.array([1.0e-4, 0.0, 0.0], dtype=np.complex128)  # x-driven SHG

# Graphene nearest-neighbor distance a0 = 1.42 Å = 2.68 Bohr
A0_BOHR = 2.68
BZ_BOUNDS_AU = (-np.pi / A0_BOHR, np.pi / A0_BOHR)
AU_TO_EV = 27.211386245988


def load_qxti_dataset(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    data = load_dataset_npz(path)
    omega = np.asarray(data.get("laser_omega_axis", data.get("omega_axis")), dtype=np.float64)
    sigma1 = None
    sigma2 = None
    if "sigma_order_1_tensor" in data:
        sigma1 = np.asarray(data["sigma_order_1_tensor"], dtype=np.complex128)
    elif "chi_order_1_tensor" in data:
        sigma1 = np.asarray(data["chi_order_1_tensor"], dtype=np.complex128)
    if "sigma_order_2_tensor" in data:
        sigma2 = np.asarray(data["sigma_order_2_tensor"], dtype=np.complex128)
    elif "chi_order_2_tensor" in data:
        sigma2 = np.asarray(data["chi_order_2_tensor"], dtype=np.complex128)
    return omega, sigma1, sigma2


def plot_sigma1(
    *,
    omega_axis: np.ndarray,
    sigma_ana: np.ndarray,
    sigma_qxti: np.ndarray | None,
) -> Path:
    sigma_xx_ana = sigma_ana[:, 0, 0]
    sigma_universal_au = SPIN_DEG / 4.0

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    omega_ev = omega_axis * AU_TO_EV

    axes[0].plot(
        omega_ev,
        sigma_xx_ana.real,
        "b-",
        lw=2,
        label=f"Analítica (Ec. A2) g={SPIN_DEG}, γ={GAMMA_AU * AU_TO_EV * 1e3:.0f} meV",
    )
    axes[0].axhline(
        sigma_universal_au,
        color="gray",
        ls="--",
        label=f"σ₁ = g/4 = {sigma_universal_au:.2f} a.u. (universal)",
    )

    if sigma_qxti is not None:
        sigma_xx_q = sigma_qxti[:, 0, 0]
        axes[0].plot(omega_ev, sigma_xx_q.real, "r--", lw=1.5, label="QXTI numérico")
        axes[1].plot(omega_ev, sigma_xx_q.imag, "r--", lw=1.5, label="QXTI numérico")

    axes[1].plot(omega_ev, sigma_xx_ana.imag, "b-", lw=2, label="Analítica")

    for ax, ylabel in zip(axes, ["Re[σ_xx] (a.u.)", "Im[σ_xx] (a.u.)"]):
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[1].set_xlabel("ℏω (eV)", fontsize=12)
    axes[0].set_title(
        "Conductividad óptica lineal σ^(1)_xx del grafeno\n"
        "Analítica (Hipolito+2018 Eq. A2) vs QXTI numérico",
        fontsize=11,
    )

    out = OUTPUT_DIR / "compare_sigma1_vs_analytic.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_sigma2(
    *,
    omega_axis: np.ndarray,
    sigma2_ana: np.ndarray,
    sigma2_qxti: np.ndarray | None,
) -> Path:
    omega_ev = omega_axis * AU_TO_EV
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 7.4), sharex=True)

    sigma_xxx_ana = sigma2_ana[:, 0]
    sigma_yxx_ana = sigma2_ana[:, 1]

    axes[0].plot(omega_ev, sigma_xxx_ana.real, color="#0072B2", lw=2, label=r"Analítica $\sigma^{(2)}_{xxx}$")
    axes[0].plot(omega_ev, sigma_yxx_ana.real, color="#D55E00", lw=2, label=r"Analítica $\sigma^{(2)}_{yxx}$")
    axes[1].plot(omega_ev, sigma_xxx_ana.imag, color="#0072B2", lw=2, label=r"Analítica $\sigma^{(2)}_{xxx}$")
    axes[1].plot(omega_ev, sigma_yxx_ana.imag, color="#D55E00", lw=2, label=r"Analítica $\sigma^{(2)}_{yxx}$")

    if sigma2_qxti is not None:
        sigma_xxx_q = sigma2_qxti[:, 0, 0, 0]
        sigma_yxx_q = sigma2_qxti[:, 1, 0, 0]
        axes[0].plot(omega_ev, sigma_xxx_q.real, color="#0072B2", lw=1.5, ls="--", label=r"QXTI $\sigma^{(2)}_{xxx}$")
        axes[0].plot(omega_ev, sigma_yxx_q.real, color="#D55E00", lw=1.5, ls="--", label=r"QXTI $\sigma^{(2)}_{yxx}$")
        axes[1].plot(omega_ev, sigma_xxx_q.imag, color="#0072B2", lw=1.5, ls="--", label=r"QXTI $\sigma^{(2)}_{xxx}$")
        axes[1].plot(omega_ev, sigma_yxx_q.imag, color="#D55E00", lw=1.5, ls="--", label=r"QXTI $\sigma^{(2)}_{yxx}$")

    for ax, ylabel in zip(axes, [r"Re[$\sigma^{(2)}_{\phi xx}$] (a.u.)", r"Im[$\sigma^{(2)}_{\phi xx}$] (a.u.)"]):
        ax.axhline(0.0, color="gray", lw=0.8, alpha=0.7)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)

    axes[1].set_xlabel("ℏω (eV)", fontsize=12)
    axes[0].set_title(
        "Segundo armónico del grafeno excitado en x\n"
        r"Comparación de $\sigma^{(2)}_{xxx}$ y $\sigma^{(2)}_{yxx}$: analítica vs QXTI",
        fontsize=11,
    )

    out = OUTPUT_DIR / "compare_sigma2_vs_analytic.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    print("Computing analytical graphene conductivity tensors ...")
    H = load_model(MODEL_FILE, FUNCTION_NAME)

    try:
        omega_q, sigma1_q, sigma2_q = load_qxti_dataset(QXTI_DATA)
        positive_mask = np.asarray(omega_q > 0.0, dtype=bool)
        omega_axis = np.asarray(omega_q[positive_mask], dtype=np.float64)
        sigma1_q_pos = None if sigma1_q is None else np.asarray(sigma1_q[positive_mask], dtype=np.complex128)
        sigma2_q_pos = None if sigma2_q is None else np.asarray(sigma2_q[positive_mask], dtype=np.complex128)
    except FileNotFoundError:
        print(f"No QXTI data found at {QXTI_DATA}. Using a default ω axis and plotting only the analytics.")
        omega_axis = np.linspace(0.005, 0.15, 80)
        sigma1_q_pos = None
        sigma2_q_pos = None

    sigma1_ana = analytical_sigma1_fast(
        H,
        kpoints=KPOINTS,
        omega_axis=omega_axis,
        gamma=GAMMA_AU,
        mu=MU_AU,
        T=T_K,
        spin_deg=SPIN_DEG,
        bz_bounds=BZ_BOUNDS_AU,
        dimension=2,
    )
    sigma_orders = sigma_analytic(
        H,
        kpoints=KPOINTS,
        omega_axis=omega_axis,
        E_field=E_FIELD_ORDER2,
        gamma=GAMMA_AU,
        mu=MU_AU,
        T_K=T_K,
        spin_deg=SPIN_DEG,
        bz_bounds=BZ_BOUNDS_AU,
        dimension=2,
        max_order=2,
        verbose=False,
    )
    sigma2_ana = np.asarray(sigma_orders[2], dtype=np.complex128)

    sigma1_plot = plot_sigma1(omega_axis=omega_axis, sigma_ana=sigma1_ana, sigma_qxti=sigma1_q_pos)
    sigma2_plot = plot_sigma2(omega_axis=omega_axis, sigma2_ana=sigma2_ana, sigma2_qxti=sigma2_q_pos)

    sigma1_xx_ana = sigma1_ana[:, 0, 0]
    sigma_universal_au = SPIN_DEG / 4.0
    print(f"Universal conductivity σ₁ = g/4 = {sigma_universal_au:.3f} a.u.")
    print(f"  = {sigma_universal_au * 7.748e-5 * 1e3:.3f} mS  (in SI, 2D)")
    print(f"Max |Re σ_xx^(1) (analytical)| = {np.max(np.abs(sigma1_xx_ana.real)):.4f} a.u.")
    print(f"Max |σ_xxx^(2) (analytical)| = {np.max(np.abs(sigma2_ana[:, 0])):.4e} a.u.")
    print(f"Max |σ_yxx^(2) (analytical)| = {np.max(np.abs(sigma2_ana[:, 1])):.4e} a.u.")

    if sigma2_q_pos is None:
        print("\nNo QXTI sigma^(2) tensor found in the dataset.")
        print("Rerun the susceptibility workflow with `susceptibility_orders = [1,2]` to include SHG.")
    else:
        diff_xxx = np.max(np.abs(sigma2_q_pos[:, 0, 0, 0] - sigma2_ana[:, 0]))
        diff_yxx = np.max(np.abs(sigma2_q_pos[:, 1, 0, 0] - sigma2_ana[:, 1]))
        print(f"Max |Δσ_xxx^(2)| = {diff_xxx:.4e} a.u.")
        print(f"Max |Δσ_yxx^(2)| = {diff_yxx:.4e} a.u.")

    print(f"\nPlots saved to:\n  - {sigma1_plot}\n  - {sigma2_plot}")
    print("\nDiagnóstico lineal:")
    print(f"  SPIN_DEG usado en analítica = {SPIN_DEG}")
    print("  QXTI no incluye el factor degeneración de spin por defecto.")
    print("  Si σ^(1) numérica sale aproximadamente a la mitad, falta g=2 en la comparación.")


if __name__ == "__main__":
    main()
