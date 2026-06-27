"""Compare analytical σ^(1) from Hipolito+2018 vs QXTI numerical result for graphene.

Run AFTER a susceptibility scan with orders=[1]:
    python main.py inputs/susceptibility/inputParams.susceptibility.cfg
    python tools/compare_sigma_graphene.py

Plots:
  - Re[σ_xx] analytical (paper Eq. A2) vs QXTI numerical
  - Im[σ_xx] same
  - Ratio analytical/numerical (should be ~1 if physics and units agree)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from qxti.analytics.hipolito2018 import analytical_sigma1_fast, load_model

# ── configuration ──────────────────────────────────────────────────────────
OUTPUT_DIR    = "outputs/susceptibility_graphene"
QXTI_DATA     = f"{OUTPUT_DIR}/data/xtp_susceptibility.npz"
MODEL_FILE    = "models/graphene.py"
FUNCTION_NAME = "H"

GAMMA_AU = 0.001      # 10 meV dephasing in a.u. (≈ 0.001 Ha)
MU_AU    = 0.0        # undoped (μ = 0)
T_K      = 10         # temperature in Kelvin (matches paper's T=10 K default)
SPIN_DEG = 2          # spin degeneracy (key factor: paper uses g=2)
KPOINTS  = (101, 101) # grid for analytical sum
BZ_BOUNDS = (-np.pi / 1.42, np.pi / 1.42)  # graphene BZ in 1/Å → but in a.u. below
# Graphene nearest-neighbor distance a0 = 1.42 Å = 2.68 Bohr
A0_BOHR  = 2.68
BZ_BOUNDS_AU = (-np.pi / A0_BOHR, np.pi / A0_BOHR)

AU_TO_EV = 27.211386245988
# ────────────────────────────────────────────────────────────────────────────

def load_qxti(path):
    d = np.load(path, allow_pickle=True)
    omega = np.asarray(d["omega_axis"])           # a.u.
    # σ_xx comes from the x-probe run: sigma[freq, out_axis, in_axis]
    sigma = np.asarray(d.get("sigma_order_1_tensor", d.get("chi_order_1_tensor")))
    return omega, sigma

def main():
    # ── 1. Analytical σ^(1) ────────────────────────────────────────────────
    print("Computing analytical σ^(1) via Hipolito+2018 Eq. A2 ...")
    H = load_model(MODEL_FILE, FUNCTION_NAME)

    # Use the same ω axis as the QXTI output
    try:
        omega_q, sigma_q = load_qxti(QXTI_DATA)
        omega_axis = omega_q[omega_q > 0]
        mask = omega_q > 0
    except FileNotFoundError:
        print(f"No QXTI data found at {QXTI_DATA}. Using default ω axis.")
        omega_axis = np.linspace(0.005, 0.15, 80)
        mask = None
        sigma_q = None

    sigma_ana = analytical_sigma1_fast(
        H,
        kpoints=KPOINTS,
        omega_axis=omega_axis,
        gamma=GAMMA_AU,
        mu=MU_AU,
        T=T_K,
        spin_deg=SPIN_DEG,
        bz_bounds=BZ_BOUNDS_AU,
        dimension=2,
    )  # shape (nw, 3, 3), a.u.

    sigma_xx_ana = sigma_ana[:, 0, 0]

    # ── 2. Universal conductivity reference ─────────────────────────────────
    # σ^(1)_xx for undoped graphene = g·e²/(4ℏ) = g/4 in a.u. (e=ℏ=1)
    # = 0.5 a.u. (with g=2) for ℏω > 0
    sigma_universal_au = SPIN_DEG / 4.0  # = 0.5 a.u.
    print(f"Universal conductivity σ₁ = g/4 = {sigma_universal_au:.3f} a.u.")
    print(f"  = {sigma_universal_au * 7.748e-5 * 1e3:.3f} mS  (in SI, 2D)")
    print(f"Max |Re σ_xx (analytical)| = {np.max(np.abs(sigma_xx_ana.real)):.4f} a.u.")

    # ── 3. Plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    omega_ev = omega_axis * AU_TO_EV

    axes[0].plot(omega_ev, sigma_xx_ana.real, "b-", lw=2,
                 label=f"Analítica (Ec. A2) g={SPIN_DEG}, γ={GAMMA_AU*AU_TO_EV*1e3:.0f} meV")
    axes[0].axhline(sigma_universal_au, color="gray", ls="--",
                    label=f"σ₁ = g/4 = {sigma_universal_au:.2f} a.u. (universal)")

    if sigma_q is not None:
        # σ_q shape: (n_freq, n_out, n_in) — take σ_xx = [freq, 0, 0]
        sigma_xx_q = sigma_q[mask, 0, 0] if sigma_q.ndim == 3 else sigma_q[mask, 0]
        axes[0].plot(omega_ev, sigma_xx_q.real, "r--", lw=1.5,
                     label="QXTI numérico")
        axes[1].plot(omega_ev, sigma_xx_q.imag, "r--", lw=1.5, label="QXTI")

    axes[1].plot(omega_ev, sigma_xx_ana.imag, "b-", lw=2, label="Analítica")

    for ax, ylabel in zip(axes, ["Re[σ_xx] (a.u.)", "Im[σ_xx] (a.u.)"]):
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[1].set_xlabel("ℏω (eV)", fontsize=12)
    axes[0].set_title(
        "Conductividad óptica lineal σ^(1)_xx del grafeno\n"
        f"Analítica (Hipolito+2018 Eq. A2) vs QXTI numérico (TD pulse + FFT)",
        fontsize=11,
    )

    out = Path(OUTPUT_DIR) / "compare_sigma_vs_analytic.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"\nPlot guardado en {out}")
    plt.show()

    # ── 4. Diagnóstico de discrepancia ──────────────────────────────────────
    print("\n── Diagnóstico de factores de discrepancia ──")
    print(f"  SPIN_DEG usado en analítica = {SPIN_DEG}")
    print(f"  QXTI NO incluye factor de spin (asume 1 en operadores)")
    print(f"  → Si QXTI da la mitad de la analítica: falta el factor g=2")
    print()
    print("Conversión SI (2D): σ_2D [S] = σ_2D [a.u.] × (e²/ℏ) = σ_2D [a.u.] × 7.748×10⁻⁵ S")
    print(f"  σ₁ = {sigma_universal_au:.3f} a.u. → {sigma_universal_au * 7.748e-5 * 1e6:.2f} μS = e²/(4ℏ) ✓")

if __name__ == "__main__":
    main()
