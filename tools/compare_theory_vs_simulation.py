"""Compare σ^(1)(ω) from analytical Kubo formula vs QXTI time-domain simulation.

Runs both calculations and plots them overlaid. The comparison uses EXACTLY the
same k-grid and BZ weights in both cases, so any disagreement is purely physical
(not a normalization artifact).

Usage:
    python tools/compare_theory_vs_simulation.py
    python tools/compare_theory_vs_simulation.py --cfg inputs/susceptibility/inputParams.graphene_comparison.cfg
"""
import argparse
import sys
from pathlib import Path
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

_AU_TO_EV = 27.211386245988
_KB_AU = 3.1668114e-6


def run_qxti_simulation(cfg_path: str) -> Path:
    """Run the QXTI susceptibility scan and return output directory."""
    from qxti.core.config import QXTIConfig
    out_dir = QXTIConfig.from_file(cfg_path).xtp.susceptibility_output_dir
    data_file = Path(out_dir) / "data" / "xtp_susceptibility.npz"
    if data_file.exists():
        print(f"[cache] Datos ya existen en {data_file}, salteando simulacion.")
        return Path(out_dir)
    print(f"[QXTI] Corriendo simulacion para {cfg_path} ...")
    result = subprocess.run(
        [sys.executable, "main.py", cfg_path],
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"QXTI fallo con exit code {result.returncode}")
    return Path(out_dir)


def load_qxti_sigma(out_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load σ^(1)(ω) from QXTI's saved npz output."""
    d = np.load(out_dir / "data" / "xtp_susceptibility.npz", allow_pickle=True)
    omega = np.asarray(d["omega_axis"])              # (nfreq,) in a.u.
    sigma = np.asarray(d["sigma_order_1_tensor"])    # (nfreq, dim, dim) in a.u.
    return omega, sigma


def kubo_sigma1_on_qxti_grid(cfg_path: str, omega_axis: np.ndarray) -> np.ndarray:
    """Analytical σ^(1)_φα(ω) on the EXACT same k-grid and BZ weights as QXTI.

    Using the same grid makes normalization identical on both sides — any
    difference between this and QXTI is purely from the physics, not from
    BZ geometry or quadrature conventions.

    Formula (Eq. A2 of Hipolito+2018 / standard Kubo):
        σ^(1)_φα = ig·Σ_k w_k/V_BZ ·  {
            Σ_{m≠n} v^φ_nm · v^α_mn · (f_n-f_m)/ε_mn / (ω̄-ε_mn)   [interband]
            + Σ_n (-i/ω̄) · v^φ_nn · ∂f_n/∂k_α                       [intraband]
        }

    where v^φ_nm = ⟨n|∂H/∂k_φ|m⟩ in the band basis.
    """
    from qxti.core.config import QXTIConfig
    from qxti.core.simulation import QXTISimulation
    from qxti.response.xtp import XTP
    from qxti.analytics.rho_analytic import _band_frame, _velocity_band, _fermi, _dfde

    cfg = QXTIConfig.from_file(cfg_path)
    sim = QXTISimulation(config=cfg)
    ham = sim.build_hamiltonian()
    kg = sim.build_kgrid(ham)
    k_pts = kg.points()             # (Nk, 3)
    dim = ham.dimension

    # Get BZ weights identical to what QXTI uses
    # Build a minimal XTP just for its _integration_weights()
    xtp_stub = XTP(
        hamiltonian=ham,
        kgrid=kg,
        timegrid=None,
        laser_system=None,
        operator_factory=None,
        band_gauge_frame=None,
        rho_orders={},
        directions=["x", "y"][:dim],
        bz_mask_enabled=False,
    )
    weights = xtp_stub._integration_weights()      # (Nk,) normalized
    V_BZ = float(np.sum(weights))                  # total BZ volume = sum of weights

    T_au = float(cfg.susceptibility_solver.temperature) * _KB_AU if hasattr(cfg.susceptibility_solver, 'temperature') else 10 * _KB_AU
    try:
        T_K = float(cfg.susceptibility_solver.temperature)
        # check if it looks like already in a.u. (< 0.1) or Kelvin (> 1)
        T_au = T_K * _KB_AU if T_K > 0.1 else T_K
    except Exception:
        T_au = 10 * _KB_AU

    mu = 0.0  # fermi level for undoped graphene
    gamma_au = 1.0 / float(cfg.susceptibility_solver.coherence_time)  # 1/T2

    try:
        spin_deg = 2 if ham.basis_size <= 2 else 1  # explicit spin in large bases
    except Exception:
        spin_deg = 2

    active = list(range(dim))
    nw = len(omega_axis)
    sigma = np.zeros((nw, 3, 3), dtype=np.complex128)

    print(f"[Kubo] {len(k_pts)} k-puntos, {nw} frecuencias, T={T_au/_KB_AU:.0f} K, γ={gamma_au:.4f} a.u.")

    for ik, (kx, ky, kz) in enumerate(k_pts):
        if ik % max(1, len(k_pts) // 5) == 0:
            print(f"  k {ik+1}/{len(k_pts)}", end="\r", flush=True)
        try:
            evals, _ = _band_frame(ham._matrix_at, float(kx), float(ky), float(kz))
        except Exception:
            try:
                evals, _ = _band_frame(
                    lambda kx2, ky2, kz2: ham._matrix_at(kx2, ky2, kz2),
                    float(kx), float(ky), float(kz)
                )
            except Exception:
                continue

        try:
            vel = _velocity_band(
                lambda kx2, ky2, kz2: ham._matrix_at(kx2, ky2, kz2),
                float(kx), float(ky), float(kz)
            )
        except Exception:
            continue

        nb = len(evals)
        f = _fermi(evals, mu, T_au)
        dfde = _dfde(f, T_au)
        w_k = float(weights[ik])

        for iw, omega in enumerate(omega_axis):
            ow = complex(omega + 1j * gamma_au)
            for phi in active:
                for alpha in active:
                    s = 0.0 + 0.0j
                    # Interband: v^φ_nm · v^α_mn · f_nm/ε_mn / (ω̄−ε_mn)
                    # v^φ_nm = vel[phi][n,m]  (paper convention: ⟨n|v^φ|m⟩)
                    for m in range(nb):
                        for n in range(nb):
                            if m == n:
                                continue
                            eps_mn = evals[m] - evals[n]
                            if abs(eps_mn) < 1e-20:
                                continue
                            f_nm = f[n] - f[m]
                            s += (vel[phi][n, m] * vel[alpha][m, n]
                                  * f_nm / eps_mn / (ow - eps_mn))
                    # Intraband: (−i/ω̄) · v^φ_nn · ∂f_n/∂k_α
                    for n in range(nb):
                        df_dk = dfde[n] * vel[alpha][n, n].real
                        s += (-1j / ow) * vel[phi][n, n] * df_dk

                    # ig·w_k / V_BZ from the Kubo prefactor (Eq. A2 paper)
                    sigma[iw, phi, alpha] += 1j * spin_deg * w_k / V_BZ * s

    print()
    return sigma[:, :dim, :dim]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="inputs/susceptibility/inputParams.graphene_comparison.cfg")
    parser.add_argument("--skip-run", action="store_true", help="No correr la simulacion, usar datos existentes")
    args = parser.parse_args()

    cfg_path = args.cfg

    # ── 1. QXTI simulation ────────────────────────────────────────────────────
    if not args.skip_run:
        out_dir = run_qxti_simulation(cfg_path)
    else:
        from qxti.core.config import QXTIConfig
        out_dir = Path(QXTIConfig.from_file(cfg_path).xtp.susceptibility_output_dir)

    omega_q, sigma_q = load_qxti_sigma(out_dir)
    print(f"[QXTI] sigma shape: {sigma_q.shape}, omega: {omega_q[0]*_AU_TO_EV:.2f}–{omega_q[-1]*_AU_TO_EV:.2f} eV")

    # ── 2. Analytical Kubo on same grid ──────────────────────────────────────
    print("[Kubo] Calculando formula analitica en el mismo grid k ...")
    sigma_ana = kubo_sigma1_on_qxti_grid(cfg_path, omega_q)

    # ── 3. Plot ───────────────────────────────────────────────────────────────
    omega_ev = omega_q * _AU_TO_EV
    dim = sigma_q.shape[1]
    components = [(0, 0, "σ_xx"), (1, 1, "σ_yy")] if dim >= 2 else [(0, 0, "σ_xx")]

    fig, axes = plt.subplots(len(components), 2, figsize=(13, 4 * len(components)), squeeze=False)

    for row, (i, j, label) in enumerate(components):
        qxti_ij = sigma_q[:, i, j]
        ana_ij  = sigma_ana[:, i, j] if sigma_ana.ndim == 3 else sigma_ana[:, i]

        for col, (part, part_name) in enumerate([("real", "Re"), ("imag", "Im")]):
            ax = axes[row, col]
            yq = getattr(qxti_ij, part)
            ya = getattr(ana_ij, part)

            ax.plot(omega_ev, yq, "b-",  lw=2.0, label="QXTI (time-domain CMD)")
            ax.plot(omega_ev, ya, "r--", lw=2.0, label="Kubo analítico (Eq. A2)")

            # Ratio to check convergence
            mask = np.abs(yq) > 1e-8 * np.max(np.abs(yq))
            if mask.sum() > 2:
                ratio = np.median(ya[mask] / yq[mask])
                ax.set_title(f"{part_name}[{label}]   ratio_analytic/QXTI ≈ {ratio:.3f}", fontsize=10)
            else:
                ax.set_title(f"{part_name}[{label}]", fontsize=10)

            ax.set_xlabel("ℏω (eV)")
            ax.set_ylabel("σ (a.u.)")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

    # Reference: universal graphene conductivity g*e²/4ħ = 0.5 a.u. with g=2
    sigma_univ = spin_deg / 4.0 if (dim <= 2) else None
    for row in range(len(components)):
        axes[row, 1].axhline(sigma_univ, color="gray", ls=":", lw=1.2,
                             label=f"σ₁=g/4={sigma_univ:.2f} a.u.") if sigma_univ else None
        axes[row, 1].legend(fontsize=9)

    fig.suptitle(
        "Comparación: σ^(1)(ω)  ·  QXTI time-domain vs Kubo analítico\n"
        f"Config: {Path(cfg_path).name}  —  mismo grid k, mismos pesos BZ",
        fontsize=11,
    )
    plt.tight_layout()
    out_fig = out_dir / "compare_theory_vs_simulation.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_fig, dpi=150)
    print(f"\n[Plot] guardado en {out_fig}")

    # ── 4. Print convergence table ────────────────────────────────────────────
    print("\n=== Tabla de convergencia (QXTI vs Kubo analitico) ===")
    print(f"{'omega(eV)':>10}  {'Im[sig_xx] QXTI':>18}  {'Im[sig_xx] Kubo':>18}  {'ratio':>8}")
    for i in range(0, len(omega_q), max(1, len(omega_q)//8)):
        sq = sigma_q[i, 0, 0].imag
        sa = sigma_ana[i, 0, 0].imag
        r = sa / sq if abs(sq) > 1e-12 else float("nan")
        print(f"{omega_ev[i]:>10.3f}  {sq:>18.6f}  {sa:>18.6f}  {r:>8.4f}")

    plt.show()


if __name__ == "__main__":
    main()
