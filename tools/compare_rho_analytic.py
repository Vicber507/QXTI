"""Compare analytical ρ^(1,2,3) vs QXTI numerical for any material.

Usage:
    # Quick sanity check on a small grid (no prior run needed):
    python tools/compare_rho_analytic.py --model graphene --order 1

    # Full comparison after running a QXTI HHG simulation:
    python tools/compare_rho_analytic.py \\
        --model graphene --order 1 \\
        --rho_path outputs/graphene_hhg/rho_order_1.npy \\
        --cfg inputs/inputParams.graphene.cfg
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from qxti.analytics.rho_analytic import rho_order_s, sigma_analytic

_AU_TO_EV = 27.211386245988
_KB_AU = 3.1668114e-6


def load_model(source_file: str, func_name: str = "H"):
    import importlib.util
    for base in [".", "models"]:
        p = Path(base) / source_file
        if p.exists():
            spec = importlib.util.spec_from_file_location("_m", p)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return getattr(m, func_name)
    raise FileNotFoundError(source_file)


def quick_check(H_func, model_name: str, order: int, dim: int,
                omega_eV: float, gamma_meV: float,
                mu_eV: float, T_K: float, spin_deg: int,
                kN: int, bz_bounds: tuple):
    """Compute σ^(order) on a small grid and show it — no prior run needed."""
    omega = omega_eV / _AU_TO_EV
    gamma = gamma_meV * 1e-3 / _AU_TO_EV
    mu = mu_eV / _AU_TO_EV

    # E field amplitude (weak, x-direction)
    E0 = 1e-4
    E_field = np.array([E0, 0.0, 0.0])

    omega_axis = np.linspace(0.01 / _AU_TO_EV, 0.15 / _AU_TO_EV, 40)

    print(f"\n=== Analítica ρ^(1..{order}) para {model_name} ===")
    print(f"  grid: {kN}^{dim}  ω: 0.01–0.15 eV  γ={gamma_meV} meV  μ={mu_eV} eV  T={T_K} K")

    sigma = sigma_analytic(
        H_func,
        kpoints=(kN,) * dim,
        omega_axis=omega_axis,
        E_field=E_field,
        gamma=gamma,
        mu=mu,
        T_K=T_K,
        spin_deg=spin_deg,
        bz_bounds=bz_bounds,
        dimension=dim,
        max_order=order,
    )

    n_panels = order
    fig, axes = plt.subplots(n_panels, 2, figsize=(12, 3.5 * n_panels), squeeze=False)
    omega_ev = omega_axis * _AU_TO_EV

    labels = {1: "σ^(1)_xx (a.u.)", 2: "σ^(2)_xx (a.u.)", 3: "σ^(3)_xx (a.u.)"}

    for s in range(1, order + 1):
        sig_xx = sigma[s][:, 0]  # x-component
        ax_re, ax_im = axes[s-1]

        ax_re.plot(omega_ev, sig_xx.real, "b-", lw=2)
        ax_re.set_ylabel(f"Re[{labels[s]}]", fontsize=11)
        ax_re.set_title(f"Orden {s} — parte real", fontsize=11)
        ax_re.grid(True, alpha=0.3)

        ax_im.plot(omega_ev, sig_xx.imag, "r-", lw=2)
        ax_im.set_ylabel(f"Im[{labels[s]}]", fontsize=11)
        ax_im.set_title(f"Orden {s} — parte imaginaria", fontsize=11)
        ax_im.grid(True, alpha=0.3)

        if s == order:
            ax_re.set_xlabel("ℏω (eV)", fontsize=11)
            ax_im.set_xlabel("ℏω (eV)", fontsize=11)

        # Universal conductivity reference for order 1
        if s == 1:
            sigma_univ = spin_deg / 4.0
            ax_re.axhline(sigma_univ, color="gray", ls="--",
                          label=f"σ₁=g/4={sigma_univ:.2f} a.u.")
            ax_re.legend(fontsize=9)

    fig.suptitle(f"Conductividad analítica (Hipolito+2018 Eqs. A1a–A1c)\n"
                 f"Modelo: {model_name}  spin_deg={spin_deg}  "
                 f"grid {kN}^{dim}  γ={gamma_meV} meV",
                 fontsize=11)
    plt.tight_layout()
    out = Path(f"outputs/analytic_{model_name}_order{order}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"  → guardado en {out}")
    plt.show()


def rho_element_plot(H_func, model_name: str, kx: float, ky: float, kz: float,
                     order: int, omega_eV: float, gamma_meV: float,
                     mu_eV: float, T_K: float):
    """Show all ρ^(s)_mn elements at one k-point for visual inspection."""
    omega = omega_eV / _AU_TO_EV
    gamma = gamma_meV * 1e-3 / _AU_TO_EV
    mu = mu_eV / _AU_TO_EV
    T_au = T_K * _KB_AU
    E0 = 1e-4
    E_field = np.array([E0, 0.0, 0.0])

    rhos = rho_order_s(H_func, kx, ky, kz, E_field, omega, gamma, mu, T_au,
                       max_order=order)

    print(f"\n=== ρ^(0..{order}) en k=({kx:.3f},{ky:.3f},{kz:.3f}), ω={omega_eV:.3f} eV ===")
    for s, rho in sorted(rhos.items()):
        print(f"\n  ρ^({s}) (normalizado por E0^{s} = {E0**s:.2e}):")
        r_norm = rho / (E0 ** max(s, 1))
        nb = r_norm.shape[0]
        for m in range(nb):
            row = "  ".join(f"{r_norm[m,n].real:+.4f}{r_norm[m,n].imag:+.4f}i" for n in range(nb))
            print(f"    [{row}]")

    # Also compare ρ^(0) diagonal with Fermi distribution
    evals, _ = __import__("qxti.analytics.rho_analytic", fromlist=["_band_frame"])._band_frame(
        H_func, kx, ky, kz
    ) if False else (None, None)
def main():
    parser = argparse.ArgumentParser(description="Comparación analítica ρ^(s) vs QXTI")
    parser.add_argument("--model", default="graphene", help="Nombre del archivo modelo (sin .py)")
    parser.add_argument("--func", default="H", help="Nombre de la función H")
    parser.add_argument("--order", type=int, default=1, help="Orden máximo (1,2,3)")
    parser.add_argument("--dim", type=int, default=2, help="Dimensión (2 para grafeno)")
    parser.add_argument("--kN", type=int, default=21, help="Puntos por eje BZ")
    parser.add_argument("--omega_eV", type=float, default=0.05, help="Frecuencia sonda (eV)")
    parser.add_argument("--gamma_meV", type=float, default=10.0, help="Ensanchamiento (meV)")
    parser.add_argument("--mu_eV", type=float, default=0.0, help="Potencial químico (eV)")
    parser.add_argument("--T_K", type=float, default=10.0, help="Temperatura (K)")
    parser.add_argument("--spin", type=int, default=2, help="Degeneración de spin (1 o 2)")
    parser.add_argument("--bz_lo", type=float, default=None, help="BZ inferior (a.u.)")
    parser.add_argument("--bz_hi", type=float, default=None, help="BZ superior (a.u.)")
    parser.add_argument("--rho_path", default=None, help="Ruta a rho_order_N.npy de QXTI")
    parser.add_argument("--cfg", default=None, help="Config .cfg de QXTI para leer k-grid")

    args = parser.parse_args()

    H = load_model(f"{args.model}.py", args.func)

    # Default BZ bounds per model
    bz_defaults = {
        "graphene": (-np.pi / 2.68, np.pi / 2.68),  # a0=1.42Å=2.68 Bohr
        "haldane": (-np.pi, np.pi),
        "wsm_orenstein": (-np.pi, np.pi),
        "wsm_two_weyl": (-np.pi, np.pi),
    }
    bz = (args.bz_lo, args.bz_hi) if args.bz_lo is not None else bz_defaults.get(args.model, (-np.pi, np.pi))

    # ── 1. Quick analytic check ──────────────────────────────────────────────
    quick_check(H, args.model, args.order, args.dim,
                args.omega_eV, args.gamma_meV, args.mu_eV, args.T_K,
                args.spin, args.kN, bz)

    # ── 2. Show ρ matrix elements at Γ and one generic k-point ──────────────
    print("\n--- ρ^(s) en k=Γ=(0,0,0) ---")
    T_au = args.T_K * _KB_AU
    gamma = args.gamma_meV * 1e-3 / _AU_TO_EV
    mu = args.mu_eV / _AU_TO_EV
    omega = args.omega_eV / _AU_TO_EV
    E_field = np.array([1e-4, 0.0, 0.0])

    for kpt, label in [((0.0, 0.0, 0.0), "Γ"), ((0.3, 0.2, 0.0), "k genérico")]:
        kx, ky, kz = kpt
        rhos = rho_order_s(H, kx, ky, kz, E_field, omega, gamma, mu, T_au,
                           max_order=args.order)
        print(f"\n  Punto k={label}: ω={args.omega_eV:.3f} eV  γ={args.gamma_meV} meV")
        for s in range(0, args.order + 1):
            rho = rhos.get(s)
            if rho is None:
                continue
            norm_factor = max((1e-4) ** s, 1e-60)
            print(f"  ρ^({s}) / E0^{s}:")
            r = rho / norm_factor
            for row in r:
                print("    [" + "  ".join(f"{v.real:+.4f}{v.imag:+.4f}i" for v in row) + "]")

    # ── 3. Compare with QXTI saved rho (if provided) ────────────────────────
    if args.rho_path and args.cfg:
        print(f"\n--- Comparando con QXTI: {args.rho_path} ---")
        from qxti.core.config import QXTIConfig
        from qxti.core.simulation import QXTISimulation
        from qxti.analytics.rho_analytic import compare_rho_vs_qxti

        cfg = QXTIConfig.from_file(args.cfg)
        sim = QXTISimulation(config=cfg)
        ham = sim.build_hamiltonian()
        kg = sim.build_kgrid(ham)
        tg = sim.build_timegrid(sim.build_laser_system())

        results = compare_rho_vs_qxti(
            H, args.rho_path, kg.points(), tg.t_values,
            E_field, omega, gamma=gamma, mu=mu, T_K=args.T_K,
            order=args.order,
        )
        print(f"\nError relativo promedio: {np.mean(results['error_rel']):.3e}")


if __name__ == "__main__":
    main()
