#!/usr/bin/env python3
"""Batch parameter-sweep runner for QXTI.

Queue several "sweeps" (each varies one or more input parameters over a range)
and this script runs every case **one at a time**, saving each run into its own
output folder. Typical use: fix an elliptical laser and rotate its in-plane axis
``phix`` from 0 to 360 deg, computing the HHG spectrum at each angle.

HOW TO USE
----------
Edit the ``SWEEPS`` list below (it is the queue) and run:

    python tools/run_parameter_sweep.py

Each sweep is a dict:

    {
      "name":   "graphene_phix",                 # folder name under outputs/sweeps/
      "config": "inputs/inputParams.graphene.cfg",# base input
      "mode":   "cmd",                            # cmd (HHG) | xtp | ldos
      "set":    {"laser.ellip": 0.5},             # FIXED overrides (section.field)
      "vary":   {"laser.phix": {"start": 0.0, "stop": 360.0, "steps": 12,
                                 "unit": "deg", "endpoint": False}},
      "plot":    True,     # also generate the standard graphics for each run
      "summary": True,     # (cmd only, single varied param) build an HHG-vs-param map
    }

``vary`` value forms:
    {"start": a, "stop": b, "steps": N, "unit": "deg"|"rad"|"native", "endpoint": bool}
    {"values": [ ... ]}                            # explicit list (native units)

Keys are ``section.field`` in atomic units (as in the .cfg), e.g. ``laser.phix``,
``laser.ellip``, ``laser.E0``, ``cmd.max_order``, ``kgrid`` values, or a model
parameter such as ``hamiltonian.M0`` / ``hamiltonian.phi0``. ``unit = deg`` maps a
0..360 range to radians (for phase-like fields such as ``phix``/``cep``).

Runs already finished are skipped (resume), unless ``--force``. A single sweep can
also be defined fully from the CLI (see ``--help``).

Outputs
-------
    outputs/sweeps/<name>/<label>/{input.cfg, cmd|xtp|ldos|hamiltonian/...}
    outputs/sweeps/<name>/summary_hhg.(png|npz)     # if summary requested
"""
from __future__ import annotations

import argparse
import configparser
import itertools
import os
import time
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import main as qxti_main
from qxti.utils.progress import ProgressTimer, format_duration

_AU_TO_EV = 27.211386245988
SWEEP_ROOT = Path("outputs/sweeps")

# =============================================================================
# EDIT HERE: your queue of sweeps. They run top to bottom, one case at a time.
# =============================================================================
SWEEPS = [
    {
        "name": "graphene_phix_ellip0.5",
        "config": "inputs/inputParams.graphene.cfg",
        "mode": "cmd",
        "set": {"laser.ellip": 0.5},
        "vary": {"laser.phix": {"start": 0.0, "stop": 360.0, "steps": 12,
                                "unit": "deg", "endpoint": False}},
        "plot": True,
        "summary": True,
    },
    # --- add more sweeps below; they are processed one after another ---
    # {
    #     "name": "haldane_M0_scan",
    #     "config": "inputs/inputParams.haldane_topological.cfg",
    #     "mode": "cmd",
    #     "set": {},
    #     "vary": {"hamiltonian.M0": {"values": [0.0, 0.03, 0.06, 0.09]}},
    #     "plot": True, "summary": True,
    # },
]


# --------------------------------------------------------------------------- #
# Parameter-value expansion
# --------------------------------------------------------------------------- #
def _axis_values(spec: dict) -> list[tuple[float, str]]:
    """Return [(native_value, display_string), ...] for one varied key."""
    unit = str(spec.get("unit", "native")).lower()
    if "values" in spec:
        raw = [float(v) for v in spec["values"]]
    else:
        n = int(spec["steps"])
        raw = list(np.linspace(float(spec["start"]), float(spec["stop"]), n,
                               endpoint=bool(spec.get("endpoint", True))))
    out = []
    for v in raw:
        if unit == "deg":
            out.append((float(np.deg2rad(v)), f"{v:g}deg"))
        else:
            out.append((float(v), f"{v:g}"))
    return out


def _expand(vary: dict) -> list[dict]:
    """Cartesian product of all varied keys. Each item: {key: (native, disp)}."""
    keys = list(vary.keys())
    per_key = [_axis_values(vary[k]) for k in keys]
    combos = []
    for pick in itertools.product(*per_key):
        combos.append({k: pick[i] for i, k in enumerate(keys)})
    return combos


def _label(combo: dict, index: int) -> str:
    parts = [f"{k.split('.')[-1]}={disp}" for k, (_v, disp) in combo.items()]
    return f"{index:03d}_" + "_".join(parts)


# --------------------------------------------------------------------------- #
# Per-run config file
# --------------------------------------------------------------------------- #
def _fmt(value) -> str:
    return repr(float(value)) if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)


def _write_run_config(base_cfg: Path, overrides: dict, run_dir: Path, mode: str) -> Path:
    """Write a per-run .cfg = base + overrides + explicit per-run output dirs."""
    parser = configparser.ConfigParser(inline_comment_prefixes=("#",))
    parser.optionxform = str
    if not parser.read(base_cfg):
        raise FileNotFoundError(f"Base config not found: {base_cfg}")

    for key, value in overrides.items():
        section, field = key.split(".", 1)
        if section not in parser:
            parser[section] = {}
        parser[section][field] = _fmt(value)

    # Explicit per-run output dirs (kept as-is by with_standard_output_dirs).
    def _setout(section: str, field: str, sub: str) -> None:
        if section not in parser:
            parser[section] = {}
        parser[section][field] = str(run_dir / sub)

    _setout("cmd", "output_dir", "cmd")
    _setout("xtp", "susceptibility_output_dir", "xtp")
    _setout("ldos", "output_dir", "ldos")
    _setout("hamiltonian_plots", "output_dir", "hamiltonian")

    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "input.cfg"
    with cfg_path.open("w") as fh:
        parser.write(fh)
    return cfg_path


def _run_done(run_dir: Path, mode: str) -> bool:
    marker = {"cmd": "cmd/data/current_spectrum.npz",
              "xtp": "xtp/data/xtp_susceptibility.npz",
              "ldos": "ldos/data/ldos.npz"}[mode]
    return (run_dir / marker).exists()


# --------------------------------------------------------------------------- #
# HHG summary map (cmd, single varied parameter)
# --------------------------------------------------------------------------- #
def _build_hhg_summary(sweep_name: str, combos: list[dict], run_dirs: list[Path],
                       varied_key: str, unit_is_deg: bool, emax_ev: float | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from qxti.graphics.plot_susceptibility_tensor import apply_paper_style
        apply_paper_style()
    except Exception:
        pass

    xvals, spectra, omega_ref = [], [], None
    for combo, rd in zip(combos, run_dirs):
        f = rd / "cmd" / "data" / "current_spectrum.npz"
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=True)
        omega = np.asarray(d["omega_axis"], dtype=float)
        mag = np.asarray(d["current_total_magnitude"], dtype=float)
        pos = omega > 0
        omega, mag = omega[pos], mag[pos]
        if omega_ref is None:
            omega_ref = omega
        # display value of the single varied parameter (degrees if unit=deg)
        native, disp = combo[varied_key]
        xvals.append(float(disp[:-3]) if unit_is_deg else native)
        spectra.append(np.interp(omega_ref, omega, mag) if omega.shape != omega_ref.shape else mag)

    if not spectra:
        print("[sweep] no HHG spectra found for the summary map.")
        return

    xvals = np.asarray(xvals)
    order = np.argsort(xvals)
    xvals = xvals[order]
    Z = np.log10(np.maximum(np.array(spectra)[order].T, 1e-30))   # (nomega, nvals)
    E_eV = omega_ref * _AU_TO_EV

    # Focus the energy axis on the harmonic region (the FFT axis runs far higher).
    if emax_ev is not None and emax_ev > 0:
        keep = E_eV <= float(emax_ev)
        E_eV, Z = E_eV[keep], Z[keep]

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    im = ax.pcolormesh(xvals, E_eV, Z, shading="auto", cmap="inferno")
    ax.set_xlabel(varied_key.split(".")[-1] + (" (deg)" if unit_is_deg else " (a.u.)"))
    ax.set_ylabel(r"$E_\mathrm{photon}\;(\mathrm{eV})$")
    ax.set_title(f"HHG vs {varied_key.split('.')[-1]} - {sweep_name}")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r"$\log_{10}\,|J(\omega)|$")
    fig.tight_layout()
    out_png = SWEEP_ROOT / sweep_name / "summary_hhg.png"
    fig.savefig(out_png, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    np.savez_compressed(SWEEP_ROOT / sweep_name / "summary_hhg.npz",
                        swept_values=xvals, energy_ev=E_eV, log_current=Z, varied_key=varied_key)
    print(f"[sweep] summary map: {out_png}")


# --------------------------------------------------------------------------- #
# Sweep execution
# --------------------------------------------------------------------------- #
def run_sweep(sweep: dict, *, force: bool, do_plot: bool | None) -> None:
    name = sweep["name"]
    base_cfg = Path(sweep["config"]).expanduser()
    mode = str(sweep.get("mode", "cmd")).lower()
    fixed = dict(sweep.get("set", {}))
    combos = _expand(sweep["vary"])
    plot = sweep.get("plot", True) if do_plot is None else do_plot

    sweep_dir = SWEEP_ROOT / name
    run_dirs: list[Path] = []
    print(f"\n===== SWEEP '{name}': {len(combos)} runs, mode={mode}, base={base_cfg.name} =====")
    timer = ProgressTimer(total=len(combos))

    for i, combo in enumerate(combos):
        label = _label(combo, i)
        run_dir = sweep_dir / label
        run_dirs.append(run_dir)
        overrides = {**fixed, **{k: v for k, (v, _d) in combo.items()}}
        pretty = ", ".join(f"{k}={d}" for k, (_v, d) in combo.items())

        if _run_done(run_dir, mode) and not force:
            print(f"[{i + 1}/{len(combos)}] {pretty}  -> already done, skipping.")
            timer.completed = i + 1
            continue

        print(f"\n[{i + 1}/{len(combos)}] {pretty}  (ETA {timer.eta_text()})")
        cfg_path = _write_run_config(base_cfg, overrides, run_dir, mode)
        try:
            qxti_main.run_from_config_path(cfg_path, mode=mode)
            if plot:
                from qxti.graphics.graphics import plot_all_graphics_from_saved_data
                plot_all_graphics_from_saved_data(cfg_path)
        except Exception as exc:  # keep the batch alive if one run fails
            print(f"[{i + 1}/{len(combos)}] FAILED: {exc!r}")
        timer.completed = i + 1

    print(f"===== SWEEP '{name}' done in {format_duration(timer.elapsed_seconds)} =====")

    if sweep.get("summary", False) and mode == "cmd" and len(sweep["vary"]) == 1:
        varied_key = next(iter(sweep["vary"]))
        unit_is_deg = str(sweep["vary"][varied_key].get("unit", "")).lower() == "deg"
        # Cap the summary energy axis at ~14 harmonics of the laser photon energy.
        try:
            from qxti.core import QXTIConfig
            omega = float(QXTIConfig.from_file(base_cfg).laser.omega)
            emax_ev = 14.0 * omega * _AU_TO_EV
        except Exception:
            emax_ev = None
        emax_ev = float(sweep.get("summary_emax_ev", emax_ev) or 0) or emax_ev
        _build_hhg_summary(name, combos, run_dirs, varied_key, unit_is_deg, emax_ev=emax_ev)


def _sweep_from_cli(args) -> dict:
    vary_spec: dict
    if args.values is not None:
        vary_spec = {"values": [float(x) for x in args.values.split(",")]}
    else:
        vary_spec = {"start": args.start, "stop": args.stop, "steps": args.steps,
                     "unit": args.unit, "endpoint": not args.no_endpoint}
    fixed = {}
    for item in (args.set or []):
        k, v = item.split("=", 1)
        fixed[k.strip()] = float(v) if _isfloat(v) else v.strip()
    return {
        "name": args.name or f"{Path(args.config).stem}_{args.vary.split('.')[-1]}",
        "config": args.config, "mode": args.mode, "set": fixed,
        "vary": {args.vary: vary_spec}, "plot": not args.no_plot, "summary": not args.no_summary,
    }


def _isfloat(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Queue and run QXTI parameter sweeps one at a time.")
    p.add_argument("--config", help="Base config (enables single-sweep CLI mode instead of the SWEEPS list).")
    p.add_argument("--mode", default="cmd", choices=("cmd", "xtp", "ldos"))
    p.add_argument("--vary", help="Parameter to vary, e.g. laser.phix")
    p.add_argument("--from", dest="start", type=float, help="Range start.")
    p.add_argument("--to", dest="stop", type=float, help="Range stop.")
    p.add_argument("--steps", type=int, default=12, help="Number of values.")
    p.add_argument("--unit", default="native", choices=("native", "deg", "rad"))
    p.add_argument("--values", help="Explicit comma-separated values (native units) instead of a range.")
    p.add_argument("--no-endpoint", action="store_true", help="Exclude the stop value (good for periodic 0..360).")
    p.add_argument("--set", action="append", help="Fixed override section.field=value (repeatable).")
    p.add_argument("--name", help="Sweep name (output folder).")
    p.add_argument("--force", action="store_true", help="Re-run cases even if their output already exists.")
    p.add_argument("--no-plot", action="store_true", help="Do not generate per-run graphics.")
    p.add_argument("--no-summary", action="store_true", help="Do not build the HHG summary map.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.config and args.vary:
        sweeps = [_sweep_from_cli(args)]
    else:
        sweeps = SWEEPS
    if not sweeps:
        print("No sweeps defined. Edit the SWEEPS list at the top of this file, "
              "or pass --config/--vary on the CLI.")
        return 0
    t0 = time.perf_counter()
    for sweep in sweeps:
        run_sweep(sweep, force=args.force, do_plot=(False if args.no_plot else None))
    print(f"\nAll sweeps finished in {format_duration(time.perf_counter() - t0)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
