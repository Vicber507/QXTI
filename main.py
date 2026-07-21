from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path

# Cross-platform runtime setup, BEFORE importing NumPy (via qxti.core):
#  - a writable matplotlib/font cache dir that exists on mac/win/linux;
#  - pin BLAS/OpenMP to 1 thread so the k-loop ThreadPool owns the parallelism
#    (no cores x BLAS oversubscription).  Override BLAS with QXTI_BLAS_THREADS.
from qxti.utils.parallel import configure_runtime_env, parallel_plan
configure_runtime_env()

from qxti.core import QXTIConfig, QXTISimulation, SusceptibilityScanRunner, LDOSRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run QXTI physics calculations from a SINGLE input file. Choose the "
            "calculation with a flag (named after its config section):\n"
            "  -cmd  : time-domain / HHG current spectrum (the [cmd] + [laser] workflow)\n"
            "  -xtp  : conductivity & susceptibility tensors sigma/chi(omega) "
            "(the [xtp] frequency sweep)\n"
            "  -ldos : density of states g(E), projected PDOS and spectral A(k,E) "
            "(the [ldos] workflow)\n"
            "(-hhg is kept as a deprecated alias of -cmd.)\n"
            "If no flag is given, main.py auto-detects from [xtp] susceptibility_enabled "
            "(backward compatible)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="inputParams.cfg",
        help="Path to the configuration file. Defaults to inputParams.cfg.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "-cmd", "--cmd", "-hhg", "--hhg", dest="cmd", action="store_true",
        help="Run the time-domain / HHG current workflow (uses [cmd], [laser]). "
        "-hhg is a deprecated alias.",
    )
    mode.add_argument(
        "-xtp", "--xtp", dest="xtp", action="store_true",
        help="Run the conductivity/susceptibility tensor sweep (uses [xtp]).",
    )
    mode.add_argument(
        "-ldos", "--ldos", dest="ldos", action="store_true",
        help="Run the density-of-states calculation (uses [ldos]).",
    )
    return parser


def run_from_config_path(
    config_path: str | Path,
    *,
    mode: str | None = None,
) -> dict[str, Path]:
    """Run one calculation from a single config.

    ``mode`` is "cmd" (alias "hhg"), "xtp", "ldos", or None. When None it falls
    back to the legacy auto-detection (``[xtp] susceptibility_enabled``). A SINGLE
    input file can hold the [cmd]/[laser] (HHG), [xtp] (tensor sweep + its solver
    params) and [ldos] (density of states) sections; the flag selects which runs.
    """
    resolved_path = Path(config_path).expanduser()
    config = QXTIConfig.from_file(resolved_path)
    # Standardize outputs to outputs/<model_name>/{cmd,xtp,ldos,hamiltonian}.
    # Guarded with hasattr so SimpleNamespace test mocks (which patch from_file)
    # still work.
    if hasattr(config, "with_standard_output_dirs"):
        config = config.with_standard_output_dirs()

    forced = mode is not None
    # Accept the deprecated "hhg" spelling as an alias of "cmd".
    if mode == "hhg":
        mode = "cmd"
    if mode is None:
        mode = "xtp" if config.xtp.susceptibility_enabled else "cmd"

    if mode == "ldos":
        if forced and not getattr(getattr(config, "ldos", None), "enabled", True):
            config = replace(config, ldos=replace(config.ldos, enabled=True))
        method = getattr(getattr(config, "ldos", None), "method", "eigenvalues")
        print(f"[main] Mode: -ldos -> density of states (method={method}).")
        return LDOSRunner(config=config).run()

    if mode == "xtp":
        # An explicit -xtp flag enables the sweep even if the config left
        # susceptibility_enabled = false (so one file can serve both modes).
        if forced and not config.xtp.susceptibility_enabled:
            config = replace(config, xtp=replace(config.xtp, susceptibility_enabled=True))
        method = getattr(getattr(config, "xtp", None), "susceptibility_method", "simulation")
        print(f"[main] Mode: -xtp -> conductivity/susceptibility tensor sweep (method={method}).")
        return SusceptibilityScanRunner(config=config).run()

    # mode == "cmd"
    if forced and not getattr(getattr(config, "cmd", None), "enabled", True):
        config = replace(config, cmd=replace(config.cmd, enabled=True))
    method = getattr(getattr(config, "cmd", None), "response_method", "simulation")
    print(f"[main] Mode: -cmd -> time-domain / HHG current spectrum (method={method}).")
    return QXTISimulation(config=config).run()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    mode = "cmd" if args.cmd else ("xtp" if args.xtp else ("ldos" if args.ldos else None))
    config_path = Path(args.config).expanduser()
    print(f"[main] Parallelism: {parallel_plan()} "
          f"(set [cmd]/[xtp]/[ldos] n_workers to override; 0 = auto).")
    outputs = run_from_config_path(config_path, mode=mode)

    if not outputs:
        print(f"No outputs were generated for {config_path}.")
        return 0

    print(f"Generated data outputs from {config_path}:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
