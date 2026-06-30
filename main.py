from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path

# Keep matplotlib/font caches in a writable place for local runs.
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

from qxti.core import QXTIConfig, QXTISimulation, SusceptibilityScanRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run QXTI physics calculations from a SINGLE input file. Choose the "
            "calculation with a flag:\n"
            "  -hhg : time-domain / HHG current spectrum (the [cmd] + [laser] workflow)\n"
            "  -xtp : conductivity & susceptibility tensors sigma/chi(omega) "
            "(the [xtp] + [susceptibility_solver] frequency sweep)\n"
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
        "-hhg", "--hhg", dest="hhg", action="store_true",
        help="Run the HHG / time-domain current workflow (uses [cmd], [laser]).",
    )
    mode.add_argument(
        "-xtp", "--xtp", dest="xtp", action="store_true",
        help="Run the conductivity/susceptibility tensor sweep (uses [xtp], [susceptibility_solver]).",
    )
    return parser


def run_from_config_path(
    config_path: str | Path,
    *,
    mode: str | None = None,
) -> dict[str, Path]:
    """Run one calculation from a single config.

    ``mode`` is "hhg", "xtp", or None. When None it falls back to the legacy
    auto-detection (``[xtp] susceptibility_enabled``). A SINGLE input file can
    hold both the [cmd]/[laser] (HHG) and [xtp]/[susceptibility_solver] (tensor)
    sections; the flag selects which one runs.
    """
    resolved_path = Path(config_path).expanduser()
    config = QXTIConfig.from_file(resolved_path)
    # Standardize outputs to outputs/<model_name>/{cmd,xtp,hamiltonian}. Guarded
    # with hasattr so SimpleNamespace test mocks (which patch from_file) still work.
    if hasattr(config, "with_standard_output_dirs"):
        config = config.with_standard_output_dirs()

    forced = mode is not None
    if mode is None:
        mode = "xtp" if config.xtp.susceptibility_enabled else "hhg"

    if mode == "xtp":
        # An explicit -xtp flag enables the sweep even if the config left
        # susceptibility_enabled = false (so one file can serve both modes).
        if forced and not config.xtp.susceptibility_enabled:
            config = replace(config, xtp=replace(config.xtp, susceptibility_enabled=True))
        method = getattr(getattr(config, "xtp", None), "susceptibility_method", "simulation")
        print(f"[main] Mode: -xtp -> conductivity/susceptibility tensor sweep (method={method}).")
        return SusceptibilityScanRunner(config=config).run()

    # mode == "hhg"
    if forced and not getattr(getattr(config, "cmd", None), "enabled", True):
        config = replace(config, cmd=replace(config.cmd, enabled=True))
    method = getattr(getattr(config, "cmd", None), "response_method", "simulation")
    print(f"[main] Mode: -hhg -> time-domain / HHG current spectrum (method={method}).")
    return QXTISimulation(config=config).run()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    mode = "hhg" if args.hhg else ("xtp" if args.xtp else None)
    config_path = Path(args.config).expanduser()
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
