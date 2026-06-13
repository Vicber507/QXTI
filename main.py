from __future__ import annotations

import argparse
import os
from pathlib import Path

# Keep matplotlib/font caches in a writable place for local runs.
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

from qxti.core import QXTIConfig, QXTISimulation, SusceptibilityScanRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run QXTI physics calculations from an input file. "
            "If [xtp] susceptibility_enabled = true, main.py automatically runs the "
            "dedicated susceptibility workflow."
        )
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="inputParams.cfg",
        help="Path to the configuration file. Defaults to inputParams.cfg.",
    )
    return parser


def run_from_config_path(config_path: str | Path) -> dict[str, Path]:
    resolved_path = Path(config_path).expanduser()
    config = QXTIConfig.from_file(resolved_path)

    if config.xtp.susceptibility_enabled:
        print(
            "[main] Detected [xtp] susceptibility_enabled = true in the input. "
            "Running the dedicated susceptibility workflow "
            "(no rho_order files or CMD datasets will be written)."
        )
        return SusceptibilityScanRunner(config=config).run()

    return QXTISimulation(config=config).run()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    outputs = run_from_config_path(config_path)

    if not outputs:
        print(f"No outputs were generated for {config_path}.")
        return 0

    print(f"Generated data outputs from {config_path}:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
