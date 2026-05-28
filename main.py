from __future__ import annotations

import argparse
import os
from pathlib import Path

# Keep matplotlib/font caches in a writable place for local runs.
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

from qxti.core import QXTISimulation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run QXTI intrinsic Hamiltonian plots from an inputParams.cfg file."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="inputParams.cfg",
        help="Path to the configuration file. Defaults to inputParams.cfg.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    outputs = QXTISimulation.from_file(config_path).run()

    if not outputs:
        print(f"No outputs were generated for {config_path}.")
        return 0

    print(f"Generated outputs from {config_path}:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
