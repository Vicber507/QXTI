#!/usr/bin/env python3
"""Compatibility wrapper for the old HHG waterfall script name.

The implementation now lives in ``plot_hhg_waterfall_phiz.py`` and uses only
phi_z folders, with linearly polarized light by default.
"""
from __future__ import annotations

from plot_hhg_waterfall_phiz import main


if __name__ == "__main__":
    raise SystemExit(main())
