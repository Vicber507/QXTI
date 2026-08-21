"""Tests for the frequency-domain perturbative single-pulse treatment.

``[cmd] pfddm_pulse`` selects how the pfddm engine handles a SINGLE pulse:
  * ``envelope``   (default) — fast closed-form χ⁽ˢ⁾(sω₀) dressed with envelope^s
                    (quasi-CW / adiabatic; exact for many-cycle pulses);
  * ``full_field`` — drive the SAME perturbative recursion with the full pulse field
                    E(t) (realistic pulsed response: true harmonic bandwidth/chirp).
Multi-colour drives always use the full-field path regardless.
"""
from __future__ import annotations

import os
from dataclasses import replace as R
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qxti.core.config import QXTIConfig, _canonical_pfddm_pulse

CFG = "inputs/inputParams.graphene.cfg"


def _cfg(mode: str):
    cfg = QXTIConfig.from_file(CFG).with_standard_output_dirs()
    return R(cfg, kgrid=R(cfg.kgrid, k_points=(16, 16)),
             cmd=R(cfg.cmd, max_order=3, response_method="pfddm", pfddm_pulse=mode))


def test_pfddm_pulse_default_is_envelope():
    assert QXTIConfig.from_file(CFG).cmd.pfddm_pulse == "envelope"


def test_pfddm_pulse_aliases_and_validation():
    assert _canonical_pfddm_pulse("full") == "full_field"
    assert _canonical_pfddm_pulse("exact") == "full_field"
    assert _canonical_pfddm_pulse("adiabatic") == "envelope"
    with pytest.raises(ValueError):
        _canonical_pfddm_pulse("euler_supreme")


def test_single_pulse_full_field_runs_and_agrees_on_linear_response():
    """full_field must run for a SINGLE pulse (relabeled 'pfddm-full-field') and agree
    with the envelope approximation on the gauge-invariant linear response H1."""
    from qxti.analytics.theory_response import compute_hhg_spectrum

    env = compute_hhg_spectrum(_cfg("envelope"), progress=False)
    full = compute_hhg_spectrum(_cfg("full_field"), progress=False)

    assert env["method"] == "theory"
    assert full["method"] == "pfddm-full-field"       # single pulse routed through full field

    w0 = float(_cfg("envelope").laser.omega)

    def h1(res):
        f = np.asarray(res["dataset"]["omega_axis"])
        s = np.asarray(res["dataset"]["current_total_magnitude"])
        assert np.isfinite(s).all()
        return s[int(np.argmin(np.abs(f - w0)))]

    ratio = h1(full) / h1(env)
    assert 0.9 < ratio < 1.1, f"H1 envelope vs full_field disagree: ratio {ratio:.3f}"
