from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main as main_module


def test_main_routes_to_susceptibility_scan_when_enabled(monkeypatch) -> None:
    config = SimpleNamespace(
        xtp=SimpleNamespace(susceptibility_enabled=True),
    )
    expected_outputs = {"xtp_susceptibility_data": Path("outputs/xtp_susceptibility.npz")}

    class FakeRunner:
        def __init__(self, *, config):
            self.config = config

        def run(self):
            return expected_outputs

    class FakeSimulation:
        def __init__(self, *, config):
            raise AssertionError("QXTISimulation should not be used for susceptibility-scan inputs.")

    monkeypatch.setattr(main_module.QXTIConfig, "from_file", lambda path: config)
    monkeypatch.setattr(main_module, "SusceptibilityScanRunner", FakeRunner)
    monkeypatch.setattr(main_module, "QXTISimulation", FakeSimulation)

    outputs = main_module.run_from_config_path("inputParams.susceptibility.cfg")

    assert outputs == expected_outputs


def test_main_routes_to_standard_simulation_when_scan_disabled(monkeypatch) -> None:
    config = SimpleNamespace(
        xtp=SimpleNamespace(susceptibility_enabled=False),
    )
    expected_outputs = {"xtp_current_spectrum_data": Path("outputs/current_spectrum.npz")}

    class FakeSimulation:
        def __init__(self, *, config):
            self.config = config

        def run(self):
            return expected_outputs

    class FakeRunner:
        def __init__(self, *, config):
            raise AssertionError("SusceptibilityScanRunner should not be used for standard inputs.")

    monkeypatch.setattr(main_module.QXTIConfig, "from_file", lambda path: config)
    monkeypatch.setattr(main_module, "QXTISimulation", FakeSimulation)
    monkeypatch.setattr(main_module, "SusceptibilityScanRunner", FakeRunner)

    outputs = main_module.run_from_config_path("inputParams.cfg")

    assert outputs == expected_outputs
