from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qxti.core.simulation import QXTISimulation
from qxti.utils.progress import ProgressTimer, format_duration


def test_format_duration_shows_subsecond_values() -> None:
    assert format_duration(0.034) == "0.03s"
    assert format_duration(2.34) == "2.3s"
    assert format_duration(12.0) == "0:12"


def test_progress_timer_can_delay_eta_until_enough_samples() -> None:
    timer = ProgressTimer(total=4, min_completed_for_eta=2)

    assert timer.eta_text() == "unknown"
    timer.advance()
    assert timer.eta_text() == "unknown"
    timer.advance()
    assert timer.eta_text() != "unknown"


def test_load_saved_rho_order_paths_reports_memmap(tmp_path: Path, capsys) -> None:
    path = tmp_path / "rho_order_0.npy"
    np.save(path, np.zeros((2, 3, 2, 2), dtype=np.complex64))

    loaded = QXTISimulation._load_saved_rho_order_paths({0: path})
    captured = capsys.readouterr().out

    assert 0 in loaded
    assert isinstance(loaded[0], np.memmap)
    assert "Rho map 1/1: memory-mapped 'rho_order_0.npy'" in captured
