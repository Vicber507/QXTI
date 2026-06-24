from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Callable


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0.0:
        return "unknown"

    if seconds < 1.0:
        return f"{seconds:.2f}s"
    if seconds < 10.0:
        return f"{seconds:.1f}s"

    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def format_bytes(num_bytes: int | float) -> str:
    value = float(num_bytes)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


@dataclass(slots=True)
class ProgressTimer:
    total: int
    completed: int = 0
    min_completed_for_eta: int = 1
    start_time: float = field(default_factory=time.perf_counter)

    @property
    def elapsed_seconds(self) -> float:
        return float(time.perf_counter() - self.start_time)

    @property
    def remaining(self) -> int:
        return max(0, int(self.total) - int(self.completed))

    def advance(self, steps: int = 1) -> None:
        self.completed += int(steps)

    def eta_seconds(self) -> float:
        minimum = max(1, int(self.min_completed_for_eta))
        if self.completed < minimum:
            return math.inf
        if self.total <= self.completed:
            return 0.0
        rate = self.completed / max(self.elapsed_seconds, 1.0e-12)
        return self.remaining / rate

    def eta_text(self) -> str:
        minimum = max(1, int(self.min_completed_for_eta))
        if self.total <= self.completed:
            return "0:00"
        if self.completed < minimum:
            return "unknown"
        return format_duration(self.eta_seconds())


@dataclass(slots=True)
class DetailedProgressReporter:
    """Throttle detailed progress messages for one long-running inner task."""

    label: str
    total: int
    emit: Callable[[str], None]
    interval_seconds: float = 15.0
    min_completed_for_eta: int = 1
    completed: int = 0
    _current_message: str = ""
    _last_emit_time: float = field(default_factory=time.perf_counter)
    _timer: ProgressTimer = field(init=False)

    def __post_init__(self) -> None:
        self._timer = ProgressTimer(
            total=max(1, int(self.total)),
            min_completed_for_eta=max(1, int(self.min_completed_for_eta)),
        )

    def advance(self, steps: int = 1, message: str = "") -> None:
        if message:
            self._current_message = str(message)
        self.completed = min(int(self.total), self.completed + max(0, int(steps)))
        self._timer.completed = self.completed
        self._emit_if_due()

    def note(self, message: str, *, force: bool = False) -> None:
        self._current_message = str(message)
        self._emit_if_due(force=force)

    def finish(self, message: str | None = None) -> None:
        if message:
            self._current_message = str(message)
        self.completed = int(self.total)
        self._timer.completed = self.completed
        self._emit_if_due(force=True)

    def _emit_if_due(self, *, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and self.interval_seconds > 0.0 and (now - self._last_emit_time) < self.interval_seconds:
            return
        self._last_emit_time = now
        fraction = min(1.0, self.completed / max(int(self.total), 1))
        percent = 100.0 * fraction
        eta = self._timer.eta_text()
        elapsed = format_duration(self._timer.elapsed_seconds)
        suffix = f"elapsed {elapsed}" if eta == "unknown" else f"elapsed {elapsed}, ETA {eta}"
        message = self._current_message or "working"
        self.emit(
            f"{self.label}: {message} "
            f"({percent:.1f}% complete, {self.completed}/{self.total} work units, {suffix})."
        )
