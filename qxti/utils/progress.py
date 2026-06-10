from __future__ import annotations

from dataclasses import dataclass, field
import math
import time


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
