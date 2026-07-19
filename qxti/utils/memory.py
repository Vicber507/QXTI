"""Cross-platform RAM accounting and block sizing for large k-grid runs.

The goal (user-directed): NEVER let a compute job exhaust physical memory and
force the OS to swap itself to death / shut the machine off.  Every heavy loop
should (a) size its working blocks so the peak fits in the RAM currently
available minus a safety reserve (default 1 GB), and (b) re-check before each
block, shrinking or aborting cleanly instead of thrashing.

`available_bytes()` works on Linux, macOS and Windows with pure-stdlib
fallbacks (no psutil required); it uses psutil when present.
"""
from __future__ import annotations

import ctypes
import os
import platform
import re
import subprocess
import sys

_GB = 1024.0 ** 3


# ---------------------------------------------------------------------------
# Physical memory query (available = what we can allocate before pressure)
# ---------------------------------------------------------------------------
def _available_psutil() -> int | None:
    try:
        import psutil  # type: ignore
        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def _available_linux() -> int | None:
    try:
        with open("/proc/meminfo", "r") as fh:
            text = fh.read()
    except OSError:
        return None
    m = re.search(r"^MemAvailable:\s+(\d+)\s*kB", text, re.MULTILINE)
    if m:
        return int(m.group(1)) * 1024
    # Older kernels: approximate with MemFree + Buffers + Cached.
    total = 0
    for key in ("MemFree", "Buffers", "Cached"):
        mm = re.search(rf"^{key}:\s+(\d+)\s*kB", text, re.MULTILINE)
        if mm:
            total += int(mm.group(1)) * 1024
    return total or None


def _page_size_darwin() -> int:
    try:
        return int(subprocess.check_output(["sysctl", "-n", "hw.pagesize"], text=True).strip())
    except Exception:
        return 4096


def _available_darwin() -> int | None:
    """macOS: (free + inactive + speculative + purgeable) pages * page size.

    These are the page classes the VM can hand back to a new allocation without
    swapping.  Wired/active/compressed pages are excluded, so this tracks the
    real headroom (it collapses toward ~0 exactly when the machine starts to
    thrash, which is what we want to guard against).
    """
    try:
        out = subprocess.check_output(["vm_stat"], text=True)
    except Exception:
        return None
    page = _page_size_darwin()
    counts = {}
    for line in out.splitlines():
        m = re.match(r'"?Pages ([^:"]+)"?:\s+(\d+)', line)
        if m:
            counts[m.group(1).strip().lower()] = int(m.group(2))
    keys = ("free", "inactive", "speculative", "purgeable")
    if not any(k in counts for k in keys):
        return None
    return sum(counts.get(k, 0) for k in keys) * page


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _available_windows() -> int | None:
    try:
        stat = _MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
            return int(stat.ullAvailPhys)
    except Exception:
        return None
    return None


def available_bytes() -> int:
    """Best-effort physically-available RAM in bytes (0 if unknowable)."""
    val = _available_psutil()
    if val is not None:
        return val
    system = platform.system()
    if system == "Linux":
        val = _available_linux()
    elif system == "Darwin":
        val = _available_darwin()
    elif system == "Windows":
        val = _available_windows()
    else:
        val = None
    return int(val) if val else 0


def total_bytes() -> int:
    """Best-effort total physical RAM in bytes (0 if unknowable)."""
    try:
        import psutil  # type: ignore
        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    try:
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except Exception:
        pass
    if platform.system() == "Windows":
        try:
            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
                return int(stat.ullTotalPhys)
        except Exception:
            pass
    return 0


def fmt_gb(nbytes: float) -> str:
    return f"{nbytes / _GB:.2f} GB"


# ---------------------------------------------------------------------------
# Block sizing: choose how many "units" (e.g. k-planes) fit in the budget
# ---------------------------------------------------------------------------
def memory_budget_bytes(reserve_gb: float = 1.0, *, fraction: float = 0.9) -> float:
    """Bytes a job may use right now while leaving ``reserve_gb`` free.

    ``fraction`` (<1) discounts the estimate so allocator overhead / temporary
    spikes don't cross the reserve line.  If RAM can't be queried we fall back
    to a conservative slice of total (or 2 GB if even that is unknown).
    """
    avail = available_bytes()
    if avail <= 0:
        tot = total_bytes()
        avail = int(tot * 0.5) if tot > 0 else int(2 * _GB)
    budget = (avail - reserve_gb * _GB) * float(fraction)
    return max(budget, 0.0)


def pick_block_count(
    n_units: int,
    bytes_per_unit: float,
    *,
    reserve_gb: float = 1.0,
    min_units_per_block: int = 1,
    halo_units: int = 0,
) -> tuple[int, int]:
    """Return ``(units_per_block, n_blocks)`` so peak memory stays in budget.

    ``bytes_per_unit`` must already fold in the per-unit *peak* live footprint
    (all simultaneous arrays).  ``halo_units`` are extra units carried by every
    block (counted against the budget but not reducing the stride).
    """
    n_units = int(n_units)
    budget = memory_budget_bytes(reserve_gb=reserve_gb)
    if bytes_per_unit <= 0:
        return n_units, 1
    fit = int(budget // bytes_per_unit) - int(halo_units)
    per_block = max(int(min_units_per_block), min(n_units, fit))
    per_block = max(per_block, 1)
    n_blocks = (n_units + per_block - 1) // per_block
    return per_block, n_blocks


# ---------------------------------------------------------------------------
# Runtime guard: verify headroom before a heavy allocation
# ---------------------------------------------------------------------------
class MemoryGuardError(RuntimeError):
    """Raised when the requested work cannot run without crossing the reserve."""


def ensure_headroom(
    need_bytes: float,
    *,
    reserve_gb: float = 1.0,
    label: str = "",
    gc_collect: bool = True,
) -> bool:
    """Check that ``need_bytes`` can be allocated while keeping ``reserve_gb`` free.

    Returns True if OK.  If not, runs a gc pass (optional) and re-checks; still
    short -> raises ``MemoryGuardError`` with a clear message instead of letting
    the OS thrash/OOM.  If RAM is unknowable it permits the work (returns True).
    """
    def _headroom() -> int:
        avail = available_bytes()
        return -1 if avail <= 0 else int(avail - reserve_gb * _GB)

    head = _headroom()
    if head < 0:
        return True  # cannot measure -> don't block the run
    if head >= need_bytes:
        return True
    if gc_collect:
        import gc
        gc.collect()
        head = _headroom()
        if head < 0 or head >= need_bytes:
            return True
    where = f" [{label}]" if label else ""
    raise MemoryGuardError(
        f"Insufficient RAM{where}: need {fmt_gb(need_bytes)} but only "
        f"{fmt_gb(max(head, 0))} is available above the {reserve_gb:g} GB reserve "
        f"({fmt_gb(available_bytes())} free now). Reduce the grid or block size."
    )


def status_line(reserve_gb: float = 1.0) -> str:
    """One-line human summary of the current memory situation."""
    avail = available_bytes()
    tot = total_bytes()
    parts = [f"RAM available {fmt_gb(avail)}"]
    if tot:
        parts.append(f"of {fmt_gb(tot)} ({100.0 * avail / tot:.0f}%)")
    parts.append(f"| budget above {reserve_gb:g} GB reserve: {fmt_gb(memory_budget_bytes(reserve_gb))}")
    return " ".join(parts)


if __name__ == "__main__":  # quick self-check: python -m qxti.utils.memory
    print(f"platform      : {platform.system()} ({platform.machine()})")
    print(f"python        : {sys.version.split()[0]}")
    print(f"total RAM     : {fmt_gb(total_bytes())}")
    print(f"available RAM : {fmt_gb(available_bytes())}")
    print(status_line(1.0))
