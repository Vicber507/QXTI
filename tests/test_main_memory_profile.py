from __future__ import annotations

import csv
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from textwrap import dedent
from typing import TextIO

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = PROJECT_ROOT / "outputs" / "memory"
ORDER_EVENT_PATTERN = re.compile(r"CMD saved order (\d+):")
LEGACY_ORDER_EVENT_PATTERN = re.compile(r"CMD order (\d+)/\d+ completed\.")


def _rss_mib(pid: int) -> float:
    status_path = Path("/proc") / str(pid) / "status"
    try:
        with status_path.open("r", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except FileNotFoundError:
        return 0.0
    return 0.0


def _write_low_memory_config(tmp_path: Path) -> Path:
    model_path = tmp_path / "memory_profile_model.py"
    model_path.write_text(
        dedent(
            """
            from __future__ import annotations

            import numpy as np

            MODEL_NAME = "memory-profile-eight-band"
            BASIS_SIZE = 8
            DIMENSION = 1
            BASIS_TYPE = "synthetic"
            IS_PERIODIC = True
            DEFAULT_PARAMS = {"bandwidth": 0.1}
            DEFAULT_LATTICE = {
                "lattice_constants": {"a": 1.0},
                "real_space_vectors": {"a1": [1.0]},
            }

            def H(kx, ky, kz, params):
                del ky, kz
                bandwidth = float(params["bandwidth"])
                centers = np.linspace(-0.8, 0.8, BASIS_SIZE)
                slopes = np.linspace(-1.0, 1.0, BASIS_SIZE)
                energies = centers + bandwidth * slopes * np.cos(kx)
                return np.diag(energies).astype(complex)
            """
        ).strip()
        + "\n",
        encoding="ascii",
    )

    config_path = tmp_path / "input_memory_profile.cfg"
    config_path.write_text(
        dedent(
            f"""
            [hamiltonian]
            source_file = {model_path}

            [hamiltonian_plots]
            enabled = false

            [kgrid]
            dimension = 1
            k_points = [81]

            [timegrid]
            dt = 1.0
            t_min = 0.0
            t_max = 120.0
            Nt = 401
            fft_window = hann
            zero_padding = false
            padding_factor = 1

            [laser]
            omega = 0.061
            E0 = 0.000049
            ellip = 0.0
            ncycles = 1.0
            cep = 0.0
            t0 = 0.0
            phix = 0.0
            thetaz = 0.0
            phiz = 0.0
            envname = gauss

            [cmd]
            enabled = true
            output_dir = {tmp_path / "cmd"}
            max_order = 4
            population_time = 110
            coherence_time = 220
            temperature = 0.02
            fermi_level = 0.0
            distribution = valence_occupation
            basis = band
            gauge = length
            include_intraband = true
            include_interband = true
            include_dephasing = true
            solver = RKF45
            solver_tolerance = 2.0e-3
            solver_max_iterations = 0
            solver_max_rejections = 20000
            solver_h_min = 1.0e-12
            solver_safety_factor = 0.90
            solver_min_factor = 0.20
            solver_max_factor = 6.00
            save_frequency_domain = false

            [xtp]
            bz_mask_enabled = false
            bz_mask_radius_percent = 80
            bz_mask_sigma = 0.5
            """
        ).strip()
        + "\n",
        encoding="ascii",
    )
    return config_path


def _collect_stream(
    stream: TextIO,
    start_time: float,
    lines: list[tuple[float, str]],
) -> None:
    for line in stream:
        lines.append((time.monotonic() - start_time, line.rstrip()))


def _profile_command(
    command: list[str],
    *,
    config_path: Path,
    sample_interval: float,
    order_event_pattern: re.Pattern[str],
    failure_name: str,
) -> tuple[list[tuple[float, float]], list[tuple[float, int, str]]]:
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", "/tmp")
    env.setdefault("XDG_CACHE_HOME", "/tmp")
    env["PYTHONUNBUFFERED"] = "1"
    env["QXTI_MEMORY_PROFILE_CONFIG"] = str(config_path)

    start = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    assert process.stdout is not None
    assert process.stderr is not None

    stdout_lines: list[tuple[float, str]] = []
    stderr_lines: list[tuple[float, str]] = []
    stdout_thread = threading.Thread(
        target=_collect_stream,
        args=(process.stdout, start, stdout_lines),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_collect_stream,
        args=(process.stderr, start, stderr_lines),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    samples: list[tuple[float, float]] = []
    while process.poll() is None:
        samples.append((time.monotonic() - start, _rss_mib(process.pid)))
        time.sleep(sample_interval)

    samples.append((time.monotonic() - start, _rss_mib(process.pid)))
    process.wait()
    stdout_thread.join(timeout=2.0)
    stderr_thread.join(timeout=2.0)
    if process.returncode != 0:
        raise AssertionError(
            f"{failure_name} failed during memory profiling.\n"
            f"stdout:\n{_format_lines(stdout_lines)}\n\n"
            f"stderr:\n{_format_lines(stderr_lines)}"
        )

    order_events: list[tuple[float, int, str]] = []
    for seconds, line in stdout_lines:
        match = order_event_pattern.search(line)
        if match is not None:
            order_events.append((seconds, int(match.group(1)), line))
    return samples, order_events


def _profile_main(
    config_path: Path,
    sample_interval: float,
) -> tuple[list[tuple[float, float]], list[tuple[float, int, str]]]:
    return _profile_command(
        [sys.executable, "-u", "main.py", str(config_path)],
        config_path=config_path,
        sample_interval=sample_interval,
        order_event_pattern=ORDER_EVENT_PATTERN,
        failure_name="streaming main.py",
    )


def _profile_streaming_cmd(
    config_path: Path,
    sample_interval: float,
) -> tuple[list[tuple[float, float]], list[tuple[float, int, str]]]:
    streaming_script = dedent(
        """
        import os
        import time

        from qxti.core import QXTISimulation

        config_path = os.environ["QXTI_MEMORY_PROFILE_CONFIG"]
        hold_seconds = float(os.environ.get("QXTI_MEMORY_PROFILE_HOLD_SECONDS", "0.5"))

        simulation = QXTISimulation.from_file(config_path)
        hamiltonian = simulation.build_hamiltonian()
        cmd = simulation.build_cmd(hamiltonian)
        paths = cmd.solve_time_domain(simulation.config.cmd.output_dir)
        print(f"[MEMORY_PROFILE] streaming saved {len(paths)} orders", flush=True)
        time.sleep(hold_seconds)
        print("[MEMORY_PROFILE] streaming done", flush=True)
        """
    )
    return _profile_command(
        [sys.executable, "-u", "-c", streaming_script],
        config_path=config_path,
        sample_interval=sample_interval,
        order_event_pattern=ORDER_EVENT_PATTERN,
        failure_name="streaming CMD solve_time_domain",
    )


def _profile_legacy_in_memory(
    config_path: Path,
    sample_interval: float,
) -> tuple[list[tuple[float, float]], list[tuple[float, int, str]]]:
    legacy_script = dedent(
        """
        import os
        import time

        from qxti.core import QXTISimulation

        config_path = os.environ["QXTI_MEMORY_PROFILE_CONFIG"]
        hold_seconds = float(os.environ.get("QXTI_MEMORY_PROFILE_HOLD_SECONDS", "0.5"))

        simulation = QXTISimulation.from_file(config_path)
        hamiltonian = simulation.build_hamiltonian()
        cmd = simulation.build_cmd(hamiltonian)
        rho_orders = cmd.solve_time_domain_in_memory()
        retained_mib = sum(tensor.nbytes for tensor in rho_orders.values()) / 1024**2
        print(f"[MEMORY_PROFILE] legacy retained {len(rho_orders)} orders, tensors={retained_mib:.1f} MiB", flush=True)
        time.sleep(hold_seconds)
        print(f"[MEMORY_PROFILE] legacy done, still retained={sum(tensor.nbytes for tensor in rho_orders.values()) / 1024**2:.1f} MiB", flush=True)
        """
    )
    return _profile_command(
        [sys.executable, "-u", "-c", legacy_script],
        config_path=config_path,
        sample_interval=sample_interval,
        order_event_pattern=LEGACY_ORDER_EVENT_PATTERN,
        failure_name="legacy in-memory solve_time_domain",
    )


def _format_lines(lines: list[tuple[float, str]]) -> str:
    return "\n".join(f"[{seconds:.3f}s] {line}" for seconds, line in lines[-80:])


def _write_profile_csv(
    runs: dict[str, list[tuple[float, float]]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "seconds", "rss_mib"])
        for mode, samples in runs.items():
            for seconds, rss_mib in samples:
                writer.writerow([mode, seconds, rss_mib])


def _write_order_events_csv(
    runs: dict[str, list[tuple[float, int, str]]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "seconds", "order", "event"])
        for mode, order_events in runs.items():
            for seconds, order, event in order_events:
                writer.writerow([mode, seconds, order, event])


def _write_profile_plot(
    runs: dict[str, list[tuple[float, float]]],
    order_runs: dict[str, list[tuple[float, int, str]]],
    output_path: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    all_rss = [rss_mib for samples in runs.values() for _seconds, rss_mib in samples]
    all_seconds = [seconds for samples in runs.values() for seconds, _rss_mib in samples]
    peak = max(all_rss, default=0.0)
    max_seconds = max(all_seconds, default=0.0)

    figure, axis = plt.subplots(figsize=(9, 4.8))
    colors = {
        "streaming": "tab:blue",
        "legacy": "tab:orange",
    }
    for mode, samples in runs.items():
        seconds = [sample[0] for sample in samples]
        rss_mib = [sample[1] for sample in samples]
        axis.plot(
            seconds,
            rss_mib,
            color=colors.get(mode, None),
            linewidth=1.8,
            label=f"{mode} peak {max(rss_mib, default=0.0):.1f} MiB",
        )
    axis.axhline(peak, color="tab:red", linestyle="--", linewidth=1.0)
    for mode, order_events in order_runs.items():
        color = colors.get(mode, "tab:green")
        for event_seconds, order, _line in order_events:
            axis.axvline(event_seconds, color=color, linestyle=":", linewidth=1.0, alpha=0.65)
            axis.text(
                event_seconds,
                peak * (0.96 if mode == "streaming" else 0.82) if peak > 0.0 else 0.0,
                f"{mode} order {order}",
                color=color,
                rotation=90,
                ha="right",
                va="top",
                fontsize=7,
            )
    axis.text(
        max_seconds,
        peak,
        f" peak {peak:.1f} MiB",
        color="tab:red",
        ha="right",
        va="bottom",
    )
    axis.set_title("QXTI memory profile")
    axis.set_xlabel("time [s]")
    axis.set_ylabel("RSS [MiB]")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


@pytest.mark.skipif(
    os.environ.get("QXTI_RUN_MEMORY_PROFILE") != "1",
    reason="Set QXTI_RUN_MEMORY_PROFILE=1 to run the main.py memory profile.",
)
def test_main_memory_profile_generates_csv_and_plot(tmp_path: Path) -> None:
    config_override = os.environ.get("QXTI_MEMORY_PROFILE_CONFIG", "").strip()
    config_path = Path(config_override) if config_override else _write_low_memory_config(tmp_path)
    sample_interval = float(os.environ.get("QXTI_MEMORY_PROFILE_INTERVAL", "0.05"))
    profile_mode = os.environ.get("QXTI_MEMORY_PROFILE_MODE", "streaming").strip().lower()

    if profile_mode == "streaming":
        runs = {"streaming": _profile_main(config_path, sample_interval)}
    elif profile_mode in {"legacy", "legacy_in_memory"}:
        runs = {"legacy": _profile_legacy_in_memory(config_path, sample_interval)}
    elif profile_mode == "compare":
        runs = {
            "streaming": _profile_streaming_cmd(config_path, sample_interval),
            "legacy": _profile_legacy_in_memory(config_path, sample_interval),
        }
    else:
        raise ValueError("QXTI_MEMORY_PROFILE_MODE must be streaming, legacy, or compare.")

    sample_runs = {mode: result[0] for mode, result in runs.items()}
    event_runs = {mode: result[1] for mode, result in runs.items()}
    assert all(samples for samples in sample_runs.values())
    assert all(events for events in event_runs.values())

    csv_path = PROFILE_DIR / "main_memory_profile.csv"
    events_csv_path = PROFILE_DIR / "main_memory_profile_events.csv"
    png_path = PROFILE_DIR / "main_memory_profile.png"
    _write_profile_csv(sample_runs, csv_path)
    _write_order_events_csv(event_runs, events_csv_path)
    _write_profile_plot(sample_runs, event_runs, png_path)

    peak_mib = max(sample[1] for samples in sample_runs.values() for sample in samples)
    max_mib = float(os.environ.get("QXTI_MEMORY_PROFILE_MAX_MIB", "1200"))
    assert peak_mib < max_mib, (
        f"Peak RSS was {peak_mib:.1f} MiB, above the configured limit "
        f"of {max_mib:.1f} MiB. CSV: {csv_path}. Plot: {png_path}."
    )
