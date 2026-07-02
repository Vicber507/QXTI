"""Runner for the density-of-states branch (``main.py <config> -ldos``).

Thin orchestration layer around :func:`qxti.analytics.dos.compute_dos_spectrum`,
mirroring :class:`SusceptibilityScanRunner`: it standardizes the output path to
``outputs/<model>/ldos`` and writes a single ``ldos.npz`` dataset that the
graphics layer reads back (total DOS, projected PDOS, cumulative N(E), and the
optional spectral map).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qxti.core.config import QXTIConfig
from qxti.utils.progress import format_duration


@dataclass(slots=True)
class LDOSRunner:
    """Compute and save the density of states for a single configuration."""

    config: QXTIConfig

    @classmethod
    def from_file(cls, config_path: str | Path) -> "LDOSRunner":
        # Standardize outputs to outputs/<model_name>/ldos so every entry point
        # (CLI, runner, graphics) reads/writes the same paths.
        return cls(config=QXTIConfig.from_file(config_path).with_standard_output_dirs())

    def run(self) -> dict[str, Path]:
        from qxti.analytics.dos import compute_dos_spectrum
        from qxti.data import save_dataset_npz

        lcfg = self.config.ldos
        method = str(getattr(lcfg, "method", "eigenvalues")).strip().lower()
        surface = method == "surface"
        finite = method == "finite"
        if surface:
            self._emit_progress(
                "Surface/edge engine: semi-infinite Lopez-Sancho decimation along "
                f"'{lcfg.surface_normal}' (side={lcfg.surface_side}). Computes the "
                "surface spectral function A_surf(k,E) (surface/edge states in the "
                "projected bulk gap; side=both shows the two faces, e.g. all Fermi arcs)"
                + (", a constant-energy map" if lcfg.spectral_plane_enabled else "")
                + (", and the surface LDOS" if lcfg.surface_ldos_enabled else "")
                + ". Progress with ETA below."
            )
        elif finite:
            self._emit_progress(
                "Finite LDOS engine: building a finite 2-D plaque with open boundaries "
                f"in both lattice directions ({lcfg.finite_nx}x{lcfg.finite_ny} cells). "
                "Computes DOS(E), a finite-state spectrum colored by edge localization, "
                "and a site-resolved LDOS(r,E) map so both edges can be seen at once."
            )
        else:
            self._emit_progress(
                "DOS engine: diagonalizing H(k) over the [kgrid] mesh and broadening "
                f"the eigenvalues ({lcfg.broadening}). Computes total DOS"
                + (", orbital-projected PDOS" if lcfg.projected else "")
                + (", spectral A(k,E)" if lcfg.spectral_enabled else "")
                + ", and cumulative N(E). Progress with ETA below."
            )

        result = compute_dos_spectrum(self.config, progress=True)

        out_dir = Path(lcfg.output_dir) / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "ldos.npz"
        save_dataset_npz(out_path, result["dataset"])

        plot_hint = "Plot it with: python qxti/graphics/graphics.py <config> -ldos"
        if surface:
            self._emit_progress(
                f"Surface spectral data saved as '{out_path.name}' (computed in "
                f"{format_duration(result['runtime_seconds'])}). {plot_hint}"
            )
        elif finite:
            integral = float(result["integral"])
            expected = int(result["expected"])
            self._emit_progress(
                f"Finite LDOS data saved as '{out_path.name}' (computed in "
                f"{format_duration(result['runtime_seconds'])}). "
                f"Integrated DOS = {integral:.4f} states (expected {expected} finite states). {plot_hint}"
            )
        else:
            integral = float(result["integral"])
            expected = int(result["expected"])
            self._emit_progress(
                f"DOS saved as '{out_path.name}' (computed in "
                f"{format_duration(result['runtime_seconds'])}). "
                f"Sum rule: \\int g(E) dE = {integral:.4f} (expected {expected} bands; "
                f"deviation {100.0 * abs(integral - expected) / max(expected, 1):.2f}% reflects "
                f"broadening leaking past the energy window). {plot_hint}"
            )
        return {"ldos_data": out_path}

    def _emit_progress(self, message: str) -> None:
        print(f"[QXTI] {message}", flush=True)
