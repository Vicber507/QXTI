from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable
from qxti.grids import KGrid
from qxti.core import QXTISimulation

import numpy as np

ObservableDict = dict[str, float]
SimulationRunner = Callable[[KGrid], Any]
ObservableExtractor = Callable[[Any], ObservableDict]


@dataclass(slots=True)
class KGridConvergenceResult:
    kgrid: KGrid
    observables: ObservableDict
    elapsed_seconds: float
    relative_errors: dict[str, float] = field(default_factory=dict)
    absolute_errors: dict[str, float] = field(default_factory=dict)
    converged: bool = False


@dataclass(slots=True)
class KGridConvergenceTester:
    """
    Runs a k-grid convergence test.

    Parameters
    ----------
    runner:
        Function that receives a KGrid and returns a simulation result.

    extractor:
        Function that receives the simulation result and returns relevant
        scalar observables, for example energy, gap, peak position, population.

    relative_tolerance:
        Maximum accepted relative change between consecutive k-grids.

    absolute_tolerance:
        Optional absolute tolerance. Useful for quantities close to zero.

    require_consecutive:
        Number of consecutive converged refinements required before stopping.
    """

    runner: SimulationRunner
    extractor: ObservableExtractor
    relative_tolerance: float = 1e-2
    absolute_tolerance: float = 1e-6
    require_consecutive: int = 2
    eps: float = 1e-12

    def run(
        self,
        kgrids: list[KGrid],
        *,
        stop_when_converged: bool = True,
    ) -> list[KGridConvergenceResult]:
        if len(kgrids) < 2:
            raise ValueError("At least two KGrid objects are required.")

        results: list[KGridConvergenceResult] = []
        consecutive_converged = 0

        previous_observables: ObservableDict | None = None

        for kgrid in kgrids:
            start = perf_counter()
            simulation_result = self.runner(kgrid)
            elapsed = perf_counter() - start

            observables = self.extractor(simulation_result)
            self._validate_observables(observables)

            relative_errors: dict[str, float] = {}
            absolute_errors: dict[str, float] = {}
            converged = False

            if previous_observables is not None:
                relative_errors, absolute_errors = self._compare(
                    previous_observables,
                    observables,
                )

                converged = self._is_converged(relative_errors, absolute_errors)

                if converged:
                    consecutive_converged += 1
                else:
                    consecutive_converged = 0

            result = KGridConvergenceResult(
                kgrid=kgrid,
                observables=observables,
                elapsed_seconds=elapsed,
                relative_errors=relative_errors,
                absolute_errors=absolute_errors,
                converged=converged,
            )

            results.append(result)
            previous_observables = observables

            if (
                stop_when_converged
                and consecutive_converged >= self.require_consecutive
            ):
                break

        return results

    def recommended_kgrid(
        self,
        results: list[KGridConvergenceResult],
    ) -> KGrid | None:
        """
        Returns the first k-grid that satisfies the convergence criterion.

        If require_consecutive = 2, this returns the first grid in the stable
        pair/sequence, not necessarily the last computed grid.
        """

        if not results:
            return None

        converged_indices = [
            index for index, result in enumerate(results) if result.converged
        ]

        if len(converged_indices) < self.require_consecutive:
            return None

        first_stable_index = converged_indices[0]
        return results[first_stable_index].kgrid

    def summary_table(
        self,
        results: list[KGridConvergenceResult],
    ) -> list[dict[str, Any]]:
        table: list[dict[str, Any]] = []

        for result in results:
            row: dict[str, Any] = {
                "shape": result.kgrid.shape,
                "total_points": result.kgrid.total_points,
                "elapsed_seconds": result.elapsed_seconds,
                "converged": result.converged,
            }

            for name, value in result.observables.items():
                row[name] = value

            for name, value in result.relative_errors.items():
                row[f"rel_error_{name}"] = value

            for name, value in result.absolute_errors.items():
                row[f"abs_error_{name}"] = value

            table.append(row)

        return table

    def _compare(
        self,
        previous: ObservableDict,
        current: ObservableDict,
    ) -> tuple[dict[str, float], dict[str, float]]:
        if previous.keys() != current.keys():
            raise ValueError(
                "Observable keys must be the same for every k-grid result."
            )

        relative_errors: dict[str, float] = {}
        absolute_errors: dict[str, float] = {}

        for name in current:
            diff = abs(current[name] - previous[name])
            scale = max(abs(current[name]), self.eps)

            absolute_errors[name] = float(diff)
            relative_errors[name] = float(diff / scale)

        return relative_errors, absolute_errors

    def _is_converged(
        self,
        relative_errors: dict[str, float],
        absolute_errors: dict[str, float],
    ) -> bool:
        for name in relative_errors:
            rel_ok = relative_errors[name] <= self.relative_tolerance
            abs_ok = absolute_errors[name] <= self.absolute_tolerance

            if not (rel_ok or abs_ok):
                return False

        return True

    @staticmethod
    def _validate_observables(observables: ObservableDict) -> None:
        if not observables:
            raise ValueError("At least one observable is required.")

        for name, value in observables.items():
            if not np.isfinite(value):
                raise ValueError(f"Observable {name!r} is not finite: {value}")

def _example() -> None:
    """Illustrative convergence workflow; intentionally not run by pytest."""

    kgrids = [
        KGrid.uniform(dimension=3, num_points=4),
        KGrid.uniform(dimension=3, num_points=6),
        KGrid.uniform(dimension=3, num_points=8),
        KGrid.uniform(dimension=3, num_points=10),
        KGrid.uniform(dimension=3, num_points=12),
    ]

    def run_simulation_for_kgrid(kgrid: KGrid):
        simulation = QXTISimulation(...)
        simulation.kgrid = kgrid
        return simulation.run()

    def extract_observables(result) -> dict[str, float]:
        return {
            "energy": result.energy,
            "gap": result.gap,
            "main_observable": result.observable,
        }

    tester = KGridConvergenceTester(
        runner=run_simulation_for_kgrid,
        extractor=extract_observables,
        relative_tolerance=1e-2,
        absolute_tolerance=1e-6,
        require_consecutive=2,
    )

    results = tester.run(kgrids)
    tester.summary_table(results)
    tester.recommended_kgrid(results)


if __name__ == "__main__":
    _example()
