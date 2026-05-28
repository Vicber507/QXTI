from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qxti.response import CMD


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class ResponseData:
    """Numerical post-processing helpers for CMD density-matrix outputs."""

    cmd: CMD

    def population_heatmap_data(
        self,
        *,
        orders: tuple[int, ...] | list[int] | None = None,
        k_aggregation: str = "mean",
        rho_orders: dict[int, ComplexArray] | None = None,
    ) -> dict[str, Any]:
        time_domain = self.cmd.solve_time_domain() if rho_orders is None else rho_orders
        resolved_orders = self._resolve_orders(orders, time_domain)
        total_rho = self._sum_orders(time_domain, resolved_orders)
        populations = np.real(np.diagonal(total_rho, axis1=2, axis2=3))
        aggregation_mode, aggregation_label = self._normalize_k_aggregation(k_aggregation)
        aggregated = self._aggregate_populations(populations, aggregation_mode)

        return {
            "orders": resolved_orders,
            "time_axis": np.asarray(self.cmd.timegrid.generate(), dtype=float),
            "band_indices": np.arange(populations.shape[2], dtype=int),
            "population_map": aggregated.T,
            "population_traces": aggregated,
            "population_frames": np.transpose(populations, (1, 2, 0)),
            "k_points": np.asarray(self.cmd.kgrid.points(), dtype=float),
            "k_point_indices": np.arange(populations.shape[0], dtype=int),
            "k_aggregation": aggregation_mode,
            "aggregation_label": aggregation_label,
        }

    def population_kxky_animation_data(
        self,
        *,
        orders: tuple[int, ...] | list[int] | None = None,
        rho_orders: dict[int, ComplexArray] | None = None,
    ) -> dict[str, Any]:
        time_domain = self.cmd.solve_time_domain() if rho_orders is None else rho_orders
        resolved_orders = self._resolve_orders(orders, time_domain)
        total_rho = self._sum_orders(time_domain, resolved_orders)
        populations = np.real(np.diagonal(total_rho, axis1=2, axis2=3))

        if self.cmd.kgrid.dimension < 2:
            raise ValueError("A kx-ky population animation requires a 2D or 3D k-grid.")
        if len(self.cmd.kgrid.kz_values) != 1:
            raise ValueError(
                "A kx-ky population animation currently requires a single kz slice."
            )

        nkx, nky, nkz = self.cmd.kgrid.shape
        _, nt, nb = populations.shape
        population_grid = populations.reshape(nkx, nky, nkz, nt, nb)
        frames = np.transpose(population_grid[:, :, 0, :, :], (2, 3, 1, 0))

        return {
            "orders": resolved_orders,
            "time_axis": np.asarray(self.cmd.timegrid.generate(), dtype=float),
            "band_indices": np.arange(nb, dtype=int),
            "kx_values": np.asarray(self.cmd.kgrid.kx_values, dtype=float),
            "ky_values": np.asarray(self.cmd.kgrid.ky_values, dtype=float),
            "population_frames": frames,
        }

    @staticmethod
    def _resolve_orders(
        requested: tuple[int, ...] | list[int] | None,
        rho_orders: dict[int, ComplexArray],
    ) -> tuple[int, ...]:
        available = tuple(sorted(int(order) for order in rho_orders))
        if requested is None:
            return available

        resolved = tuple(int(order) for order in requested)
        missing = tuple(order for order in resolved if order not in rho_orders)
        if missing:
            raise ValueError(
                f"Requested CMD orders {missing} are not available. "
                f"Available orders are {available}."
            )
        return resolved

    @staticmethod
    def _sum_orders(
        rho_orders: dict[int, ComplexArray],
        resolved_orders: tuple[int, ...],
    ) -> ComplexArray:
        reference = np.asarray(rho_orders[resolved_orders[0]], dtype=np.complex128)
        total = np.zeros_like(reference)
        for order in resolved_orders:
            total += np.asarray(rho_orders[order], dtype=np.complex128)
        return total

    @staticmethod
    def _normalize_k_aggregation(k_aggregation: str) -> tuple[str, str]:
        key = k_aggregation.strip().lower()
        aliases = {
            "mean": ("mean", "k-average"),
            "average": ("mean", "k-average"),
            "avg": ("mean", "k-average"),
            "sum": ("sum", "k-sum"),
            "first": ("first", "first k-point"),
            "first_k": ("first", "first k-point"),
        }
        if key not in aliases:
            raise ValueError("k_aggregation must be one of: mean, sum, first.")
        return aliases[key]

    @staticmethod
    def _aggregate_populations(
        populations: FloatArray,
        aggregation_mode: str,
    ) -> FloatArray:
        if aggregation_mode == "mean":
            return np.mean(populations, axis=0)
        if aggregation_mode == "sum":
            return np.sum(populations, axis=0)
        if aggregation_mode == "first":
            return populations[0]
        raise ValueError(f"Unsupported aggregation mode '{aggregation_mode}'.")
