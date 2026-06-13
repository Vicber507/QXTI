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
        value_mode: str = "absolute",
        rho_orders: dict[int, ComplexArray] | None = None,
    ) -> dict[str, Any]:
        time_domain = self.cmd.solve_time_domain_in_memory() if rho_orders is None else rho_orders
        resolved_orders = self._resolve_orders(orders, time_domain)
        populations = self._population_values_from_orders(
            time_domain,
            resolved_orders=resolved_orders,
            value_mode=value_mode,
        )
        aggregation_mode, aggregation_label = self._normalize_k_aggregation(k_aggregation)
        aggregated = self._aggregate_populations(populations, aggregation_mode)

        return {
            "orders": resolved_orders,
            "value_mode": self._normalize_population_value_mode(value_mode),
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
        value_mode: str = "absolute",
        time_indices: NDArray[np.int_] | None = None,
        rho_orders: dict[int, ComplexArray] | None = None,
    ) -> dict[str, Any]:
        time_domain = self.cmd.solve_time_domain_in_memory() if rho_orders is None else rho_orders
        resolved_orders = self._resolve_orders(orders, time_domain)
        full_time_axis = np.asarray(self.cmd.timegrid.generate(), dtype=float)
        resolved_time_indices = self._resolve_time_indices(len(full_time_axis), time_indices)
        populations = self._population_values_from_orders(
            time_domain,
            resolved_orders=resolved_orders,
            value_mode=value_mode,
            time_indices=resolved_time_indices,
        )

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
            "value_mode": self._normalize_population_value_mode(value_mode),
            "time_axis": full_time_axis if resolved_time_indices is None else full_time_axis[resolved_time_indices],
            "band_indices": np.arange(nb, dtype=int),
            "kx_values": np.asarray(self.cmd.kgrid.kx_values, dtype=float),
            "ky_values": np.asarray(self.cmd.kgrid.ky_values, dtype=float),
            "population_frames": frames,
            "equilibrium_population_frame": self._equilibrium_population_frame_from_saved_rho(
                time_domain,
                kx_values=np.asarray(self.cmd.kgrid.kx_values, dtype=float),
                ky_values=np.asarray(self.cmd.kgrid.ky_values, dtype=float),
                kz_values=np.asarray(self.cmd.kgrid.kz_values, dtype=float),
            ),
        }

    def coherence_heatmap_data(
        self,
        *,
        orders: tuple[int, ...] | list[int] | None = None,
        k_aggregation: str = "mean",
        component: str = "magnitude",
        rho_orders: dict[int, ComplexArray] | None = None,
    ) -> dict[str, Any]:
        time_domain = self.cmd.solve_time_domain_in_memory() if rho_orders is None else rho_orders
        resolved_orders = self._resolve_orders(orders, time_domain)
        coherence_values, pair_indices, pair_labels = self._coherence_series_from_orders(
            time_domain,
            resolved_orders=resolved_orders,
            component=component,
        )
        aggregation_mode, aggregation_label = self._normalize_k_aggregation(k_aggregation)
        aggregated = self._aggregate_populations(coherence_values, aggregation_mode)

        return {
            "orders": resolved_orders,
            "time_axis": np.asarray(self.cmd.timegrid.generate(), dtype=float),
            "pair_indices": np.asarray(pair_indices, dtype=int),
            "pair_labels": list(pair_labels),
            "coherence_map": aggregated.T,
            "coherence_traces": aggregated,
            "coherence_frames": np.transpose(coherence_values, (1, 2, 0)),
            "k_points": np.asarray(self.cmd.kgrid.points(), dtype=float),
            "k_point_indices": np.arange(coherence_values.shape[0], dtype=int),
            "k_aggregation": aggregation_mode,
            "aggregation_label": aggregation_label,
            "component": component,
        }

    def coherence_kxky_animation_data(
        self,
        *,
        orders: tuple[int, ...] | list[int] | None = None,
        component: str = "magnitude",
        time_indices: NDArray[np.int_] | None = None,
        rho_orders: dict[int, ComplexArray] | None = None,
    ) -> dict[str, Any]:
        time_domain = self.cmd.solve_time_domain_in_memory() if rho_orders is None else rho_orders
        resolved_orders = self._resolve_orders(orders, time_domain)
        full_time_axis = np.asarray(self.cmd.timegrid.generate(), dtype=float)
        resolved_time_indices = self._resolve_time_indices(len(full_time_axis), time_indices)
        coherence_values, pair_indices, pair_labels = self._coherence_series_from_orders(
            time_domain,
            resolved_orders=resolved_orders,
            component=component,
            time_indices=resolved_time_indices,
        )

        if self.cmd.kgrid.dimension < 2:
            raise ValueError("A kx-ky coherence animation requires a 2D or 3D k-grid.")
        if len(self.cmd.kgrid.kz_values) != 1:
            raise ValueError(
                "A kx-ky coherence animation currently requires a single kz slice."
            )

        nkx, nky, nkz = self.cmd.kgrid.shape
        _, nt, npairs = coherence_values.shape
        coherence_grid = coherence_values.reshape(nkx, nky, nkz, nt, npairs)
        frames = np.transpose(coherence_grid[:, :, 0, :, :], (2, 3, 1, 0))

        return {
            "orders": resolved_orders,
            "time_axis": full_time_axis if resolved_time_indices is None else full_time_axis[resolved_time_indices],
            "pair_indices": np.asarray(pair_indices, dtype=int),
            "pair_labels": list(pair_labels),
            "kx_values": np.asarray(self.cmd.kgrid.kx_values, dtype=float),
            "ky_values": np.asarray(self.cmd.kgrid.ky_values, dtype=float),
            "coherence_frames": frames,
            "component": component,
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
    def _resolve_time_indices(
        num_times: int,
        time_indices: NDArray[np.int_] | None,
    ) -> NDArray[np.int_] | None:
        if time_indices is None:
            return None
        resolved = np.asarray(time_indices, dtype=int).reshape(-1)
        if resolved.size == 0:
            raise ValueError("time_indices cannot be empty when provided.")
        if np.any(resolved < 0) or np.any(resolved >= num_times):
            raise ValueError(
                f"time_indices must stay within [0, {num_times - 1}] for the selected time grid."
            )
        return resolved

    @staticmethod
    def _sum_orders(
        rho_orders: dict[int, ComplexArray],
        resolved_orders: tuple[int, ...],
    ) -> ComplexArray:
        reference = ResponseData._as_complex_tensor(rho_orders[resolved_orders[0]])
        total = np.zeros(reference.shape, dtype=np.complex128)
        for order in resolved_orders:
            total += ResponseData._as_complex_tensor(rho_orders[order])
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

    @classmethod
    def population_heatmap_data_from_saved_rho(
        cls,
        rho_orders: dict[int, ComplexArray],
        *,
        time_axis: FloatArray,
        k_points: FloatArray,
        orders: tuple[int, ...] | list[int] | None = None,
        k_aggregation: str = "mean",
        value_mode: str = "absolute",
    ) -> dict[str, Any]:
        resolved_orders = cls._resolve_orders(orders, rho_orders)
        populations = cls._population_values_from_orders(
            rho_orders,
            resolved_orders=resolved_orders,
            value_mode=value_mode,
        )
        aggregation_mode, aggregation_label = cls._normalize_k_aggregation(k_aggregation)
        aggregated = cls._aggregate_populations(populations, aggregation_mode)
        return {
            "orders": resolved_orders,
            "value_mode": cls._normalize_population_value_mode(value_mode),
            "time_axis": np.asarray(time_axis, dtype=float),
            "band_indices": np.arange(populations.shape[2], dtype=int),
            "population_map": aggregated.T,
            "population_traces": aggregated,
            "population_frames": np.transpose(populations, (1, 2, 0)),
            "k_points": np.asarray(k_points, dtype=float),
            "k_point_indices": np.arange(populations.shape[0], dtype=int),
            "k_aggregation": aggregation_mode,
            "aggregation_label": aggregation_label,
        }

    @classmethod
    def population_kxky_animation_data_from_saved_rho(
        cls,
        rho_orders: dict[int, ComplexArray],
        *,
        time_axis: FloatArray,
        kx_values: FloatArray,
        ky_values: FloatArray,
        kz_values: FloatArray,
        orders: tuple[int, ...] | list[int] | None = None,
        value_mode: str = "absolute",
    ) -> dict[str, Any]:
        resolved_orders = cls._resolve_orders(orders, rho_orders)
        populations = cls._population_values_from_orders(
            rho_orders,
            resolved_orders=resolved_orders,
            value_mode=value_mode,
        )

        if len(ky_values) < 2:
            raise ValueError("A kx-ky population animation requires at least two ky points.")
        if len(kz_values) != 1:
            raise ValueError("A kx-ky population animation currently requires a single kz slice.")

        nkx = len(kx_values)
        nky = len(ky_values)
        nkz = len(kz_values)
        _, nt, nb = populations.shape
        population_grid = populations.reshape(nkx, nky, nkz, nt, nb)
        frames = np.transpose(population_grid[:, :, 0, :, :], (2, 3, 1, 0))

        return {
            "orders": resolved_orders,
            "value_mode": cls._normalize_population_value_mode(value_mode),
            "time_axis": np.asarray(time_axis, dtype=float),
            "band_indices": np.arange(nb, dtype=int),
            "kx_values": np.asarray(kx_values, dtype=float),
            "ky_values": np.asarray(ky_values, dtype=float),
            "population_frames": frames,
            "equilibrium_population_frame": cls._equilibrium_population_frame_from_saved_rho(
                rho_orders,
                kx_values=kx_values,
                ky_values=ky_values,
                kz_values=kz_values,
            ),
        }

    @classmethod
    def _population_values(
        cls,
        total_rho: ComplexArray,
        *,
        rho_orders: dict[int, ComplexArray],
        resolved_orders: tuple[int, ...],
        value_mode: str,
    ) -> FloatArray:
        mode = cls._normalize_population_value_mode(value_mode)
        populations = np.real(np.diagonal(total_rho, axis1=2, axis2=3))
        if mode == "absolute":
            return populations

        if 0 in rho_orders:
            reference = np.real(
                np.diagonal(
                    cls._as_complex_tensor(rho_orders[0]),
                    axis1=2,
                    axis2=3,
                )
            )
        else:
            reference = populations[:, :1, :]
        if 0 not in resolved_orders and reference.shape == populations.shape:
            return populations
        return populations - reference

    @classmethod
    def _population_values_from_orders(
        cls,
        rho_orders: dict[int, ComplexArray],
        *,
        resolved_orders: tuple[int, ...],
        value_mode: str,
        time_indices: NDArray[np.int_] | None = None,
    ) -> FloatArray:
        reference_tensor = cls._as_complex_tensor(rho_orders[resolved_orders[0]])
        if time_indices is not None:
            reference_tensor = np.take(reference_tensor, time_indices, axis=1)
        populations = np.zeros(
            reference_tensor.shape[:2] + (reference_tensor.shape[2],),
            dtype=np.float64,
        )
        for order in resolved_orders:
            tensor = cls._as_complex_tensor(rho_orders[order])
            if time_indices is not None:
                tensor = np.take(tensor, time_indices, axis=1)
            populations += np.real(np.diagonal(tensor, axis1=2, axis2=3))

        mode = cls._normalize_population_value_mode(value_mode)
        if mode == "absolute":
            return populations

        if 0 in rho_orders:
            reference_tensor = cls._as_complex_tensor(rho_orders[0])
            if time_indices is not None:
                reference_tensor = np.take(reference_tensor, time_indices, axis=1)
            reference = np.real(
                np.diagonal(
                    reference_tensor,
                    axis1=2,
                    axis2=3,
                )
            )
        else:
            reference = populations[:, :1, :]
        if 0 not in resolved_orders and reference.shape == populations.shape:
            return populations
        return populations - reference

    @staticmethod
    def _normalize_population_value_mode(value_mode: str) -> str:
        key = value_mode.strip().lower()
        aliases = {
            "delta": "delta",
            "change": "delta",
            "difference": "delta",
            "absolute": "absolute",
            "total": "absolute",
        }
        if key not in aliases:
            raise ValueError("population value_mode must be one of: delta, absolute.")
        return aliases[key]

    @classmethod
    def _equilibrium_population_frame_from_saved_rho(
        cls,
        rho_orders: dict[int, ComplexArray],
        *,
        kx_values: FloatArray,
        ky_values: FloatArray,
        kz_values: FloatArray,
    ) -> FloatArray | None:
        if 0 not in rho_orders:
            return None
        tensor = cls._as_complex_tensor(rho_orders[0])
        if tensor.shape[1] == 0:
            return None
        diagonal = np.real(np.diagonal(tensor[:, 0], axis1=1, axis2=2))
        nkx = len(kx_values)
        nky = len(ky_values)
        nkz = len(kz_values)
        frame = diagonal.reshape(nkx, nky, nkz, diagonal.shape[1])
        return np.transpose(frame[:, :, 0, :], (2, 1, 0))

    @classmethod
    def coherence_heatmap_data_from_saved_rho(
        cls,
        rho_orders: dict[int, ComplexArray],
        *,
        time_axis: FloatArray,
        k_points: FloatArray,
        orders: tuple[int, ...] | list[int] | None = None,
        k_aggregation: str = "mean",
        component: str = "magnitude",
    ) -> dict[str, Any]:
        resolved_orders = cls._resolve_orders(orders, rho_orders)
        coherence_values, pair_indices, pair_labels = cls._coherence_series_from_orders(
            rho_orders,
            resolved_orders=resolved_orders,
            component=component,
        )
        aggregation_mode, aggregation_label = cls._normalize_k_aggregation(k_aggregation)
        aggregated = cls._aggregate_populations(coherence_values, aggregation_mode)
        return {
            "orders": resolved_orders,
            "time_axis": np.asarray(time_axis, dtype=float),
            "pair_indices": np.asarray(pair_indices, dtype=int),
            "pair_labels": list(pair_labels),
            "coherence_map": aggregated.T,
            "coherence_traces": aggregated,
            "coherence_frames": np.transpose(coherence_values, (1, 2, 0)),
            "k_points": np.asarray(k_points, dtype=float),
            "k_point_indices": np.arange(coherence_values.shape[0], dtype=int),
            "k_aggregation": aggregation_mode,
            "aggregation_label": aggregation_label,
            "component": component,
        }

    @classmethod
    def coherence_kxky_animation_data_from_saved_rho(
        cls,
        rho_orders: dict[int, ComplexArray],
        *,
        time_axis: FloatArray,
        kx_values: FloatArray,
        ky_values: FloatArray,
        kz_values: FloatArray,
        orders: tuple[int, ...] | list[int] | None = None,
        component: str = "magnitude",
    ) -> dict[str, Any]:
        resolved_orders = cls._resolve_orders(orders, rho_orders)
        coherence_values, pair_indices, pair_labels = cls._coherence_series_from_orders(
            rho_orders,
            resolved_orders=resolved_orders,
            component=component,
        )

        if len(ky_values) < 2:
            raise ValueError("A kx-ky coherence animation requires at least two ky points.")
        if len(kz_values) != 1:
            raise ValueError("A kx-ky coherence animation currently requires a single kz slice.")

        nkx = len(kx_values)
        nky = len(ky_values)
        nkz = len(kz_values)
        _, nt, npairs = coherence_values.shape
        coherence_grid = coherence_values.reshape(nkx, nky, nkz, nt, npairs)
        frames = np.transpose(coherence_grid[:, :, 0, :, :], (2, 3, 1, 0))

        return {
            "orders": resolved_orders,
            "time_axis": np.asarray(time_axis, dtype=float),
            "pair_indices": np.asarray(pair_indices, dtype=int),
            "pair_labels": list(pair_labels),
            "kx_values": np.asarray(kx_values, dtype=float),
            "ky_values": np.asarray(ky_values, dtype=float),
            "coherence_frames": frames,
            "component": component,
        }

    @staticmethod
    def _coherence_pairs(num_bands: int) -> list[tuple[int, int]]:
        return [(row, col) for row in range(num_bands) for col in range(row + 1, num_bands)]

    @classmethod
    def _coherence_series(
        cls,
        rho_tensor: ComplexArray,
        *,
        component: str,
    ) -> tuple[FloatArray, list[tuple[int, int]], list[str]]:
        num_bands = rho_tensor.shape[2]
        pair_indices = cls._coherence_pairs(num_bands)
        if not pair_indices:
            raise ValueError("At least two bands are required to build coherence plots.")

        extracted = []
        pair_labels: list[str] = []
        for row, col in pair_indices:
            pair_labels.append(f"{row}-{col}")
            extracted.append(cls._coherence_component(rho_tensor[:, :, row, col], component=component))
        coherence_values = np.stack(extracted, axis=2)
        return coherence_values, pair_indices, pair_labels

    @classmethod
    def _coherence_series_from_orders(
        cls,
        rho_orders: dict[int, ComplexArray],
        *,
        resolved_orders: tuple[int, ...],
        component: str,
        time_indices: NDArray[np.int_] | None = None,
    ) -> tuple[FloatArray, list[tuple[int, int]], list[str]]:
        reference_tensor = cls._as_complex_tensor(rho_orders[resolved_orders[0]])
        if time_indices is not None:
            reference_tensor = np.take(reference_tensor, time_indices, axis=1)
        num_bands = reference_tensor.shape[2]
        pair_indices = cls._coherence_pairs(num_bands)
        if not pair_indices:
            raise ValueError("At least two bands are required to build coherence plots.")

        rows = np.asarray([row for row, _ in pair_indices], dtype=int)
        cols = np.asarray([col for _, col in pair_indices], dtype=int)
        pair_labels = [f"{row}-{col}" for row, col in pair_indices]
        coherence_accumulated = np.zeros(
            reference_tensor.shape[:2] + (len(pair_indices),),
            dtype=np.complex128,
        )

        for order in resolved_orders:
            tensor = cls._as_complex_tensor(rho_orders[order])
            if time_indices is not None:
                tensor = np.take(tensor, time_indices, axis=1)
            coherence_accumulated += np.asarray(
                tensor[:, :, rows, cols],
                dtype=np.complex128,
            )

        coherence_values = cls._coherence_component(coherence_accumulated, component=component)
        return coherence_values, pair_indices, pair_labels

    @staticmethod
    def _coherence_component(values: ComplexArray, *, component: str) -> FloatArray:
        key = component.strip().lower()
        if key == "complex":
            return np.asarray(values, dtype=np.complex128)
        if key == "magnitude":
            return np.abs(values)
        if key == "real":
            return np.real(values)
        if key == "imag":
            return np.imag(values)
        raise ValueError("component must be one of: complex, magnitude, real, imag.")

    @staticmethod
    def _as_complex_tensor(tensor: ComplexArray) -> ComplexArray:
        if isinstance(tensor, np.ndarray) and np.issubdtype(tensor.dtype, np.complexfloating):
            return tensor
        return np.asarray(tensor, dtype=np.complex128)
