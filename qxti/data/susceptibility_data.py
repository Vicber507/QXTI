from __future__ import annotations

from typing import Any

import numpy as np

from qxti.response import XTP


class SusceptibilityData:
    """Build reusable susceptibility datasets from one XTP calculation."""

    def __init__(self, xtp: XTP) -> None:
        self.xtp = xtp

    def tensor_spectrum_data(
        self,
        *,
        orders: tuple[int, ...] | None = None,
        input_direction: str = "auto",
        input_omega: float | None = None,
        eps: float = 1.0e-14,
    ) -> dict[str, Any]:
        selected_orders = self._resolve_orders(orders)
        if not selected_orders:
            raise ValueError("Susceptibility datasets require at least one positive perturbative order.")

        direction_labels = self.xtp._direction_labels(self.xtp.hamiltonian.dimension)
        data: dict[str, Any] = {
            "orders": selected_orders,
            "dimension": int(self.xtp.hamiltonian.dimension),
            "direction_labels": direction_labels,
            "requested_input_direction": input_direction,
            "requested_input_omega": None if input_omega is None else float(input_omega),
            "bz_mask": self.xtp.bz_mask_summary(),
        }

        reference_omega_axis: np.ndarray | None = None
        for order in selected_orders:
            order_data = self.xtp.susceptibility_tensor_spectrum(
                order=order,
                input_direction=input_direction,
                input_omega=input_omega,
                eps=eps,
            )
            conductivity_data = self.xtp.conductivity_tensor_spectrum(
                order=order,
                input_direction=input_direction,
                input_omega=input_omega,
                eps=eps,
            )
            omega_axis = np.asarray(order_data["omega_axis"], dtype=np.float64)
            if reference_omega_axis is None:
                reference_omega_axis = omega_axis
                data["omega_axis"] = omega_axis
            elif reference_omega_axis.shape != omega_axis.shape or not np.allclose(reference_omega_axis, omega_axis):
                raise ValueError("All susceptibility orders must share the same frequency axis.")

            conductivity_omega_axis = np.asarray(conductivity_data["omega_axis"], dtype=np.float64)
            if reference_omega_axis.shape != conductivity_omega_axis.shape or not np.allclose(
                reference_omega_axis,
                conductivity_omega_axis,
            ):
                raise ValueError("Conductivity and susceptibility orders must share the same frequency axis.")

            base = f"chi_order_{order}"
            data[f"{base}_tensor"] = np.asarray(order_data["tensor"], dtype=np.complex128)
            data[f"{base}_available_indices"] = np.asarray(order_data["available_indices"], dtype=np.int16)
            data[f"{base}_field_normalization"] = np.asarray(
                order_data["field_normalization"],
                dtype=np.complex128,
            )
            data[f"{base}_component_labels"] = list(order_data["component_labels"])
            data[f"{base}_input_direction"] = str(order_data["input_direction"])
            data[f"{base}_input_axis"] = int(order_data["input_axis"])
            data[f"{base}_input_omega"] = float(order_data["input_omega"])
            data[f"{base}_input_frequency_index"] = int(order_data["input_frequency_index"])
            data[f"{base}_normalization_mode"] = str(order_data["normalization_mode"])
            data[f"{base}_single_component_drive"] = bool(order_data["single_component_drive"])
            data[f"{base}_input_component_magnitudes"] = np.asarray(
                order_data["input_component_magnitudes"],
                dtype=np.float64,
            )

            conductivity_base = f"sigma_order_{order}"
            data[f"{conductivity_base}_tensor"] = np.asarray(conductivity_data["tensor"], dtype=np.complex128)
            data[f"{conductivity_base}_available_indices"] = np.asarray(
                conductivity_data["available_indices"],
                dtype=np.int16,
            )
            data[f"{conductivity_base}_field_normalization"] = np.asarray(
                conductivity_data["field_normalization"],
                dtype=np.complex128,
            )
            data[f"{conductivity_base}_component_labels"] = list(conductivity_data["component_labels"])
            data[f"{conductivity_base}_input_direction"] = str(conductivity_data["input_direction"])
            data[f"{conductivity_base}_input_axis"] = int(conductivity_data["input_axis"])
            data[f"{conductivity_base}_input_omega"] = float(conductivity_data["input_omega"])
            data[f"{conductivity_base}_input_frequency_index"] = int(conductivity_data["input_frequency_index"])
            data[f"{conductivity_base}_normalization_mode"] = str(conductivity_data["normalization_mode"])
            data[f"{conductivity_base}_single_component_drive"] = bool(conductivity_data["single_component_drive"])
            data[f"{conductivity_base}_input_component_magnitudes"] = np.asarray(
                conductivity_data["input_component_magnitudes"],
                dtype=np.float64,
            )

        return data

    def _resolve_orders(self, orders: tuple[int, ...] | None) -> tuple[int, ...]:
        if orders is None:
            requested = tuple(int(order) for order in sorted(self.xtp.orders))
        else:
            requested = tuple(int(order) for order in orders)
        return tuple(order for order in requested if order > 0 and order in self.xtp.rho_orders)
