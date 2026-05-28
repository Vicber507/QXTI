# Librerías estándar

from __future__ import annotations
from typing import Dict, List, Any

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]

"""
Atributos de la clase XTP
"""

from qxti.physics import Hamiltonian
from qxti.grids import KGrid
from qxti.grids import TimeGrid
from qxti.grids import FrequencyGrid
from qxti.physics import OperatorFactory


class XTP:
    """
    Responsibility:
    Computes macroscopic optical response quantities
    from perturbative density matrix orders.

    Outputs:
        - Polarization P(t)
        - Current density J(t)
        - Optical susceptibility chi(w)
        - Harmonic spectra
    """

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        rho_orders: Dict[int, ComplexArray],
        kgrid: KGrid,
        timegrid: TimeGrid,
        frequencygrid: FrequencyGrid,
        operator_factory: OperatorFactory,
        directions: List[str],
        orders: List[int]
    ):

        # --- Attributes ---

        self.hamiltonian: Hamiltonian = hamiltonian

        self.rho_orders: Dict[int, ComplexArray] = rho_orders

        self.kgrid: KGrid = kgrid

        self.timegrid: TimeGrid = timegrid

        self.frequencygrid: FrequencyGrid = frequencygrid

        self.operator_factory: OperatorFactory = operator_factory

        self.directions: List[str] = directions

        self.orders: List[int] = orders

    # =========================================================
    # Polarization
    # =========================================================

    def polarization(
        self,
        order: int
    ) -> RealArray:
        """
        Computes the macroscopic polarization.

        Input:
            order: int

        Output:
            ndarray[Nt, 3]

        Polarization is computed from:

            P(t) = Tr[rho * r]
        """

        rho_tensor = self.rho_orders[order]

        Nk = rho_tensor.shape[0]
        Nt = rho_tensor.shape[1]

        polarization = np.zeros(
            (Nt, 3),
            dtype=np.float64
        )

        k_points = self.kgrid.get_all_points()

        # Loop over reciprocal-space points
        for ik, k in enumerate(k_points):

            kx, ky, kz = k

            # Loop over Cartesian directions
            for idir, direction in enumerate(
                self.directions
            ):

                # Position operator
                r_op = (
                    self.operator_factory.position_operator(
                        kx,
                        ky,
                        kz,
                        direction
                    )
                )

                # Loop over time
                for it in range(Nt):

                    rho_t = rho_tensor[ik, it]

                    # Expectation value:
                    # Tr[rho * r]
                    valor = np.trace(
                        rho_t @ r_op
                    )

                    polarization[it, idir] += (
                        np.real(valor)
                    )

        # Normalize by number of k-points
        polarization /= Nk

        return polarization

    # =========================================================
    # Current Density
    # =========================================================

    def current(
        self,
        order: int,
        direction: str
    ) -> RealArray:
        """
        Computes the macroscopic current density.

        Input:
            order: int
            direction: str

        Output:
            ndarray[Nt]

        Current is computed from:

            J(t) = Tr[rho * v]
        """

        rho_tensor = self.rho_orders[order]

        Nk = rho_tensor.shape[0]
        Nt = rho_tensor.shape[1]

        current = np.zeros(
            Nt,
            dtype=np.float64
        )

        k_points = self.kgrid.get_all_points()

        for ik, k in enumerate(k_points):

            kx, ky, kz = k

            # Velocity operator
            v_op = (
                self.operator_factory.velocity_operator(
                    kx,
                    ky,
                    kz,
                    direction
                )
            )

            for it in range(Nt):

                rho_t = rho_tensor[ik, it]

                valor = np.trace(
                    rho_t @ v_op
                )

                current[it] += np.real(valor)

        current /= Nk

        return current

    # =========================================================
    # Susceptibility
    # =========================================================

    def susceptibility(
        self,
        order: int
    ) -> ComplexArray:
        """
        Computes the optical susceptibility.

        Input:
            order: int

        Output:
            ndarray

        Applies FFT to the polarization signal.
        """

        polarization = self.polarization(order)

        chi = np.fft.fft(
            polarization,
            axis=0
        )

        return chi

    # =========================================================
    # Harmonic Spectrum
    # =========================================================

    def harmonic_spectrum(
        self,
        signal: RealArray
    ) -> ComplexArray:
        """
        Computes harmonic spectrum from a
        time-domain signal.

        Input:
            signal: ndarray

        Output:
            ndarray
        """

        spectrum = np.fft.fft(signal)

        return spectrum

    # =========================================================
    # Total Polarization
    # =========================================================

    def total_polarization(self) -> RealArray:
        """
        Computes total polarization from all
        perturbative orders.

        Output:
            ndarray[Nt, 3]
        """

        Nt = len(self.timegrid)

        total_P = np.zeros(
            (Nt, 3),
            dtype=np.float64
        )

        for order in self.orders:

            total_P += self.polarization(order)

        return total_P

    # =========================================================
    # Total Current
    # =========================================================

    def total_current(self) -> RealArray:
        """
        Computes total current density.

        Output:
            ndarray[Nt, 3]
        """

        Nt = len(self.timegrid)

        total_J = np.zeros(
            (Nt, 3),
            dtype=np.float64
        )

        for idir, direction in enumerate(
            self.directions
        ):

            for order in self.orders:

                total_J[:, idir] += self.current(
                    order,
                    direction
                )

        return total_J

    # =========================================================
    # Compute All Observables
    # =========================================================

    def compute_all(self) -> Dict[str, Any]:
        """
        Computes all optical response quantities.

        Output:
            dict[str, object]
        """

        resultados = {

            "polarization":
                self.total_polarization(),

            "current":
                self.total_current(),

            "susceptibility": {

                orden: self.susceptibility(orden)

                for orden in self.orders
            }

        }

        return resultados