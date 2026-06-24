from __future__ import annotations

import threading
from typing import Callable

import numpy as np
from numpy.typing import NDArray


ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]


class StreamingCurrentAccumulator:
    """Accumulate macroscopic current and polarization on the fly during CMD solve.

    For each solved k-point ``(ik, rho_series)`` the accumulator contributes:

        J^(s)_i(t) += weight[ik] * Tr[ v_i(ik) * rho^(s)(ik, t) ]
        P^(s)_i(t) += weight[ik] * Tr[ d_i(ik) * rho^(s)(ik, t) ]

    where ``v_i`` is the current operator and ``d_i`` is the dipole/connection
    operator in the band gauge.

    This avoids saving and re-loading the full ``(Nk, Nt, Nb, Nb)`` density-matrix
    tensor for observable computation.  The callback is thread-safe: multiple
    worker threads call :meth:`accumulate` concurrently without data races.

    Parameters
    ----------
    nt:
        Number of time points.
    integration_weights:
        Per-k-point BZ integration weights, shape ``(Nk,)``.
    current_operators:
        Dict mapping ``"x"``/``"y"``/``"z"`` to the cached current operator
        tensor, shape ``(Nk, Nb, Nb)``.
    dipole_operators:
        Dict mapping ``"x"``/``"y"``/``"z"`` to the cached dipole/connection
        tensor, shape ``(Nk, Nb, Nb)``.
    active_dimension:
        Number of active Cartesian directions (1, 2, or 3).
    """

    _DIRECTIONS = ("x", "y", "z")

    def __init__(
        self,
        nt: int,
        integration_weights: RealArray,
        current_operators: dict[str, ComplexArray],
        dipole_operators: dict[str, ComplexArray],
        active_dimension: int,
    ) -> None:
        self._nt = int(nt)
        self._weights = np.asarray(integration_weights, dtype=np.float64)
        self._current_ops = {k: np.asarray(v, dtype=np.complex128) for k, v in current_operators.items()}
        self._dipole_ops = {k: np.asarray(v, dtype=np.complex128) for k, v in dipole_operators.items()}
        self._directions = self._DIRECTIONS[:max(1, min(3, int(active_dimension)))]
        self._current: dict[int, ComplexArray] = {}
        self._polarization: dict[int, ComplexArray] = {}
        self._lock = threading.Lock()

    def accumulate(self, order: int, ik: int, rho_series: ComplexArray) -> None:
        """Accumulate one k-point ``ik`` for perturbative order ``order``.

        Parameters
        ----------
        order:
            Perturbative order (0, 1, 2, ...).
        ik:
            K-point flat index into the integration-weight array.
        rho_series:
            Density matrix at this k-point, shape ``(Nt, Nb, Nb)``.
        """
        w = float(self._weights[ik])
        rho = np.asarray(rho_series, dtype=np.complex128)
        nt = rho.shape[0]

        # Compute contributions locally (no lock needed for reads).
        j_contrib = np.zeros((nt, 3), dtype=np.complex128)
        p_contrib = np.zeros((nt, 3), dtype=np.complex128)
        for axis, direction in enumerate(self._directions):
            v = self._current_ops[direction][ik]   # (Nb, Nb)
            d = self._dipole_ops[direction][ik]    # (Nb, Nb)
            j_contrib[:, axis] = np.einsum("mn,tnm->t", v, rho, optimize=True)
            p_contrib[:, axis] = np.einsum("mn,tnm->t", d, rho, optimize=True)
        j_contrib *= w
        p_contrib *= w

        # Atomic accumulation — only the += needs the lock.
        with self._lock:
            if order not in self._current:
                self._current[order] = np.zeros((nt, 3), dtype=np.complex128)
                self._polarization[order] = np.zeros((nt, 3), dtype=np.complex128)
            self._current[order] += j_contrib
            self._polarization[order] += p_contrib

    def accumulate_equilibrium(self, order: int, ik: int, rho0: ComplexArray) -> None:
        """Accumulate equilibrium order (rho constant in time, shape ``(Nb, Nb)``).

        Uses direct matrix trace instead of expanding to ``(Nt, Nb, Nb)``.
        """
        w = float(self._weights[ik])
        rho = np.asarray(rho0, dtype=np.complex128)

        j_contrib = np.zeros(3, dtype=np.complex128)
        p_contrib = np.zeros(3, dtype=np.complex128)
        for axis, direction in enumerate(self._directions):
            v = self._current_ops[direction][ik]
            d = self._dipole_ops[direction][ik]
            j_contrib[axis] = w * np.einsum("mn,nm->", v, rho, optimize=True)
            p_contrib[axis] = w * np.einsum("mn,nm->", d, rho, optimize=True)

        with self._lock:
            if order not in self._current:
                self._current[order] = np.zeros((self._nt, 3), dtype=np.complex128)
                self._polarization[order] = np.zeros((self._nt, 3), dtype=np.complex128)
            self._current[order] += j_contrib[np.newaxis, :]
            self._polarization[order] += p_contrib[np.newaxis, :]

    def make_callback(self, order: int) -> Callable[[int, ComplexArray], None]:
        """Return a thread-safe accumulation callback for ``order``."""
        return lambda ik, rho: self.accumulate(order, ik, rho)

    def make_equilibrium_callback(self, order: int) -> Callable[[int, ComplexArray], None]:
        """Return a thread-safe accumulation callback for the equilibrium order."""
        return lambda ik, rho0: self.accumulate_equilibrium(order, ik, rho0)

    def current_time(self, order: int) -> RealArray:
        """Return the real part of J^(order)(t), shape ``(Nt, 3)``."""
        arr = self._current.get(order)
        if arr is None:
            return np.zeros((self._nt, 3), dtype=np.float64)
        return np.asarray(np.real(arr), dtype=np.float64)

    def polarization_time(self, order: int) -> RealArray:
        """Return the real part of P^(order)(t), shape ``(Nt, 3)``."""
        arr = self._polarization.get(order)
        if arr is None:
            return np.zeros((self._nt, 3), dtype=np.float64)
        return np.asarray(np.real(arr), dtype=np.float64)

    def available_orders(self) -> tuple[int, ...]:
        return tuple(sorted(self._current))

    def driven_orders(self) -> tuple[int, ...]:
        return tuple(s for s in self.available_orders() if s > 0)
