from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Tuple, List, Dict, Any

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]

class Solver(ABC):
    """
    Abstract numerical solver interface.

    Responsibility
    --------------
    Provide a generic interface for all numerical propagators
    used in QXTI.

    Every solver must:
        - integrate differential equations
        - estimate convergence quality
        - expose diagnostic information
        - support complex-valued tensors
    """

    def __init__(
        self,
        tolerance: float = 1e-6,
        max_iterations: int = 100000,
        dtype: str = "complex128"
    ):


        self.tolerance: float = tolerance

        self.max_iterations: int = max_iterations

        self.dtype: str = dtype


        self.last_error: float = np.inf

        self.iterations: int = 0

        self.converged: bool = False

    @abstractmethod
    def solve(
        self,
        derivative_function: Callable,
        t0: float,
        tf: float,
        y0: ComplexArray,
        h_initial: float,
        *args,
        **kwargs
    ) -> Tuple[List[float], List[ComplexArray]]:
        """
        Main integration routine.

        Must be implemented by all subclasses.
        """
        pass

    def check_convergence(self) -> bool:
        """
        Checks whether the solver converged according
        to the requested tolerance.
        """

        self.converged = (
            self.last_error < self.tolerance
        )

        return self.converged

    def residual(self) -> float:
        """
        Returns the last numerical residual/error estimate.
        """

        return self.last_error

    def summary(self) -> Dict[str, Any]:
        """
        Returns diagnostic information about the solver.
        """

        return {

            "solver": self.__class__.__name__,

            "tolerance": self.tolerance,

            "max_iterations": self.max_iterations,

            "dtype": self.dtype,

            "iterations": self.iterations,

            "last_error": self.last_error,

            "converged": self.converged
        }

class RKF45Solver(Solver):
    """
    Adaptive Runge-Kutta-Fehlberg 4(5) solver.

    Features
    --------
    - Adaptive timestep
    - Frobenius norm error estimation
    - Hermiticity correction
    - Optional trace normalization
    """

    def __init__(
        self,
        tolerance: float = 1e-6,
        max_iterations: int = 100000,
        dtype: str = "complex128",
        h_min: float = 1e-12,
        h_max: float = 1.0,
        enforce_hermiticity: bool = True,
        enforce_trace: bool = False
    ):

        super().__init__(
            tolerance=tolerance,
            max_iterations=max_iterations,
            dtype=dtype
        )

        self.h_min = h_min

        self.h_max = h_max

        self.enforce_hermiticity = (
            enforce_hermiticity
        )

        self.enforce_trace = enforce_trace

    def solve(
        self,
        derivative_function: Callable,
        t0: float,
        tf: float,
        y0: ComplexArray,
        h_initial: float,
        *args,
        **kwargs
    ) -> Tuple[List[float], List[ComplexArray]]:

        t = t0
        reached_final_time = False

        y = y0.astype(np.complex128)

        h = h_initial

        t_l = [t]

        y_l = [y.copy()]

        self.iterations = 0

        while t < tf:

            if t + h > tf:

                h = tf - t

            ti = t

            yi = y

            k1 = h * derivative_function(ti,yi,*args)

            k2 = h * derivative_function(ti + h / 4,yi + k1 / 4,*args)

            k2 = h * derivative_function(ti + h / 4, yi + k1 / 4, *args)

            k3 = h * derivative_function(ti + 3 * h / 8, yi+ 3 * k1 / 32 + 9 * k2 / 32, *args)

            k4 = h * derivative_function(ti + 12 * h / 13,yi+ 1932 * k1 / 2197 - 7200 * k2 / 2197 + 7296 * k3 / 2197,*args)

            k5 = h * derivative_function(ti + h,yi+ 439 * k1 / 216 - 8 * k2+ 3680 * k3 / 513 - 845 * k4 / 4104,
                *args
            )

            k6 = h * derivative_function(
                ti + h / 2,
                yi
                - 8 * k1 / 27
                + 2 * k2
                + 3544 * k3 / 2565
                - 1859 * k4 / 4140
                - 11 * k5 / 40,
                *args
            )

            # =================================================
            # Fifth-order approximation
            # =================================================

            y5 = (
                yi
                + 16 * k1 / 135
                + 6656 * k3 / 12825
                + 28561 * k4 / 56430
                - 9 * k5 / 50
                + 2 * k6 / 55
            )

            # =================================================
            # Error estimation
            # =================================================

            error_tensor = (
                k1 / 360
                - 128 * k3 / 4275
                - 2197 * k4 / 75240
                + k5 / 50
                + 2* k6 / 55
            )

            err = np.linalg.norm(error_tensor)

            if err == 0:

                err = 1e-16

            self.last_error = err

            # =================================================
            # Accept timestep
            # =================================================

            if err < self.tolerance:

                t = t + h

                y = y5

                # Hermiticity enforcement
                if self.enforce_hermiticity:

                    y = (
                        y + y.conj().T
                    ) / 2.0

                # Trace conservation
                if self.enforce_trace:

                    trace = np.trace(y)

                    if np.abs(trace) > 1e-14:

                        y = y / trace

                t_l.append(t)

                y_l.append(y.copy())

                if t >= tf:
                    reached_final_time = True
                    break

            # =================================================
            # Adaptive timestep update
            # =================================================

            factor = (
                0.84
                * (self.tolerance / err) ** (1 / 4)
            )

            factor = min(
                2.0,
                max(0.1, factor)
            )

            h = h * factor

            h = min(self.h_max, h)

            h = max(self.h_min, h)

            # =================================================
            # Iteration counter
            # =================================================

            self.iterations += 1

            if self.iterations >= self.max_iterations:

                break

        if not reached_final_time and t < tf:
            raise RuntimeError(
                "RKF45Solver stopped before reaching the final time. "
                f"Reached t={t:.16e} while tf={tf:.16e}. "
                "Increase solver_max_iterations, increase solver_h_max, "
                "or relax the tolerance."
            )

        self.check_convergence()

        return t_l, y_l


# ============================================================
# Adams-Bashforth 2-Step Solver
# ============================================================

class AdamsBashforth2Solver(Solver):
    """
    Explicit two-step Adams-Bashforth solver.

    Uses:
        - RK4 bootstrap
        - Adams-Bashforth propagation
    """

    def __init__(
        self,
        tolerance: float = 1e-6,
        max_iterations: int = 100000,
        dtype: str = "complex128",
        enforce_hermiticity: bool = True,
        enforce_trace: bool = False
    ):

        super().__init__(
            tolerance=tolerance,
            max_iterations=max_iterations,
            dtype=dtype
        )

        self.enforce_hermiticity = (
            enforce_hermiticity
        )

        self.enforce_trace = enforce_trace

    # ========================================================
    # Main Adams-Bashforth Solver
    # ========================================================

    def solve(
        self,
        derivative_function: Callable,
        t0: float,
        tf: float,
        y0: ComplexArray,
        h_initial: float,
        *args,
        **kwargs
    ) -> Tuple[List[float], List[ComplexArray]]:

        h = h_initial

        time_values = np.arange(
            t0,
            tf + h,
            h
        )

        t_list = list(time_values)

        y_list = []

        self.iterations = 0

        # ----------------------------------------------------
        # Initial condition
        # ----------------------------------------------------

        y = y0.astype(np.complex128)

        y_list.append(y.copy())

        # ====================================================
        # RK4 bootstrap
        # ====================================================

        k1 = h * derivative_function(
            time_values[0],
            y,
            *args
        )

        k2 = h * derivative_function(
            time_values[0] + h / 2,
            y + k1 / 2,
            *args
        )

        k3 = h * derivative_function(
            time_values[0] + h / 2,
            y + k2 / 2,
            *args
        )

        k4 = h * derivative_function(
            time_values[0] + h,
            y + k3,
            *args
        )

        y1 = y + (
            1 / 6
        ) * (
            k1
            + 2 * k2
            + 2 * k3
            + k4
        )

        # Hermiticity correction
        if self.enforce_hermiticity:

            y1 = (
                y1 + y1.conj().T
            ) / 2.0

        # Trace normalization
        if self.enforce_trace:

            trace = np.trace(y1)

            if np.abs(trace) > 1e-14:

                y1 = y1 / trace

        y_list.append(y1.copy())

        # ====================================================
        # Adams-Bashforth propagation
        # ====================================================

        reached_final_time = False

        for n in range(1, len(time_values) - 1):

            t_n = time_values[n]

            t_nm1 = time_values[n - 1]

            y_n = y_list[n]

            y_nm1 = y_list[n - 1]

            # Derivatives
            f_n = derivative_function(
                t_n,
                y_n,
                *args
            )

            f_nm1 = derivative_function(
                t_nm1,
                y_nm1,
                *args
            )

            # Adams-Bashforth update
            y_next = (y_n + h / 2 * (3 * f_n- f_nm1))

    
            self.last_error = np.linalg.norm(
                y_next - y_n
            )

            # Hermiticity correction
            if self.enforce_hermiticity:

                y_next = (
                    y_next
                    + y_next.conj().T
                ) / 2.0

            # Trace normalization
            if self.enforce_trace:

                trace = np.trace(y_next)

                if np.abs(trace) > 1e-14:

                    y_next = y_next / trace

            y_list.append(y_next.copy())

            self.iterations += 1

            if n == len(time_values) - 2:
                reached_final_time = True

            if self.iterations >= self.max_iterations:

                break

        if not reached_final_time:
            last_time = float(t_list[len(y_list) - 1])
            raise RuntimeError(
                "AdamsBashforth2Solver stopped before reaching the final time. "
                f"Reached t={last_time:.16e} while tf={tf:.16e}. "
                "Increase solver_max_iterations or reduce the requested time window."
            )

        self.check_convergence()

        return t_list[:len(y_list)], y_list
