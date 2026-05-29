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
        max_iterations: int | None = None,
        dtype: str = "complex128"
    ):
        if tolerance <= 0.0:
            raise ValueError("tolerance must be strictly positive.")

        self.tolerance: float = tolerance

        self.max_iterations: int | None = max_iterations

        self.dtype: str = dtype

        self.last_error: float = np.inf
        self.last_allowed_error: float = np.inf
        self.last_scale: float = 0.0

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

        self.converged = self.last_error < self.last_allowed_error

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
            "tolerance_mode": "relative",

            "max_iterations": self.max_iterations,

            "dtype": self.dtype,

            "iterations": self.iterations,

            "last_error": self.last_error,

            "last_allowed_error": self.last_allowed_error,

            "last_scale": self.last_scale,

            "converged": self.converged
        }

    def _error_budget(self, *references: ComplexArray) -> float:
        """Return one scale-aware error budget for the current tentative step.

        The user-facing ``tolerance`` acts as a relative tolerance. The solver
        converts it into an absolute step-acceptance threshold by looking at the
        magnitude of the states/increments involved in the current step.
        """

        scales = [
            float(np.linalg.norm(np.asarray(reference, dtype=np.complex128).ravel()))
            for reference in references
        ]
        scale = max(scales, default=0.0)
        floor = 1.0e-14
        allowed_error = max(floor, self.tolerance * max(scale, floor))
        self.last_scale = scale
        self.last_allowed_error = allowed_error
        return allowed_error

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
        max_iterations: int | None = None,
        dtype: str = "complex128",
        h_min: float = 1e-12,
        h_max: float = 1.0,
        safety_factor: float = 0.9,
        min_factor: float = 0.2,
        max_factor: float = 4.0,
        max_rejections: int = 10000,
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

        self.safety_factor = safety_factor

        self.min_factor = min_factor

        self.max_factor = max_factor

        self.max_rejections = max_rejections

        if self.h_min <= 0.0:
            raise ValueError("h_min must be strictly positive.")
        if self.h_max <= 0.0:
            raise ValueError("h_max must be strictly positive.")
        if self.h_max < self.h_min:
            raise ValueError("h_max must be greater than or equal to h_min.")
        if self.safety_factor <= 0.0:
            raise ValueError("safety_factor must be strictly positive.")
        if self.min_factor <= 0.0:
            raise ValueError("min_factor must be strictly positive.")
        if self.max_factor < self.min_factor:
            raise ValueError("max_factor must be greater than or equal to min_factor.")
        if self.max_rejections <= 0:
            raise ValueError("max_rejections must be strictly positive.")

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
        rejected_steps = 0

        while t < tf:

            if t + h > tf:

                h = tf - t

            ti = t

            yi = y

            k1 = h * derivative_function(ti,yi,*args)

            k2 = h * derivative_function(ti + h / 4, yi + k1 / 4, *args)

            k3 = h * derivative_function(ti + 3 * h / 8, yi+ 3 * k1 / 32 + 9 * k2 / 32, *args)

            k4 = h * derivative_function(ti + 12 * h / 13,yi+ 1932 * k1 / 2197 - 7200 * k2 / 2197 + 7296 * k3 / 2197,*args)

            k5 = h * derivative_function(ti + h,yi+ 439 * k1 / 216 - 8 * k2+ 3680 * k3 / 513 - 845 * k4 / 4104,*args)

            k6 = h * derivative_function(ti + h / 2,yi- 8 * k1 / 27 + 2 * k2 + 3544 * k3 / 2565 - 1859 * k4 / 4140 - 11 * k5 / 40,*args)

            # =================================================
            # Fifth-order approximation
            # =================================================

            y5 = (yi + 16 * k1 / 135 + 6656 * k3 / 12825 + 28561 * k4 / 56430 - 9 * k5 / 50 + 2 * k6 / 55)

            # =================================================
            # Error estimation
            # =================================================

            error_tensor = (k1 / 360 - 128 * k3 / 4275 - 2197 * k4 / 75240 + k5 / 50 + 2* k6 / 55 )

            err = np.linalg.norm(error_tensor)
            allowed_error = self._error_budget(yi, y5, k1, k2, k3, k4, k5, k6)

            if err == 0:

                err = 1e-16

            self.last_error = err

            # =================================================
            # Accept timestep
            # =================================================

            if err < allowed_error:

                t = t + h

                y = y5
                rejected_steps = 0

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
            else:
                rejected_steps += 1

            # =================================================
            # Adaptive timestep update
            # =================================================

            factor = (
                self.safety_factor
                * (allowed_error / err) ** (1 / 4)
            )

            factor = min(
                self.max_factor,
                max(self.min_factor, factor)
            )

            h = h * factor

            h = min(self.h_max, h)

            h = max(self.h_min, h)

            if err >= allowed_error and h <= self.h_min:
                raise RuntimeError(
                    "RKF45Solver reached solver_h_min before satisfying the "
                    f"requested scale-aware tolerance. Current error={err:.16e}, "
                    f"allowed_error={allowed_error:.16e}, scale={self.last_scale:.16e}, "
                    f"h_min={self.h_min:.16e}."
                )

            if rejected_steps >= self.max_rejections:
                raise RuntimeError(
                    "RKF45Solver accumulated too many consecutive rejected steps. "
                    f"Reached {rejected_steps} rejected steps at t={t:.16e}. "
                    "Relax the tolerance, increase solver_h_max, or increase "
                    "solver_max_rejections."
                )

            # =================================================
            # Iteration counter
            # =================================================

            self.iterations += 1

            if self.max_iterations is not None and self.max_iterations > 0 and self.iterations >= self.max_iterations:

                break

        if not reached_final_time and t < tf:
            raise RuntimeError(
                "RKF45Solver stopped before reaching the final time. "
                f"Reached t={t:.16e} while tf={tf:.16e}. "
                "Increase solver_max_iterations, increase solver_h_max, "
                "or relax the relative tolerance."
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
        max_iterations: int | None = None,
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
            self._error_budget(y_nm1, y_n, y_next, h * f_nm1, h * f_n)

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

            if self.max_iterations is not None and self.max_iterations > 0 and self.iterations >= self.max_iterations:

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
