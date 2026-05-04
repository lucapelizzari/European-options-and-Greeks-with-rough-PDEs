
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union, Literal

import numpy as np


from scipy.special import ndtr

@dataclass(frozen=True)
class CurveEstimate:
    """
    Monte Carlo estimate of a curve over space.
    """
    x: Array
    mean: Array
    std: Array
    stderr: Array
    samples: Optional[Array] = None


@dataclass(frozen=True)
class FiniteDifferenceGreeks:
    """
    Compute price, Delta and Gamma curves from sample-wise RPDE solutions.

    Expected default solution layout is

        U[m, j, n] = u^{Y_m}(t_j, x_n),

    i.e. shape (M, J+1, N+1), which is the layout of solve_rpde_paper_scheme_1d.

    If using old arrays with shape (M, N+1, J+1), set

        layout="space_time".

    The class supports two ways of computing Greeks:

    1. Grid-based finite differences:
           delta(), gamma()

       These use the PDE spatial mesh size dx.

    2. Bumped finite differences:
           delta_bumped(), gamma_bumped()

       These use a user-chosen bump h and interpolate the sample-wise PDE
       solution curves to x-h, x, x+h.
    """
    solution: Array
    grid: SpaceTimeGrid
    layout: Literal["time_space", "space_time"] = "time_space"

    def _values_at_time(self, time_index: int = 0) -> Array:
        """
        Extract sample-wise spatial curves at a fixed time index.

        Returns
        -------
        values : array, shape (M, N+1)
            values[m, n] = u^{Y_m}(t_j, x_n).
        """
        U = np.asarray(self.solution, dtype=float)

        if U.ndim == 2:
            values = U

        elif U.ndim == 3:
            if self.layout == "time_space":
                values = U[:, time_index, :]
            elif self.layout == "space_time":
                values = U[:, :, time_index]
            else:
                raise ValueError(f"Unknown layout {self.layout!r}.")

        else:
            raise ValueError(
                "solution must have shape (M,N+1), "
                "(M,J+1,N+1), or (M,N+1,J+1)."
            )

        expected_n = len(self.grid.x)
        if values.shape[1] != expected_n:
            raise ValueError(
                f"Spatial dimension mismatch: got {values.shape[1]} values, "
                f"but grid has {expected_n} points."
            )

        return values

    @staticmethod
    def _estimate(
        x: Array,
        samples: Array,
        include_samples: bool,
    ) -> CurveEstimate:
        """
        Average sample-wise curves and compute standard errors.
        """
        samples = np.asarray(samples, dtype=float)
        M = samples.shape[0]

        mean = np.mean(samples, axis=0)

        if M > 1:
            std = np.std(samples, axis=0, ddof=1)
            stderr = std / np.sqrt(M)
        else:
            std = np.zeros_like(mean)
            stderr = np.zeros_like(mean)

        return CurveEstimate(
            x=np.asarray(x, dtype=float),
            mean=mean,
            std=std,
            stderr=stderr,
            samples=samples if include_samples else None,
        )

    def _interp_sample_curves(
        self,
        values: Array,
        x_query: Array,
    ) -> Array:
        """
        Interpolate sample-wise spatial curves.

        Parameters
        ----------
        values : array, shape (M, N+1)
            Sample-wise PDE solution curves.
        x_query : array, shape (L,)
            Spatial points where the curves should be evaluated.

        Returns
        -------
        out : array, shape (M, L)
            Interpolated values.
        """
        x_grid = np.asarray(self.grid.x, dtype=float)
        x_query = np.atleast_1d(np.asarray(x_query, dtype=float))

        if np.any(x_query < x_grid[0]) or np.any(x_query > x_grid[-1]):
            raise ValueError(
                "Interpolation points must lie inside the spatial grid. "
                f"Grid interval is [{x_grid[0]}, {x_grid[-1]}], "
                f"but requested range is [{np.min(x_query)}, {np.max(x_query)}]."
            )

        out = np.empty((values.shape[0], len(x_query)), dtype=float)

        for m in range(values.shape[0]):
            out[m] = np.interp(x_query, x_grid, values[m])

        return out

    def price(
        self,
        *,
        time_index: int = 0,
        include_samples: bool = False,
    ) -> CurveEstimate:
        """
        Price curve on the PDE grid:

            x_n -> E[u^Y(t_j, x_n)].
        """
        values = self._values_at_time(time_index)
        return self._estimate(self.grid.x, values, include_samples)

    def price_at(
        self,
        x_eval: Array,
        *,
        time_index: int = 0,
        include_samples: bool = False,
    ) -> CurveEstimate:
        """
        Price curve evaluated at arbitrary spatial points by interpolation.

        Parameters
        ----------
        x_eval : array-like
            Evaluation points inside the spatial grid.
        """
        x_eval = np.atleast_1d(np.asarray(x_eval, dtype=float))
        values = self._values_at_time(time_index)

        samples = self._interp_sample_curves(values, x_eval)

        return self._estimate(x_eval, samples, include_samples)

    def delta(
        self,
        *,
        time_index: int = 0,
        method: Literal["central", "forward", "backward"] = "central",
        include_samples: bool = False,
    ) -> CurveEstimate:
        """
        Delta curve computed by grid-based finite differences.

        central:
            Delta(x_n) approx (u(x_{n+1}) - u(x_{n-1})) / (2 dx),
            returned on x[1:-1].

        forward:
            Delta(x_n) approx (u(x_{n+1}) - u(x_n)) / dx,
            returned on x[:-1].

        backward:
            Delta(x_n) approx (u(x_n) - u(x_{n-1})) / dx,
            returned on x[1:].
        """
        values = self._values_at_time(time_index)
        x = np.asarray(self.grid.x, dtype=float)
        dx = self.grid.dx

        if method == "central":
            samples = (values[:, 2:] - values[:, :-2]) / (2.0 * dx)
            x_delta = x[1:-1]

        elif method == "forward":
            samples = (values[:, 1:] - values[:, :-1]) / dx
            x_delta = x[:-1]

        elif method == "backward":
            samples = (values[:, 1:] - values[:, :-1]) / dx
            x_delta = x[1:]

        else:
            raise ValueError(f"Unknown delta method {method!r}.")

        return self._estimate(x_delta, samples, include_samples)

    def delta_bumped(
        self,
        x_eval: Array,
        *,
        bump: float,
        time_index: int = 0,
        method: Literal["central", "forward", "backward"] = "central",
        include_samples: bool = False,
    ) -> CurveEstimate:
        """
        Delta at arbitrary spatial points using a user-chosen bump.

        central:
            Delta(x) approx (u(x+h) - u(x-h)) / (2h)

        forward:
            Delta(x) approx (u(x+h) - u(x)) / h

        backward:
            Delta(x) approx (u(x) - u(x-h)) / h
        """
        x_eval = np.atleast_1d(np.asarray(x_eval, dtype=float))
        h = float(bump)

        if h <= 0.0:
            raise ValueError("bump must be positive.")

        values = self._values_at_time(time_index)

        if method == "central":
            u_plus = self._interp_sample_curves(values, x_eval + h)
            u_minus = self._interp_sample_curves(values, x_eval - h)
            samples = (u_plus - u_minus) / (2.0 * h)

        elif method == "forward":
            u_plus = self._interp_sample_curves(values, x_eval + h)
            u_0 = self._interp_sample_curves(values, x_eval)
            samples = (u_plus - u_0) / h

        elif method == "backward":
            u_0 = self._interp_sample_curves(values, x_eval)
            u_minus = self._interp_sample_curves(values, x_eval - h)
            samples = (u_0 - u_minus) / h

        else:
            raise ValueError(f"Unknown delta method {method!r}.")

        return self._estimate(x_eval, samples, include_samples)

    def gamma(
        self,
        *,
        time_index: int = 0,
        include_samples: bool = False,
    ) -> CurveEstimate:
        """
        Gamma curve computed by the centered grid-based second difference:

            Gamma(x_n) approx
            (u(x_{n+1}) - 2u(x_n) + u(x_{n-1})) / dx^2,

        returned on x[1:-1].
        """
        values = self._values_at_time(time_index)
        x = np.asarray(self.grid.x, dtype=float)
        dx = self.grid.dx

        samples = (
            values[:, 2:]
            - 2.0 * values[:, 1:-1]
            + values[:, :-2]
        ) / (dx * dx)

        return self._estimate(x[1:-1], samples, include_samples)

    def gamma_bumped(
        self,
        x_eval: Array,
        *,
        bump: float,
        time_index: int = 0,
        include_samples: bool = False,
    ) -> CurveEstimate:
        """
        Gamma at arbitrary spatial points using a user-chosen bump:

            Gamma(x) approx
            (u(x+h) - 2u(x) + u(x-h)) / h^2.
        """
        x_eval = np.atleast_1d(np.asarray(x_eval, dtype=float))
        h = float(bump)

        if h <= 0.0:
            raise ValueError("bump must be positive.")

        values = self._values_at_time(time_index)

        u_plus = self._interp_sample_curves(values, x_eval + h)
        u_0 = self._interp_sample_curves(values, x_eval)
        u_minus = self._interp_sample_curves(values, x_eval - h)

        samples = (u_plus - 2.0 * u_0 + u_minus) / (h * h)

        return self._estimate(x_eval, samples, include_samples)

    @staticmethod
    def interpolate(
        estimate: CurveEstimate,
        x0: float,
    ) -> tuple[float, float]:
        """
        Interpolate mean and standard error of a CurveEstimate at x0.

        Returns
        -------
        mean_x0 : float
        stderr_x0 : float
        """
        mean_x0 = float(np.interp(x0, estimate.x, estimate.mean))
        stderr_x0 = float(np.interp(x0, estimate.x, estimate.stderr))
        return mean_x0, stderr_x0