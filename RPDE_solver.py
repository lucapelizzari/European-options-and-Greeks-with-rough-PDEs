
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union, Literal

import numpy as np


from scipy.special import ndtr



Array = np.ndarray
Coeff = Callable[[float, Array], Array]
Payoff = Callable[[Array], Array]
Boundary = Union[Array, Callable[[float], Array]]


@dataclass(frozen=True)
class SpaceTimeGrid:
    t: Array  # shape (J + 1,)
    x: Array  # shape (N + 1,)

    @property
    def dt_count(self) -> int:
        return len(self.t) - 1

    @property
    def dx(self) -> float:
        return float(self.x[1] - self.x[0])

    @property
    def n_space_intervals(self) -> int:
        return len(self.x) - 1

    @property
    def x_interior(self) -> Array:
        return self.x[1:-1]


@dataclass(frozen=True)
class OneDimensionalRoughPath:
    """
    Cumulative path data on the PDE time grid.

    Y[m, j]  = Y^m_{t_j}
    qv[m, j] = [Y^m]_{t_j}, or the variance clock used in the L-part.

    If second_level is None, the geometric one-dimensional lift is used:
        YY_{j,j+1} = 0.5 * (Y_{j+1} - Y_j)^2.

    Otherwise, second_level[m, j] should contain YY^m_{t_j,t_{j+1}}.
    """
    Y: Array              # shape (M, J + 1)
    qv: Array            # shape (M, J + 1)
    second_level: Optional[Array] = None  # shape (M, J), optional


def cumulative_from_increments(dX: Array) -> Array:
    """
    Convert increments dX of shape (M, J) into cumulative paths of shape (M, J + 1).
    """
    X = np.zeros((dX.shape[0], dX.shape[1] + 1), dtype=float)
    X[:, 1:] = np.cumsum(dX, axis=1)
    return X


def interpolate_paths_to_grid(source_t: Array, values: Array, target_t: Array) -> Array:
    """
    Interpolate cumulative path samples onto the PDE time grid.

    values has shape (M, len(source_t)).
    Returns array of shape (M, len(target_t)).
    """
    values = np.asarray(values, dtype=float)
    out = np.empty((values.shape[0], len(target_t)), dtype=float)
    for m in range(values.shape[0]):
        out[m] = np.interp(target_t, source_t, values[m])
    return out


def _eval_coeff(func: Coeff, t: float, x: Array) -> Array:
    y = np.asarray(func(float(t), x), dtype=float)
    return np.broadcast_to(y, x.shape).astype(float, copy=False)


def _prepare_boundary(boundary: Boundary, t_grid: Array, M: int, name: str) -> Array:
    """
    Return boundary values with shape (M, J + 1).

    boundary can be:
    - array of shape (M, J + 1),
    - array of shape (J + 1,), broadcast over samples,
    - callable t -> scalar or array of shape (M,).
    """
    J_plus_1 = len(t_grid)

    if callable(boundary):
        values = []
        for t in t_grid:
            v = np.asarray(boundary(float(t)), dtype=float)
            if v.ndim == 0:
                v = np.full(M, float(v))
            else:
                v = np.broadcast_to(v, (M,))
            values.append(v)
        return np.stack(values, axis=1)

    boundary = np.asarray(boundary, dtype=float)

    if boundary.shape == (M, J_plus_1):
        return boundary

    if boundary.shape == (J_plus_1,):
        return np.broadcast_to(boundary[None, :], (M, J_plus_1)).copy()

    raise ValueError(
        f"{name} must have shape (M, J+1), shape (J+1,), or be callable. "
        f"Got shape {boundary.shape}."
    )


def _f0_on_grid(
    f: Coeff,
    t: float,
    x_grid: Array,
    dx: float,
    f_x: Optional[Coeff],
) -> Array:
    """
    Compute f0(t,x) = -0.5 f(t,x) partial_x f(t,x) on the interior grid.

    If f_x is not given, use the same forward difference as in the paper code:
        partial_x f(t,x_n) approx (f(t,x_{n+1}) - f(t,x_n)) / dx.
    """
    x_int = x_grid[1:-1]
    f_val = _eval_coeff(f, t, x_int)

    if f_x is None:
        f_right = _eval_coeff(f, t, x_grid[2:])
        fx_val = (f_right - f_val) / dx
    else:
        fx_val = _eval_coeff(f_x, t, x_int)

    return -0.5 * f_val * fx_val


def solve_tridiagonal_batch(lower: Array, diag: Array, upper: Array, rhs: Array) -> Array:
    """
    Solve batched tridiagonal systems.

    lower, diag, upper, rhs all have shape (M, n).
    lower[:, 0] and upper[:, -1] are ignored and should be zero.
    """
    lower = lower.copy()
    diag = diag.copy()
    upper = upper.copy()
    rhs = rhs.copy()

    M, n = rhs.shape

    if n == 1:
        return rhs / diag

    for k in range(1, n):
        multiplier = lower[:, k] / diag[:, k - 1]
        diag[:, k] -= multiplier * upper[:, k - 1]
        rhs[:, k] -= multiplier * rhs[:, k - 1]

    sol = np.empty_like(rhs)
    sol[:, -1] = rhs[:, -1] / diag[:, -1]

    for k in range(n - 2, -1, -1):
        sol[:, k] = (rhs[:, k] - upper[:, k] * sol[:, k + 1]) / diag[:, k]

    return sol


def solve_rpde_paper_scheme_1d(
    *,
    f: Coeff,
    g: Coeff,
    payoff: Payoff,
    grid: SpaceTimeGrid,
    rough_path: OneDimensionalRoughPath,
    boundary_left: Boundary,
    boundary_right: Boundary,
    f_x: Optional[Coeff] = None,
) -> Array:
    """
    Solve the one-dimensional backward RPDE from the paper.

    Equation:
        -d_t u = L_t u d[qv]_t + Gamma_t u dY_t

    with
        L u      = 0.5 g^2 u_xx + f0 u_x,
        Gamma u = f u_x,
        Gamma' u = f d_x(f u_x)
                 = f^2 u_xx - 2 f0 u_x,

    and
        f0 = -0.5 f f_x.

    Paper scheme:
        u_j = u_{j+1}
              + L_j u_j Delta qv_j
              + Gamma_{j+1} u_{j+1} Delta Y_j
              + Gamma'_j u_j YY_{j,j+1}.

    In the one-dimensional geometric case:
        YY_{j,j+1} = 0.5 * (Delta Y_j)^2.

    Returns
    -------
    U : array, shape (M, J + 1, N + 1)
        U[m, j, n] approximates u^{Y_m}(t_j, x_n).
    """
    t_grid = np.asarray(grid.t, dtype=float)
    x_grid = np.asarray(grid.x, dtype=float)

    J = len(t_grid) - 1
    N = len(x_grid) - 1
    n_int = N - 1

    if n_int < 1:
        raise ValueError("Need at least one interior spatial grid point.")

    dx = grid.dx
    if not np.allclose(np.diff(x_grid), dx):
        raise ValueError("Only uniform spatial grids are currently supported.")

    Y = np.asarray(rough_path.Y, dtype=float)
    qv = np.asarray(rough_path.qv, dtype=float)

    if Y.shape != qv.shape:
        raise ValueError(f"Y and qv must have the same shape. Got {Y.shape} and {qv.shape}.")

    M, J_plus_1 = Y.shape
    if J_plus_1 != J + 1:
        raise ValueError(f"rough_path arrays must have shape (M, J+1) = (M, {J+1}).")

    left = _prepare_boundary(boundary_left, t_grid, M, "boundary_left")
    right = _prepare_boundary(boundary_right, t_grid, M, "boundary_right")

    U = np.zeros((M, J + 1, N + 1), dtype=float)

    # Boundary values for all times.
    U[:, :, 0] = left
    U[:, :, -1] = right

    # Terminal condition takes priority at t = T, including the corners.
    U[:, J, :] = np.broadcast_to(np.asarray(payoff(x_grid), dtype=float), (M, N + 1))

    x_int = x_grid[1:-1]
    dx2 = dx * dx

    for j in range(J - 1, -1, -1):
        t_j = float(t_grid[j])
        t_next = float(t_grid[j + 1])

        dy = Y[:, j + 1] - Y[:, j]
        dqv = qv[:, j + 1] - qv[:, j]

        if rough_path.second_level is None:
            dYY = 0.5 * dy * dy
        else:
            dYY = np.asarray(rough_path.second_level[:, j], dtype=float)

        # Coefficients at t_j for the implicit L and Gamma' part.
        f_j = _eval_coeff(f, t_j, x_int)
        g_j = _eval_coeff(g, t_j, x_int)
        f0_j = _f0_on_grid(f, t_j, x_grid, dx, f_x)

        # L = a D2 + b D+
        a_L = 0.5 * g_j**2 / dx2
        b_L = f0_j / dx

        L_lower = a_L
        L_diag = -2.0 * a_L - b_L
        L_upper = a_L + b_L

        # Gamma' = f^2 D2 - 2 f0 D+
        c_G = f_j**2 / dx2
        e_G = -2.0 * f0_j / dx

        Gp_lower = c_G
        Gp_diag = -2.0 * c_G - e_G
        Gp_upper = c_G + e_G

        # A = I - dqv L_j - dYY Gamma'_j.
        lower = -(dqv[:, None] * L_lower[None, :] + dYY[:, None] * Gp_lower[None, :])
        diag = 1.0 - (dqv[:, None] * L_diag[None, :] + dYY[:, None] * Gp_diag[None, :])
        upper = -(dqv[:, None] * L_upper[None, :] + dYY[:, None] * Gp_upper[None, :])

        # RHS = u_{j+1} + dy Gamma_{j+1} u_{j+1}.
        u_next = U[:, j + 1, :]
        rhs = u_next[:, 1:-1].copy()

        f_next = _eval_coeff(f, t_next, x_int)
        right_neighbour_next = u_next[:, 2:]       # includes right boundary
        current_next = u_next[:, 1:-1]

        rhs += dy[:, None] * (f_next[None, :] / dx) * (
            right_neighbour_next - current_next
        )

        # Move implicit boundary terms to RHS.
        rhs[:, 0] -= lower[:, 0] * U[:, j, 0]
        rhs[:, -1] -= upper[:, -1] * U[:, j, -1]

        lower[:, 0] = 0.0
        upper[:, -1] = 0.0

        U[:, j, 1:-1] = solve_tridiagonal_batch(lower, diag, upper, rhs)

    return U



def _normal_cdf(x: Array) -> Array:
    return ndtr(x)

def _normal_pdf(x: Array) -> Array:
    """
    Standard normal density.
    """
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


@dataclass(frozen=True)
class RomanoTouziBoundary:
    """
    Romano--Touzi explicit conditional boundary values for vanilla puts/calls.

    This is for the two explicitly solvable cases:

    1. Black--Scholes/lognormal case:
           dX = rho X dI + sqrt(1-rho^2) X dM

    2. Bachelier/normal case:
           dX = rho dI + sqrt(1-rho^2) dM

    Here I is the frozen conditional driver and qv is its quadratic variation /
    variance clock. For each sample m and time t_j, this computes

           E[ payoff(X_T) | I, X_{t_j} = x_boundary ]

    at x_boundary = a and x_boundary = b.
    """
    strike: float
    rho: float
    mode: Literal["black_scholes", "bachelier"] = "black_scholes"
    option_type: Literal["put", "call"] = "put"
    variance_floor: float = 1.0e-12

    def compute(
        self,
        *,
        grid: SpaceTimeGrid,
        rough_path: OneDimensionalRoughPath,
    ) -> tuple[Array, Array]:
        left = self.values_at_x(float(grid.x[0]), rough_path)
        right = self.values_at_x(float(grid.x[-1]), rough_path)
        return left, right

    def values_at_x(
        self,
        x0: float,
        rough_path: OneDimensionalRoughPath,
    ) -> Array:
        Y = np.asarray(rough_path.Y, dtype=float)
        qv = np.asarray(rough_path.qv, dtype=float)

        dY_to_T = Y[:, [-1]] - Y
        dqv_to_T = qv[:, [-1]] - qv

        var = (1.0 - self.rho**2) * dqv_to_T
        var_safe = np.maximum(var, self.variance_floor)
        sd = np.sqrt(var_safe)

        if self.mode == "black_scholes":
            if x0 <= 0.0:
                raise ValueError("Black--Scholes Romano--Touzi boundary requires x0 > 0.")

            mu = np.log(x0) - 0.5 * dqv_to_T + self.rho * dY_to_T
            return self._black_scholes_price(mu=mu, var=var, var_safe=var_safe, sd=sd)

        if self.mode == "bachelier":
            mu = x0 + self.rho * dY_to_T
            return self._bachelier_price(mu=mu, var=var, sd=sd)

        raise ValueError(f"Unknown mode {self.mode!r}.")

    def _black_scholes_price(
        self,
        *,
        mu: Array,
        var: Array,
        var_safe: Array,
        sd: Array,
    ) -> Array:
        K = float(self.strike)

        logK = np.log(K)
        d_put_1 = (logK - mu) / sd
        d_put_2 = (logK - mu - var_safe) / sd

        forward = np.exp(mu + 0.5 * var_safe)

        put = K * _normal_cdf(d_put_1) - forward * _normal_cdf(d_put_2)

        deterministic = np.maximum(K - np.exp(mu), 0.0)
        put = np.where(var <= self.variance_floor, deterministic, put)

        if self.option_type == "put":
            return put

        if self.option_type == "call":
            call = put + forward - K
            deterministic_call = np.maximum(np.exp(mu) - K, 0.0)
            return np.where(var <= self.variance_floor, deterministic_call, call)

        raise ValueError(f"Unknown option_type {self.option_type!r}.")

    def _bachelier_price(
        self,
        *,
        mu: Array,
        var: Array,
        sd: Array,
    ) -> Array:
        K = float(self.strike)

        z = (K - mu) / sd
        put = (K - mu) * _normal_cdf(z) + sd * _normal_pdf(z)

        deterministic = np.maximum(K - mu, 0.0)
        put = np.where(var <= self.variance_floor, deterministic, put)

        if self.option_type == "put":
            return put

        if self.option_type == "call":
            call = put + mu - K
            deterministic_call = np.maximum(mu - K, 0.0)
            return np.where(var <= self.variance_floor, deterministic_call, call)

        raise ValueError(f"Unknown option_type {self.option_type!r}.")


@dataclass(frozen=True)
class MonteCarloEulerBoundary:
    """
    Conditional Monte Carlo boundary values for general coefficients.

    For each outer sample m, each boundary point x in {a,b}, and each PDE time t_j,
    this estimates

        E[ payoff(X_T) | I^m, X_{t_j} = x ]

    by simulating

        X_{k+1} = X_k
                  + f(t_k, X_k) Delta I_k
                  + g(t_k, X_k) sqrt(Delta qv_k) Z_k.

    This is expensive but robust and works for beta != 1.
    """
    f: Coeff
    g: Coeff
    payoff: Payoff
    n_inner: int = 10_000
    seed: Optional[int] = None
    floor: Optional[float] = 0.0
    outer_batch_size: int = 128

    def compute(
        self,
        *,
        grid: SpaceTimeGrid,
        rough_path: OneDimensionalRoughPath,
    ) -> tuple[Array, Array]:
        left = self.values_at_x(float(grid.x[0]), grid, rough_path)
        right = self.values_at_x(float(grid.x[-1]), grid, rough_path)
        return left, right

    def values_at_x(
        self,
        x0: float,
        grid: SpaceTimeGrid,
        rough_path: OneDimensionalRoughPath,
    ) -> Array:
        rng = np.random.default_rng(self.seed)

        t_grid = np.asarray(grid.t, dtype=float)
        Y = np.asarray(rough_path.Y, dtype=float)
        qv = np.asarray(rough_path.qv, dtype=float)

        M, J_plus_1 = Y.shape
        J = J_plus_1 - 1

        dY = Y[:, 1:] - Y[:, :-1]
        dqv = qv[:, 1:] - qv[:, :-1]
        dqv = np.maximum(dqv, 0.0)

        out = np.empty((M, J + 1), dtype=float)

        # At maturity the boundary value is just the payoff.
        out[:, J] = self.payoff(np.full(M, x0, dtype=float))

        for j in range(J - 1, -1, -1):
            values_j = np.empty(M, dtype=float)

            for m0 in range(0, M, self.outer_batch_size):
                m1 = min(m0 + self.outer_batch_size, M)
                batch_size = m1 - m0

                X = np.full((batch_size, self.n_inner), x0, dtype=float)

                for k in range(j, J):
                    t_k = float(t_grid[k])

                    drift_increment = dY[m0:m1, k][:, None]
                    diffusion_sd = np.sqrt(dqv[m0:m1, k])[:, None]

                    Z = rng.standard_normal((batch_size, self.n_inner))

                    X = (
                        X
                        + self.f(t_k, X) * drift_increment
                        + self.g(t_k, X) * diffusion_sd * Z
                    )

                    if self.floor is not None:
                        X = np.maximum(X, self.floor)

                values_j[m0:m1] = np.mean(self.payoff(X), axis=1)

            out[:, j] = values_j

        return out

@dataclass(frozen=True)
class SimpleAsymptoticBoundary:
    """
    Simple time-independent Dirichlet/asymptotic boundary.

    By default this sets

        u(t,a) = payoff(a),
        u(t,b) = payoff(b),

    for all times and all outer samples.

    For a put with a close to 0 and b large, this corresponds to

        u(t,a) approx K - a,
        u(t,b) approx 0.

    You can also manually override the left/right constants.
    """
    payoff: Payoff
    left_value: Optional[float] = None
    right_value: Optional[float] = None

    def compute(
        self,
        *,
        grid: SpaceTimeGrid,
        rough_path: OneDimensionalRoughPath,
    ) -> tuple[Array, Array]:
        M, J_plus_1 = rough_path.Y.shape

        if self.left_value is None:
            left_value = float(np.ravel(self.payoff(np.array([grid.x[0]], dtype=float)))[0])
        else:
            left_value = float(self.left_value)

        if self.right_value is None:
            right_value = float(np.ravel(self.payoff(np.array([grid.x[-1]], dtype=float)))[0])
        else:
            right_value = float(self.right_value)

        boundary_left = np.full((M, J_plus_1), left_value, dtype=float)
        boundary_right = np.full((M, J_plus_1), right_value, dtype=float)

        return boundary_left, boundary_right




