"""
Parametric airfoil geometry (pure NumPy — no FEM dependencies).

The NACA 4-digit family: thickness t (chord fraction), maximum camber m
(chord fraction), camber position p (chord fraction). Classic equations with
the closed-trailing-edge thickness polynomial (the -0.1036 coefficient), so
the surface is a clean closed polygon for meshing.

Also provides the signed distance function of a polygon on a grid — the
operator-learning input encoding for geometry (FlowBench-style SDF channels).
"""

from typing import Tuple

import numpy as np


def naca4_coords(
    thickness: float = 0.12,
    camber: float = 0.0,
    camber_pos: float = 0.4,
    n_points: int = 120,
) -> np.ndarray:
    """
    Closed surface polygon of a NACA 4-digit airfoil, chord on [0, 1].

    Returns an (N, 2) array tracing the upper surface TE -> LE then the lower
    surface LE -> TE; the last point closes onto the first (sharp TE).
    Cosine spacing clusters points at the leading and trailing edges.
    """
    x = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_points)))  # cosine spacing

    t = thickness
    # closed-TE thickness distribution (-0.1036 instead of -0.1015)
    yt = (
        5.0
        * t
        * (
            0.2969 * np.sqrt(x)
            - 0.1260 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1036 * x**4
        )
    )

    m, p = camber, camber_pos
    if m > 0.0 and 0.0 < p < 1.0:
        yc = np.where(
            x < p,
            m / p**2 * (2.0 * p * x - x**2),
            m / (1.0 - p) ** 2 * ((1.0 - 2.0 * p) + 2.0 * p * x - x**2),
        )
        dyc = np.where(
            x < p,
            2.0 * m / p**2 * (p - x),
            2.0 * m / (1.0 - p) ** 2 * (p - x),
        )
    else:
        yc = np.zeros_like(x)
        dyc = np.zeros_like(x)

    theta = np.arctan(dyc)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    # upper TE -> LE, then lower LE -> TE (skip duplicated LE point)
    upper = np.column_stack([xu[::-1], yu[::-1]])
    lower = np.column_stack([xl[1:], yl[1:]])
    poly = np.vstack([upper, lower])

    # weld the trailing edge shut exactly
    te = 0.5 * (poly[0] + poly[-1])
    poly[0] = te
    poly[-1] = te
    return poly


def rotate_airfoil(
    points: np.ndarray,
    angle_of_attack_deg: float,
    center: Tuple[float, float] = (0.25, 0.0),
) -> np.ndarray:
    """
    Pitch the airfoil by the angle of attack (positive = nose up), rotating
    about the quarter-chord point. With flow in +x, positive AoA produces
    positive lift for symmetric foils.
    """
    a = -np.deg2rad(angle_of_attack_deg)  # nose-up = clockwise in flow frame
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    c = np.asarray(center)
    return (points - c) @ R.T + c


def polygon_sdf(X: np.ndarray, Y: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """
    Signed distance from grid points (X, Y) to a closed polygon:
    negative INSIDE the polygon, positive outside.

    Distance: exact min distance to the polygon's segments (vectorised).
    Sign: even-odd ray casting.
    """
    P = np.column_stack([X.ravel(), Y.ravel()])  # (M, 2)
    A = polygon[:-1]  # (S, 2) segment starts
    B = polygon[1:]  # (S, 2) segment ends

    AB = B - A  # (S, 2)
    AB2 = (AB**2).sum(axis=1)  # (S,)
    AB2[AB2 == 0.0] = 1e-300

    # projection parameter of every point onto every segment
    AP = P[:, None, :] - A[None, :, :]  # (M, S, 2)
    s = np.clip((AP * AB[None, :, :]).sum(axis=2) / AB2[None, :], 0.0, 1.0)
    closest = A[None, :, :] + s[:, :, None] * AB[None, :, :]
    dist = np.sqrt(((P[:, None, :] - closest) ** 2).sum(axis=2)).min(axis=1)

    # even-odd rule for the sign
    x, y = P[:, 0], P[:, 1]
    x1, y1 = A[:, 0], A[:, 1]
    x2, y2 = B[:, 0], B[:, 1]
    inside = np.zeros(len(P), dtype=bool)
    for k in range(len(A)):
        crosses = (y1[k] > y) != (y2[k] > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = x1[k] + (y - y1[k]) * (x2[k] - x1[k]) / (y2[k] - y1[k])
        inside ^= crosses & (x < xint)

    sdf = np.where(inside, -dist, dist)
    return sdf.reshape(X.shape)


def _stretch_profile(first: float, n: int, length: np.ndarray) -> np.ndarray:
    """
    Per-station tanh clustering on [0, 1] whose first physical step is ~first.

    Solved per station because the body-to-far-field distance varies by more
    than an order of magnitude around a C-grid; a single normalised stretch
    would put the first cell of a 20-chord line 20x too far off the wall.
    """
    length = np.atleast_1d(length)
    target = np.clip(first / length, 1e-9, 0.4)
    t1 = 1.0 / (n - 1)
    lo = np.full_like(target, 1e-3)
    hi = np.full_like(target, 14.0)
    for _ in range(80):  # vectorised bisection
        b = 0.5 * (lo + hi)
        step = 1.0 - np.tanh(b * (1.0 - t1)) / np.tanh(b)
        coarse = step > target
        lo = np.where(coarse, b, lo)
        hi = np.where(coarse, hi, b)
    b = 0.5 * (lo + hi)[:, None]
    t = np.linspace(0.0, 1.0, n)[None, :]
    return 1.0 - np.tanh(b * (1.0 - t)) / np.tanh(b)


def _winslow(X: np.ndarray, Y: np.ndarray, iters: int, omega: float = 0.5):
    """Winslow elliptic smoothing of interior nodes; boundaries stay put."""
    for _ in range(iters):
        xe = 0.5 * (X[2:, 1:-1] - X[:-2, 1:-1])
        ye = 0.5 * (Y[2:, 1:-1] - Y[:-2, 1:-1])
        xn = 0.5 * (X[1:-1, 2:] - X[1:-1, :-2])
        yn = 0.5 * (Y[1:-1, 2:] - Y[1:-1, :-2])
        a = xn**2 + yn**2
        b = xe * xn + ye * yn
        c = xe**2 + ye**2
        den = 2.0 * (a + c) + 1e-30
        for F in (X, Y):
            Fn = (
                a * (F[2:, 1:-1] + F[:-2, 1:-1])
                + c * (F[1:-1, 2:] + F[1:-1, :-2])
                - 0.5 * b * (F[2:, 2:] - F[2:, :-2] - F[:-2, 2:] + F[:-2, :-2])
            ) / den
            F[1:-1, 1:-1] = (1.0 - omega) * F[1:-1, 1:-1] + omega * Fn
    return X, Y


def _recluster(X: np.ndarray, Y: np.ndarray, eta: np.ndarray):
    """Redistribute nodes along each eta line onto the target spacing."""
    s = np.concatenate(
        [
            np.zeros((X.shape[0], 1)),
            np.cumsum(np.hypot(np.diff(X, axis=1), np.diff(Y, axis=1)), axis=1),
        ],
        axis=1,
    )
    s = s / s[:, -1:]
    Xn, Yn = np.empty_like(X), np.empty_like(Y)
    for i in range(X.shape[0]):
        Xn[i] = np.interp(eta[i], s[i], X[i])
        Yn[i] = np.interp(eta[i], s[i], Y[i])
    return Xn, Yn


def airfoil_c_grid(
    thickness: float = 0.12,
    camber: float = 0.0,
    camber_pos: float = 0.4,
    n_surf: int = 161,
    n_wake: int = 49,
    n_eta: int = 65,
    radius: float = 20.0,
    x_out: float = 21.0,
    first_cell: float = 1e-3,
    smooth_iters: int = 100,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Body-fitted C-grid around a NACA 4-digit airfoil.

    xi runs along the lower wake cut inward to the trailing edge, round the
    airfoil (lower surface to the nose, then the upper surface back to the
    trailing edge), then out along the upper wake cut. eta runs from the body
    outward to the far field. This is the standard airfoil topology: the sharp
    trailing edge sits at a grid corner and the wake cut carries the Kutta
    condition without any explicit enforcement.

    Returns (X, Y, n_wall, n_wake) with X, Y shaped (n_xi, n_eta) where
    n_xi = 2 * n_wake + n_wall, and n_wall points lie on the airfoil.
    """
    # naca4_coords(n_points=P) returns 2P-1 points: index 0 is the trailing
    # edge, 0..P-1 the upper surface TE -> LE, P-1..2P-2 the lower surface
    # LE -> TE, closing back on the trailing edge.
    n_half = (n_surf + 1) // 2
    surf = naca4_coords(thickness, camber, camber_pos, n_points=n_half)
    upper = surf[:n_half]  # TE -> LE
    lower = surf[n_half - 1 : 2 * n_half - 1]  # LE -> TE
    wall = np.vstack([lower[::-1], upper[::-1][1:]])
    n_wall = wall.shape[0]  # = n_surf for odd n_surf

    y_te = wall[0, 1]
    s = np.linspace(0.0, 1.0, n_wake + 1)[1:]
    wake = np.column_stack([1.0 + (x_out - 1.0) * s**2, np.full(n_wake, y_te)])
    inner = np.vstack([wake[::-1], wall, wake])
    n_xi = inner.shape[0]

    # Outer boundary must be traversed in the same sense as the inner one
    # (lower outflow, round the front, upper outflow) or the lines tangle.
    xc = 0.25
    n_arc = n_xi - 2 * n_wake
    bot = np.column_stack(
        [np.linspace(x_out, xc, n_wake + 1)[:-1], np.full(n_wake, -radius)]
    )
    th = np.linspace(-0.5 * np.pi, -1.5 * np.pi, n_arc)
    arc = np.column_stack([xc + radius * np.cos(th), radius * np.sin(th)])
    top = np.column_stack(
        [np.linspace(xc, x_out, n_wake + 1)[1:], np.full(n_wake, radius)]
    )
    outer = np.vstack([bot, arc, top])

    length = np.hypot(outer[:, 0] - inner[:, 0], outer[:, 1] - inner[:, 1])
    eta = _stretch_profile(first_cell, n_eta, length)

    X = inner[:, 0:1] + (outer[:, 0:1] - inner[:, 0:1]) * eta
    Y = inner[:, 1:2] + (outer[:, 1:2] - inner[:, 1:2]) * eta

    if smooth_iters:
        # Smoothing fixes the line SHAPES but relaxes spacing back towards
        # uniform, so the clustering is restored afterwards.
        X, Y = _winslow(X, Y, smooth_iters)
        X, Y = _recluster(X, Y, eta)
    return X, Y, n_wall, n_wake


def quad_areas(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Signed areas of the structured quad cells of a node grid (shoelace)."""
    x1, y1 = X[:-1, :-1], Y[:-1, :-1]
    x2, y2 = X[1:, :-1], Y[1:, :-1]
    x3, y3 = X[1:, 1:], Y[1:, 1:]
    x4, y4 = X[:-1, 1:], Y[:-1, 1:]
    return 0.5 * (
        (x1 * y2 - x2 * y1)
        + (x2 * y3 - x3 * y2)
        + (x3 * y4 - x4 * y3)
        + (x4 * y1 - x1 * y4)
    )
