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
