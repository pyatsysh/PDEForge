"""
Generate the website gallery — every figure is PDEForge output, seeded and
regenerable. Style: dark surface, fields edge-to-edge, no chart junk;
diverging maps for signed fields, perceptually-uniform sequential for
magnitudes.

    python scripts/make_gallery.py --only spectral   # numpy/jax models
    python scripts/make_gallery.py --only fem        # needs dolfinx
    python scripts/make_gallery.py --only volume     # needs pyvista
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
SURFACE = "#0d1117"
INK = "#8b949e"
DPI = 200


def _field_fig(width=8.0, height=None, aspect=1.0):
    height = height or width * aspect
    fig = plt.figure(figsize=(width, height), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(SURFACE)
    ax.axis("off")
    return fig, ax


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {OUT / name}")


def _sym(v):
    m = np.abs(v).max()
    return dict(vmin=-m, vmax=m)


def kolmogorov():
    import pdeforge

    d = pdeforge.generate_dataset(
        "kolmogorov_flow_2d",
        n_samples=1,
        resolution={"x": 256, "y": 256},
        params={"viscosity": 1 / 70, "time_horizon": 30.0, "_dt": 0.004},
        seed=7,
        verbose=False,
        backend="jax",
    )
    w = np.asarray(d.outputs[0])
    fig, ax = _field_fig(8.0)
    ax.imshow(w, cmap="RdBu_r", origin="lower", interpolation="bilinear", **_sym(w))
    _save(fig, "kolmogorov_vorticity.png")


def ks_spacetime():
    import pdeforge

    d = pdeforge.generate_dataset(
        "ks_1d",
        n_samples=1,
        resolution={"x": 512},
        params={"time_horizon": 150.0, "_n_time_steps": 400},
        seed=3,
        verbose=False,
        outputs="trajectory",
    )
    U = np.asarray(d.outputs[0])  # (t, x)
    fig, ax = _field_fig(10.0, height=4.4)
    ax.imshow(
        U.T,
        cmap="RdBu_r",
        origin="lower",
        aspect="auto",
        interpolation="bilinear",
        **_sym(U),
    )
    _save(fig, "ks_spacetime.png")


def gray_scott():
    import pdeforge

    d = pdeforge.generate_dataset(
        "gray_scott_2d",
        n_samples=1,
        resolution={"x": 256, "y": 256},
        params={"feed": 0.054, "kill": 0.063, "time_end": 8000.0},
        seed=11,
        verbose=False,
        backend="jax",
    )
    V = np.asarray(d.outputs[0][1])
    fig, ax = _field_fig(8.0)
    ax.imshow(V, cmap="magma", origin="lower", interpolation="bilinear")
    _save(fig, "gray_scott.png")


def heterogeneous_wave():
    import pdeforge

    d = pdeforge.generate_dataset(
        "heterogeneous_wave_2d",
        n_samples=1,
        resolution={"x": 384, "y": 384},
        params={"time_end": 0.35, "c_min": 0.6, "c_max": 1.6},
        seed=5,
        verbose=False,
    )
    u = np.asarray(d.outputs[0])
    fig, ax = _field_fig(8.0)
    ax.imshow(u, cmap="RdBu_r", origin="lower", interpolation="bilinear", **_sym(u))
    _save(fig, "wave_random_medium.png")


def banner():
    """Six-tile strip for the README top: pure texture, no text."""
    import pdeforge

    tiles = []
    # 1 kolmogorov filaments
    d = pdeforge.generate_dataset(
        "kolmogorov_flow_2d",
        n_samples=1,
        resolution={"x": 192, "y": 192},
        params={"viscosity": 1 / 70, "time_horizon": 25.0, "_dt": 0.005},
        seed=17,
        verbose=False,
        backend="jax",
    )
    tiles.append((np.asarray(d.outputs[0]), "RdBu_r", True))
    # 2 gray-scott labyrinth, seeded everywhere so it fills the tile
    d = pdeforge.generate_dataset(
        "gray_scott_2d",
        n_samples=1,
        resolution={"x": 192, "y": 192},
        params={"feed": 0.0367, "kill": 0.0649, "time_end": 3500.0},
        ic_params={"n_patches": 40, "noise": 0.02},
        seed=2,
        verbose=False,
        backend="jax",
    )
    tiles.append((np.asarray(d.outputs[0][1]), "magma", False))
    # 3 kuramoto-sivashinsky spacetime
    d = pdeforge.generate_dataset(
        "ks_1d",
        n_samples=1,
        resolution={"x": 192},
        params={"time_horizon": 120.0, "_n_time_steps": 192},
        seed=3,
        verbose=False,
        outputs="trajectory",
    )
    tiles.append((np.asarray(d.outputs[0]).T, "RdBu_r", True))
    # 4 spinodal maze (early-time cahn-hilliard)
    d = pdeforge.generate_dataset(
        "cahn_hilliard",
        n_samples=1,
        resolution={"x": 192, "y": 192},
        params={"time_end": 2.0},
        seed=29,
        verbose=False,
    )
    tiles.append((np.asarray(d.outputs[0]), "RdBu_r", True))
    # 5 wavefronts refracting through a random medium
    d = pdeforge.generate_dataset(
        "heterogeneous_wave_2d",
        n_samples=1,
        resolution={"x": 192, "y": 192},
        params={"time_end": 0.28, "c_min": 0.5, "c_max": 1.8, "pulse_width": 0.03},
        ic_params={"cutoff": 3},
        seed=31,
        verbose=False,
    )
    w = np.asarray(d.outputs[0])
    m0 = 0.35 * np.abs(w).max()  # saturate the fronts
    tiles.append((np.clip(w, -m0, m0), "RdBu_r", True))
    # 6 two-phase darcy permeability (the canonical input measure)
    from pdeforge import get_model

    m = get_model("darcy_fno_2d")(
        resolution={"x": 193, "y": 193}, coeff="piececonst", tau=9.0
    )
    tiles.append((m.generate_ic(seed=41), "magma", False))

    fig = plt.figure(figsize=(14.4, 2.4), facecolor=SURFACE)
    for i, (f, cmap, sym) in enumerate(tiles):
        ax = fig.add_axes([i / 6 + 0.0015, 0.02, 1 / 6 - 0.003, 0.96])
        ax.axis("off")
        kw = _sym(f) if sym else {}
        ax.imshow(f, cmap=cmap, origin="lower", interpolation="bilinear", **kw)
    _save(fig, "banner.png")


def naca():
    """The product shot: needs dolfinx."""
    import pdeforge
    from pdeforge import get_model

    m = get_model("naca_flow_2d")(
        resolution={"x": 384, "y": 192}, _mesh_resolution=0.07
    )
    thickness, camber, camber_pos, aoa = 0.12, 0.04, 0.4, 6.0
    sol = m.solve(thickness=thickness, camber=camber, camber_pos=camber_pos, aoa=aoa)
    forces = m._last_forces
    sdf = m.sdf_input(thickness, camber, camber_pos, aoa)
    u, v = sol[:, :, 0].T, sol[:, :, 1].T  # -> (ny, nx) for plotting
    speed = np.sqrt(u**2 + v**2)
    mask = (sdf < 0).T

    x = m.grids["x"]
    y = m.grids["y"]
    X, Y = np.meshgrid(x, y)

    fig, ax = _field_fig(10.0, height=5.0)
    sp = np.ma.array(speed, mask=mask)
    ax.imshow(
        sp,
        cmap="inferno",
        origin="lower",
        extent=[x[0], x[-1], y[0], y[-1]],
        interpolation="bilinear",
    )
    ax.streamplot(
        X,
        Y,
        np.where(mask, np.nan, u),
        np.where(mask, np.nan, v),
        color="#c9d1d9",
        linewidth=0.6,
        density=1.4,
        arrowsize=0.7,
    )
    ax.contourf(X, Y, mask.astype(float), levels=[0.5, 1.5], colors=[SURFACE])
    ax.set_xlim(-0.6, 2.4)
    ax.set_ylim(-0.75, 0.75)
    ax.text(
        0.985,
        0.03,
        f"NACA 4412 · AoA {aoa:.0f}°· Cl {forces['Cl']:.2f} · Cd {forces['Cd']:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=INK,
        fontsize=9,
    )
    _save(fig, "naca_flow.png")


def volume_shots():
    """Restyled 3D renders: needs pyvista."""
    import pyvista as pv

    import pdeforge
    from pdeforge.visualization import dataset_to_imagedata

    # canonical Darcy 3D, styled
    d = pdeforge.generate_dataset(
        "darcy_fno_3d",
        n_samples=1,
        resolution={"x": 49, "y": 49, "z": 49},
        seed=1,
        verbose=False,
    )
    img = dataset_to_imagedata(d)
    name = img.point_data.keys()[0]
    lo, hi = img.get_data_range(name)
    p = pv.Plotter(off_screen=True, window_size=(1400, 1400))
    p.set_background(SURFACE)
    p.add_mesh(
        img.contour(isosurfaces=np.linspace(lo, hi, 9)[1:-1], scalars=name),
        cmap="magma",
        opacity=0.55,
        smooth_shading=True,
        show_scalar_bar=False,
    )
    p.camera_position = "iso"
    p.camera.zoom(1.25)
    p.screenshot(str(OUT / "darcy3d_hero.png"))
    print(f"  wrote {OUT / 'darcy3d_hero.png'}")

    # 3D spinodal decomposition: the u = 0 interface
    d = pdeforge.generate_dataset(
        "cahn_hilliard",
        n_samples=1,
        resolution={"x": 48, "y": 48, "z": 48},
        params={"time_end": 30.0},
        seed=4,
        verbose=False,
    )
    img = dataset_to_imagedata(d)
    name = img.point_data.keys()[0]
    p = pv.Plotter(off_screen=True, window_size=(1400, 1400))
    p.set_background(SURFACE)
    p.add_mesh(
        img.contour(isosurfaces=[0.0], scalars=name),
        color="#e3963e",
        smooth_shading=True,
        specular=0.4,
        show_scalar_bar=False,
    )
    p.camera_position = "iso"
    p.camera.zoom(1.2)
    p.screenshot(str(OUT / "spinodal3d.png"))
    print(f"  wrote {OUT / 'spinodal3d.png'}")


SPECTRAL = [kolmogorov, ks_spacetime, gray_scott, heterogeneous_wave, banner]
FEM = [naca]
VOLUME = [volume_shots]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only", choices=["spectral", "fem", "volume", "all"], default="all"
    )
    args = ap.parse_args()
    groups = {
        "spectral": SPECTRAL,
        "fem": FEM,
        "volume": VOLUME,
        "all": SPECTRAL + VOLUME + FEM,
    }
    for fn in groups[args.only]:
        print(f"[{fn.__name__}]")
        fn()


if __name__ == "__main__":
    main()
