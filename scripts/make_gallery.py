"""
Generate the website gallery — every figure is PDEForge output, seeded and
regenerable. Style: dark surface, fields edge-to-edge, no chart junk;
diverging maps for signed fields, perceptually-uniform sequential for
magnitudes.

    python scripts/make_gallery.py --only spectral     # numpy/jax models
    python scripts/make_gallery.py --only fem          # needs dolfinx
    python scripts/make_gallery.py --only volume       # needs pyvista
    python scripts/make_gallery.py --only motion       # motion loops (mp4; needs ffmpeg)
    python scripts/make_gallery.py --only fem-motion   # LES cylinder loop (slow)
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
SURFACE = "#2E3440"
SURFACE_RGB = (0x2E, 0x34, 0x40)
INK = "#D8DEE9"
DPI = 200

# Diverging map for signed fields on the polar-night surface: the midpoint
# IS the page background, so zero melts into the page and the two signs
# glow out of it — frost/ice for negative, aurora fire for positive.
NORD_GLOW = LinearSegmentedColormap.from_list(
    "nord_glow",
    [
        (0.00, "#B8E2EC"),
        (0.10, "#88C0D0"),
        (0.26, "#81A1C1"),
        (0.40, "#5E81AC"),
        (0.50, "#2E3440"),
        (0.60, "#BF616A"),
        (0.78, "#D08770"),
        (1.00, "#EBCB8B"),
    ],
)

# Sequential frost ramp for magnitudes that should read as water/ice.
NORD_ICE = LinearSegmentedColormap.from_list(
    "nord_ice",
    ["#2E3440", "#3B4252", "#4C566A", "#5E81AC", "#81A1C1", "#88C0D0", "#C8E4EA", "#ECEFF4"],
)


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

    # the same six textures as individual tiles for the gallery grid
    for i, (f, cmap, sym) in enumerate(tiles, start=1):
        fig, ax = _field_fig(3.2)
        kw = _sym(f) if sym else {}
        ax.imshow(f, cmap=cmap, origin="lower", interpolation="bilinear", **kw)
        _save(fig, f"banner_tile_{i}.png")


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


# ---------- motion: animated gallery ----------


def _frame_rgb(field, cmap, vmin, vmax, shade=None, mask=None):
    """Colormap a 2D field straight to uint8 RGB — no figure machinery."""
    t = np.clip((field - vmin) / (vmax - vmin), 0.0, 1.0)
    rgb = cmap(t)[..., :3]
    if shade is not None:
        rgb = np.clip(rgb * shade[..., None], 0.0, 1.0)
    if mask is not None:
        rgb[mask] = np.array(SURFACE_RGB) / 255.0
    return (rgb * 255).astype(np.uint8)


def _hillshade(h, strength=1.2):
    """Upper-left lighting from the field's own gradients (liquid look)."""
    gy, gx = np.gradient(h)
    scale = max(np.std(gx), np.std(gy), 1e-12)
    gx, gy = gx / (4 * scale), gy / (4 * scale)
    return 0.72 + strength * 0.30 * np.clip(-0.707 * gx + 0.707 * gy, -1, 1)


def _save_mp4(frames, name, fps=14, upscale=1):
    """Write a looping mp4 (h264, crf 27, yuv420p, faststart; needs ffmpeg
    on PATH). Motion ships as mp4, never GIF; the caller trims transients
    so the loop opens on a developed frame."""
    import subprocess

    from PIL import Image

    imgs = []
    for f in frames:
        im = Image.fromarray(f[::-1])
        if upscale != 1:
            im = im.resize((im.width * upscale, im.height * upscale), Image.BILINEAR)
        imgs.append(np.asarray(im))
    h, w = imgs[0].shape[:2]
    w2, h2 = w - w % 2, h - h % 2  # h264 wants even dimensions
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.mp4"
    proc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w2}x{h2}", "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-crf", "27", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(path),
        ],
        stdin=subprocess.PIPE,
    )
    for f in imgs:
        proc.stdin.write(np.ascontiguousarray(f[:h2, :w2]).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {name}")
    print(f"  wrote {path} ({path.stat().st_size / 1e3:.0f} kB, {len(frames)} frames)")


def motion_kolmogorov():
    """Forced 2D turbulence in motion; cores saturated into ice/amber."""
    import pdeforge

    d = pdeforge.generate_dataset(
        "kolmogorov_flow_2d",
        n_samples=1,
        resolution={"x": 256, "y": 256},
        params={"viscosity": 1 / 70, "time_horizon": 30.0, "_dt": 0.004, "_n_time_steps": 121},
        seed=7,
        verbose=False,
        backend="jax",
        outputs="trajectory",
    )
    w = np.asarray(d.outputs[0])[48:]  # drop spin-up
    m = 0.6 * np.abs(w).max()
    frames = [_frame_rgb(f, NORD_GLOW, -m, m) for f in w]
    _save_mp4(frames, "kolmogorov_motion", fps=14)


def motion_gray_scott():
    """Mitosis: nine seeds divide and grow into a labyrinth."""
    import pdeforge

    d = pdeforge.generate_dataset(
        "gray_scott_2d",
        n_samples=1,
        resolution={"x": 256, "y": 256},
        params={"feed": 0.054, "kill": 0.063, "time_end": 9000.0, "_n_time_steps": 136},
        ic_params={"n_patches": 9, "noise": 0.02},
        seed=5,
        verbose=False,
        backend="jax",
        outputs="trajectory",
    )
    v = np.asarray(d.outputs[0])[:, 1]
    vmin, vmax = float(v.min()), float(v.max())
    frames = [_frame_rgb(f, plt.get_cmap("magma"), vmin, vmax) for f in v]
    _save_mp4(frames, "gray_scott_motion", fps=14)


def motion_fhn_spiral():
    """Excitable medium at beta = 0.5: broken wavefronts curl, collide
    and re-seed instead of dying out (at the default beta = 0.7 the
    medium is sub-excitable and free wave ends retract)."""
    from pdeforge import get_model

    beta, gamma = 0.5, 0.8
    u_rest = -0.76  # most negative root of gamma u^3 + (1-gamma) u + beta
    v_rest = (u_rest + beta) / gamma
    m = get_model("fitzhugh_nagumo_2d")(
        resolution={"x": 256, "y": 256},
        domain={"x": (0.0, 120.0), "y": (0.0, 120.0)},
        epsilon=0.02,
        beta=beta,
        time_end=400.0,
        _n_time_steps=161,
        _dt=0.05,
    )
    x, y = m.grids["x"], m.grids["y"]
    X, Y = np.meshgrid(x, y)
    u0 = np.full_like(X, u_rest)
    v0 = np.full_like(X, v_rest)
    u0[(np.abs(Y - 60.0) < 8.0) & (X < 66.0)] = 1.0  # broken stripe
    v0[(Y < 52.0) & (X < 66.0)] = v_rest + 0.45  # refractory block: tips curl
    traj = m.solve(u0, ic_v=v0, return_full=True)
    u = traj[16:, :, :, 0]  # drop the initial stripe flash
    frames = [_frame_rgb(f, plt.get_cmap("inferno"), -1.0, 1.1) for f in u]
    _save_mp4(frames, "fhn_spiral_motion", fps=14)


def motion_shallow_water():
    """Gravity waves criss-crossing a periodic pool, lit like water."""
    import pdeforge

    d = pdeforge.generate_dataset(
        "shallow_water_2d",
        n_samples=1,
        resolution={"x": 256, "y": 256},
        params={"time_end": 2.2, "_n_time_steps": 181},
        ic_params={"amplitude": 0.08, "cutoff": 3},
        seed=21,
        verbose=False,
        outputs="trajectory",
    )
    h = np.asarray(d.outputs[0])[60:, 0]  # developed criss-cross phase only
    vmin, vmax = np.percentile(h, 1.0), np.percentile(h, 99.5)
    frames = [
        _frame_rgb(f, NORD_ICE, vmin, vmax, shade=_hillshade(f, strength=1.5))
        for f in h
    ]
    _save_mp4(frames, "shallow_water_motion", fps=14)


def motion_cylinder_turbulent(traj=None):
    """The hero: LES vortex street at Re 2000, vorticity in nord glow."""
    from pdeforge import get_model

    m = get_model("cylinder_flow_2d_turbulent")(
        resolution={"x": 512, "y": 96},
        viscosity=5e-5,  # Re = 2000
        time_end=12.0,
        n_time_steps=241,
        _mesh_resolution=0.015,
    )
    cx, cy = 0.325, 0.2
    if traj is None:
        traj = m.solve(cx=cx, cy=cy)
    u = np.transpose(traj[..., 0], (0, 2, 1))  # -> (t, ny, nx)
    v = np.transpose(traj[..., 1], (0, 2, 1))
    x, y = m.grids["x"], m.grids["y"]
    dx, dy = x[1] - x[0], y[1] - y[0]
    w = np.gradient(v, dx, axis=2) - np.gradient(u, dy, axis=1)
    w = w[120:]  # developed wake only (t >= 6)
    # subtract the base channel-shear profile (measured upstream of the
    # cylinder) so the parabolic-inlet vorticity doesn't tint the whole
    # frame and the street glows out of the dark surface
    w_bg = w[:, :, x < 0.24].mean(axis=(0, 2))
    w = w - w_bg[None, :, None]
    X, Y = np.meshgrid(x, y)
    disk = (X - cx) ** 2 + (Y - cy) ** 2 <= (1.08 * m.r) ** 2
    m_abs = np.percentile(np.abs(w[:, :, x > cx + 2 * m.r]), 98)
    frames = [_frame_rgb(f, NORD_GLOW, -m_abs, m_abs, mask=disk) for f in w]
    _save_mp4(frames, "cylinder_turbulent_motion", fps=18, upscale=2)


SPECTRAL = [kolmogorov, ks_spacetime, gray_scott, heterogeneous_wave, banner]
FEM = [naca]
VOLUME = [volume_shots]
MOTION = [motion_kolmogorov, motion_gray_scott, motion_fhn_spiral, motion_shallow_water]
FEM_MOTION = [motion_cylinder_turbulent]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only",
        choices=["spectral", "fem", "volume", "motion", "fem-motion", "all"],
        default="all",
    )
    args = ap.parse_args()
    groups = {
        "spectral": SPECTRAL,
        "fem": FEM,
        "volume": VOLUME,
        "motion": MOTION,
        "fem-motion": FEM_MOTION,
        "all": SPECTRAL + VOLUME + MOTION + FEM + FEM_MOTION,
    }
    for fn in groups[args.only]:
        print(f"[{fn.__name__}]")
        fn()


if __name__ == "__main__":
    main()
