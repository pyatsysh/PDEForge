"""
One figure per model, for the per-model documentation pages.

Every figure is 2:1 and shows the operator the model defines: what goes in on
the left, what comes out on the right. Where the dynamics carry the interest
rather than the endpoints, the panel is a space-time diagram instead. Palette
and surface come from ``make_gallery`` so the two sets match.

    python scripts/make_model_figures.py --only spectral   # numpy/jax models
    python scripts/make_model_figures.py --only fem        # needs dolfinx
    python scripts/make_model_figures.py --only heat_1d    # a single model
    python scripts/make_model_figures.py --list

Figures land in ``docs/figures/model_<name>.png``, seeded and regenerable.
"""

import argparse
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_gallery import DPI, INK, NORD_GLOW, NORD_ICE, OUT, SURFACE  # noqa: E402

# 2:1 at 200 dpi: 1600 x 800, which is the widest a docs column ever renders.
FIGW, FIGH = 8.0, 4.0
GAP = 0.012  # gutter between panels, in figure fractions; reads as framing


def _sym(v):
    m = float(np.abs(v).max()) or 1.0
    return dict(vmin=-m, vmax=m)


def _canvas():
    fig = plt.figure(figsize=(FIGW, FIGH), facecolor=SURFACE)
    return fig


def _axes(fig, n):
    """n equal panels across the 2:1 canvas, separated by surface-coloured gutters."""
    axes = []
    w = (1.0 - (n - 1) * GAP) / n
    for i in range(n):
        ax = fig.add_axes([i * (w + GAP), 0.0, w, 1.0])
        ax.set_facecolor(SURFACE)
        ax.axis("off")
        axes.append(ax)
    return axes


def _fig_label(fig, text):
    """Caption centred on the whole canvas: a multi-panel montage's label is
    wider than any one panel, so anchoring it to a panel clips it."""
    t = fig.text(0.5, 0.022, text, ha="center", va="bottom",
                 color=INK, fontsize=8.5)
    t.set_path_effects(
        [patheffects.withStroke(linewidth=2.2, foreground=SURFACE, alpha=0.85)]
    )


def _label(ax, text):
    # A dark halo: several of these fields run pale edge to edge, and plain
    # snow-coloured text disappears into them.
    t = ax.text(
        0.5,
        0.022,
        text,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color=INK,
        fontsize=8.5,
    )
    t.set_path_effects(
        [patheffects.withStroke(linewidth=2.2, foreground=SURFACE, alpha=0.85)]
    )


def _save(fig, model):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"model_{model}.png"
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {path}")


def _show(ax, field, cmap, sym=False, sat=1.0, **kw):
    field = np.asarray(field)
    lim = _sym(field) if sym else {}
    if sym and sat != 1.0:
        lim = dict(vmin=lim["vmin"] * sat, vmax=lim["vmax"] * sat)
    ax.imshow(
        field,
        cmap=cmap,
        origin="lower",
        interpolation="bilinear",
        aspect="auto",
        **lim,
        **kw,
    )


def _gen(model, **kw):
    import pdeforge

    kw.setdefault("verbose", False)
    kw.setdefault("n_samples", 1)
    return pdeforge.generate_dataset(model, **kw)


# ---------- renderers ----------


def _pair(model, a, b, la, lb, cmap_a=NORD_ICE, cmap_b=NORD_GLOW,
          sym_a=False, sym_b=True, shared=False):
    """Two panels. ``shared`` when both panels show the SAME quantity at two
    times: one colormap and one colour scale across both, so the eye reads the
    change in the field rather than a change of scale."""
    fig = _canvas()
    ax1, ax2 = _axes(fig, 2)
    if shared:
        both = np.concatenate([np.asarray(a).ravel(), np.asarray(b).ravel()])
        lim = _sym(both) if sym_a else dict(vmin=both.min(), vmax=both.max())
        ax1.imshow(np.asarray(a), cmap=cmap_a, origin="lower",
                   interpolation="bilinear", aspect="auto", **lim)
        ax2.imshow(np.asarray(b), cmap=cmap_a, origin="lower",
                   interpolation="bilinear", aspect="auto", **lim)
    else:
        _show(ax1, a, cmap_a, sym=sym_a)
        _show(ax2, b, cmap_b, sym=sym_b)
    _label(ax1, la)
    _label(ax2, lb)
    _save(fig, model)


def _wide(model, field, cmap, sym=False, label=None, sat=1.0):
    fig = _canvas()
    (ax,) = _axes(fig, 1)
    _show(ax, field, cmap, sym=sym, sat=sat)
    if label:
        _label(ax, label)
    _save(fig, model)


def _spacetime(model, U, label, cmap=NORD_GLOW, sym=True):
    """U is (n_t, n_x); time runs left to right, space up the panel."""
    _wide(model, np.asarray(U).T, cmap, sym=sym, label=label)


def _curves(model, x, ins, outs, la, lb):
    """Two panels of 1D profiles: the input measure left, its images right."""
    fig = _canvas()
    ax1, ax2 = _axes(fig, 2)
    frost = ["#88C0D0", "#81A1C1", "#5E81AC", "#B48EAD", "#8FBCBB"]
    aurora = ["#EBCB8B", "#D08770", "#BF616A", "#A3BE8C", "#D8DEE9"]
    for ax, curves, colors in ((ax1, ins, frost), (ax2, outs, aurora)):
        for i, u in enumerate(curves):
            ax.plot(x, u, color=colors[i % len(colors)], lw=1.6, alpha=0.95)
        ax.set_facecolor(SURFACE)
        ax.margins(x=0.02, y=0.08)
    lo = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
    hi = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim(lo, hi)
    ax2.set_ylim(lo, hi)
    _label(ax1, la)
    _label(ax2, lb)
    _save(fig, model)


def _slices(model, vol, label, cmap=NORD_GLOW, sym=True, n=4):
    """A 3D volume as n evenly spaced slices across the last axis."""
    vol = np.asarray(vol)
    fig = _canvas()
    axes = _axes(fig, n)
    idx = np.linspace(0, vol.shape[-1] - 1, n).astype(int)
    lim = _sym(vol) if sym else {}
    for ax, k in zip(axes, idx):
        ax.imshow(
            vol[..., k],
            cmap=cmap,
            origin="lower",
            interpolation="bilinear",
            aspect="auto",
            **lim,
        )
    _fig_label(fig, label)
    _save(fig, model)


# ---------- spectral and finite-volume models ----------


def advection_1d():
    d = _gen("advection_1d", resolution={"x": 256},
             params={"speed": 1.0, "time_end": 12.0, "_n_time_steps": 240},
             ic_generator="depression_box", seed=3, outputs="trajectory")
    _spacetime("advection_1d", d.outputs[0],
               "u(x, t): a sharp-edged profile translates, shape intact")


def allen_cahn_1d():
    # Small epsilon so many interfaces form (width ~ sqrt(eps)); a rough IC so
    # there are plenty of zero crossings to seed them; a horizon short enough
    # that coarsening is still in progress rather than finished.
    d = _gen("allen_cahn_1d", resolution={"x": 1024},
             params={"epsilon": 2e-4, "time_end": 12.0, "_n_time_steps": 300},
             ic_params={"n_modes": 24, "decay": 0.5, "amplitude": 0.3},
             seed=5, outputs="trajectory")
    _spacetime("allen_cahn_1d", d.outputs[0],
               "u(x, t): interfaces form fast, then drift together and annihilate in pairs")


def allen_cahn_3d():
    # Stop while coarsening is still in progress: by t = 2 at eps = 0.03 the
    # box has collapsed into a single phase and every slice is featureless.
    d = _gen("allen_cahn_3d", resolution={"x": 64, "y": 64, "z": 64},
             params={"epsilon": 0.004, "time_end": 0.6}, seed=7,
             ic_params={"n_modes": 8, "decay": 0.6, "amplitude": 0.4})
    _slices("allen_cahn_3d", d.outputs[0],
            "u at four z-slices, partway through coarsening")


def eggshell_droplets_3d():
    # The subject of this model is the dichotomy, not one trajectory, so the
    # figure runs both regimes and shows each one's start and end. The slice
    # is the plane y = centre, the only plane containing both droplet centres.
    from pdeforge import get_model

    def run(**kw):
        m = get_model("eggshell_droplets_3d")(
            resolution={"x": 96, "y": 96, "z": 96}, epsilon=0.0135, **kw
        )
        ic = m.generate_ic(seed=0)
        uT = m.solve(ic)
        mid = ic[0].shape[1] // 2
        return ic[0][:, mid, :], uT[:, mid, :], m.last_diagnostics["regime"]

    # Equal partners already touching: no ripening drive, so they can only merge.
    c0, c1, c_regime = run(size_asymmetry=0.0, droplet_gap=0.15, time_end=0.05)
    # Unequal partners held apart: Gibbs-Thomson dissolves the smaller one.
    # Stopped shortly after the event (t ~ 0.03): run much further and the shell
    # matrix slowly eats the survivor too, which is a separate, slower story.
    r0, r1, r_regime = run(size_asymmetry=0.30, droplet_gap=1.0, time_end=0.06)

    fig = _canvas()
    axes = _axes(fig, 4)
    panels = [c0, c1, r0, r1]
    both = np.concatenate([p.ravel() for p in panels])
    lim = _sym(both)
    for ax, field in zip(axes, panels):
        ax.imshow(field, cmap=NORD_GLOW, origin="lower",
                  interpolation="bilinear", aspect="auto", **lim)
    _label(axes[0], "equal pair, touching")
    _label(axes[1], f"-> {c_regime}")
    _label(axes[2], "unequal pair, apart")
    _label(axes[3], f"-> {r_regime}")
    _save(fig, "eggshell_droplets_3d")


def burgers_2d():
    d = _gen("burgers_2d", resolution={"x": 256, "y": 256},
             params={"viscosity": 0.002, "time_horizon": 1.0}, seed=11)
    w0 = np.hypot(d.inputs[0][0], d.inputs[0][1])
    wT = np.hypot(d.outputs[0][0], d.outputs[0][1])
    _pair("burgers_2d", w0, wT, "speed at t = 0", "speed at t = T: fronts have formed",
          cmap_a=NORD_ICE, shared=True)


def darcy_fno_2d():
    d = _gen("darcy_fno_2d", resolution={"x": 421, "y": 421}, seed=0)
    _pair("darcy_fno_2d", d.inputs[0], d.outputs[0],
          "coefficient a(x, y), log-normal", "pressure u(x, y)",
          cmap_a=NORD_ICE, cmap_b=NORD_ICE, sym_b=False)


def fitzhugh_nagumo_1d():
    from pdeforge import get_model

    # beta = 0.5 is the fully excitable regime. The medium must START at rest:
    # leaving ic_v to default puts EVERY point on the v-nullcline, including the
    # stimulus, which then cannot fire.
    beta, gamma = 0.5, 0.8
    u_rest = -0.76                        # most negative root of gamma u^3 + (1-gamma) u + beta
    v_rest = (u_rest + beta) / gamma
    m = get_model("fitzhugh_nagumo_1d")(
        resolution={"x": 512}, domain={"x": (0.0, 200.0)},
        epsilon=0.02, beta=beta, time_end=250.0, _n_time_steps=320,
    )
    x = m.grids["x"]
    u0 = np.full_like(x, u_rest)
    v0 = np.full_like(x, v_rest)
    u0[(x > 90.0) & (x < 110.0)] = 1.0    # one supra-threshold stimulus, 20 units wide
    traj = np.asarray(m.solve(u0, ic_v=v0, return_full=True))
    U = traj[..., 0] if traj.ndim == 3 else traj
    _spacetime("fitzhugh_nagumo_1d", U,
               "u(x, t): a single stimulus launches two counter-propagating pulses")


def fitzhugh_nagumo_2d():
    # The excitable recipe (beta = 0.5), same as the gallery's motion loop: a
    # broken stripe with a refractory block behind one end, started from rest.
    # The default measure gives a single bump that simply decays.
    from pdeforge import get_model

    beta, gamma = 0.5, 0.8
    u_rest = -0.76
    v_rest = (u_rest + beta) / gamma
    m = get_model("fitzhugh_nagumo_2d")(
        resolution={"x": 256, "y": 256},
        domain={"x": (0.0, 120.0), "y": (0.0, 120.0)},
        epsilon=0.02, beta=beta, time_end=400.0, _n_time_steps=81, _dt=0.05,
    )
    X, Y = np.meshgrid(m.grids["x"], m.grids["y"])
    u0 = np.full_like(X, u_rest)
    v0 = np.full_like(X, v_rest)
    u0[(np.abs(Y - 60.0) < 8.0) & (X < 66.0)] = 1.0    # the broken stripe
    v0[(Y < 52.0) & (X < 66.0)] = v_rest + 0.45        # refractory block: tips curl
    traj = np.asarray(m.solve(u0, ic_v=v0, return_full=True))
    u = traj[..., 0] if traj.ndim == 4 else traj
    _pair("fitzhugh_nagumo_2d", u[2], u[-1],
          "u shortly after the broken stripe is applied",
          "u at t = T: the free ends have curled into spirals",
          cmap_a=NORD_GLOW, sym_a=True, shared=True)


def heat_1d():
    d = _gen("heat_1d", resolution={"x": 256}, n_samples=4,
             params={"diffusivity": 0.02, "time_end": 1.0}, seed=1)
    x = np.linspace(0, 2 * np.pi, d.inputs.shape[-1], endpoint=False)
    _curves("heat_1d", x, d.inputs, d.outputs,
            "four draws from the input measure", "the same four at t = T")


def heat_2d():
    d = _gen("heat_2d", resolution={"x": 256, "y": 256},
             params={"diffusivity": 0.005, "time_end": 1.0}, seed=6)
    _pair("heat_2d", d.inputs[0], d.outputs[0], "u at t = 0", "u at t = T",
          cmap_a=NORD_GLOW, sym_a=True, shared=True)


def heat_3d():
    d = _gen("heat_3d", resolution={"x": 48, "y": 48, "z": 48},
             params={"diffusivity": 0.005, "time_end": 0.5}, seed=8)
    _slices("heat_3d", d.outputs[0], "u at four z-slices, t = T")


def helmholtz_2d():
    d = _gen("helmholtz_2d", resolution={"x": 256, "y": 256},
             params={"wavenumber": 30.0, "damping": 1.0}, seed=9)
    _pair("helmholtz_2d", d.inputs[0], d.outputs[0],
          "source f(x, y)", "field u(x, y) at k = 30")


def kdv_1d():
    d = _gen("kdv_1d", resolution={"x": 512},
             params={"time_end": 4.0, "_n_time_steps": 300}, seed=0, outputs="trajectory")
    _spacetime("kdv_1d", d.outputs[0],
               "u(x, t): solitons overtake and emerge unchanged")


def lotka_volterra_2d():
    d = _gen("lotka_volterra_2d", resolution={"x": 256, "y": 256},
             params={"time_end": 12.0}, seed=13)
    _pair("lotka_volterra_2d", d.outputs[0][0], d.outputs[0][1],
          "prey u at t = T", "predator v at t = T",
          cmap_a=NORD_ICE, cmap_b="magma", sym_b=False)


def ns_vorticity_2d():
    # nu = 1e-4 keeps the cascade alive over the horizon; at 1e-3 viscosity
    # has flattened the field by T and the figure shows smoothing, not rollup.
    d = _gen("ns_vorticity_2d", resolution={"x": 256, "y": 256},
             params={"viscosity": 1e-4, "time_horizon": 8.0}, seed=15, backend="jax")
    _pair("ns_vorticity_2d", d.inputs[0], d.outputs[0],
          "vorticity at t = 0", "vorticity at t = T",
          cmap_a=NORD_GLOW, sym_a=True, shared=True)


def schrodinger_1d():
    d = _gen("schrodinger_1d", resolution={"x": 512},
             params={"g": -1.0, "time_end": 4.0, "_n_time_steps": 300},
             seed=17, outputs="trajectory")
    U = np.asarray(d.outputs[0])
    dens = U[:, 0] ** 2 + U[:, 1] ** 2 if U.ndim == 3 else U**2
    _spacetime("schrodinger_1d", dens,
               "|psi|^2 (x, t): dispersion spreading the field, focusing pulling it back",
               cmap=NORD_ICE, sym=False)


def shallow_water_2d():
    d = _gen("shallow_water_2d", resolution={"x": 256, "y": 256},
             params={"time_end": 0.35}, ic_params={"amplitude": 0.08, "cutoff": 3},
             seed=19)
    # h sits at the mean depth (1.0) plus a ~0.05 wave; plotting h itself puts
    # the whole signal in one sliver of the colour range. The anomaly is the
    # physical quantity anyway.
    h0 = np.asarray(d.inputs[0][0])
    hT = np.asarray(d.outputs[0][0])
    ref = float(h0.mean())
    _pair("shallow_water_2d", h0 - ref, hT - ref,
          "surface anomaly h - H at t = 0", "h - H at t = T: the waves have spread",
          cmap_a=NORD_GLOW, sym_a=True, shared=True)


def wave_1d():
    from pdeforge import get_model

    m = get_model("wave_1d")(
        resolution={"x": 512}, wave_speed=1.0, time_end=12.0, _n_time_steps=300,
    )
    x = m.grids["x"]
    L = x[-1] - x[0] + (x[1] - x[0])
    u0 = np.exp(-(((x - 0.5 * L) / (0.03 * L)) ** 2))  # one localised bump
    traj = np.asarray(m.solve(u0, return_full=True))
    _spacetime("wave_1d", traj,
               "u(x, t): one pulse splits into left- and right-going characteristics")


def wave_2d():
    d = _gen("wave_2d", resolution={"x": 256, "y": 256},
             params={"wave_speed": 1.0, "time_end": 1.2}, seed=23)
    _pair("wave_2d", d.inputs[0], d.outputs[0], "u at t = 0", "u at t = T",
          cmap_a=NORD_GLOW, sym_a=True, shared=True)


def stochastic_burgers_1d():
    # Low viscosity so a front actually forms, and enough noise that the
    # members separate: at nu = 0.05 the solution has already diffused into a
    # smooth wave and the realisations sit on top of each other.
    d = _gen("stochastic_burgers_1d", resolution={"x": 512},
             params={"viscosity": 0.002, "noise_intensity": 0.6,
                     "n_realizations": 12, "time_horizon": 1.5},
             seed=25)
    real = np.asarray(d.outputs[0])
    x = np.linspace(0, 2 * np.pi, real.shape[-1], endpoint=False)
    _curves("stochastic_burgers_1d", x, [np.asarray(d.inputs[0])], real[:5],
            "one initial condition", "five realisations of the same solve")


def stochastic_heat_1d():
    d = _gen("stochastic_heat_1d", resolution={"x": 256},
             params={"noise_intensity": 0.3, "n_realizations": 20, "time_end": 1.0},
             seed=27)
    out = np.asarray(d.outputs[0])
    x = np.linspace(0, 2 * np.pi, out.shape[-1], endpoint=False)
    curves = out[:5] if out.ndim == 2 else [out]
    _curves("stochastic_heat_1d", x, [np.asarray(d.inputs[0])], curves,
            "one initial condition", "the ensemble at t = T")


def stochastic_heat_2d():
    d = _gen("stochastic_heat_2d", resolution={"x": 128, "y": 128},
             params={"noise_intensity": 0.3, "n_realizations": 20, "time_end": 1.0},
             seed=29)
    out = np.asarray(d.outputs[0])
    mean = out.mean(0) if out.ndim == 3 else out
    std = out.std(0) if out.ndim == 3 else np.zeros_like(out)
    _pair("stochastic_heat_2d", mean, std,
          "ensemble mean at t = T", "ensemble standard deviation",
          cmap_a=NORD_GLOW, cmap_b=NORD_ICE, sym_a=True, sym_b=False)


def stochastic_allen_cahn_2d():
    d = _gen("stochastic_allen_cahn_2d", resolution={"x": 128, "y": 128},
             params={"noise_intensity": 0.12, "n_realizations": 8, "time_end": 3.0},
             seed=31)
    out = np.asarray(d.outputs[0])
    _pair("stochastic_allen_cahn_2d", out[0], out[1],
          "realisation 1", "realisation 2, same initial condition")


def airfoil_euler_2d():
    # Degenerate ranges pin the sample to the published benchmark condition:
    # NACA 0012, M = 0.8, AoA 1.25 deg.
    d = _gen("airfoil_euler_2d", resolution={"xi": 256, "eta": 64},
             params={"mach_range": (0.8, 0.8), "aoa_range": (1.25, 1.25),
                     "thickness_range": (0.12, 0.12), "camber_range": (0.0, 0.0),
                     "camber_pos_range": (0.4, 0.4)},
             seed=0)
    sol = np.asarray(d.outputs[0])
    rho, u, v, p = (sol[..., i] for i in range(4))
    gamma = 1.4
    mach = np.hypot(u, v) / np.sqrt(gamma * np.maximum(p, 1e-9) / np.maximum(rho, 1e-9))
    xy = np.asarray(d.inputs[0])
    fig = _canvas()
    (ax,) = _axes(fig, 1)
    pc = ax.pcolormesh(xy[..., 0], xy[..., 1], mach, cmap="magma", shading="gouraud")
    pc.set_clim(0.0, 1.4)
    ax.contour(xy[..., 0], xy[..., 1], mach, levels=[1.0], colors=["#ECEFF4"], linewidths=0.9)
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(-0.5, 0.5)
    ax.set_aspect("equal")
    _label(ax, "Mach number, NACA 0012 at M = 0.8, AoA 1.25 deg; the white line is M = 1")
    _save(fig, "airfoil_euler_2d")


# ---------- FEniCSx models ----------


# These models put the component axis LAST, and the FEniCSx ones store space as
# (nx, ny) rather than (ny, nx), so their fields need a transpose before imshow.
# Verified empirically on an asymmetric 48 x 24 grid.


def stokes_2d():
    d = _gen("stokes_2d", resolution={"x": 128, "y": 128},
             params={"viscosity": 1.0, "force_complexity": 6}, seed=33)
    f = np.hypot(d.inputs[0][..., 0], d.inputs[0][..., 1])       # (ny, nx, 2)
    speed = np.hypot(d.outputs[0][..., 0], d.outputs[0][..., 1])  # (ny, nx, 3)
    _pair("stokes_2d", f, speed, "forcing magnitude |f|", "flow speed |u|",
          cmap_a=NORD_ICE, cmap_b=NORD_ICE, sym_b=False)


def elasticity_2d():
    d = _gen("elasticity_2d", resolution={"x": 128, "y": 128},
             params={"e_inclusion": 10.0, "traction_y": -1.0, "n_inclusions": 6}, seed=35)
    _pair("elasticity_2d", np.asarray(d.inputs[0]).T, np.asarray(d.outputs[0][..., 2]).T,
          "Young's modulus E(x, y)", "von Mises stress",
          cmap_a=NORD_ICE, cmap_b="magma", sym_b=False)


def rayleigh_benard_2d():
    d = _gen("rayleigh_benard_2d", resolution={"x": 96, "y": 96},
             params={"rayleigh": 5e4, "prandtl": 0.71, "time_end": 0.6}, seed=37)
    out = np.asarray(d.outputs[0])           # (nx, ny, 3) = u, v, T
    T = out[..., 2].T
    v = out[..., 1].T
    _pair("rayleigh_benard_2d", T, v,
          "temperature at Ra = 5e4", "vertical velocity: the rolls that carry the heat",
          cmap_a=NORD_ICE, cmap_b=NORD_GLOW, sym_b=True)


def porous_darcy_fem():
    d = _gen("porous_darcy_fem", resolution={"x": 128, "y": 128},
             params={"ch_time": 8.0, "permeability_contrast": 1e3}, seed=39)
    _pair("porous_darcy_fem", np.asarray(d.inputs[0]).T, np.asarray(d.outputs[0][..., 0]).T,
          "permeability from a Cahn-Hilliard morphology", "pressure under a unit drop",
          cmap_a=NORD_ICE, cmap_b=NORD_ICE, sym_b=False)


def cylinder_flow_2d():
    d = _gen("cylinder_flow_2d", resolution={"x": 256, "y": 128},
             params={"viscosity": 1e-3, "inlet_velocity": 0.3}, seed=41)
    out = np.asarray(d.outputs[0])           # (nx, ny, 3)
    speed = np.hypot(out[..., 0], out[..., 1]).T
    _wide("cylinder_flow_2d", speed, NORD_ICE,
          label="speed |u|: the steady wake behind the cylinder")


def cylinder_flow_2d_parameterized():
    d = _gen("cylinder_flow_2d_parameterized", n_samples=2,
             resolution={"x": 256, "y": 128}, params={"viscosity": 1e-3}, seed=43)
    def _speed(k):
        o = np.asarray(d.outputs[k])
        return np.hypot(o[..., 0], o[..., 1]).T
    cx = d.inputs[:, 1] if np.ndim(d.inputs) > 1 else [0, 0]
    _pair("cylinder_flow_2d_parameterized", _speed(0), _speed(1),
          f"sample 1: cylinder at x = {float(cx[0]):.2f}",
          f"sample 2: cylinder at x = {float(cx[1]):.2f}",
          cmap_a=NORD_ICE, cmap_b=NORD_ICE, sym_b=False)


SPECTRAL = [
    advection_1d, allen_cahn_1d, allen_cahn_3d, burgers_2d, darcy_fno_2d,
    eggshell_droplets_3d,
    fitzhugh_nagumo_1d, fitzhugh_nagumo_2d, heat_1d, heat_2d, heat_3d,
    helmholtz_2d, kdv_1d, lotka_volterra_2d, ns_vorticity_2d, schrodinger_1d,
    shallow_water_2d, wave_1d, wave_2d, stochastic_burgers_1d,
    stochastic_heat_1d, stochastic_heat_2d, stochastic_allen_cahn_2d,
    airfoil_euler_2d,
]

FEM = [
    stokes_2d, elasticity_2d, rayleigh_benard_2d, porous_darcy_fem,
    cylinder_flow_2d, cylinder_flow_2d_parameterized,
]

ALL = {f.__name__: f for f in SPECTRAL + FEM}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="spectral",
                    help="spectral | fem | all | <model name>")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for name in ALL:
            print(name)
        return

    if args.only == "spectral":
        jobs = SPECTRAL
    elif args.only == "fem":
        jobs = FEM
    elif args.only == "all":
        jobs = SPECTRAL + FEM
    elif args.only in ALL:
        jobs = [ALL[args.only]]
    else:
        raise SystemExit(f"unknown target {args.only!r}; try --list")

    failed = []
    for job in jobs:
        print(f"{job.__name__} ...")
        try:
            job()
        except Exception:
            traceback.print_exc()
            failed.append(job.__name__)

    print(f"\n{len(jobs) - len(failed)}/{len(jobs)} written")
    if failed:
        print("failed: " + ", ".join(failed))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
