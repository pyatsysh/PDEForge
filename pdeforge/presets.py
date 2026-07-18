"""
Named presets reproducing canonical operator-learning benchmark SETUPS —
"the canon, with knobs."

A preset pins the model, physical parameters, and input measure of a
well-known dataset so it can be regenerated at ANY resolution with one call:

    generate_dataset(preset="fno_darcy_2d", n_samples=1000,
                     resolution={"x": 421, "y": 421}, seed=0)

...and every hyperparameter of the original generator stays exposed as a
knob (params= / ic_params= overrides win over the preset).

Fidelity statements are backed by tests against distributed data where we
hold a copy (see tests/test_canonical.py):
- fno_darcy_2d: input measure recovered from the distributed Darcy421 data
  (spectrum fit alpha = 2.00, tau = 3.04, R^2 = 0.998); solving THEIR stored
  coefficients reproduces THEIR stored solutions to 0.49% rel-L2.
- burgers_{smooth,canonical,rough}_1d: reproduces an independent ETDRK4
  reference implementation to ~3e-8 rel-L2.
Presets are statistical twins of the published setups (same equation,
parameters, and input measure), not byte-identical replicas — the originals'
random draws are not recoverable.
"""

PRESETS = {
    # ------------------------------------------------------------------
    # Li et al. 2020 (FNO) — Darcy flow, unit square, Dirichlet, f = 1.
    # ------------------------------------------------------------------
    # The Darcy421 family: log-normal coefficients a = exp(psi),
    # psi ~ N(0, (-Lap + 9)^(-2))  [alpha=2, tau=3; sigma measured 0.2918].
    "fno_darcy_2d": {
        "model": "darcy_fno_2d",
        "params": {
            "coeff": "lognormal",
            "alpha": 2.0,
            "tau": 3.0,
            "sigma": 0.2918,
            "forcing": 1.0,
        },
        "notes": "Canonical Darcy421 (log-normal). Validated vs distributed data.",
    },
    # The piececonst family: two-phase {12, 3} thresholded pushforward.
    "fno_darcy_piececonst_2d": {
        "model": "darcy_fno_2d",
        "params": {
            "coeff": "piececonst",
            "alpha": 2.0,
            "tau": 3.0,
            "kappa_plus": 12.0,
            "kappa_minus": 3.0,
            "threshold": 0.0,
            "forcing": 1.0,
        },
        "notes": "Canonical piecewise-constant Darcy ({12,3} pushforward).",
    },
    # ------------------------------------------------------------------
    # Li et al. 2020 (FNO) — Burgers, unit circle, nu = 0.1, u0 -> u(1).
    # ------------------------------------------------------------------
    # Official .mat measure: u0 ~ N(0, 625(-Lap + 25)^(-2))  [tau=5, alpha=2].
    "fno_burgers_grf_1d": {
        "model": "burgers_1d",
        "params": {"viscosity": 0.1, "time_horizon": 1.0, "advection": 1.0},
        "ic_generator": "grf_periodic",
        "ic_params": {"alpha": 2.0, "tau": 5.0, "scale": 625.0},
        "notes": "FNO-paper Burgers with the official GRF measure N(0,625(-Lap+25)^-2).",
    },
    # Sine-series prior variant (a_n ~ N(0, 0.49/n^3)) at the canonical
    # sharp viscosity — pdeforge's native Burgers defaults ARE this measure.
    "fno_burgers_1d": {
        "model": "burgers_1d",
        "params": {"viscosity": 0.01 / 3.141592653589793, "time_horizon": 1.0},
        "ic_generator": "fourier",
        "ic_params": {"n_modes": 9, "decay": 1.5, "amplitude": 0.7, "use_cos": False},
        "notes": "Sine-prior Burgers at nu = 0.01/pi (a_n ~ N(0, 0.49/n^3)).",
    },
    # ------------------------------------------------------------------
    # Li et al. 2020 (FNO) — Navier-Stokes vorticity, forced, nu = 1e-3.
    # ------------------------------------------------------------------
    "fno_ns_vorticity_2d": {
        "model": "ns_vorticity_2d",
        "params": {
            "viscosity": 1e-3,
            "time_horizon": 50.0,
            "forcing": "fno",
            "forcing_amplitude": 0.1,
        },
        "notes": "FNO-paper forced NS (nu=1e-3, T=50). Long horizon: expect "
        "minutes/sample at 64^2 on numpy; use backend='jax' or shorten T.",
    },
    # ------------------------------------------------------------------
    # Burgers regularity ladder: three families dialling front sharpness
    # from assumption-friendly to front-dominated — designed for coverage
    # and discretisation studies. Validated against an independent ETDRK4
    # reference implementation (~3e-8 rel-L2).
    # ------------------------------------------------------------------
    "burgers_smooth_1d": {
        "model": "burgers_1d",
        "params": {
            "viscosity": 0.1 / 3.141592653589793,
            "time_horizon": 1.0,
            "_dt": 1e-3,
        },
        "ic_generator": "fourier",
        "ic_params": {"n_modes": 3, "decay": 1.5, "amplitude": 0.7, "use_cos": False},
        "notes": "Regularity ladder, smooth end: nearly featureless solutions.",
    },
    "burgers_canonical_1d": {
        "model": "burgers_1d",
        "params": {
            "viscosity": 0.01 / 3.141592653589793,
            "time_horizon": 1.0,
            "_dt": 1e-3,
        },
        "ic_generator": "fourier",
        "ic_params": {"n_modes": 9, "decay": 1.5, "amplitude": 0.7, "use_cos": False},
        "notes": "Regularity ladder, middle: paper-baseline fronts.",
    },
    "burgers_rough_1d": {
        "model": "burgers_1d",
        "params": {
            "viscosity": 0.0025 / 3.141592653589793,
            "time_horizon": 1.0,
            "_dt": 1e-3,
        },
        "ic_generator": "fourier",
        "ic_params": {"n_modes": 15, "decay": 1.5, "amplitude": 0.7, "use_cos": False},
        "notes": "Regularity ladder, sharp end: front-dominated solutions.",
    },
    # ------------------------------------------------------------------
    # PDEBench-flavoured low-viscosity Burgers.
    # ------------------------------------------------------------------
    "pdebench_burgers_1d": {
        "model": "burgers_1d",
        "params": {"viscosity": 0.001, "time_horizon": 2.0},
        "ic_generator": "fourier",
        "ic_params": {"n_modes": 10, "decay": 1.5, "amplitude": 0.7},
        "notes": "PDEBench-style low-viscosity Burgers (shock-rich).",
    },
}


def get_preset(name):
    """Return a preset config dict (raises with the available list)."""
    if name not in PRESETS:
        raise ValueError(
            f"Unknown preset {name!r}. Available: {sorted(PRESETS.keys())}"
        )
    return PRESETS[name]


def list_presets():
    """Names of all available presets."""
    return sorted(PRESETS.keys())
