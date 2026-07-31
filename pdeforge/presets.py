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
- mp_pde_kdv_1d: our ETDRK4 reproduces THEIR solver (scipy Radau + psdiff)
  to ~2e-4 rel-L2 at nx = 256, T = 100 — and that residual is theirs, not
  ours: against a converged nx = 1024 reference we sit at 8e-7 while their
  Radau sits at 2.9e-4. Short-horizon agreement is ~1e-6.
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
    # KdV in the dispersive-shock-wave (undular-bore) regime.
    # ------------------------------------------------------------------
    # Same equation and same ETDRK4 solver as every other KdV setup here, but
    # a small dispersion scale and a depression input measure. A localised
    # smooth depression does not steepen into a thin front; it dissolves into
    # a train of high-wavenumber oscillations filling a LARGE contiguous
    # fraction of the domain. Where a Burgers shock mis-samples only a
    # sqrt(nu)-thin front, the bore is hard for a band-limited operator
    # everywhere it lives — a stringent operator / UQ benchmark.
    "kdv_dsw_1d": {
        "model": "kdv_1d",
        "domain": {"x": (0.0, 1.0)},
        "resolution": {"x": 512},
        "params": {
            "advection": 6.0,
            "dispersion": 8.0e-6,
            "time_end": 1.0e-2,
            "_n_time_steps": 101,
            "_dt": 1.0e-6,
        },
        "ic_generator": "depression_box",
        "notes": "KdV undular bore, vigorous: ~155 oscillations at nx=512, but "
        "un-resolved there (429 at nx=2048) — a large un-resolvable BIAS set.",
    },
    # The same bore at a longer wavelength: a moderate operator can nearly
    # resolve it, so hardness becomes epistemic (data-limited) rather than a
    # bias floor — the pairing is what makes the two useful for method
    # comparison.
    "kdv_dsw_epistemic_1d": {
        "model": "kdv_1d",
        "domain": {"x": (0.0, 1.0)},
        "resolution": {"x": 512},
        "params": {
            "advection": 6.0,
            "dispersion": 4.0e-5,
            "time_end": 1.0e-2,
            "_n_time_steps": 101,
            "_dt": 1.0e-6,
        },
        "ic_generator": "depression_box",
        "notes": "KdV undular bore, near-resolvable: ~83 oscillations and "
        "grid-converged at nx=512 — hardness is EPISTEMIC (data-limited), "
        "not a band-limit bias floor.",
    },
    # ------------------------------------------------------------------
    # Brandstetter et al. — KdV on a long periodic box, the input measure and
    # protocol shared by "Message Passing Neural PDE Solvers" (arXiv:2202.03376)
    # and "Lie Point Symmetry Data Augmentation" (arXiv:2202.07643).
    # ------------------------------------------------------------------
    # u_t + u u_x + u_xxx = 0 (mu = delta2 = 1, NOT the textbook mu = 6) on
    # L = 128 with nx = 256, from a 10-wave random sine series restricted to
    # wavenumbers {1, 2}. Their generator randomises L and T by +/-10% per
    # trajectory (scale_jitter) and keeps only the last 140 of 250 frames, so
    # a sample starts at t ~ 0.44 T from a developed soliton gas, not the IC.
    # Un-dealiased to match their psdiff right-hand side; at nx = 256 the
    # 2/3 mask would also cut genuine spectral content.
    "mp_pde_kdv_1d": {
        "model": "kdv_1d",
        "domain": {"x": (0.0, 128.0)},
        "resolution": {"x": 256},
        "params": {
            "advection": 1.0,
            "dispersion": 1.0,
            "time_end": 100.0,
            "dealias": False,
            "scale_jitter": 0.1,
            "_n_time_steps": 250,
            "_n_frames_kept": 140,
            "_dt": 5e-3,
        },
        "ic_generator": "sine_series",
        "ic_params": {"n_waves": 10, "lmin": 1, "lmax": 3, "amplitude": 0.5},
        "outputs": "trajectory",
        "notes": "Brandstetter et al. KdV (MP-PDE / LPSDA). ETDRK4 sits 8e-7 "
        "from a converged nx=1024 reference at T=100, where their own Radau "
        "sits at 2.9e-4; dt = 5e-3 is converged.",
    },
    # Their "easy" variant: same measure, half the horizon.
    "mp_pde_kdv_easy_1d": {
        "model": "kdv_1d",
        "domain": {"x": (0.0, 128.0)},
        "resolution": {"x": 256},
        "params": {
            "advection": 1.0,
            "dispersion": 1.0,
            "time_end": 50.0,
            "dealias": False,
            "scale_jitter": 0.1,
            "_n_time_steps": 250,
            "_n_frames_kept": 140,
            "_dt": 5e-3,
        },
        "ic_generator": "sine_series",
        "ic_params": {"n_waves": 10, "lmin": 1, "lmax": 3, "amplitude": 0.5},
        "outputs": "trajectory",
        "notes": "Brandstetter et al. KdV, 'easy' end_time = 50 variant.",
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
