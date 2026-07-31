"""
1D Korteweg-de Vries equation.

    u_t + mu u u_x + delta2 u_xxx = 0

The canonical dispersive-soliton PDE: sech^2 solitary waves propagate with
speed proportional to their amplitude and interact elastically. Defaults are
the textbook normalisation (mu = 6, delta2 = 1); ``advection`` and
``dispersion`` open both coefficients up, which is what the neural-emulator
benchmarks need — Brandstetter et al. use mu = delta2 = 1 on a long domain
(see the ``mp_pde_kdv_1d`` preset).

Operator learning task: u(x, 0) -> u(x, T), or the full trajectory with
``outputs="trajectory"``.
"""

from typing import Callable, Dict, Tuple, Union

import numpy as np

from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.core.registry import register_model
from pdeforge.generators.initial_conditions import get_ic_generator
from pdeforge.solvers.semilinear import SemiLinearSpectralModel


@register_model("kdv_1d")
class KdV1D(SemiLinearSpectralModel):
    """
    KdV on a periodic domain (default [0, 20]). The dispersive u_xxx term
    (symbol i delta2 k^3) is integrated exactly by ETDRK4 — the stiffness that
    cripples explicit schemes never appears; only mu u u_x is stepped.

    ``scale_jitter`` reproduces the per-trajectory grid randomisation used by
    the Brandstetter et al. generators: each sample draws its own domain
    length and horizon uniformly within +/- that fraction. KdV's scaling
    symmetry (u, x, t) -> (lam^2 u, x/lam, t/lam^3) means this genuinely
    enriches the dataset rather than merely rescaling it. The dataset's stored
    grid stays NOMINAL — per-sample dx and dt differ from it by up to the
    jitter fraction, and the shared coordinate is really x/L.
    """

    NDIM = 1
    INPUT_NAMES = ["u0"]
    OUTPUT_NAMES = ["u_T"]

    USER_PARAMS = [
        ParamSpec(
            name="time_end",
            description="Final time",
            default=1.0,
            param_type=ParamType.PHYSICAL,
            bounds=(0.01, 200.0),
        ),
        ParamSpec(
            name="advection",
            description="Coefficient mu on u u_x (textbook KdV: 6)",
            default=6.0,
            param_type=ParamType.PHYSICAL,
            bounds=(0.1, 20.0),
            affects="Soliton amplitude-speed relation; mu = 1 is the "
            "neural-emulator benchmark normalisation",
        ),
        ParamSpec(
            name="dispersion",
            description="Coefficient delta2 on u_xxx",
            default=1.0,
            param_type=ParamType.PHYSICAL,
            bounds=(1e-6, 10.0),
            affects="Soliton width and the wavenumber content of the solution",
        ),
        ParamSpec(
            name="dealias",
            description="Apply the 2/3-rule mask to the nonlinear term",
            default=True,
            param_type=ParamType.INPUT,
            choices=[True, False],
            affects="At coarse nx the mask also removes GENUINE spectral "
            "content, so an un-dealiased run can be the more accurate one — "
            "and is what the reference generators do (see mp_pde_kdv_1d)",
        ),
        ParamSpec(
            name="scale_jitter",
            description="Per-sample fractional randomisation of domain length "
            "and horizon (0 = fixed grid; 0.1 = the Brandstetter et al. +/-10%)",
            default=0.0,
            param_type=ParamType.INPUT,
            bounds=(0.0, 0.5),
        ),
    ]

    DEFAULT_PARAMS = {
        "time_end": 1.0,
        "advection": 6.0,
        "dispersion": 1.0,
        "dealias": True,
        "scale_jitter": 0.0,
        "_n_time_steps": 101,
        "_n_frames_kept": None,
        "_dt": 2e-4,
    }

    def __init__(self, resolution, domain=None, **params):
        if domain is None:
            domain = {"x": (0.0, 20.0)}
        super().__init__(resolution, domain, **params)

        self.T = self.params.get("time_end", 1.0)
        self.n_t = self.params.get("_n_time_steps", 101)
        # Trajectories may drop a leading transient: solve n_t frames over
        # [0, T] but hand back only the last n_kept of them.
        self.n_kept = self.params.get("_n_frames_kept") or self.n_t
        if self.n_kept > self.n_t:
            raise ValueError(
                f"_n_frames_kept ({self.n_kept}) exceeds _n_time_steps ({self.n_t})"
            )
        self.mu = self.params.get("advection", 6.0)
        self.delta2 = self.params.get("dispersion", 1.0)
        self.scale_jitter = self.params.get("scale_jitter", 0.0)

        self._setup_spectral()
        if not self.params.get("dealias", True):
            self.dealias = np.ones_like(self.dealias)
        self.nx = resolution["x"]
        self.k = self.K[0]
        self.dt = self.params.get("_dt") or 2e-4
        # Per-sample (L, T) drawn by generate_ic and consumed by solve.
        self._sample_scale = None

        if self.scale_jitter and self.backend == "jax":
            raise ValueError(
                "scale_jitter requires backend='numpy': the JAX path batches "
                "all initial conditions before solving, which cannot carry a "
                "per-sample domain length."
            )

    def linear_symbol(self):
        # u_t = -delta2 u_xxx -> L = -delta2 (i k)^3 = i delta2 k^3
        return 1j * self.delta2 * self.k**3

    def nonlinear_hat(self, v, u, ops):
        # -mu u u_x = -mu/2 d/dx(u^2)
        return -0.5j * self.mu * self.k * (self.dealias * self._fft(u * u, ops))

    def solve(self, ic, return_full=False):
        """Solve one sample, honouring a pending per-sample scale draw."""
        if self._sample_scale is None:
            out = super().solve(ic, return_full=return_full)
        else:
            L, T = self._sample_scale
            self._sample_scale = None  # consume: never reuse a stale draw
            k_nominal, T_nominal = self.k, self.T
            try:
                self.k = 2.0 * np.pi * np.fft.fftfreq(self.nx, d=L / self.nx)
                self.T = T
                out = super().solve(ic, return_full=return_full)
            finally:
                # Restore nominal state so dataset_grid() and repeated solves
                # never inherit one sample's draw.
                self.k, self.T = k_nominal, T_nominal

        if return_full and self.n_kept < self.n_t:
            out = out[-self.n_kept :]  # discard the burn-in transient
        return out

    def dataset_grid(self, outputs="final"):
        """Time coordinate trimmed to the kept (post-burn-in) frames."""
        grid = super().dataset_grid(outputs)
        if "t" in grid and self.n_kept < self.n_t:
            grid["t"] = grid["t"][-self.n_kept :]
        return grid

    def _draw_scale(self, seed):
        """
        Draw this sample's (L, T) for scale_jitter, on a stream DERIVED from
        but independent of the IC seed — so switching jitter on perturbs the
        grid without also redrawing the input measure.
        """
        j = self.scale_jitter
        rng = np.random.default_rng(None if seed is None else int(seed) ^ 0x5CA1E5)
        self._sample_scale = (
            self.domain.size("x") * rng.uniform(1.0 - j, 1.0 + j),
            self.T * rng.uniform(1.0 - j, 1.0 + j),
        )

    def _sample_grid(self):
        """(grid, L) for the sample being built — rescaled when jitter is on."""
        L0 = self.domain.size("x")
        if self._sample_scale is None:
            return self.grids, L0
        L = self._sample_scale[0]
        return {**self.grids, "x": self.grids["x"] * (L / L0)}, L

    def generate_ic(self, generator="solitons", generator_params=None, seed=None):
        """
        Default IC: superposition of 1-3 well-separated solitons with random
        speeds — the natural input measure for KdV. "sine_series" is the
        neural-emulator benchmark measure; "fourier" gives generic smooth
        fields.
        """
        if generator_params is None:
            generator_params = {}

        if self.scale_jitter:
            self._draw_scale(seed)
        grids, L = self._sample_grid()

        if generator == "solitons":
            rng = np.random.default_rng(seed)
            x = grids["x"]
            n_sol = generator_params.get("n_solitons", int(rng.integers(1, 4)))
            u0 = np.zeros_like(x)
            for _ in range(n_sol):
                c = rng.uniform(*generator_params.get("speed_range", (1.0, 6.0)))
                x0 = rng.uniform(0.0, L)
                u0 += self._soliton_profile(c, x0, x, L)
            return u0

        if isinstance(generator, str):
            gen = get_ic_generator(generator, **generator_params)
        else:
            gen = generator
        return gen.generate(shape=(self.nx,), seed=seed, grid=grids)

    def _soliton_profile(self, c, x0, x, L):
        """
        Periodic sech^2 soliton of speed c for u_t + mu u u_x + delta2 u_xxx = 0:

            u = (3 c / mu) sech^2( sqrt(c / delta2) (x - x0) / 2 )

        which reduces to the textbook c/2 sech^2(sqrt(c)(x-x0)/2) at mu = 6,
        delta2 = 1.
        """
        dxp = (x - x0 + L / 2) % L - L / 2
        width = 0.5 * np.sqrt(c / self.delta2)
        return (3.0 * c / self.mu) / np.cosh(width * dxp) ** 2

    def soliton(self, c, x0=None):
        """Exact single-soliton profile of speed c on the nominal grid."""
        x = self.grids["x"]
        L = self.domain.size("x")
        x0 = L / 2 if x0 is None else x0
        return self._soliton_profile(c, x0, x, L)

    def validate_solution(self, ic, solution, tol=1e-6):
        is_valid = (
            not np.isnan(solution).any()
            and not np.isinf(solution).any()
            and np.abs(solution).max() < 1e3
        )
        return {"valid": is_valid, "max_value": float(np.abs(solution).max())}
