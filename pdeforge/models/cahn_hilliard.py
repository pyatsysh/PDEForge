"""
N-dimensional Cahn-Hilliard Equation Solver (Spinodal Decomposition)

The Cahn-Hilliard equation models phase separation with *conserved* dynamics:

    ∂u/∂t = M ∇²(u³ − u − ε²∇²u)

Starting from a near-uniform mixture perturbed by small random noise, the
spinodal instability spontaneously amplifies a band of wavenumbers and the
conserved nonlinear dynamics sharpen the result into the characteristic
interconnected (labyrinthine) or droplet morphology of spinodal decomposition.

Unlike Allen-Cahn (non-conserved), the spatial mean of u is preserved exactly,
so `mean_composition` controls the morphology: 0 → bicontinuous labyrinth,
|m| → 0.4 → minority-phase droplets.

The same code path handles 2D and 3D — the dimension is inferred from the
`resolution` dict. Randomised samples come from seeding the white-noise IC;
the spinodal *appearance* is produced by the PDE dynamics, not the IC.

with periodic boundary conditions.

Operator Learning Task:
    u(x, t=0) → u(x, t=T)
"""

from typing import Callable, Dict, Tuple, Union

import numpy as np

from pdeforge.core.base import PDEModel
from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.core.registry import register_model


@register_model("cahn_hilliard")
class CahnHilliard(PDEModel):
    """
    N-dimensional Cahn-Hilliard equation for spinodal decomposition.

    ∂u/∂t = M ∇²(u³ − u − ε²∇²u)

    Generates randomised patterns reminiscent of spinodal decomposition in
    2D and 3D. The same model handles both: the dimension is inferred from
    the `resolution` dict ({"x", "y"} → 2D, {"x", "y", "z"} → 3D).

    Each sample starts from a near-uniform field plus small white noise; the
    spinodal instability grows that noise into a pattern with a characteristic
    length scale λ* ≈ 2π√2·ε. Different seeds give distinct realisations that
    share the same morphology statistics.

    Set ``binarize=True`` to threshold the result at u = 0 and return hard
    ``{0, 1}`` two-phase masks instead of the continuous phase field.

    Examples
    --------
    >>> dataset = generate_dataset(
    ...     model="cahn_hilliard",
    ...     n_samples=50,
    ...     resolution={"x": 128, "y": 128},
    ...     params={"epsilon": 0.01, "mean_composition": 0.0, "time_end": 0.1},
    ... )
    >>> # 3D: just add a z axis to the resolution
    >>> dataset = generate_dataset(
    ...     model="cahn_hilliard",
    ...     n_samples=10,
    ...     resolution={"x": 64, "y": 64, "z": 64},
    ... )
    >>> # binary {0, 1} masks instead of the continuous field
    >>> masks = generate_dataset(
    ...     model="cahn_hilliard",
    ...     n_samples=50,
    ...     resolution={"x": 128, "y": 128},
    ...     params={"binarize": True},
    ... )
    """

    NDIM = None  # works in 2D and 3D; inferred from resolution
    INPUT_NAMES = ["u0"]
    OUTPUT_NAMES = ["u_T"]

    USER_PARAMS = [
        ParamSpec(
            name="epsilon",
            description="Interface width parameter",
            default=0.01,
            param_type=ParamType.PHYSICAL,
            bounds=(0.004, 0.025),
            affects=(
                "Sets the pattern length scale λ* ≈ 2π√2·ε (smaller → finer). "
                "Below ~0.006 use resolution ≥ 256 to resolve interfaces."
            ),
        ),
        ParamSpec(
            name="mobility",
            description="Cahn-Hilliard mobility M",
            default=1.0,
            param_type=ParamType.PHYSICAL,
            bounds=(0.01, 10.0),
            affects="Higher mobility → faster phase separation and coarsening",
        ),
        ParamSpec(
            name="mean_composition",
            description="Spatial mean of u (conserved exactly by the dynamics)",
            default=0.0,
            param_type=ParamType.INPUT,
            bounds=(-0.6, 0.6),
            affects=(
                "0 → bicontinuous labyrinth; |m| → 0.4 → droplets. "
                "Beyond ±0.577 leaves the spinodal regime."
            ),
        ),
        ParamSpec(
            name="time_end",
            description="Final simulation time",
            default=0.1,
            param_type=ParamType.PHYSICAL,
            bounds=(0.001, 10.0),
            units="s",
            affects="Longer time → coarser domains (coarsening ~ t^(1/3))",
        ),
        ParamSpec(
            name="binarize",
            description="Return a binary {0, 1} mask instead of the continuous field",
            default=False,
            param_type=ParamType.OUTPUT,
            choices=[True, False],
            affects=(
                "True → outputs are hard two-phase masks (threshold at u = 0); "
                "False → continuous phase field u in ~[-1, 1]"
            ),
        ),
    ]

    DEFAULT_PARAMS = {
        "epsilon": 0.01,
        "mobility": 1.0,
        "mean_composition": 0.0,
        "time_end": 0.1,
        "binarize": False,
        "_dt": 1e-3,
        "_n_time_steps": 51,
        "_stabilization": 2.0,
        "_noise_amplitude": 0.02,
    }

    def __init__(
        self,
        resolution: Dict[str, int],
        domain: Dict[str, Tuple[float, float]] = None,
        **params,
    ):
        super().__init__(resolution, domain, **params)

        self.eps = self.params["epsilon"]
        self.mobility = self.params["mobility"]
        self.mean_composition = self.params["mean_composition"]
        self.T = self.params["time_end"]
        self.dt = self.params["_dt"]
        self.n_t = self.params["_n_time_steps"]
        self.stab = self.params["_stabilization"]
        self.noise_amplitude = self.params["_noise_amplitude"]
        self.binarize = self.params["binarize"]

        # Array axes are the dimensions in reverse-sorted order:
        # 2D -> (ny, nx), 3D -> (nz, ny, nx). This matches the convention
        # used by the existing 2D spectral models (e.g. allen_cahn_2d).
        self.dim_order = sorted(resolution.keys())[::-1]
        self.field_shape = tuple(resolution[d] for d in self.dim_order)

        # Wavenumber grids — N-dimensional via meshgrid + fftn.
        ks = []
        for d in self.dim_order:
            grid = self.grids[d]
            dx = grid[1] - grid[0]
            ks.append(2 * np.pi * np.fft.fftfreq(resolution[d], d=dx))
        K = np.meshgrid(*ks, indexing="ij")
        self.K2 = sum(Ki**2 for Ki in K)

        # Implicit denominator of the stabilised IMEX scheme:
        #   denom = 1 + dt·M·k²·(A + ε²k²)
        self._denom = 1.0 + self.dt * self.mobility * self.K2 * (
            self.stab + self.eps**2 * self.K2
        )

    def solve(self, ic: np.ndarray, return_full: bool = False) -> np.ndarray:
        """
        Solve Cahn-Hilliard with a linearly-stabilised (Eyre-type) IMEX scheme.

        The biharmonic term and a stabilising term are treated implicitly, the
        cubic explicitly. The naive IMEX scheme (cf. allen_cahn) is unstable
        here because the uphill-diffusion term is k²-amplified; the stabiliser
        A makes the step robust. The k=0 (mean) mode is preserved exactly each
        step, so the spatial mean of u is conserved to machine precision.

        With ``binarize=True`` the result is thresholded at u = 0 into a hard
        ``{0, 1}`` two-phase mask.
        """
        u = ic.copy()
        dt = self.dt
        M = self.mobility
        A = self.stab

        n_substeps = int(np.ceil(self.T / dt))
        output_interval = max(1, n_substeps // max(1, self.n_t - 1))

        solutions = [u.copy()]
        for step in range(n_substeps):
            u_hat = np.fft.fftn(u)
            u3_hat = np.fft.fftn(u**3)
            # numer = û + dt·M·k²·[(1 + A)·û − FFT(u³)]
            numer = u_hat + dt * M * self.K2 * ((1.0 + A) * u_hat - u3_hat)
            u = np.fft.ifftn(numer / self._denom).real

            if (step + 1) % output_interval == 0 and len(solutions) < self.n_t:
                solutions.append(u.copy())

        # Pad / trim so return_full always has exactly n_t frames.
        while len(solutions) < self.n_t:
            solutions.append(u.copy())
        solutions = solutions[: self.n_t]

        if return_full:
            result = np.stack(solutions, axis=0)
        else:
            result = solutions[-1]

        if self.binarize:
            # Threshold at u = 0 → hard {0, 1} two-phase mask.
            result = (result > 0.0).astype(np.float64)
        return result

    def generate_ic(
        self,
        generator: Union[str, Callable] = "default",
        generator_params: Dict = None,
        seed: int = None,
    ) -> np.ndarray:
        """
        Generate a spinodal initial condition: a near-uniform field at
        `mean_composition` perturbed by small white noise.

        The IC itself is structureless — the spinodal morphology is produced
        by the instability during `solve`, not drawn here. Different seeds
        give distinct realisations with the same morphology statistics.
        """
        if generator_params is None:
            generator_params = {}

        if seed is not None:
            np.random.seed(seed)

        m = self.mean_composition
        eta = generator_params.get("noise_amplitude", self.noise_amplitude)

        return m + eta * np.random.randn(*self.field_shape)

    def validate_solution(
        self,
        ic: np.ndarray,
        solution: np.ndarray,
        tol: float = 1e-6,
    ) -> Dict:
        """
        Validate the solution.

        Continuous output: finite, bounded, and mass-conserving (mean preserved).
        Binarized output: finite and a clean {0, 1} mask (reports fill fraction).
        """
        finite = not np.isnan(solution).any() and not np.isinf(solution).any()

        if self.binarize:
            is_binary = bool(np.isin(solution, (0.0, 1.0)).all())
            return {
                "valid": finite and is_binary,
                "fill_fraction": float(solution.mean()),
            }

        mass_drift = float(np.abs(solution.mean() - ic.mean()))
        is_valid = (
            finite
            and np.abs(solution).max() < 1.5
            and mass_drift < 1e-4
        )
        return {
            "valid": is_valid,
            "mass_drift": mass_drift,
            "max_value": float(np.abs(solution).max()),
        }
