"""
2D Darcy flow solver.

Flow through porous media:
    -div(kappa(x,y) grad u) = f

Operator learning task: kappa(x,y) -> u(x,y)
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from pdeforge.core.base import PDEModel

# ensure field is positive
_clip_positive = lambda x: np.maximum(x, 1e-6)
from pdeforge.core.registry import register_model
from pdeforge.generators.initial_conditions import (
    GaussianRandomFieldGenerator,
    SigmoidTransformGenerator,
    get_ic_generator,
)


@register_model("darcy_2d")
class Darcy2D(PDEModel):
    """
    2D Darcy flow with spectral solver. Maps permeability to pressure.
    Uses periodic BCs.
    """

    NDIM = 2
    TIME_DEPENDENT = False  # steady elliptic solve
    DEFAULT_PARAMS = {
        "kappa_min": 0.1,
        "kappa_max": 10.0,
        "source_type": "sine",
        "cg_tol": 1e-10,
        "cg_maxiter": 1000,
    }
    INPUT_NAMES = ["kappa"]
    OUTPUT_NAMES = ["pressure"]

    def __init__(self, resolution, domain=None, **params):
        super().__init__(resolution, domain, **params)

        self.nx = resolution.get("x", 64)
        self.ny = resolution.get("y", 64)
        self.dx = self.grids["x"][1] - self.grids["x"][0]
        self.dy = self.grids["y"][1] - self.grids["y"][0]

        self.X, self.Y = np.meshgrid(self.grids["x"], self.grids["y"])

        # wavenumbers
        self.kx = 2 * np.pi * np.fft.fftfreq(self.nx, d=self.dx)
        self.ky = 2 * np.pi * np.fft.fftfreq(self.ny, d=self.dy)
        self.KX, self.KY = np.meshgrid(self.kx, self.ky)
        self.K2 = self.KX**2 + self.KY**2

        # source term
        if self.params["source_type"] == "sine":
            self.f = np.sin(2 * np.pi * self.X) * np.sin(2 * np.pi * self.Y)
        else:
            self.f = np.ones((self.ny, self.nx))

    def _apply_operator(self, u, kappa):
        # A(u) = -div(kappa grad u)
        u = u.reshape((self.ny, self.nx))

        u_hat = np.fft.fft2(u)
        du_dx = np.fft.ifft2(1j * self.KX * u_hat).real
        du_dy = np.fft.ifft2(1j * self.KY * u_hat).real
        lap_u = np.fft.ifft2(-self.K2 * u_hat).real

        kappa_hat = np.fft.fft2(kappa)
        dkappa_dx = np.fft.ifft2(1j * self.KX * kappa_hat).real
        dkappa_dy = np.fft.ifft2(1j * self.KY * kappa_hat).real

        result = -kappa * lap_u - (dkappa_dx * du_dx + dkappa_dy * du_dy)

        return result.flatten()

    def solve(self, kappa, return_info=False):
        """
        Solve Darcy eqn with conjugate gradient.

        kappa: permeability field
        return_info: if True, also return solver info
        """
        n = self.nx * self.ny

        def matvec(u):
            return self._apply_operator(u, kappa)

        A = LinearOperator((n, n), matvec=matvec)
        b = self.f.flatten()

        # XXX: initial guess could be better
        f_hat = np.fft.fft2(self.f)
        K2_safe = self.K2.copy()
        K2_safe[0, 0] = 1e-10
        u0_hat = f_hat / K2_safe / kappa.mean()
        u0 = np.fft.ifft2(u0_hat).real.flatten()

        # scipy 1.14+ uses rtol, older uses tol
        import scipy

        cg_kwargs = {
            "x0": u0,
            "maxiter": self.params["cg_maxiter"],
        }
        if tuple(map(int, scipy.__version__.split(".")[:2])) >= (1, 14):
            cg_kwargs["rtol"] = self.params["cg_tol"]
        else:
            cg_kwargs["tol"] = self.params["cg_tol"]
        u, info = cg(A, b, **cg_kwargs)

        u = u.reshape((self.ny, self.nx))
        u = u - u.mean()  # zero mean

        if return_info:
            return u, {"cg_converged": info == 0}
        return u

    def generate_ic(self, generator="sigmoid", generator_params=None, seed=None):
        """
        Generate random permeability field.
        Uses sigmoid transform to ensure kappa in [kappa_min, kappa_max].
        """
        if generator_params == None:
            generator_params = {}

        if generator == "sigmoid" or generator == "default":
            gen = SigmoidTransformGenerator(
                u_min=self.params["kappa_min"],
                u_max=self.params["kappa_max"],
                base_generator=GaussianRandomFieldGenerator(
                    alpha=generator_params.get("alpha", 2.0),
                    amplitude=generator_params.get("amplitude", 1.0),
                ),
            )
        elif isinstance(generator, str):
            gen = get_ic_generator(generator, **generator_params)
        else:
            gen = generator

        kappa = gen.generate(
            shape=(self.ny, self.nx),
            seed=seed,
            grid=self.grids,
        )

        kappa = _clip_positive(kappa)

        return kappa

    def validate_solution(self, kappa, u, tol=1e-6):
        is_valid = not np.isnan(u).any() and not np.isinf(u).any()
        return {"valid": is_valid}
