"""
Transonic compressible Euler over parameterized NACA airfoils.

    d/dt (rho, rho u, rho v, rho E) + div F = 0

solved to steady state on a body-fitted C-grid with HLLC + MUSCL finite
volumes (see ``pdeforge.solvers.euler_fv``). At the transonic condition the
flow accelerates past Mach 1 over the upper surface and closes with a shock —
the feature that makes this a different learning problem from every smooth
model in the catalogue, and one a Fourier method would ring across.

GEOMETRY IS THE DATA. Every sample draws its own airfoil (thickness, camber,
camber position) and flow condition (freestream Mach, angle of attack); the
C-grid is rebuilt around it and the solution is returned ON that mesh. The
operator-learning task is therefore the Geo-FNO one — a deformed mesh in, the
flow field on it out — rather than a fixed Cartesian grid:

    (x, y) mesh coordinates  ->  (rho, u, v, p)

Per-sample parameters, C_l, C_d and the residual drop achieved are recorded in
the dataset metadata.

This is airfoil data WITH KNOBS at a shock-carrying condition. It is not a
byte-level recreation of the Geo-FNO airfoil dataset (their mesh, solver and
sampling are their own), and it is emphatically not RANS: for the viscous,
turbulence-closed regime PDEForge reads AirfRANS rather than rebuilding it
(see ``pdeforge.load_airfrans``).
"""

from concurrent.futures import as_completed
from typing import Dict

import numpy as np
from tqdm import tqdm

from pdeforge.core.base import PDEModel
from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.core.registry import register_model
from pdeforge.geometry import airfoil_c_grid
from pdeforge.solvers.euler_fv import GAMMA, EulerCGrid


def _solve_one(model, ic_generator, ic_params, seed, validate):
    """Module-level worker so process pools can pickle it."""
    return model.generate_sample(
        generator=ic_generator,
        generator_params=ic_params,
        seed=seed,
        validate=validate,
    )


@register_model("airfoil_euler_2d")
class AirfoilEuler2D(PDEModel):
    """
    Steady transonic Euler around a random NACA 4-digit airfoil.

    ``resolution`` sets the C-grid cell counts, ``{"xi": ..., "eta": ...}``:
    xi wraps the airfoil and both wake cuts, eta runs from the wall to the far
    field. xi is rounded to even so the surface point count stays odd (the
    trailing edge must land on a node).
    """

    NDIM = 2
    INPUT_NAMES = ["x", "y"]
    OUTPUT_NAMES = ["rho", "u", "v", "p"]
    TIME_DEPENDENT = False  # steady state: no trajectory to hand back
    BACKENDS = {"numpy"}

    USER_PARAMS = [
        ParamSpec(
            name="mach_range",
            description="Freestream Mach number range",
            default=(0.70, 0.82),
            param_type=ParamType.PHYSICAL,
            affects="Above ~0.72 a supersonic pocket and a shock appear",
        ),
        ParamSpec(
            name="aoa_range",
            description="Angle of attack range (degrees)",
            default=(-1.5, 3.0),
            param_type=ParamType.PHYSICAL,
        ),
        ParamSpec(
            name="thickness_range",
            description="NACA maximum thickness (chord fraction)",
            default=(0.08, 0.15),
            param_type=ParamType.GEOMETRY,
        ),
        ParamSpec(
            name="camber_range",
            description="NACA maximum camber (chord fraction)",
            default=(0.0, 0.04),
            param_type=ParamType.GEOMETRY,
        ),
        ParamSpec(
            name="camber_pos_range",
            description="Chordwise position of maximum camber",
            default=(0.3, 0.5),
            param_type=ParamType.GEOMETRY,
        ),
        ParamSpec(
            name="far_field_radius",
            description="Far-field distance in chords",
            default=20.0,
            param_type=ParamType.GEOMETRY,
            bounds=(5.0, 200.0),
            affects="With the vortex correction on, 20 chords is enough",
        ),
        ParamSpec(
            name="cfl",
            description="CFL number for the local time step",
            default=0.7,
            param_type=ParamType.INPUT,
            bounds=(0.05, 1.0),
        ),
        ParamSpec(
            name="max_iterations",
            description="Cap on steady-state iterations per sample",
            default=12000,
            param_type=ParamType.INPUT,
            bounds=(100, 200000),
        ),
        ParamSpec(
            name="residual_tol",
            description="Stop when the density residual has dropped this far",
            default=1e-5,
            param_type=ParamType.INPUT,
            bounds=(1e-12, 1e-1),
        ),
    ]

    DEFAULT_PARAMS = {
        "mach_range": (0.70, 0.82),
        "aoa_range": (-1.5, 3.0),
        "thickness_range": (0.08, 0.15),
        "camber_range": (0.0, 0.04),
        "camber_pos_range": (0.3, 0.5),
        "far_field_radius": 20.0,
        "cfl": 0.7,
        "max_iterations": 12000,
        "residual_tol": 1e-5,
        "vortex_correction": True,
        "first_cell": 1e-3,
        "smooth_iters": 100,
        "wake_fraction": 0.19,
        "order": 2,
    }

    def __init__(self, resolution=None, domain=None, **params):
        if resolution is None:
            resolution = {"xi": 256, "eta": 64}
        if "xi" not in resolution or "eta" not in resolution:
            raise ValueError(
                "airfoil_euler_2d takes resolution={'xi': ..., 'eta': ...} "
                "(C-grid cell counts), got %r" % (resolution,)
            )
        # xi must be even so the surface node count comes out odd
        resolution = dict(resolution)
        resolution["xi"] = int(resolution["xi"]) + (int(resolution["xi"]) % 2)

        if domain is None:
            # computational coordinates; the PHYSICAL mesh is per sample and
            # travels in the inputs
            domain = {"xi": (0.0, 1.0), "eta": (0.0, 1.0)}
        super().__init__(resolution, domain, **params)

        self.n_xi = resolution["xi"]
        self.n_eta = resolution["eta"]
        self.n_wake = max(9, int(round(self.params["wake_fraction"] * self.n_xi)))
        self.n_surf = self.n_xi + 1 - 2 * self.n_wake
        if self.n_surf < 41:
            raise ValueError(
                f"resolution xi={self.n_xi} leaves only {self.n_surf} surface "
                "points; raise xi or lower wake_fraction"
            )
        self._last_info: Dict = {}

    # ------------------------------------------------------------------ mesh
    def build_mesh(self, thickness, camber, camber_pos):
        return airfoil_c_grid(
            thickness=thickness,
            camber=camber,
            camber_pos=camber_pos,
            n_surf=self.n_surf,
            n_wake=self.n_wake,
            n_eta=self.n_eta + 1,
            radius=self.params["far_field_radius"],
            x_out=self.params["far_field_radius"] + 1.0,
            first_cell=self.params["first_cell"],
            smooth_iters=self.params["smooth_iters"],
        )

    # ------------------------------------------------------- sample plumbing
    def generate_ic(self, generator="default", generator_params=None, seed=None):
        """Draw the sample's identity: airfoil shape plus flow condition."""
        if generator_params is None:
            generator_params = {}
        rng = np.random.default_rng(seed)

        def draw(key):
            lo, hi = generator_params.get(key, self.params[key])
            return float(rng.uniform(lo, hi))

        return np.array(
            [
                draw("thickness_range"),
                draw("camber_range"),
                draw("camber_pos_range"),
                draw("mach_range"),
                draw("aoa_range"),
            ]
        )

    def solve(self, ic, return_full=False):
        """
        Solve one sample. ``ic`` is the parameter vector from generate_ic:
        (thickness, camber, camber_pos, mach, aoa_deg).
        """
        if return_full:
            raise ValueError(
                "airfoil_euler_2d is a steady-state model; there is no "
                "trajectory to return."
            )
        thickness, camber, camber_pos, mach, aoa = (float(v) for v in ic)

        X, Y, n_wall, n_wake = self.build_mesh(thickness, camber, camber_pos)
        s = EulerCGrid(
            X,
            Y,
            n_wall,
            n_wake,
            mach=mach,
            aoa_deg=aoa,
            cfl=self.params["cfl"],
            order=self.params["order"],
            vortex_correction=self.params["vortex_correction"],
        )
        drop = s.solve(
            iters=int(self.params["max_iterations"]),
            tol=self.params["residual_tol"],
        )
        r, u, v, p = s.fields()
        cl, cd = s.force_coefficients()

        self._last_info = {
            "params": [thickness, camber, camber_pos, mach, aoa],
            "Cl": cl,
            "Cd": cd,
            "residual_drop": float(drop),
            "mach_max": float(s.mach_field().max()),
            "mesh": np.stack([s.g["xc"], s.g["yc"]], axis=-1),
        }
        return np.stack([r, u, v, p], axis=-1)

    def generate_sample(
        self,
        generator="default",
        generator_params=None,
        seed=None,
        validate=True,
        max_attempts=10,
        return_full=False,
    ):
        """The input is the deformed MESH, so it is only known after solving."""
        for attempt in range(max_attempts):
            current_seed = None if seed is None else seed + attempt
            ic = self.generate_ic(
                generator=generator,
                generator_params=generator_params,
                seed=current_seed,
            )
            try:
                solution = self.solve(ic)
            except (FloatingPointError, ValueError):
                if attempt == max_attempts - 1:
                    raise
                continue
            info = dict(self._last_info)
            mesh = info.pop("mesh")
            info["valid"] = bool(np.isfinite(solution).all())
            if not validate or info["valid"]:
                return mesh, solution, info
        raise RuntimeError(f"no valid sample after {max_attempts} attempts")

    def generate_dataset(
        self,
        n_samples,
        ic_generator="default",
        ic_params=None,
        seed=None,
        validate=True,
        n_jobs=1,
        verbose=True,
        outputs="final",
    ):
        """Records per-sample geometry, flow condition, and force coefficients."""
        if outputs != "final":
            raise ValueError(
                "airfoil_euler_2d is steady state; outputs='trajectory' is "
                "not available."
            )
        from pdeforge.core.base import _legacy_seed, _seed_sequence
        from pdeforge.core.types import PDEDataset

        seqs = _seed_sequence(seed).spawn(n_samples)
        seeds = [_legacy_seed(s) if seed is not None else None for s in seqs]
        args = [(ic_generator, ic_params, s, validate) for s in seeds]

        # Own loop rather than the base one: each sample carries per-sample
        # metadata (geometry, forces) that the base worker cannot return, and
        # a closure-based recorder would make this model unpicklable and so
        # silently break n_jobs > 1.
        if n_jobs == 1:
            iterator = tqdm(args, disable=not verbose, desc="Generating samples")
            results = [_solve_one(self, *a) for a in iterator]
        else:
            from concurrent.futures import ProcessPoolExecutor

            max_workers = None if n_jobs in (-1, 0) else n_jobs
            results = [None] * n_samples
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(_solve_one, self, *a): i for i, a in enumerate(args)
                }
                done = tqdm(
                    as_completed(futures),
                    total=n_samples,
                    disable=not verbose,
                    desc=f"Generating samples ({n_jobs} workers)",
                )
                for fut in done:
                    results[futures[fut]] = fut.result()

        meshes = [r[0] for r in results]
        solutions = [r[1] for r in results]
        records = [r[2] for r in results]

        dataset = PDEDataset(
            inputs=np.stack(meshes, axis=0),
            outputs=np.stack(solutions, axis=0),
            grid=self.dataset_grid("final"),
            metadata=self.dataset_metadata(
                n_samples, ic_generator, ic_params, seed, "final"
            ),
            input_names=self.INPUT_NAMES,
            output_names=self.OUTPUT_NAMES,
        )

        recs = records[-n_samples:]
        names = ["thickness", "camber", "camber_pos", "mach", "aoa_deg"]
        dataset.metadata["param_samples"] = {
            k: [r["params"][i] for r in recs] for i, k in enumerate(names)
        }
        for key in ("Cl", "Cd", "residual_drop", "mach_max"):
            dataset.metadata[key] = [r[key] for r in recs]
        dataset.metadata["transonic_fraction"] = float(
            np.mean([r["mach_max"] > 1.0 for r in recs])
        )
        return dataset

    def validate_solution(self, ic, solution, tol=1e-6):
        finite = np.isfinite(solution).all()
        rho, p = solution[..., 0], solution[..., 3]
        return {
            "valid": bool(finite and rho.min() > 0.0 and p.min() > 0.0),
            "rho_min": float(rho.min()),
            "p_min": float(p.min()),
        }
