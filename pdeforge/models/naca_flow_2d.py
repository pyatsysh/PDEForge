"""
2D incompressible flow past a NACA 4-digit airfoil family (FEniCSx).

Steady Navier-Stokes in a channel around a parameterized airfoil:

    rho (u . grad) u - mu Laplacian(u) + grad(p) = 0,   div(u) = 0

with uniform far-field velocity on inlet/top/bottom, natural outlet, no-slip
on the airfoil. GEOMETRY IS THE DATA: every sample draws its own airfoil
(thickness, camber, camber position, angle of attack) plus an inlet-velocity
scale, the channel is re-meshed, and the solve returns (u, v, p) on the
regular grid together with the lift and drag coefficients from the surface
stress integral.

Operator learning task: geometry (as a signed-distance-function channel) ->
(u, v, p). Per-sample parameters and (C_l, C_d) are recorded in the dataset
metadata — the FlowBench-style targets, but from a generator with knobs
rather than a frozen download. "Airfoil-class data with knobs": laminar
incompressible NS — deliberately NOT a recreation of the transonic Geo-FNO
airfoil or of RANS-based AirfRANS.
"""

from typing import Dict, Tuple

import numpy as np

from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.core.registry import register_model
from pdeforge.geometry import naca4_coords, polygon_sdf, rotate_airfoil

try:
    import basix
    import dolfinx
    import ufl
    from dolfinx import fem
    from dolfinx.fem.petsc import NonlinearProblem
    from mpi4py import MPI
    from petsc4py import PETSc
    from ufl import FacetNormal, Identity, div, dx, grad, inner

    from pdeforge.core.fenics_base import FEniCSModel
    from pdeforge.solvers.fenics_utils import create_channel_with_polygon

    HAS_FENICSX = True
except ImportError:
    HAS_FENICSX = False

    class FEniCSModel:  # type: ignore[no-redef]
        pass


if HAS_FENICSX:

    @register_model("naca_flow_2d")
    class NACAFlow2D(FEniCSModel):
        """
        Steady laminar flow past NACA 4-digit airfoils, geometry sampled per
        dataset sample. Taylor-Hood elements, Newton/SNES, direct LU.
        """

        NDIM = 2
        BACKEND = "fenicsx"
        TIME_DEPENDENT = False
        INPUT_NAMES = ["sdf"]
        OUTPUT_NAMES = ["u", "v", "p"]

        USER_PARAMS = [
            ParamSpec(
                name="viscosity",
                description="Dynamic viscosity (Re = U * chord / nu)",
                default=0.02,
                param_type=ParamType.PHYSICAL,
                bounds=(1e-3, 1.0),
                affects="Default U=1, chord=1 -> Re = 50 (robust steady flow)",
            ),
            ParamSpec(
                name="inlet_velocity",
                description="Far-field velocity U",
                default=1.0,
                param_type=ParamType.PHYSICAL,
                bounds=(0.1, 5.0),
            ),
            ParamSpec(
                name="thickness_range",
                description="NACA thickness draw range (chord fraction)",
                default=(0.08, 0.18),
                param_type=ParamType.GEOMETRY,
            ),
            ParamSpec(
                name="camber_range",
                description="Max camber draw range (chord fraction)",
                default=(0.0, 0.06),
                param_type=ParamType.GEOMETRY,
            ),
            ParamSpec(
                name="camber_pos_range",
                description="Camber position draw range (chord fraction)",
                default=(0.3, 0.6),
                param_type=ParamType.GEOMETRY,
            ),
            ParamSpec(
                name="aoa_range",
                description="Angle-of-attack draw range (degrees)",
                default=(-8.0, 8.0),
                param_type=ParamType.GEOMETRY,
            ),
        ]

        DEFAULT_PARAMS = {
            "viscosity": 0.02,
            "inlet_velocity": 1.0,
            "density": 1.0,
            "thickness_range": (0.08, 0.18),
            "camber_range": (0.0, 0.06),
            "camber_pos_range": (0.3, 0.6),
            "aoa_range": (-8.0, 8.0),
            "inlet_scale_range": (0.8, 1.2),
            "_mesh_resolution": 0.12,
            "_n_surface_points": 100,
            "_x_range": (-1.0, 3.0),
            "_y_range": (-1.0, 1.0),
        }

        def __init__(self, resolution, domain=None, **params):
            merged = {**self.DEFAULT_PARAMS, **params}
            if domain is None:
                domain = {"x": merged["_x_range"], "y": merged["_y_range"]}

            self.mu = merged["viscosity"]
            self.rho = merged.get("density", 1.0)
            self.U = merged["inlet_velocity"]

            # geometry of the CURRENT mesh (rebuilt per sample in solve)
            self._geo = {
                "thickness": 0.12,
                "camber": 0.0,
                "camber_pos": 0.4,
                "aoa": 0.0,
            }

            super().__init__(resolution, domain, **params)

        # -- geometry / mesh ------------------------------------------------

        def _airfoil_polygon(self, thickness, camber, camber_pos, aoa):
            poly = naca4_coords(
                thickness=thickness,
                camber=camber,
                camber_pos=camber_pos,
                n_points=self.params.get("_n_surface_points", 100),
            )
            return rotate_airfoil(poly, aoa, center=(0.25, 0.0))

        def create_mesh(self):
            poly = self._airfoil_polygon(**self._geo)
            x_rng = self.params.get("_x_range", (-1.0, 3.0))
            y_rng = self.params.get("_y_range", (-1.0, 1.0))
            return create_channel_with_polygon(
                poly,
                x_min=x_rng[0],
                x_max=x_rng[1],
                y_min=y_rng[0],
                y_max=y_rng[1],
                resolution=self.params.get("_mesh_resolution", 0.12),
                comm=MPI.COMM_WORLD,
            )

        def create_function_spaces(self):
            P2 = basix.ufl.element("Lagrange", self.mesh.basix_cell(), 2, shape=(2,))
            P1 = basix.ufl.element("Lagrange", self.mesh.basix_cell(), 1)
            self.W = fem.functionspace(self.mesh, basix.ufl.mixed_element([P2, P1]))

        def _rebuild_geometry(self, thickness, camber, camber_pos, aoa):
            """Re-mesh for a new airfoil (geometry IS the per-sample input)."""
            geo = {
                "thickness": thickness,
                "camber": camber,
                "camber_pos": camber_pos,
                "aoa": aoa,
            }
            if geo == self._geo and hasattr(self, "mesh"):
                return
            self._geo = geo
            self.mesh = self.create_mesh()
            self.comm = self.mesh.comm
            self.create_function_spaces()
            self._setup_output_grid()

        # -- solve ----------------------------------------------------------

        def solve(
            self,
            thickness=None,
            camber=None,
            camber_pos=None,
            aoa=None,
            inlet_scale=1.0,
            return_functions=False,
        ):
            g = self._geo
            self._rebuild_geometry(
                thickness if thickness is not None else g["thickness"],
                camber if camber is not None else g["camber"],
                camber_pos if camber_pos is not None else g["camber_pos"],
                aoa if aoa is not None else g["aoa"],
            )

            W = self.W
            V, _ = W.sub(0).collapse()
            facet_tags = self.mesh.facet_tags
            U_far = self.U * inlet_scale

            def freestream(x):
                values = np.zeros((2, x.shape[1]))
                values[0] = U_far
                return values

            free_func = fem.Function(V)
            free_func.interpolate(freestream)
            noslip_func = fem.Function(V)
            noslip_func.interpolate(lambda x: np.zeros((2, x.shape[1])))

            bcs = []
            # far-field velocity on inlet (1) and top/bottom walls (3)
            for tag in (1, 3):
                facets = facet_tags.find(tag)
                dofs = fem.locate_dofs_topological(
                    (W.sub(0), V), self.mesh.topology.dim - 1, facets
                )
                bcs.append(fem.dirichletbc(free_func, dofs, W.sub(0)))
            # no-slip on the airfoil (4)
            facets = facet_tags.find(4)
            dofs = fem.locate_dofs_topological(
                (W.sub(0), V), self.mesh.topology.dim - 1, facets
            )
            bcs.append(fem.dirichletbc(noslip_func, dofs, W.sub(0)))

            w = fem.Function(W)
            u, p = ufl.split(w)
            v, q = ufl.TestFunctions(W)

            F = (
                self.rho * inner(grad(u) * u, v) * dx
                + self.mu * inner(grad(u), grad(v)) * dx
                - p * div(v) * dx
                - q * div(u) * dx
            )

            problem = NonlinearProblem(
                F,
                w,
                bcs=bcs,
                petsc_options_prefix="naca_flow_",
                petsc_options={
                    "snes_type": "newtonls",
                    "snes_linesearch_type": "bt",
                    "snes_rtol": 1e-6,
                    "snes_max_it": 50,
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                    "pc_factor_mat_solver_type": "mumps",
                },
            )
            w = problem.solve()
            if problem.solver.getConvergedReason() <= 0:
                raise RuntimeError("Newton solver did not converge")

            self._last_forces = self._compute_forces(w, facet_tags, U_far)

            if return_functions:
                return w

            u_sol = w.sub(0).collapse()
            p_sol = w.sub(1).collapse()
            u_grid = self.interpolate_to_grid(u_sol, fill_value=0.0)
            p_grid = self.interpolate_to_grid(p_sol, fill_value=0.0)
            return np.stack([u_grid[:, :, 0], u_grid[:, :, 1], p_grid], axis=-1)

        def _compute_forces(self, w, facet_tags, U_far):
            """Lift/drag coefficients from the surface stress integral."""
            u, p = ufl.split(w)
            n = FacetNormal(self.mesh)
            ds_obs = ufl.Measure("ds", domain=self.mesh, subdomain_data=facet_tags)(4)
            # fluid stress; ds normal points OUT of the fluid = INTO the
            # airfoil, so the force on the airfoil is -sigma.n integrated.
            sigma = -p * Identity(2) + self.mu * (grad(u) + grad(u).T)
            traction = -sigma * n
            Fx = fem.assemble_scalar(fem.form(traction[0] * ds_obs))
            Fy = fem.assemble_scalar(fem.form(traction[1] * ds_obs))
            q_dyn = 0.5 * self.rho * U_far**2  # chord = 1
            return {
                "Fx": float(Fx),
                "Fy": float(Fy),
                "Cd": float(Fx / q_dyn),
                "Cl": float(Fy / q_dyn),
            }

        # -- sampling -------------------------------------------------------

        def generate_ic(self, generator="default", generator_params=None, seed=None):
            """Draw airfoil parameters + inlet scale (the sample's identity)."""
            if generator_params is None:
                generator_params = {}
            rng = np.random.default_rng(seed)
            p = self.params

            def draw(rng_key, default_key):
                lo, hi = generator_params.get(rng_key, p[default_key])
                return rng.uniform(lo, hi)

            thickness = draw("thickness_range", "thickness_range")
            camber = draw("camber_range", "camber_range")
            camber_pos = draw("camber_pos_range", "camber_pos_range")
            aoa = draw("aoa_range", "aoa_range")
            inlet_scale = draw("inlet_scale_range", "inlet_scale_range")
            return np.array([thickness, camber, camber_pos, aoa, inlet_scale])

        def sdf_input(self, thickness, camber, camber_pos, aoa):
            """The operator-learning input: airfoil SDF on the output grid."""
            poly = self._airfoil_polygon(thickness, camber, camber_pos, aoa)
            X, Y = np.meshgrid(self.grids["x"], self.grids["y"], indexing="ij")
            return polygon_sdf(X, Y, np.vstack([poly, poly[:1]]))

        def generate_sample(
            self,
            generator="default",
            generator_params=None,
            seed=None,
            validate=True,
            max_attempts=10,
        ):
            for attempt in range(max_attempts):
                current_seed = seed + attempt if seed is not None else None
                params_vec = self.generate_ic(
                    generator=generator,
                    generator_params=generator_params,
                    seed=current_seed,
                )
                thickness, camber, camber_pos, aoa, inlet_scale = params_vec
                try:
                    solution = self.solve(
                        thickness=thickness,
                        camber=camber,
                        camber_pos=camber_pos,
                        aoa=aoa,
                        inlet_scale=inlet_scale,
                    )
                    sdf = self.sdf_input(thickness, camber, camber_pos, aoa)
                    info = {
                        "valid": bool(np.isfinite(solution).all()),
                        "params": params_vec.tolist(),
                        **self._last_forces,
                    }
                    if not validate or info["valid"]:
                        return sdf, solution, info
                except Exception:
                    if attempt == max_attempts - 1:
                        raise
                    continue
            raise RuntimeError(
                f"Failed to generate valid sample after {max_attempts} attempts"
            )

        def generate_dataset(
            self,
            n_samples,
            ic_generator="default",
            ic_params=None,
            seed=None,
            validate=True,
            n_jobs=1,
            verbose=True,
        ):
            """Adds per-sample airfoil params and (Cl, Cd) to metadata."""
            self._sample_records = []
            orig_generate_sample = self.generate_sample

            def recording_generate_sample(*args, **kwargs):
                sdf, sol, info = orig_generate_sample(*args, **kwargs)
                self._sample_records.append(
                    {k: info[k] for k in ("params", "Cl", "Cd")}
                )
                return sdf, sol, info

            self.generate_sample = recording_generate_sample  # type: ignore
            try:
                dataset = super().generate_dataset(
                    n_samples=n_samples,
                    ic_generator=ic_generator,
                    ic_params=ic_params,
                    seed=seed,
                    validate=validate,
                    n_jobs=1,  # geometry rebuild is stateful: sequential
                    verbose=verbose,
                )
            finally:
                self.generate_sample = orig_generate_sample  # type: ignore

            recs = self._sample_records[-n_samples:]
            names = ["thickness", "camber", "camber_pos", "aoa_deg", "inlet_scale"]
            dataset.metadata["param_samples"] = {
                name: [r["params"][i] for r in recs] for i, name in enumerate(names)
            }
            dataset.metadata["Cl"] = [r["Cl"] for r in recs]
            dataset.metadata["Cd"] = [r["Cd"] for r in recs]
            return dataset

        def validate_solution(self, ic, solution, tol=1e-6):
            is_valid = (
                not np.isnan(solution).any()
                and not np.isinf(solution).any()
                and np.abs(solution).max() < 1e4
            )
            return {"valid": is_valid, "max_value": float(np.abs(solution).max())}
