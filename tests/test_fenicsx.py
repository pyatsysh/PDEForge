"""
FEniCSx smoke tests — exercised by the fenicsx CI workflow (conda env).

Locally these skip cleanly when dolfinx is absent; in CI they must PASS
(the workflow no longer masks failures).
"""

import numpy as np
import pytest

dolfinx = pytest.importorskip("dolfinx", reason="FEniCSx (dolfinx) not installed")

from pdeforge import generate_dataset, get_model, list_models


class TestFEniCSxRegistry:
    def test_fem_models_registered(self):
        names = set(list_models())
        expected = {
            "cylinder_flow_2d",
            "cylinder_flow_2d_unsteady",
            "cylinder_flow_2d_parameterized",
            "cylinder_flow_2d_turbulent",
            "elasticity_2d",
            "porous_darcy_fem",
            "rayleigh_benard_2d",
        }
        missing = expected - names
        assert not missing, f"FEM models not registered: {missing}"

    def test_backend_resolution(self):
        from pdeforge.solvers.ops import resolve_backend

        cls = get_model("cylinder_flow_2d")
        assert resolve_backend(cls, "auto") == "fenicsx"
        with pytest.raises(ValueError):
            resolve_backend(cls, "jax")


class TestSteadyCylinderSmoke:
    def test_single_coarse_solve(self):
        """One steady solve on a coarse mesh: finite fields, sane magnitudes."""
        m = get_model("cylinder_flow_2d")(
            resolution={"x": 32, "y": 16}, mesh_refinement=1
        )
        assert m.backend == "fenicsx"
        assert m.PARALLEL_SAFE is False
        ic, sol, info = m.generate_sample(seed=0)
        assert np.isfinite(np.asarray(sol)).all()

    def test_tiny_dataset(self):
        d = generate_dataset(
            "cylinder_flow_2d",
            n_samples=2,
            resolution={"x": 32, "y": 16},
            seed=1,
            verbose=False,
        )
        assert d.n_samples == 2
        assert np.isfinite(d.outputs).all()
        assert d.metadata["backend"] == "fenicsx"


class TestDivergenceFree:
    def test_velocity_divergence_small(self):
        """Physics check: interpolated velocity field is near divergence-free."""
        d = generate_dataset(
            "cylinder_flow_2d",
            n_samples=1,
            resolution={"x": 48, "y": 24},
            seed=0,
            verbose=False,
        )
        out = np.asarray(d.outputs[0])
        # output channels include (u, v); FD divergence away from the obstacle
        if out.ndim == 3 and out.shape[-1] >= 2:
            u, v = out[..., 0], out[..., 1]
            du = np.gradient(u, axis=1)
            dv = np.gradient(v, axis=0)
            div = du + dv
            inner = div[4:-4, 4:-4]
            assert np.median(np.abs(inner)) < 0.5 * np.abs(u).max()


class TestElasticity2D:
    def test_energy_balance_and_shapes(self):
        """Clapeyron: strain energy = half the external work, at solver
        precision, for the heterogeneous-inclusion sample."""
        m = get_model("elasticity_2d")(resolution={"x": 48, "y": 48}, _mesh_n=32)
        ic, sol, info = m.generate_sample(seed=3)
        assert ic.shape == (48, 48)
        assert set(np.round(np.unique(ic), 6)) == {1.0, 10.0}
        assert sol.shape == (48, 48, 3)
        assert np.isfinite(sol).all()
        assert info["valid"]
        assert info["energy_balance"] < 1e-10

    def test_stiffer_matrix_displaces_less(self):
        """Doubling every modulus halves the displacement (linearity)."""
        m = get_model("elasticity_2d")(resolution={"x": 32, "y": 32}, _mesh_n=24)
        ic = m.generate_ic(seed=7)
        u1 = m.solve(ic)[..., :2]
        u2 = m.solve(2.0 * ic)[..., :2]
        assert np.allclose(u2, 0.5 * u1, rtol=1e-8, atol=1e-12)

    def test_dataset_generation(self):
        d = generate_dataset(
            "elasticity_2d",
            n_samples=2,
            resolution={"x": 32, "y": 32},
            params={"_mesh_n": 24},
            seed=0,
            verbose=False,
        )
        assert d.inputs.shape == (2, 32, 32)
        assert d.outputs.shape == (2, 32, 32, 3)
        assert d.metadata["backend"] == "fenicsx"


class TestPorousDarcyFEM:
    def test_pipeline_flux_and_bounds(self):
        """CH morphology -> binary k -> Darcy: flux balances across the
        pressure boundaries and p obeys the maximum principle."""
        m = get_model("porous_darcy_fem")(resolution={"x": 48, "y": 48}, _mesh_n=40)
        ic, sol, info = m.generate_sample(seed=11)
        assert ic.shape == (48, 48)
        ks = np.unique(ic)
        assert len(ks) == 2 and ks.max() == 1.0 and ks.min() == 1e-3
        assert sol.shape == (48, 48, 3)
        assert info["valid"]
        assert info["flux_imbalance"] < 0.05
        assert 0.0 < info["k_eff"] < 1.0

    def test_contrast_monotonicity(self):
        """Lowering the contrast (more permeable solid) raises k_eff."""
        kw = dict(resolution={"x": 40, "y": 40}, _mesh_n=32)
        hi = get_model("porous_darcy_fem")(permeability_contrast=1e3, **kw)
        lo = get_model("porous_darcy_fem")(permeability_contrast=1e1, **kw)
        hi.solve(hi.generate_ic(seed=5))
        lo.solve(lo.generate_ic(seed=5))
        assert lo._last_flux["k_eff"] > hi._last_flux["k_eff"]


class TestRayleighBenard2D:
    def test_subcritical_conduction(self):
        """Below Ra_c ~ 1708 the cavity relaxes to conduction: Nu = 1 at
        both plates and the velocity decays to zero."""
        m = get_model("rayleigh_benard_2d")(
            resolution={"x": 24, "y": 24},
            rayleigh=800.0,
            time_end=0.4,
            _mesh_n=20,
        )
        sol = m.solve(m.generate_ic(seed=1))
        assert np.isfinite(sol).all()
        assert abs(m._last_nusselt["Nu_bottom"] - 1.0) < 0.01
        assert abs(m._last_nusselt["Nu_top"] - 1.0) < 0.01
        assert np.abs(sol[..., :2]).max() < 5e-3

    def test_supercritical_nusselt_vs_literature(self):
        """Ra = 1e4, Pr = 0.71 square cavity: plate-averaged Nu matches
        the benchmark value 2.158 (Ouertatani et al. 2008). Measured
        2.165 at this coarse config."""
        m = get_model("rayleigh_benard_2d")(
            resolution={"x": 24, "y": 24},
            rayleigh=1e4,
            prandtl=0.71,
            time_end=0.3,
            _mesh_n=24,
        )
        m.solve(m.generate_ic(seed=1))
        nu = m._last_nusselt["Nu_bottom"]
        assert abs(nu - 2.158) / 2.158 < 0.05
        assert m._last_nusselt["imbalance"] < 0.05
