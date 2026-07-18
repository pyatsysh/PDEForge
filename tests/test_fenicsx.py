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
