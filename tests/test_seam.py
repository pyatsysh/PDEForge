"""
Tests for the semi-linear spectral seam: ETDRK4 correctness, exact linear
propagation, model physics, and numpy/jax cross-backend consistency.
"""

import numpy as np
import pytest

from pdeforge import generate_dataset, get_model
from pdeforge.solvers.semilinear import etdrk4_coeffs


class TestETDRK4Coeffs:
    def test_scalar_exact_decay(self):
        """For pure linear problems the E coefficient is the exact propagator."""
        lam, dt = -37.5, 0.01
        C = etdrk4_coeffs(np.array([lam]), dt)
        assert np.allclose(C["E"][0], np.exp(lam * dt), rtol=1e-12)

    def test_small_and_large_z_stable(self):
        """Contour evaluation must not blow up for tiny or huge |L dt|."""
        L = np.array([-1e-12, -1.0, -1e6])
        C = etdrk4_coeffs(L, 0.01)
        for key in ("Q", "f1", "f2", "f3"):
            assert np.all(np.isfinite(C[key]))

    def test_complex_symbol(self):
        """Dispersive (imaginary) symbols are supported."""
        L = 1j * np.linspace(-100, 100, 7)
        C = etdrk4_coeffs(L, 0.01)
        # |exp(i w dt)| = 1: energy-preserving propagator
        assert np.allclose(np.abs(C["E"]), 1.0, rtol=1e-12)


class TestHeatExact:
    """Heat models on the seam must match the analytic Fourier solution."""

    def test_heat_1d_exact_mode_decay(self):
        m = get_model("heat_1d")(resolution={"x": 128}, diffusivity=0.05, time_end=0.7)
        x = m.grids["x"]
        k0 = 2 * np.pi * 3  # mode 3 on the unit domain
        ic = np.sin(k0 * x)
        u = m.solve(ic)
        expected = np.exp(-0.05 * k0**2 * 0.7) * np.sin(k0 * x)
        assert np.allclose(u, expected, atol=1e-10)

    def test_heat_2d_exact_mode_decay(self):
        m = get_model("heat_2d")(
            resolution={"x": 32, "y": 32}, diffusivity=0.02, time_end=0.5
        )
        # field_shape is (ny, nx); build the mode on the meshgrid
        X, Y = np.meshgrid(m.grids["x"], m.grids["y"])
        kx, ky = 2 * np.pi * 2, 2 * np.pi * 5
        ic = np.sin(kx * X) * np.cos(ky * Y)
        u = m.solve(ic)
        expected = np.exp(-0.02 * (kx**2 + ky**2) * 0.5) * ic
        assert np.allclose(u, expected, atol=1e-10)


class TestBurgersSeam:
    def test_temporal_convergence(self):
        """Halving dt should barely change the ETDRK4 solution."""
        m1 = get_model("burgers_1d")(resolution={"x": 256})
        ic = m1.generate_ic(seed=5)
        u1 = m1.solve(ic)
        m2 = get_model("burgers_1d")(resolution={"x": 256}, _dt=m1.dt / 2)
        u2 = m2.solve(ic)
        rel = np.max(np.abs(u1 - u2)) / np.max(np.abs(u2))
        assert rel < 1e-7

    def test_no_warnings_on_stiff_params(self, recwarn):
        """The old odeint 'excess work' regime must be warning-free now."""
        d = generate_dataset(
            "burgers_1d",
            n_samples=2,
            resolution={"x": 128},
            params={"viscosity": 0.1, "time_horizon": 0.5},
            seed=42,
            verbose=False,
        )
        assert np.isfinite(d.outputs).all()
        runtime = [w for w in recwarn.list if issubclass(w.category, RuntimeWarning)]
        assert not runtime

    def test_trajectory_frames(self):
        m = get_model("burgers_1d")(resolution={"x": 64}, _n_time_steps=11)
        U = m.solve(m.generate_ic(seed=0), return_full=True)
        assert U.shape == (11, 64)
        # frame 0 is the IC, last frame the final state; they must differ
        assert not np.allclose(U[0], U[-1])


class TestAllenCahnSeam:
    def test_bounded_and_phase_separating(self):
        m = get_model("allen_cahn_1d")(resolution={"x": 128}, time_end=5.0)
        u = m.solve(m.generate_ic(seed=2))
        # Solutions approach the wells at +-1 without overshooting far
        assert np.abs(u).max() < 1.5
        assert np.abs(u).max() > 0.8

    def test_reproducible(self):
        kw = dict(n_samples=3, resolution={"x": 64}, seed=11, verbose=False)
        a = generate_dataset("allen_cahn_1d", **kw)
        b = generate_dataset("allen_cahn_1d", **kw)
        assert np.array_equal(a.outputs, b.outputs)


class TestJaxBackend:
    """Cross-backend consistency: same spec, two engines."""

    @pytest.fixture(autouse=True)
    def _need_jax(self):
        pytest.importorskip("jax")

    @pytest.mark.parametrize("model", ["heat_1d", "burgers_1d", "allen_cahn_1d"])
    def test_numpy_vs_jax_single(self, model):
        m_np = get_model(model)(resolution={"x": 64})
        ic = m_np.generate_ic(seed=7)
        u_np = m_np.solve(ic)
        m_jx = get_model(model)(resolution={"x": 64}, backend="jax")
        u_jx = m_jx.solve(ic)
        assert u_jx.dtype == np.float64 or np.isrealobj(u_jx)
        assert np.allclose(u_np, u_jx, rtol=1e-8, atol=1e-10)

    def test_numpy_vs_jax_dataset(self):
        kw = dict(n_samples=4, resolution={"x": 64}, seed=3, verbose=False)
        d_np = generate_dataset("burgers_1d", backend="numpy", **kw)
        d_jx = generate_dataset("burgers_1d", backend="jax", **kw)
        # ICs are ALWAYS numpy-generated: bit-identical inputs across backends
        assert np.array_equal(d_np.inputs, d_jx.inputs)
        assert np.allclose(d_np.outputs, d_jx.outputs, rtol=1e-8, atol=1e-10)
        assert d_jx.metadata["backend"] == "jax"
        assert isinstance(d_jx.outputs, np.ndarray)

    def test_heat_2d_jax(self):
        m = get_model("heat_2d")(resolution={"x": 32, "y": 32}, backend="jax")
        X, Y = np.meshgrid(m.grids["x"], m.grids["y"])
        ic = np.sin(2 * np.pi * 3 * X)
        u = m.solve(ic)
        expected = np.exp(-m.alpha * (2 * np.pi * 3) ** 2 * m.T) * ic
        assert np.allclose(u, expected, atol=1e-9)

    def test_fem_backend_guard(self):
        from pdeforge.solvers.ops import resolve_backend

        class FakeFEM:
            BACKENDS = {"fenicsx"}

        assert resolve_backend(FakeFEM, "auto") == "fenicsx"
        with pytest.raises(ValueError):
            resolve_backend(FakeFEM, "jax")


class TestBackendResolution:
    def test_auto_is_numpy_for_spectral(self):
        from pdeforge.solvers.ops import resolve_backend

        assert resolve_backend(get_model("burgers_1d"), "auto") == "numpy"

    def test_jax_on_unsupported_model_errors(self):
        from pdeforge.solvers.ops import resolve_backend

        with pytest.raises(ValueError):
            resolve_backend(get_model("cahn_hilliard"), "jax")
