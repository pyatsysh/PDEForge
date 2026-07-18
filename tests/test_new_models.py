"""
Physics-validation tests for the 2026 catalogue expansion.

Every model gets at least one test that checks PHYSICS (an exact solution,
a conservation law, a convergence property, or a known regime), not just
shapes — the sum-rule ethos.
"""

import numpy as np
import pytest

from pdeforge import generate_dataset, get_model


class TestAdvection1D:
    def test_exact_translation(self):
        """The solution IS the band-limited translated IC — machine precision."""
        m = get_model("advection_1d")(resolution={"x": 128}, speed=1.3, time_end=0.37)
        ic = m.generate_ic(seed=4)
        u = m.solve(ic)
        assert np.allclose(u, m.exact_solution(ic), atol=1e-11)

    def test_norm_preserved(self):
        m = get_model("advection_1d")(resolution={"x": 64})
        ic = m.generate_ic(seed=1)
        u = m.solve(ic)
        assert np.isclose(np.linalg.norm(u), np.linalg.norm(ic), rtol=1e-10)


class TestNSVorticity2D:
    def test_taylor_green_exact_decay(self):
        """Taylor-Green vortex: w(t) = w0 exp(-2 nu t) exactly (on [0,2pi]^2)."""
        nu, T = 0.01, 1.0
        m = get_model("ns_vorticity_2d")(
            resolution={"x": 48, "y": 48},
            domain={"x": (0, 2 * np.pi), "y": (0, 2 * np.pi)},
            viscosity=nu,
            time_horizon=T,
        )
        X, Y = np.meshgrid(m.grids["x"], m.grids["y"])
        w0 = 2.0 * np.cos(X) * np.cos(Y)
        w = m.solve(w0)
        expected = np.exp(-2.0 * nu * T) * w0
        assert np.allclose(w, expected, atol=2e-4)

    def test_unforced_energy_decays(self):
        m = get_model("ns_vorticity_2d")(
            resolution={"x": 32, "y": 32}, viscosity=1e-2, time_horizon=1.0
        )
        w0 = m.generate_ic(seed=3)
        w = m.solve(w0)
        assert np.sum(w**2) < np.sum(w0**2)

    def test_dataset_generation(self):
        d = generate_dataset(
            "ns_vorticity_2d",
            n_samples=2,
            resolution={"x": 32, "y": 32},
            seed=0,
            verbose=False,
        )
        assert d.outputs.shape == (2, 32, 32)
        assert np.isfinite(d.outputs).all()


class TestKS1D:
    def test_dt_convergence(self):
        m1 = get_model("ks_1d")(resolution={"x": 128}, time_end=5.0, _dt=0.25)
        # NOTE: KS param is time_horizon; time_end is silently absorbed into
        # params — use the documented name.
        m1 = get_model("ks_1d")(resolution={"x": 128}, time_horizon=5.0, _dt=0.25)
        ic = m1.generate_ic(seed=2)
        u1 = m1.solve(ic)
        m2 = get_model("ks_1d")(resolution={"x": 128}, time_horizon=5.0, _dt=0.125)
        u2 = m2.solve(ic)
        rel = np.max(np.abs(u1 - u2)) / (np.max(np.abs(u2)) + 1e-12)
        assert rel < 1e-3  # 4th-order in the pre-chaotic window

    def test_chaotic_but_bounded(self):
        m = get_model("ks_1d")(resolution={"x": 128}, time_horizon=60.0)
        u = m.solve(m.generate_ic(seed=7))
        assert np.abs(u).max() < 10.0  # bounded attractor
        assert np.std(u) > 0.3  # ...but genuinely active


class TestGrayScott2D:
    def test_trivial_state_is_steady(self):
        """(U, V) = (1, 0) is an exact fixed point of the dynamics."""
        m = get_model("gray_scott_2d")(resolution={"x": 32, "y": 32}, time_end=200.0)
        ic = np.stack([np.ones((32, 32)), np.zeros((32, 32))], axis=0)
        out = m.solve(ic)
        assert np.allclose(out[0], 1.0, atol=1e-8)
        assert np.allclose(out[1], 0.0, atol=1e-8)

    def test_patterns_emerge(self):
        m = get_model("gray_scott_2d")(resolution={"x": 48, "y": 48}, time_end=1500.0)
        out = m.solve(m.generate_ic(seed=1))
        # V develops structure (nontrivial variance), U departs from 1
        assert np.std(out[1]) > 1e-3
        assert np.std(out[0]) > 1e-3
        assert np.abs(out).max() < 2.0


class TestKdV1D:
    def test_soliton_propagation(self):
        """A single soliton translates at speed c with preserved shape."""
        m = get_model("kdv_1d")(resolution={"x": 512}, time_end=0.5, _dt=1e-4)
        c = 4.0
        L = m.domain.size("x")
        ic = m.soliton(c, x0=L / 2)
        u = m.solve(ic)
        expected = m.soliton(c, x0=L / 2 + c * 0.5)
        rel = np.max(np.abs(u - expected)) / np.max(np.abs(expected))
        assert rel < 5e-3

    def test_mass_conserved(self):
        m = get_model("kdv_1d")(resolution={"x": 256}, time_end=0.2, _dt=2e-4)
        ic = m.generate_ic(seed=3)
        u = m.solve(ic)
        assert np.isclose(np.mean(u), np.mean(ic), atol=1e-10)


class TestKolmogorov2D:
    def test_energy_bounded_and_sustained(self):
        m = get_model("kolmogorov_flow_2d")(
            resolution={"x": 48, "y": 48}, time_horizon=8.0
        )
        w = m.solve(m.generate_ic(seed=5))
        E = float(np.mean(w**2))
        assert np.isfinite(E)
        assert 1e-3 < E < 1e4  # forced-dissipative: neither dead nor blown up


class TestLotkaVolterra2D:
    def test_uniform_ic_matches_ode(self):
        """Spatially uniform IC must follow the classic LV ODE exactly."""
        from scipy.integrate import solve_ivp

        m = get_model("lotka_volterra_2d")(
            resolution={"x": 16, "y": 16}, time_end=2.0, _dt=0.002
        )
        u0, v0 = 1.3, 0.7
        ic = np.stack([np.full((16, 16), u0), np.full((16, 16), v0)], axis=0)
        out = m.solve(ic)

        def rhs(t, y):
            return [y[0] * (m.a - m.b * y[1]), y[1] * (-m.c + m.d * y[0])]

        ref = solve_ivp(rhs, (0, 2.0), [u0, v0], rtol=1e-10, atol=1e-12)
        assert np.allclose(out[0], ref.y[0, -1], rtol=1e-3)
        assert np.allclose(out[1], ref.y[1, -1], rtol=1e-3)
        # and the field stays spatially uniform
        assert np.std(out[0]) < 1e-8


class TestBurgers2D:
    def test_dt_convergence(self):
        m1 = get_model("burgers_2d")(resolution={"x": 48, "y": 48})
        ic = m1.generate_ic(seed=1)
        u1 = m1.solve(ic)
        m2 = get_model("burgers_2d")(resolution={"x": 48, "y": 48}, _dt=m1.dt / 2)
        u2 = m2.solve(ic)
        rel = np.max(np.abs(u1 - u2)) / (np.max(np.abs(u2)) + 1e-12)
        assert rel < 1e-6

    def test_energy_decays(self):
        m = get_model("burgers_2d")(resolution={"x": 32, "y": 32})
        ic = m.generate_ic(seed=2)
        out = m.solve(ic)
        assert np.sum(out**2) < np.sum(ic**2)


class TestShallowWater2D:
    def test_mass_conserved_machine_precision(self):
        m = get_model("shallow_water_2d")(resolution={"x": 48, "y": 48})
        ic = m.generate_ic(seed=1)
        out = m.solve(ic)
        assert np.isclose(np.mean(out[0]), np.mean(ic[0]), rtol=1e-12)

    def test_gravity_wave_speed(self):
        """Small-amplitude standing wave oscillates at omega = sqrt(gH)|k|."""
        g, H = 9.81, 1.0
        m = get_model("shallow_water_2d")(
            resolution={"x": 64, "y": 4},
            gravity=g,
            mean_depth=H,
            time_end=0.5,
            _n_time_steps=201,
        )
        X = np.meshgrid(m.grids["x"], m.grids["y"])[0]
        kx = 2 * np.pi  # mode 1 on unit domain
        eps = 1e-4 * H
        ic = np.stack(
            [H + eps * np.cos(kx * X), np.zeros_like(X), np.zeros_like(X)], axis=0
        )
        traj = m.solve(ic, return_full=True)  # (n_t, 3, ny, nx)
        amp = traj[:, 0, 0, :] - H  # surface elevation along x at y=0
        # project onto the cos(kx x) mode over time
        proj = amp @ np.cos(kx * m.grids["x"]) / (0.5 * len(m.grids["x"]))
        t = np.linspace(0, m.T, m.n_t)
        # count the first zero crossing: quarter period T/4 = pi/(2 omega)
        omega_theory = np.sqrt(g * H) * kx
        crossings = np.where(np.diff(np.sign(proj)))[0]
        assert len(crossings) > 0
        t_quarter = t[crossings[0]]
        omega_measured = np.pi / (2 * t_quarter)
        assert abs(omega_measured - omega_theory) / omega_theory < 0.1


class TestSchrodinger1D:
    def test_norm_conserved_machine_precision(self):
        m = get_model("schrodinger_1d")(resolution={"x": 256}, time_end=0.5)
        ic = m.generate_ic(seed=3)
        out = m.solve(ic)
        assert np.isclose(m.norm(out), m.norm(ic), rtol=1e-12)

    def test_bright_soliton_shape_preserved(self):
        """|psi| of the g=-1 bright soliton is time-invariant.

        Domain (-20, 20): the sech tails are ~1e-9 at the periodic boundary,
        so periodicity does not pollute the comparison.
        """
        m = get_model("schrodinger_1d")(
            resolution={"x": 512},
            domain={"x": (-20.0, 20.0)},
            g=-1.0,
            time_end=1.0,
            _dt=5e-4,
        )
        ic = m.bright_soliton(a=1.0, v=0.0, x0=0.0)
        out = m.solve(ic)
        amp_in = np.abs(ic[0] + 1j * ic[1])
        amp_out = np.abs(out[0] + 1j * out[1])
        assert np.allclose(amp_out, amp_in, atol=1e-6)


class TestHeterogeneousWave2D:
    def test_homogeneous_medium_second_order_to_exact(self):
        """With c = const the exact propagator is known (u_hat cos(c|k|t));
        the leapfrog solution must converge to it at clean 2nd order."""
        c0, T = 1.0, 0.25
        medium = np.full((48, 48), c0)

        def err(dt_div):
            m = get_model("heterogeneous_wave_2d")(
                resolution={"x": 48, "y": 48},
                c_min=c0,
                c_max=c0,
                time_end=T,
                _dt=None if dt_div == 1 else 0.00469 / dt_div,
            )
            u = m.solve(medium)
            pulse_hat = np.fft.fft2(m._pulse)
            omega = c0 * np.sqrt(m.K2)
            exact = np.fft.ifft2(pulse_hat * np.cos(omega * T)).real
            return np.max(np.abs(u - exact)) / np.max(np.abs(exact))

        e1, e4 = err(1), err(4)
        assert e1 < 1e-2  # accurate at the default CFL step
        ratio = e1 / e4
        assert 10.0 < ratio < 22.0  # O(dt^2): ratio ~ 16

    def test_dataset_medium_to_field(self):
        d = generate_dataset(
            "heterogeneous_wave_2d",
            n_samples=2,
            resolution={"x": 32, "y": 32},
            seed=0,
            verbose=False,
        )
        assert d.inputs.shape == (2, 32, 32)  # the medium
        assert d.outputs.shape == (2, 32, 32)  # the wavefield
        assert not np.allclose(d.outputs[0], d.outputs[1])  # medium matters


class TestHelmholtz2D:
    def test_operator_residual_machine_precision(self):
        """Apply (Lap + kappa^2 + i gamma kappa) to the solution: must give f
        back to machine precision (non-circular operator-level check)."""
        m = get_model("helmholtz_2d")(
            resolution={"x": 32, "y": 32}, wavenumber=15.0, damping=1.0
        )
        f = m.generate_ic(seed=8)
        u = m.solve_complex(f)
        lap_u = np.fft.ifft2(-m.K2 * np.fft.fft2(u))
        residual = lap_u + (m.kappa**2 + 1j * m.gamma * m.kappa) * u - f
        assert np.max(np.abs(residual)) < 1e-9

    def test_steady_flag(self):
        with pytest.raises(ValueError, match="steady"):
            generate_dataset(
                "helmholtz_2d",
                n_samples=1,
                resolution={"x": 16, "y": 16},
                verbose=False,
                outputs="trajectory",
            )


class Test3DModels:
    def test_heat_3d_exact_decay(self):
        m = get_model("heat_3d")(
            resolution={"x": 16, "y": 16, "z": 16}, diffusivity=0.05, time_end=0.4
        )
        # single mode along z: field_shape is (z, y, x) reverse-sorted
        z = m.grids["z"]
        kz = 2 * np.pi * 2
        ic = np.tile(np.sin(kz * z)[:, None, None], (1, 16, 16))
        u = m.solve(ic)
        expected = np.exp(-0.05 * kz**2 * 0.4) * ic
        assert np.allclose(u, expected, atol=1e-10)

    def test_allen_cahn_3d_bounded(self):
        m = get_model("allen_cahn_3d")(
            resolution={"x": 16, "y": 16, "z": 16}, time_end=2.0
        )
        u = m.solve(m.generate_ic(seed=1))
        assert np.abs(u).max() < 1.5


class TestStochasticBurgers1D:
    def test_zero_noise_matches_deterministic_step_scheme(self):
        """sigma=0 must be reproducible and finite (deterministic limit)."""
        m = get_model("stochastic_burgers_1d")(
            resolution={"x": 128}, noise_intensity=0.0, n_realizations=2
        )
        ic = m.generate_ic(seed=5)
        out = m.solve(ic, seed=1)
        # all realizations identical when sigma = 0
        assert np.allclose(out[0], out[1], atol=1e-14)
        assert np.isfinite(out).all()

    def test_variance_grows_with_sigma(self):
        outs = {}
        for sigma in (0.05, 0.2):
            m = get_model("stochastic_burgers_1d")(
                resolution={"x": 64}, noise_intensity=sigma, n_realizations=8
            )
            ic = m.generate_ic(seed=2)
            out = m.solve(ic, seed=3)
            outs[sigma] = float(np.mean(np.var(out, axis=0)))
        assert outs[0.2] > outs[0.05]

    def test_dataset_shape(self):
        d = generate_dataset(
            "stochastic_burgers_1d",
            n_samples=2,
            resolution={"x": 64},
            params={"n_realizations": 4},
            seed=0,
            verbose=False,
        )
        assert d.outputs.shape == (2, 4, 64)


class TestStochasticAllenCahn2D:
    def test_zero_noise_deterministic(self):
        m = get_model("stochastic_allen_cahn_2d")(
            resolution={"x": 32, "y": 32}, noise_intensity=0.0, n_realizations=2
        )
        ic = m.generate_ic(seed=1)
        out = m.solve(ic, seed=0)
        assert np.allclose(out[0], out[1], atol=1e-14)
        assert np.abs(out).max() < 1.5

    def test_noise_roughens(self):
        base = get_model("stochastic_allen_cahn_2d")(
            resolution={"x": 32, "y": 32}, noise_intensity=0.0, n_realizations=1
        )
        noisy = get_model("stochastic_allen_cahn_2d")(
            resolution={"x": 32, "y": 32}, noise_intensity=0.3, n_realizations=1
        )
        ic = base.generate_ic(seed=4)
        u0 = base.solve(ic, seed=1)[0]
        u1 = noisy.solve(ic, seed=1)[0]
        assert not np.allclose(u0, u1)


class TestRegistryCount:
    def test_all_new_models_registered(self):
        from pdeforge import list_models

        names = set(list_models())
        expected = {
            "ns_vorticity_2d",
            "kolmogorov_flow_2d",
            "ks_1d",
            "gray_scott_2d",
            "advection_1d",
            "kdv_1d",
            "lotka_volterra_2d",
            "burgers_2d",
            "shallow_water_2d",
            "schrodinger_1d",
            "heterogeneous_wave_2d",
            "helmholtz_2d",
            "heat_3d",
            "allen_cahn_3d",
            "stochastic_burgers_1d",
            "stochastic_allen_cahn_2d",
        }
        missing = expected - names
        assert not missing, f"unregistered models: {missing}"
