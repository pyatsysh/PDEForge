"""
Canonical-recreation tests — "the canon, with knobs."

Fidelity is tested at three levels: (1) solver correctness (operator
residual, exact solutions), (2) input-measure statistics, and (3) where a
distributed copy of the original data exists on this machine, the decisive
shootout: THEIR stored inputs through OUR solver vs THEIR stored outputs.
Level-3 tests skip cleanly on machines without the data.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from pdeforge import generate_dataset, get_model

# Locally held copies of distributed reference data (not shipped with the
# repo): point these env vars at the files to enable the shootout tests.
DARCY421 = Path(os.environ.get("PDEFORGE_DARCY421_NPZ", "/nonexistent"))
BURGERS_REF = Path(os.environ.get("PDEFORGE_BURGERS_REF_NPZ", "/nonexistent"))


class TestDarcyFNOSolver:
    def test_operator_residual_machine_precision(self):
        m = get_model("darcy_fno_2d")(resolution={"x": 65, "y": 65})
        a = m.generate_ic(seed=0)
        u = m.solve(a)
        res = m.apply_operator(a, u)
        assert np.abs(res - m.f_const).max() < 1e-9

    def test_constant_coefficient_vs_exact_series(self):
        """a = 1: -Lap u = 1 on the unit square has a classical series
        solution; the FD solve must match it to O(h^2)."""
        n = 85
        u = get_model("darcy_fno_2d")(resolution={"x": n, "y": n}).solve(
            np.ones((n, n))
        )
        x = np.linspace(0, 1, n)
        X, Y = np.meshgrid(x, x)
        S = np.zeros((n, n))
        for i in range(1, 60, 2):
            for j in range(1, 60, 2):
                S += (
                    (16 / np.pi**4)
                    * np.sin(i * np.pi * X)
                    * np.sin(j * np.pi * Y)
                    / (i * j * (i**2 + j**2))
                )
        assert np.linalg.norm(u - S) / np.linalg.norm(S) < 5e-4

    def test_dirichlet_boundary_exact_zero(self):
        m = get_model("darcy_fno_2d")(resolution={"x": 33, "y": 33})
        u = m.solve(m.generate_ic(seed=1))
        assert np.abs(u[0, :]).max() == 0.0
        assert np.abs(u[:, -1]).max() == 0.0


class TestCanonicalMeasures:
    def test_lognormal_family_stats(self):
        """Fixed-constant calibration: per-sample std FLUCTUATES around
        sigma (a per-sample rescale would condition the measure)."""
        m = get_model("darcy_fno_2d")(resolution={"x": 129, "y": 129})
        stds = [np.log(m.generate_ic(seed=s)).std() for s in range(30)]
        assert 0.2 < np.mean(stds) < 0.4  # around sigma = 0.2918
        assert np.std(stds) > 0.03  # genuine measure fluctuation

    def test_piececonst_values_and_fraction(self):
        m = get_model("darcy_fno_2d")(
            resolution={"x": 129, "y": 129}, coeff="piececonst"
        )
        vals = [m.generate_ic(seed=s) for s in range(10)]
        assert set(np.unique(vals[0])) == {3.0, 12.0}
        frac = np.mean([(v == 12.0).mean() for v in vals])
        assert 0.35 < frac < 0.65  # threshold at 0 -> ~half/half

    def test_grf_periodic_variance_analytic(self):
        from pdeforge.generators.initial_conditions import GRFPeriodicGenerator

        g = GRFPeriodicGenerator()  # canonical tau=5, alpha=2, scale=625
        n = 1024
        samples = np.stack([g.generate((n,), seed=s) for s in range(400)])
        assert np.isclose(
            samples.var(), g.expected_variance(n), rtol=0.15
        )  # variance-of-variance is large for strongly correlated fields

    def test_knobs_change_the_measure(self):
        """The point of the exercise: hyperparameters are live knobs."""

        # rougher field has more relative high-frequency content
        def hf_fraction(a):
            F = np.abs(np.fft.fft2(np.log(a))) ** 2
            k = np.fft.fftfreq(65) * 65
            KX, KY = np.meshgrid(k, k)
            hf = np.sqrt(KX**2 + KY**2) > 10
            return F[hf].sum() / F.sum()

        def mean_hf(alpha, tau):
            m = get_model("darcy_fno_2d")(
                resolution={"x": 65, "y": 65}, alpha=alpha, tau=tau
            )
            return np.mean([hf_fraction(m.generate_ic(seed=s)) for s in range(5)])

        rough = mean_hf(alpha=1.2, tau=12.0)  # measured ~0.31
        smooth = mean_hf(alpha=4.0, tau=1.0)  # measured ~0.043
        assert rough > 4 * smooth


class TestCanonicalPresets:
    def test_presets_registered(self):
        from pdeforge import list_presets

        names = set(list_presets())
        assert {
            "fno_darcy_2d",
            "fno_darcy_piececonst_2d",
            "fno_burgers_1d",
            "fno_burgers_grf_1d",
            "fno_ns_vorticity_2d",
            "burgers_smooth_1d",
            "burgers_canonical_1d",
            "burgers_rough_1d",
        } <= names

    def test_darcy_preset_any_resolution(self):
        d = generate_dataset(
            preset="fno_darcy_2d",
            n_samples=2,
            resolution={"x": 61, "y": 61},
            seed=0,
            verbose=False,
        )
        assert d.outputs.shape == (2, 61, 61)
        assert np.isfinite(d.outputs).all()

    def test_burgers_grf_preset(self):
        d = generate_dataset(
            preset="fno_burgers_grf_1d",
            n_samples=2,
            resolution={"x": 256},
            seed=0,
            verbose=False,
        )
        assert np.isfinite(d.outputs).all()
        # the GRF measure has O(1) pointwise std (canonical ~1.16)
        assert 0.3 < d.inputs.std() < 3.0


@pytest.mark.skipif(
    not DARCY421.exists(),
    reason="Darcy421 data not available (set PDEFORGE_DARCY421_NPZ)",
)
class TestDarcy421Shootout:
    def test_their_inputs_reproduce_their_outputs(self):
        d = np.load(DARCY421)
        m = get_model("darcy_fno_2d")(resolution={"x": 421, "y": 421})
        rels = []
        for k in range(2):
            a = d["x"][k].astype(np.float64)
            u_ours = m.solve(a)
            u_theirs = d["y"][k].astype(np.float64)
            rels.append(np.linalg.norm(u_ours - u_theirs) / np.linalg.norm(u_theirs))
        # 0.49% measured: two consistent discretisations of one continuum
        # problem (their stored data has ~0.8% boundary residue from their
        # own pipeline, bounding what is achievable).
        assert max(rels) < 2e-2

    def test_their_measure_spectrum(self):
        """Spectrum fit of THEIR coefficients recovers alpha=2, tau=3."""
        from scipy.fft import dctn

        d = np.load(DARCY421)
        psi = np.log(d["x"][:60].astype(np.float64))
        C = dctn(psi, axes=(-2, -1), norm="ortho")
        P = (C**2).mean(axis=0)
        n = psi.shape[-1]
        i = np.arange(n)
        I, J = np.meshgrid(i, i, indexing="ij")
        K2 = np.pi**2 * (I**2 + J**2)
        mask = (I + J > 0) & (I < 40) & (J < 40)
        x, y = K2[mask], np.log(P[mask])
        best = None
        for tau in np.linspace(1.0, 8.0, 120):
            X = np.log(x + tau**2)
            A = np.stack([np.ones_like(X), -X], axis=1)
            coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
            r2 = 1 - (res[0] / (len(y) * y.var()) if len(res) else 0)
            if best is None or r2 > best[2]:
                best = (tau, coef[1], r2)
        tau, alpha, r2 = best
        assert abs(alpha - 2.0) < 0.1
        assert abs(tau - 3.0) < 0.3
        assert r2 > 0.99


@pytest.mark.skipif(
    not BURGERS_REF.exists(),
    reason="Burgers reference data not available (set PDEFORGE_BURGERS_REF_NPZ)",
)
class TestBurgersReferenceShootout:
    def test_reproduce_reference_solutions(self):
        d = np.load(BURGERS_REF)
        m = get_model("burgers_1d")(
            resolution={"x": 2048},
            viscosity=0.1 / np.pi,
            time_horizon=1.0,
            _dt=1e-3,
        )
        for k in range(2):
            u1 = m.solve(d["test_u0"][k].astype(np.float64))
            rel = np.linalg.norm(u1 - d["test_u1"][k]) / np.linalg.norm(d["test_u1"][k])
            assert rel < 1e-6  # measured 2.7e-8


class TestDarcyFNO3D:
    """The canonical measure extended to 3D — dimension as a knob."""

    def test_operator_residual_machine_precision(self):
        m = get_model("darcy_fno_3d")(resolution={"x": 17, "y": 17, "z": 17})
        a = m.generate_ic(seed=0)
        u = m.solve(a)
        res = m.apply_operator(a, u)
        assert np.abs(res - m.f_const).max() < 1e-9

    def test_constant_coefficient_vs_exact_series(self):
        """a = 1: -Lap u = 1 on the unit cube, triple-sine series solution."""
        n = 21
        m = get_model("darcy_fno_3d")(resolution={"x": n, "y": n, "z": n})
        u = m.solve(np.ones((n, n, n)))
        x = np.linspace(0, 1, n)
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        S = np.zeros((n, n, n))
        for i in range(1, 14, 2):
            for j in range(1, 14, 2):
                for k in range(1, 14, 2):
                    S += (
                        64.0
                        / (np.pi**5 * i * j * k * (i**2 + j**2 + k**2))
                        * np.sin(i * np.pi * Z)
                        * np.sin(j * np.pi * Y)
                        * np.sin(k * np.pi * X)
                    )
        assert np.linalg.norm(u - S) / np.linalg.norm(S) < 1e-2

    def test_cg_matches_direct(self):
        """The large-grid CG path agrees with direct LU on a small grid."""
        n = 15
        m_direct = get_model("darcy_fno_3d")(resolution={"x": n, "y": n, "z": n})
        m_cg = get_model("darcy_fno_3d")(
            resolution={"x": n, "y": n, "z": n}, _direct_max_unknowns=0
        )
        a = m_direct.generate_ic(seed=3)
        u_d, info_d = m_direct.solve(a, return_info=True)
        u_c, info_c = m_cg.solve(a, return_info=True)
        assert info_d["solver"] == "spsolve" and info_c["solver"] == "cg"
        assert np.linalg.norm(u_d - u_c) / np.linalg.norm(u_d) < 1e-7

    def test_measure_matches_analytic_lambda_3d(self):
        """Shell-averaged empirical mode variances track the analytic
        eigenvalues lambda = (pi^2|m|^2 + tau^2)^(-alpha) with unit slope
        in log-log (fit-free up to the overall calibration constant; a
        parametric tau-fit is too noisy at feasible sample counts)."""
        from scipy.fft import dctn

        m = get_model("darcy_fno_3d")(resolution={"x": 25, "y": 25, "z": 25})
        psi = np.stack([np.log(m.generate_ic(seed=s)) for s in range(40)])
        C = dctn(psi, axes=(-3, -2, -1), norm="ortho")
        P = (C**2).mean(axis=0)

        n = 25
        i = np.arange(n)
        I, J, K = np.meshgrid(i, i, i, indexing="ij")
        mode2 = I**2 + J**2 + K**2
        lam = (np.pi**2 * mode2 + 3.0**2) ** (-2.0)

        # average P and lam over shells of |mode|^2 to beat chi^2 noise
        shells = [(1, 3), (3, 8), (8, 16), (16, 32), (32, 64), (64, 128)]
        emp, ana = [], []
        for lo, hi in shells:
            sel = (mode2 >= lo) & (mode2 < hi)
            emp.append(P[sel].mean())
            ana.append(lam[sel].mean())
        emp, ana = np.log(emp), np.log(ana)
        slope = np.polyfit(ana, emp, 1)[0]
        r2 = np.corrcoef(ana, emp)[0, 1] ** 2
        assert abs(slope - 1.0) < 0.06  # unit log-log slope = same law
        assert r2 > 0.998

    def test_piececonst_3d(self):
        m = get_model("darcy_fno_3d")(
            resolution={"x": 21, "y": 21, "z": 21}, coeff="piececonst"
        )
        a = m.generate_ic(seed=1)
        assert set(np.unique(a)) == {3.0, 12.0}
        u = m.solve(a)
        assert np.isfinite(u).all() and u.max() > 0

    def test_dataset_generation(self):
        d = generate_dataset(
            "darcy_fno_3d",
            n_samples=2,
            resolution={"x": 17, "y": 17, "z": 17},
            seed=0,
            verbose=False,
        )
        assert d.inputs.shape == (2, 17, 17, 17)
        assert d.outputs.shape == (2, 17, 17, 17)
        # boundary exactly zero (Dirichlet)
        assert np.abs(d.outputs[:, 0]).max() == 0.0


class TestMPPDEKdV:
    """
    Brandstetter et al. KdV — the generator shared by "Message Passing Neural
    PDE Solvers" (arXiv:2202.03376) and "Lie Point Symmetry Data Augmentation"
    (arXiv:2202.07643): u_t + u u_x + u_xxx = 0 on a periodic L = 128 box,
    nx = 256, from a 10-wave random sine series over wavenumbers {1, 2}.
    """

    @staticmethod
    def _reference(u0, L, T, nt, tol=1e-9):
        """Their own scheme, verbatim: psdiff derivatives stepped by Radau."""
        from scipy.fftpack import diff as psdiff
        from scipy.integrate import solve_ivp

        def rhs(t, u, L):
            return -u * psdiff(u, period=L) - psdiff(u, order=3, period=L)

        sol = solve_ivp(
            rhs,
            [0.0, T],
            u0,
            method="Radau",
            t_eval=np.linspace(0.0, T, nt),
            args=(L,),
            atol=tol,
            rtol=tol,
        )
        assert sol.success, sol.message
        return sol.y.T

    def test_reproduces_their_reference_solver(self):
        """
        The decisive check: OUR ETDRK4 against THEIR Radau + psdiff scheme on
        the same initial field. Un-dealiased, because their right-hand side is
        — with the 2/3 mask on, the two schemes genuinely differ (the mask
        also cuts real spectral content at nx = 256).
        """
        L, nx, T, nt = 128.0, 128, 5.0, 6
        m = get_model("kdv_1d")(
            resolution={"x": nx},
            domain={"x": (0.0, L)},
            advection=1.0,
            dispersion=1.0,
            dealias=False,
            time_end=T,
            _n_time_steps=nt,
            _dt=2e-3,
        )
        for seed in (0, 1, 2):
            u0 = m.generate_ic(generator="sine_series", seed=seed)
            got = m.solve(u0, return_full=True)
            ref = self._reference(u0, L, T, nt)
            rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
            assert rel < 1e-5, f"seed {seed}: rel-L2 {rel:.3e}"

    def test_dealias_mask_matters_once_the_spectrum_broadens(self):
        """
        The mask is not cosmetic here. Early on (narrow spectrum) it changes
        nothing, but once KdV has broadened the spectrum past nx/3 the 2/3 cut
        removes GENUINE content and the two runs separate — which is why the
        preset turns it off rather than treating it as a free choice.
        Measured against an nx = 1024 reference at T = 100: un-dealiased sits
        ~8e-7 from the converged solution, dealiased ~1e-4.
        """
        kw = dict(
            resolution={"x": 256},
            domain={"x": (0.0, 128.0)},
            advection=1.0,
            dispersion=1.0,
            _n_time_steps=2,
            _dt=5e-3,
        )
        u0 = get_model("kdv_1d")(**kw, time_end=20.0, dealias=False).generate_ic(
            generator="sine_series", seed=0
        )

        def run(T, mask):
            return get_model("kdv_1d")(
                **{**kw, "time_end": T}, dealias=mask
            ).solve(u0)

        def rel(a, b):
            return np.linalg.norm(a - b) / np.linalg.norm(b)

        assert rel(run(0.5, True), run(0.5, False)) < 1e-9  # narrow spectrum
        assert rel(run(20.0, True), run(20.0, False)) > 1e-5  # broadened

    def test_soliton_speed_at_unit_advection(self):
        """
        mu and delta2 must both be wired: u = 3c sech^2(sqrt(c)(x-x0-ct)/2)
        is the exact soliton of u_t + u u_x + u_xxx = 0 and must translate
        rigidly at speed c.
        """
        L, nx, T, c = 128.0, 512, 2.0, 1.0
        m = get_model("kdv_1d")(
            resolution={"x": nx},
            domain={"x": (0.0, L)},
            advection=1.0,
            dispersion=1.0,
            time_end=T,
            _dt=1e-4,
        )
        u = m.solve(m.soliton(c, x0=0.3 * L))
        expected = m.soliton(c, x0=0.3 * L + c * T)
        assert np.linalg.norm(u - expected) / np.linalg.norm(expected) < 1e-4

    def test_sine_series_measure(self):
        """Zero-mean, and ONLY wavenumbers 1 and 2 carry power — the
        half-open randint(lmin, lmax) convention, not a closed range."""
        from pdeforge.generators.initial_conditions import TruncatedSineGenerator

        gen = TruncatedSineGenerator(n_waves=10, lmin=1, lmax=3, amplitude=0.5)
        nx, L = 256, 128.0
        grid = {"x": np.linspace(0.0, (1 - 1.0 / nx) * L, nx)}
        power = np.zeros(nx // 2)
        for seed in range(40):
            u0 = gen.generate(shape=(nx,), seed=seed, grid=grid)
            assert abs(u0.mean()) < 1e-12  # exactly zero-mean on the grid
            power += np.abs(np.fft.rfft(u0)[: nx // 2]) ** 2
        assert power[1] > 0 and power[2] > 0
        # every other mode, mode 3 included, is empty to machine precision
        assert power[3] / power[1:3].max() < 1e-24
        assert power[[0, *range(3, nx // 2)]].max() / power[1:3].max() < 1e-24

    def test_scale_jitter_leaves_this_measure_invariant(self):
        """
        The sine series is a function of x/L, so a jittered box must not
        change the IC array — only the dynamics. That separation is the point:
        jitter enriches the trajectories without disturbing the input measure.
        """
        kw = dict(
            resolution={"x": 128},
            domain={"x": (0.0, 128.0)},
            advection=1.0,
            dispersion=1.0,
            time_end=5.0,
            dealias=False,
            _dt=2e-3,
        )
        plain = get_model("kdv_1d")(**kw, scale_jitter=0.0)
        jittered = get_model("kdv_1d")(**kw, scale_jitter=0.1)

        ic_a = plain.generate_ic(generator="sine_series", seed=7)
        ic_b = jittered.generate_ic(generator="sine_series", seed=7)
        assert np.allclose(ic_a, ic_b, atol=1e-14)

        # ...but the solve sees a different box, so the trajectories differ
        sol_b = jittered.solve(ic_b)
        assert not np.allclose(plain.solve(ic_a), sol_b, atol=1e-6)

    def test_jitter_restores_nominal_state(self):
        """A consumed draw must not leak into the next solve or the grid."""
        m = get_model("kdv_1d")(
            resolution={"x": 64},
            domain={"x": (0.0, 128.0)},
            advection=1.0,
            dispersion=1.0,
            time_end=2.0,
            scale_jitter=0.1,
            _dt=2e-3,
        )
        k0, T0 = m.k.copy(), m.T
        ic = m.generate_ic(generator="sine_series", seed=1)
        assert m._sample_scale is not None
        m.solve(ic)
        assert m._sample_scale is None
        assert np.array_equal(m.k, k0) and m.T == T0
        # a bare solve (no pending draw) is deterministic
        assert np.array_equal(m.solve(ic), m.solve(ic))

    def test_preset_shape_burn_in_and_grid(self):
        """140 frames kept out of 250 => trajectories start at ~0.44 T."""
        d = generate_dataset(
            preset="mp_pde_kdv_easy_1d",
            n_samples=2,
            seed=0,
            verbose=False,
            params={"_n_time_steps": 25, "_n_frames_kept": 14, "_dt": 0.02},
        )
        assert d.inputs.shape == (2, 256)
        assert d.outputs.shape == (2, 14, 256)
        assert np.isfinite(d.outputs).all()
        t = d.grid["t"]
        assert len(t) == 14 and np.isclose(t[-1], 50.0)
        assert np.isclose(t[0], 50.0 * 11 / 24)  # last 14 of 25 frames
        assert np.isclose(d.grid["x"][1] - d.grid["x"][0], 0.5)  # L/nx = 128/256

    def test_preset_registered_and_pins_the_regime(self):
        from pdeforge.presets import get_preset

        for name in ("mp_pde_kdv_1d", "mp_pde_kdv_easy_1d"):
            cfg = get_preset(name)
            assert cfg["model"] == "kdv_1d"
            assert cfg["params"]["advection"] == 1.0
            assert cfg["params"]["dispersion"] == 1.0
            assert cfg["params"]["dealias"] is False
            assert cfg["params"]["scale_jitter"] == 0.1
            assert cfg["domain"] == {"x": (0.0, 128.0)}
            assert cfg["resolution"] == {"x": 256}
            assert cfg["outputs"] == "trajectory"
            assert cfg["ic_params"]["lmax"] == 3

    def test_mass_conserved_over_the_horizon(self):
        """KdV conserves mass; the measure is zero-mean, so it must stay so."""
        m = get_model("kdv_1d")(
            resolution={"x": 128},
            domain={"x": (0.0, 128.0)},
            advection=1.0,
            dispersion=1.0,
            dealias=False,
            time_end=20.0,
            _dt=5e-3,
        )
        ic = m.generate_ic(generator="sine_series", seed=4)
        assert abs(m.solve(ic).mean() - ic.mean()) < 1e-11


class TestKdVBackwardsCompatible:
    def test_textbook_defaults_unchanged(self):
        """kdv_1d's new coefficient knobs must default to the old hardcoded
        mu = 6, delta2 = 1 behaviour, mask and all."""
        m = get_model("kdv_1d")(resolution={"x": 256})
        assert m.mu == 6.0 and m.delta2 == 1.0
        assert m.dealias.min() == 0.0  # 2/3 mask still on by default
        assert m.scale_jitter == 0.0 and m.n_kept == m.n_t
        # textbook soliton: u = c/2 sech^2(sqrt(c)(x-x0)/2)
        c, L = 4.0, m.domain.size("x")
        x = m.grids["x"]
        d = (x - 0.5 * L + L / 2) % L - L / 2
        assert np.allclose(m.soliton(c), 0.5 * c / np.cosh(0.5 * np.sqrt(c) * d) ** 2)


AIRFRANS_ROOT = Path(os.environ.get("PDEFORGE_AIRFRANS_ROOT", "/nonexistent"))


def _vtk_xml_doc(arrays, compressed, header_type="UInt32"):
    """Build a minimal .vtu in memory, the way VTK writes it."""
    import base64
    import zlib

    htype = {"UInt32": np.uint32, "UInt64": np.uint64}[header_type]

    def encode(a):
        raw = a.tobytes()
        if not compressed:
            head = np.array([len(raw)], dtype=htype).tobytes()
            return base64.b64encode(head).decode() + base64.b64encode(raw).decode()
        # header and payload are two INDEPENDENT base64 streams
        blob = zlib.compress(raw)
        head = np.array([1, len(raw), len(raw), len(blob)], dtype=htype).tobytes()
        return base64.b64encode(head).decode() + base64.b64encode(blob).decode()

    vtk_type = {
        np.dtype(np.float32): "Float32",
        np.dtype(np.float64): "Float64",
        np.dtype(np.int64): "Int64",
    }
    n = len(next(iter(arrays.values())))
    comp = ' compressor="vtkZLibDataCompressor"' if compressed else ""
    body = []
    for name, a in arrays.items():
        nc = a.shape[1] if a.ndim > 1 else 1
        body.append(
            f'<DataArray type="{vtk_type[a.dtype]}" Name="{name}" '
            f'NumberOfComponents="{nc}" format="binary">{encode(a)}</DataArray>'
        )
    pts = encode(np.zeros((n, 3), dtype=np.float64))
    return (
        f'<?xml version="1.0"?>\n<VTKFile type="UnstructuredGrid" '
        f'header_type="{header_type}"{comp}><UnstructuredGrid>'
        f'<Piece NumberOfPoints="{n}" NumberOfCells="0">'
        f'<Points><DataArray type="Float64" Name="Points" '
        f'NumberOfComponents="3" format="binary">{pts}</DataArray></Points>'
        f'<PointData>{"".join(body)}</PointData>'
        f"</Piece></UnstructuredGrid></VTKFile>"
    )


class TestVTKXMLReader:
    """
    The reader exists so AirfRANS interop does not drag vtk/pyvista/meshio
    into the install. These round-trips need no external data.
    """

    @pytest.mark.parametrize("compressed", [True, False])
    @pytest.mark.parametrize("header_type", ["UInt32", "UInt64"])
    def test_round_trip(self, tmp_path, compressed, header_type):
        from pdeforge.io.vtk_xml import read_vtk_xml

        rng = np.random.default_rng(0)
        arrays = {
            "scalar": rng.standard_normal(97).astype(np.float32),
            "vector": rng.standard_normal((97, 3)).astype(np.float32),
            "ids": np.arange(97, dtype=np.int64),
        }
        f = tmp_path / "t.vtu"
        f.write_text(_vtk_xml_doc(arrays, compressed, header_type))

        got = read_vtk_xml(f)
        assert got["n_points"] == 97
        assert got["points"].shape == (97, 3)
        for name, a in arrays.items():
            assert got[name].shape == a.shape
            assert np.array_equal(got[name], a)

    def test_appended_data_raises_not_silently_wrong(self, tmp_path):
        from pdeforge.io.vtk_xml import read_vtk_xml

        f = tmp_path / "a.vtu"
        f.write_text(
            '<?xml version="1.0"?><VTKFile type="UnstructuredGrid">'
            '<UnstructuredGrid><Piece NumberOfPoints="0" NumberOfCells="0">'
            "</Piece></UnstructuredGrid>"
            '<AppendedData encoding="raw">_</AppendedData></VTKFile>'
        )
        with pytest.raises(NotImplementedError, match="appended"):
            read_vtk_xml(f)


class TestAirfRANSNames:
    """Name parsing needs no data files."""

    def test_four_and_five_digit_series(self):
        from pdeforge.io.airfrans import NU_AIR, parse_case_name

        p4 = parse_case_name("airFoil2D_SST_31.283_-4.156_0.919_6.98_14.32")
        assert p4["turbulence"] == "SST"
        assert p4["inlet_velocity_m_s"] == 31.283
        assert p4["angle_of_attack_deg"] == -4.156
        assert p4["naca_params"] == [0.919, 6.98, 14.32]
        assert p4["naca_series"] == 4
        assert np.isclose(p4["reynolds"], 31.283 / NU_AIR)

        p5 = parse_case_name("airFoil2D_SST_31.382_3.588_1.994_6.206_0.0_13.271")
        assert p5["naca_series"] == 5
        assert len(p5["naca_params"]) == 4

    def test_rejects_foreign_names(self):
        from pdeforge.io.airfrans import parse_case_name

        with pytest.raises(ValueError, match="not an AirfRANS case name"):
            parse_case_name("burgers_1d_seed0")


@pytest.mark.skipif(
    not AIRFRANS_ROOT.exists(),
    reason="set PDEFORGE_AIRFRANS_ROOT to the AirfRANS Dataset directory",
)
class TestAirfRANSInterop:
    """
    Shootout-tier tests against the distributed AirfRANS data. These assert
    PHYSICS, not just shapes: if the pressure normalisation or the freestream
    convention were wrong, C_p would not peak at 1.
    """

    def test_loads_split_with_expected_shape(self):
        from pdeforge import load_airfrans

        d = load_airfrans(
            AIRFRANS_ROOT, split="full_train", n_samples=3, n_points=8192,
            seed=0, verbose=False,
        )
        assert d.inputs.shape == (3, 8192, 8)
        assert d.outputs.shape == (3, 8192, 4)
        assert d.metadata["source"] == "airfrans"
        assert np.isfinite(d.inputs).all() and np.isfinite(d.outputs).all()

    def test_stagnation_pressure_coefficient_is_one(self):
        """
        C_p = (p/rho)/(0.5|U_inf|^2) must reach +1 at the stagnation point and
        go negative over the suction side. This pins the kinematic-pressure
        convention AND the freestream vector U_inf(cos a, sin a) at once.
        """
        from pdeforge import load_airfrans
        from pdeforge.io.airfrans import surface_pressure

        d = load_airfrans(
            AIRFRANS_ROOT, split="full_train", n_samples=5, n_points=8192,
            seed=0, verbose=False,
        )
        for i in range(d.n_samples):
            cp = surface_pressure(d, i)["cp"]
            assert 0.95 < cp.max() <= 1.05, f"case {i}: C_p max {cp.max():.3f}"
            assert cp.min() < -0.5, f"case {i}: no suction peak"

    def test_wall_nodes_kept_and_no_slip(self):
        """keep_surface must retain every wall node, and RANS wall nodes
        carry exactly zero velocity."""
        from pdeforge import load_airfrans

        d = load_airfrans(
            AIRFRANS_ROOT, split="full_train", n_samples=2, n_points=8192,
            keep_surface=True, seed=0, verbose=False,
        )
        names = d.input_names
        for i in range(d.n_samples):
            wall = d.inputs[i, :, names.index("surface")] > 0.5
            assert wall.sum() > 500  # ~1000 wall nodes per case
            uv = d.outputs[i, wall][:, :2]
            assert np.abs(uv).max() == 0.0  # no-slip, exactly
            # wall nodes sit at zero distance and carry unit normals
            assert np.abs(d.inputs[i, wall, names.index("sdf")]).max() < 1e-12
            n = d.inputs[i, wall][:, [names.index("n_x"), names.index("n_y")]]
            assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-5)

    def test_manifest_splits_are_disjoint_and_sized(self):
        import json

        m = json.loads((AIRFRANS_ROOT / "manifest.json").read_text())
        assert len(m["full_train"]) == 800 and len(m["full_test"]) == 200
        assert not set(m["full_train"]) & set(m["full_test"])

    def test_uniform_draw_loses_the_wall(self):
        """The reason keep_surface defaults to True: the wall is ~0.6% of the
        cloud, so a uniform draw keeps almost none of it."""
        from pdeforge import load_airfrans

        d = load_airfrans(
            AIRFRANS_ROOT, split="full_train", n_samples=1, n_points=8192,
            keep_surface=False, seed=0, verbose=False,
        )
        wall = d.inputs[0, :, d.input_names.index("surface")] > 0.5
        assert wall.sum() < 200
