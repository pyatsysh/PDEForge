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
