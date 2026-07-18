"""
Tests for the UQ layer: parameter distributions, QMC designs, OOD splits,
multi-fidelity generation, observation operators, split conformal, and the
verification harness.
"""

import numpy as np
import pytest

from pdeforge.uq import (
    Choice,
    LogUniform,
    Normal,
    Uniform,
    conformal_quantile,
    empirical_coverage,
    generate_multifidelity,
    generate_parametric_dataset,
    observe,
    params_array,
    spectral_downsample,
    split_ood,
)


class TestDistributions:
    def test_uniform_bounds(self):
        rng = np.random.default_rng(0)
        s = Uniform(2.0, 5.0).sample(rng, 1000)
        assert s.min() >= 2.0 and s.max() <= 5.0

    def test_loguniform_decades(self):
        rng = np.random.default_rng(0)
        s = LogUniform(1e-4, 1e-1).sample(rng, 4000)
        # roughly equal mass per decade
        frac_low = np.mean(s < 1e-3)
        assert 0.25 < frac_low < 0.42

    def test_normal_moments(self):
        rng = np.random.default_rng(0)
        s = Normal(3.0, 0.5).sample(rng, 20000)
        assert abs(s.mean() - 3.0) < 0.02
        assert abs(s.std() - 0.5) < 0.02

    def test_choice(self):
        rng = np.random.default_rng(0)
        s = Choice([4.0, 12.0]).sample(rng, 100)
        assert set(np.unique(s)) <= {4.0, 12.0}


class TestParametricDataset:
    def test_per_sample_params_recorded(self):
        d = generate_parametric_dataset(
            "heat_1d",
            n_samples=6,
            resolution={"x": 32},
            param_dists={"diffusivity": LogUniform(1e-3, 1e-1)},
            sampler="lhs",
            seed=1,
            verbose=False,
        )
        vals, names = params_array(d)
        assert vals.shape == (6, 1) and names == ["diffusivity"]
        assert (vals > 0).all()
        assert d.metadata["param_sampler"] == "lhs"
        # different parameters -> different outputs for same-ish ICs
        assert len(np.unique(vals)) == 6

    def test_sobol_design_spreads(self):
        d = generate_parametric_dataset(
            "heat_1d",
            n_samples=8,
            resolution={"x": 16},
            param_dists={"diffusivity": Uniform(0.01, 0.11)},
            sampler="sobol",
            seed=0,
            verbose=False,
        )
        v, _ = params_array(d)
        # Sobol at n=8 stratifies: each of 8 equal bins gets one point
        bins = np.floor((v[:, 0] - 0.01) / 0.1 * 8).astype(int)
        assert len(set(bins.tolist())) == 8


class TestOODSplit:
    def test_ood_split_shapes_and_ranges(self):
        d = generate_parametric_dataset(
            "heat_1d",
            n_samples=40,
            resolution={"x": 16},
            param_dists={"diffusivity": Uniform(0.01, 0.2)},
            sampler="lhs",
            seed=3,
            verbose=False,
        )
        splits = split_ood(
            d,
            by="diffusivity",
            train_range=(0.01, 0.1),
            ood_range=(0.1, 0.2),
            seed=0,
        )
        assert "ood" in splits and "cal" in splits
        ood_vals, _ = params_array(splits["ood"])
        assert (ood_vals > 0.1).all()
        in_vals, _ = params_array(splits["train"])
        assert (in_vals <= 0.1).all()
        assert splits["ood"].metadata["split"] == "ood"


class TestMultiFidelity:
    def test_spectral_downsample_exact_on_modes(self):
        """Truncation is exact for band-limited fields."""
        x = np.linspace(0, 1, 64, endpoint=False)
        u = np.sin(2 * np.pi * 3 * x) + 0.5 * np.cos(2 * np.pi * 5 * x)
        u_c = spectral_downsample(u, (32,))
        x_c = np.linspace(0, 1, 32, endpoint=False)
        expected = np.sin(2 * np.pi * 3 * x_c) + 0.5 * np.cos(2 * np.pi * 5 * x_c)
        assert np.allclose(u_c, expected, atol=1e-12)

    def test_pairs_share_realisation(self):
        pair = generate_multifidelity(
            "heat_1d",
            resolutions=[{"x": 32}, {"x": 128}],
            n_samples=3,
            seed=5,
            verbose=False,
        )
        assert set(pair.keys()) == {"32", "128"}
        coarse, fine = pair["32"], pair["128"]
        # the coarse IC IS the truncated fine IC
        assert np.allclose(
            coarse.inputs[0], spectral_downsample(fine.inputs[0], (32,)), atol=1e-12
        )
        # heat is linear+diagonal: coarse solve == truncated fine solve
        assert np.allclose(
            coarse.outputs[0],
            spectral_downsample(fine.outputs[0], (32,)),
            atol=1e-8,
        )


class TestObserve:
    def _d(self):
        from pdeforge import generate_dataset

        return generate_dataset(
            "heat_1d", n_samples=3, resolution={"x": 64}, seed=2, verbose=False
        )

    def test_sensors(self):
        d = observe(self._d(), sensors=10, seed=0)
        assert d.outputs.shape == (3, 10)
        assert d.metadata["observation"]["type"] == "sensors"
        assert len(d.metadata["observation"]["indices"]) == 10

    def test_subsample_and_noise(self):
        base = self._d()
        d = observe(base, subsample=4, noise_std=0.01, seed=0)
        assert d.outputs.shape == (3, 16)
        assert not np.allclose(d.outputs, base.outputs[:, ::4])  # noise added
        assert d.metadata["observation"]["noise_std"] == 0.01


class TestConformal:
    def test_coverage_guarantee_on_linear_surrogate(self):
        """
        End-to-end split conformal on heat_1d with a least-squares linear
        surrogate: empirical test coverage must be >= 1 - alpha - slack.
        (Heat is linear, so the surrogate is excellent and the test tight.)
        """
        from pdeforge import generate_dataset

        d = generate_dataset(
            "heat_1d", n_samples=200, resolution={"x": 32}, seed=7, verbose=False
        )
        s = d.split(train=0.5, val=0.0, cal=0.25, test=0.25, seed=1)
        Xtr = s["train"].inputs.reshape(len(s["train"].inputs), -1)
        Ytr = s["train"].outputs.reshape(len(s["train"].outputs), -1)
        W, *_ = np.linalg.lstsq(Xtr, Ytr, rcond=None)

        alpha = 0.1
        cal_pred = s["cal"].inputs.reshape(len(s["cal"].inputs), -1) @ W
        qhat = conformal_quantile(
            cal_pred, s["cal"].outputs.reshape(cal_pred.shape), alpha=alpha
        )
        test_pred = s["test"].inputs.reshape(len(s["test"].inputs), -1) @ W
        cov = empirical_coverage(
            test_pred, s["test"].outputs.reshape(test_pred.shape), qhat
        )
        assert cov >= 1 - alpha - 0.08  # finite-sample slack

    def test_quantile_monotone_in_alpha(self):
        rng = np.random.default_rng(0)
        pred, true = rng.random((50, 8)), rng.random((50, 8))
        q10 = conformal_quantile(pred, true, alpha=0.1)
        q30 = conformal_quantile(pred, true, alpha=0.3)
        assert q10 >= q30


class TestVerifyHarness:
    def test_burgers_convergence_order(self):
        """Spectral Burgers: error vs the fine reference must drop fast."""
        from pdeforge.verify import convergence_study

        study = convergence_study(
            "burgers_1d",
            resolutions=[{"x": 32}, {"x": 64}, {"x": 128}, {"x": 256}],
            n_samples=2,
            seed=0,
        )
        errs = [study["errors"][k] for k in study["resolutions"][:-1]]
        assert errs[0] > errs[1] > errs[2]  # monotone under refinement
        assert errs[2] < 1e-3  # accurate at 128 (shock-regime default nu)
        assert min(study["orders"]) > 2.0  # super-algebraic spectral orders

    def test_heat_exact_so_error_tiny(self):
        from pdeforge.verify import convergence_study

        study = convergence_study(
            "heat_1d",
            resolutions=[{"x": 32}, {"x": 128}],
            n_samples=2,
            seed=0,
        )
        # heat is diagonal in Fourier space: truncation commutes with solve
        assert study["errors"]["32"] < 1e-10
