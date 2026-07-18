"""
UQ-first data utilities — the calibration-native layer.

This module is what makes PDEForge UQ-native rather than UQ-compatible:

- parameter DISTRIBUTIONS as first-class inputs (per-sample draws recorded in
  metadata, QMC designs via scipy.stats.qmc);
- out-of-distribution SPLITS by parameter range (train in-range, probe OOD);
- MULTI-FIDELITY pairs: the same spectral IC realised at several resolutions;
- OBSERVATION operators (sensors, masking, noise) with recorded settings;
- a dependency-free SPLIT-CONFORMAL helper + empirical-coverage check.

Everything stays NumPy; nothing here needs torch/jax.
"""

from typing import Dict, List, Optional, Union

import numpy as np

from pdeforge.core.registry import get_model
from pdeforge.core.types import PDEDataset

# ---------------------------------------------------------------------------
# Parameter distributions
# ---------------------------------------------------------------------------


class ParamDist:
    """Base class for parameter distributions (unit-cube transformable)."""

    def from_unit(self, u):
        """Map u in [0,1) to the distribution's support (QMC-compatible)."""
        raise NotImplementedError

    def sample(self, rng, n):
        return self.from_unit(rng.random(n))

    def to_meta(self):
        return {"dist": self.__class__.__name__, **self.__dict__}


class Uniform(ParamDist):
    def __init__(self, low, high):
        self.low, self.high = float(low), float(high)

    def from_unit(self, u):
        return self.low + (self.high - self.low) * u


class LogUniform(ParamDist):
    def __init__(self, low, high):
        assert low > 0 and high > low
        self.low, self.high = float(low), float(high)

    def from_unit(self, u):
        return np.exp(np.log(self.low) + (np.log(self.high) - np.log(self.low)) * u)


class Normal(ParamDist):
    def __init__(self, mean, std):
        self.mean, self.std = float(mean), float(std)

    def from_unit(self, u):
        from scipy.special import ndtri  # inverse standard-normal CDF

        u = np.clip(u, 1e-12, 1 - 1e-12)
        return self.mean + self.std * ndtri(u)


class Choice(ParamDist):
    def __init__(self, values):
        self.values = list(values)

    def from_unit(self, u):
        idx = np.minimum((u * len(self.values)).astype(int), len(self.values) - 1)
        return np.asarray(self.values)[idx]


def _unit_design(n, d, sampler, seed):
    """n x d design on the unit cube: random / lhs / sobol / halton."""
    if sampler == "random":
        return np.random.default_rng(seed).random((n, d))
    from scipy.stats import qmc

    if sampler == "lhs":
        return qmc.LatinHypercube(d=d, seed=seed).random(n)
    if sampler == "sobol":
        return qmc.Sobol(d=d, seed=seed).random(n)
    if sampler == "halton":
        return qmc.Halton(d=d, seed=seed).random(n)
    raise ValueError(f"Unknown sampler: {sampler!r}")


def generate_parametric_dataset(
    model,
    n_samples,
    resolution,
    param_dists,
    fixed_params=None,
    domain=None,
    ic_generator="fourier",
    ic_params=None,
    sampler="random",
    seed=None,
    verbose=True,
):
    """
    Generate a dataset whose PHYSICAL PARAMETERS vary per sample.

    param_dists: {"viscosity": LogUniform(1e-4, 1e-1), ...}. Draws come from
    a random or QMC design (lhs/sobol/halton) on the unit cube, mapped
    through each distribution. Every sample's parameter values are recorded
    in metadata["param_samples"] — the calibration-grade provenance that
    parametric operator learning and OOD splitting need.
    """
    names = sorted(param_dists.keys())
    design = _unit_design(n_samples, len(names), sampler, seed)
    values = {
        name: np.asarray(param_dists[name].from_unit(design[:, j]))
        for j, name in enumerate(names)
    }

    model_cls = get_model(model)
    fixed_params = fixed_params or {}
    ic_params = ic_params or {}

    root = np.random.SeedSequence(seed)
    sample_seqs = root.spawn(n_samples)

    inputs, outputs = [], []
    iterator = range(n_samples)
    if verbose:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc=f"Parametric {model} ({sampler})")

    template = None
    for i in iterator:
        params_i = {**fixed_params, **{k: float(values[k][i]) for k in names}}
        m = model_cls(resolution=resolution, domain=domain, **params_i)
        s = int(sample_seqs[i].generate_state(1)[0] % (2**31))
        ic, sol, _ = m.generate_sample(
            generator=ic_generator, generator_params=ic_params, seed=s
        )
        inputs.append(ic)
        outputs.append(sol)
        template = m

    metadata = template.dataset_metadata(
        n_samples, ic_generator, ic_params, seed, "final"
    )
    metadata["param_samples"] = {k: values[k].tolist() for k in names}
    metadata["param_dists"] = {k: param_dists[k].to_meta() for k in names}
    metadata["param_sampler"] = sampler
    # params in metadata hold the LAST draw; the per-sample truth is above.
    return PDEDataset(
        inputs=np.stack(inputs, axis=0),
        outputs=np.stack(outputs, axis=0),
        grid=template.dataset_grid("final"),
        metadata=metadata,
        input_names=template.INPUT_NAMES,
        output_names=template.OUTPUT_NAMES,
    )


def params_array(dataset, names=None):
    """(n_samples, n_params) array of per-sample parameter draws."""
    ps = dataset.metadata.get("param_samples")
    if ps is None:
        raise ValueError("Dataset has no per-sample parameters (param_samples).")
    names = names or sorted(ps.keys())
    return np.stack([np.asarray(ps[k]) for k in names], axis=1), names


# ---------------------------------------------------------------------------
# OOD splitting
# ---------------------------------------------------------------------------


def split_ood(
    dataset,
    by,
    train_range,
    ood_range,
    train=0.6,
    val=0.15,
    cal=0.15,
    test=0.1,
    seed=None,
):
    """
    Distribution-shift splits from a parametric dataset.

    Samples whose parameter `by` lies in train_range are split 4-way as
    usual; samples in ood_range become the "ood" split. Conformal coverage
    calibrated on `cal` is guaranteed (exchangeably) for `test` but NOT for
    `ood` — measuring exactly that degradation is the point.
    """
    ps = dataset.metadata.get("param_samples", {})
    if by not in ps:
        raise ValueError(f"No per-sample values recorded for parameter {by!r}.")
    v = np.asarray(ps[by])

    in_mask = (v >= train_range[0]) & (v <= train_range[1])
    ood_mask = (v >= ood_range[0]) & (v <= ood_range[1]) & ~in_mask
    if not ood_mask.any():
        raise ValueError("ood_range selected no samples.")

    def subset(mask):
        idx = np.where(mask)[0]
        meta = {**dataset.metadata}
        meta["param_samples"] = {k: np.asarray(a)[idx].tolist() for k, a in ps.items()}
        return PDEDataset(
            inputs=dataset.inputs[idx],
            outputs=dataset.outputs[idx],
            grid=dataset.grid.copy(),
            metadata=meta,
            input_names=dataset.input_names,
            output_names=dataset.output_names,
        )

    in_dist = subset(in_mask)
    splits = in_dist.split(train=train, val=val, cal=cal, test=test, seed=seed)
    ood = subset(ood_mask)
    ood.metadata["split"] = "ood"
    ood.metadata["ood_by"] = by
    ood.metadata["ood_range"] = list(ood_range)
    splits["ood"] = ood
    return splits


# ---------------------------------------------------------------------------
# Multi-fidelity
# ---------------------------------------------------------------------------


def spectral_downsample(u, target_shape):
    """Downsample a (periodic, spectrally smooth) field by Fourier truncation."""
    u = np.asarray(u)
    src = u.shape
    U = np.fft.fftn(u)
    out = np.zeros(target_shape, dtype=complex)
    slices = []
    for n_src, n_tgt in zip(src, target_shape):
        h = n_tgt // 2
        slices.append((slice(0, h), slice(n_src - (n_tgt - h), n_src), h, n_tgt))
    # build index grids per axis: copy low+high frequency blocks
    idx_src = [
        np.r_[0 : s[2], src[i] - (target_shape[i] - s[2]) : src[i]]
        for i, s in enumerate(slices)
    ]
    idx_tgt = [np.r_[0 : s[2], s[2] : target_shape[i]] for i, s in enumerate(slices)]
    out[np.ix_(*idx_tgt)] = U[np.ix_(*idx_src)]
    scale = np.prod(target_shape) / np.prod(src)
    return np.fft.ifftn(out).real * scale


def generate_multifidelity(
    model,
    resolutions,
    n_samples,
    params=None,
    domain=None,
    seed=None,
    verbose=True,
):
    """
    Same physical samples at several fidelities.

    ICs are synthesised once on the FINEST grid (seeded), spectrally
    truncated to each coarser grid, and each fidelity solves its own copy —
    so pairs/triples share the same underlying realisation. Returns
    {resolution_key: PDEDataset} with aligned sample order.

    resolutions: list of resolution dicts, e.g. [{"x": 64}, {"x": 256}].
    """
    params = params or {}
    model_cls = get_model(model)

    # sort fine -> coarse by total points
    order = sorted(resolutions, key=lambda r: -int(np.prod(list(r.values()))))
    finest = order[0]

    fine_model = model_cls(resolution=finest, domain=domain, **params)
    root = np.random.SeedSequence(seed)
    seqs = root.spawn(n_samples)

    fine_ics = [
        fine_model.generate_ic(seed=int(s.generate_state(1)[0] % (2**31))) for s in seqs
    ]

    results = {}
    for res in order:
        m = model_cls(resolution=res, domain=domain, **params)
        shape = tuple(res[d] for d in sorted(res.keys())[::-1])
        ins, outs = [], []
        it = fine_ics
        if verbose:
            from tqdm import tqdm

            it = tqdm(fine_ics, desc=f"fidelity {shape}")
        for ic in it:
            ic_r = ic if ic.shape == shape else spectral_downsample(ic, shape)
            ins.append(ic_r)
            outs.append(m.solve(ic_r))
        meta = m.dataset_metadata(n_samples, "multifidelity", {}, seed, "final")
        meta["fidelities"] = [dict(r) for r in order]
        key = "x".join(str(res[d]) for d in sorted(res.keys()))
        results[key] = PDEDataset(
            inputs=np.stack(ins, axis=0),
            outputs=np.stack(outs, axis=0),
            grid=m.dataset_grid("final"),
            metadata=meta,
            input_names=m.INPUT_NAMES,
            output_names=m.OUTPUT_NAMES,
        )
    return results


# ---------------------------------------------------------------------------
# Observation operators
# ---------------------------------------------------------------------------


def observe(dataset, sensors=None, subsample=None, noise_std=0.0, seed=None):
    """
    Apply an observation operator to the OUTPUTS: random point sensors OR
    regular subsampling, plus optional additive Gaussian noise. Settings are
    recorded in metadata["observation"] — inverse-problem realism with full
    provenance.
    """
    rng = np.random.default_rng(seed)
    out = np.asarray(dataset.outputs)
    obs_meta = {"noise_std": float(noise_std), "seed": seed}

    if sensors is not None:
        flat = out.reshape(out.shape[0], -1)
        idx = rng.choice(flat.shape[1], size=int(sensors), replace=False)
        idx.sort()
        out = flat[:, idx]
        obs_meta.update(
            {"type": "sensors", "n_sensors": int(sensors), "indices": idx.tolist()}
        )
    elif subsample is not None:
        sl = (slice(None),) + tuple(
            slice(None, None, int(subsample)) for _ in out.shape[1:]
        )
        out = out[sl]
        obs_meta.update({"type": "subsample", "factor": int(subsample)})
    else:
        obs_meta.update({"type": "identity"})

    if noise_std > 0.0:
        out = out + noise_std * rng.standard_normal(out.shape)

    return PDEDataset(
        inputs=dataset.inputs,
        outputs=out,
        grid=dataset.grid.copy(),
        metadata={**dataset.metadata, "observation": obs_meta},
        input_names=dataset.input_names,
        output_names=["observed_" + n for n in dataset.output_names],
    )


# ---------------------------------------------------------------------------
# Split conformal (dependency-free reference implementation)
# ---------------------------------------------------------------------------


def conformal_quantile(cal_pred, cal_true, alpha=0.1, score="max"):
    """
    Split-conformal nonconformity quantile on the CALIBRATION split.

    score="max": sup-norm over the field per sample (simultaneous bands);
    score="mean": mean absolute error per sample. Returns qhat such that
    predict +- qhat covers with probability >= 1 - alpha (exchangeable data).
    """
    cal_pred, cal_true = np.asarray(cal_pred), np.asarray(cal_true)
    err = np.abs(cal_pred - cal_true).reshape(cal_pred.shape[0], -1)
    if score == "max":
        s = err.max(axis=1)
    elif score == "mean":
        s = err.mean(axis=1)
    else:
        raise ValueError(f"Unknown score: {score!r}")
    n = len(s)
    q = np.ceil((n + 1) * (1 - alpha)) / n
    return float(np.quantile(s, min(q, 1.0), method="higher"))


def empirical_coverage(pred, true, qhat, score="max"):
    """Fraction of samples whose field lies inside predict +- qhat."""
    pred, true = np.asarray(pred), np.asarray(true)
    err = np.abs(pred - true).reshape(pred.shape[0], -1)
    s = err.max(axis=1) if score == "max" else err.mean(axis=1)
    return float(np.mean(s <= qhat))
