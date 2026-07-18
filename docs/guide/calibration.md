# The Calibration Protocol

This page specifies how PDEForge's four-way split supports uncertainty
quantification with **split conformal prediction** — and the rules that keep
its guarantee valid. As of 2026, no other PDE dataset or benchmark ships a
native calibration split; published conformal-neural-operator workflows carve
one out of the training set by hand. PDEForge makes the split a first-class,
recorded, reproducible object.

## The four-way split

```python
splits = dataset.split(train=0.6, val=0.15, cal=0.15, test=0.1, seed=0)
```

| Split | Role | May influence the model? |
|-------|------|--------------------------|
| `train` | fit the surrogate | yes |
| `val` | tune hyperparameters, early stopping | yes |
| `cal` | compute nonconformity scores ONLY | **never** |
| `test` | report coverage and error | never |

The split fractions and seed are recorded in each split's metadata
(`split`, `split_seed`, `split_fractions`), so the exact partition is
reproducible from the saved dataset alone.

## The guarantee

Split conformal prediction: compute a nonconformity score on each
calibration sample, take the `ceil((n+1)(1-alpha))/n` empirical quantile
`qhat`, and attach `prediction +- qhat` to every test prediction. If
calibration and test samples are **exchangeable**, coverage is at least
`1 - alpha` in finite samples — no distributional or model assumptions.

```python
from pdeforge.uq import conformal_quantile, empirical_coverage

qhat = conformal_quantile(cal_pred, cal_true, alpha=0.1, score="max")
cov  = empirical_coverage(test_pred, test_true, qhat, score="max")
```

`score="max"` gives simultaneous (whole-field) bands; `score="mean"` gives
an average-error band. Both are provided dependency-free; MAPIE, crepes, or
TorchCP drop in wherever you prefer their estimators.

## Rules that keep the guarantee honest

1. **Never touch `cal` during training or model selection.** A calibration
   set that influenced the model is a validation set, and the guarantee is
   void. This is the failure mode hand-carved splits invite.
2. **Trajectories: split by INITIAL CONDITION, never by frame.** Frames of
   one rollout are strongly dependent; putting frames of the same trajectory
   in both `train` and `cal` destroys exchangeability. PDEForge's split
   operates on the sample axis (one IC = one sample), which does this
   correctly by construction for `outputs="trajectory"` datasets.
3. **Exchangeability holds within the generating distribution.** If you
   train on one parameter range and predict on another, the guarantee does
   NOT transfer — measure the degradation deliberately with
   `pdeforge.uq.split_ood` (an `ood` split by parameter range).
4. **Calibration size:** coverage estimates stabilise from a few tens of
   samples (~50 is a practical floor; more tightens the quantile). With
   `n_samples=1000` and the default fractions you get 150 — comfortable.
5. **Distribution shift studies:** draw parameters with
   `pdeforge.uq.generate_parametric_dataset` (LHS/Sobol designs, per-sample
   values recorded in metadata), then `split_ood(by="viscosity", ...)` for
   controlled shift experiments.

## End-to-end sketch

```python
import pdeforge
from pdeforge.uq import conformal_quantile, empirical_coverage

data = pdeforge.generate_dataset("darcy_2d", n_samples=2000,
                                 resolution={"x": 64, "y": 64}, seed=0)
s = data.split(train=0.6, val=0.15, cal=0.15, test=0.1, seed=1)

surrogate.fit(s["train"], val=s["val"])            # any model, any library

qhat = conformal_quantile(surrogate(s["cal"].inputs), s["cal"].outputs,
                          alpha=0.1)
coverage = empirical_coverage(surrogate(s["test"].inputs),
                              s["test"].outputs, qhat)   # >= 0.9 expected
```

The `examples/` directory contains a complete FNO + conformal walkthrough
against the `neuraloperator` library.
