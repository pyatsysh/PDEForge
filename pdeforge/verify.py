"""
Solver verification: convergence studies and discretisation-error estimates.

No competitor quantifies the numerical error of its own "ground truth"; this
module does. For a model it runs the SAME physical realisations across a
resolution ladder, measures each level against the finest (restricted
spectrally to the coarse grid), and reports the observed convergence order
and an error estimate at each resolution — the number a user should quote as
the data's ground-truth error bar.
"""

from typing import Dict, List, Optional

import numpy as np

from pdeforge.core.registry import get_model
from pdeforge.uq import spectral_downsample


def convergence_study(
    model,
    resolutions,
    n_samples=3,
    params=None,
    domain=None,
    seed=0,
    verbose=False,
):
    """
    Spatial-convergence study for a (periodic, spectral) model.

    resolutions: ascending list of resolution dicts, finest last — the finest
    level is the reference. ICs are synthesised on the finest grid and
    spectrally restricted downward, so every level solves the SAME
    realisation.

    Returns {"errors": {res_key: rel_err}, "orders": [...], "resolutions":
    [...], "reference": res_key} — rel_err is the mean (over samples)
    relative L2 error against the restricted reference solution.
    """
    params = params or {}
    model_cls = get_model(model)

    order = sorted(resolutions, key=lambda r: int(np.prod(list(r.values()))))
    finest = order[-1]

    fine_model = model_cls(resolution=finest, domain=domain, **params)
    root = np.random.SeedSequence(seed)
    seqs = root.spawn(n_samples)
    fine_ics = [
        fine_model.generate_ic(seed=int(s.generate_state(1)[0] % (2**31))) for s in seqs
    ]
    fine_outs = [fine_model.solve(ic) for ic in fine_ics]

    def key_of(res):
        return "x".join(str(res[d]) for d in sorted(res.keys()))

    errors = {}
    hs = []
    errs_seq = []
    for res in order[:-1]:
        m = model_cls(resolution=res, domain=domain, **params)
        shape = tuple(res[d] for d in sorted(res.keys())[::-1])
        rels = []
        for ic_f, out_f in zip(fine_ics, fine_outs):
            ic_c = spectral_downsample(ic_f, shape)
            out_c = m.solve(ic_c)
            ref_c = spectral_downsample(out_f, shape)
            rels.append(
                np.linalg.norm(out_c - ref_c) / (np.linalg.norm(ref_c) + 1e-300)
            )
        errors[key_of(res)] = float(np.mean(rels))
        hs.append(1.0 / min(shape))
        errs_seq.append(errors[key_of(res)])
        if verbose:
            print(f"  {key_of(res)}: rel L2 err = {errors[key_of(res)]:.3e}")

    orders = []
    for i in range(1, len(errs_seq)):
        if errs_seq[i] > 0 and errs_seq[i - 1] > 0:
            orders.append(
                float(np.log(errs_seq[i - 1] / errs_seq[i]) / np.log(hs[i - 1] / hs[i]))
            )

    return {
        "model": model,
        "errors": errors,
        "orders": orders,
        "resolutions": [key_of(r) for r in order],
        "reference": key_of(finest),
        "n_samples": n_samples,
        "seed": seed,
    }


def verify_model(model, resolutions=None, **kwargs):
    """
    One-call verification: convergence study with sensible defaults.
    Returns the study dict; models with exact solutions should ALSO be
    checked against them in the test suite (see tests/).
    """
    if resolutions is None:
        cls = get_model(model)
        if cls.NDIM == 1:
            resolutions = [{"x": 32}, {"x": 64}, {"x": 128}, {"x": 256}]
        elif cls.NDIM == 2:
            resolutions = [
                {"x": 16, "y": 16},
                {"x": 32, "y": 32},
                {"x": 64, "y": 64},
            ]
        else:
            resolutions = [
                {"x": 8, "y": 8, "z": 8},
                {"x": 16, "y": 16, "z": 16},
                {"x": 32, "y": 32, "z": 32},
            ]
    return convergence_study(model, resolutions, **kwargs)
