"""
JAX engine for the semi-linear spectral seam.

Runs the SAME model spec (linear_symbol / nonlinear_hat) as the NumPy engine,
but jit-compiled, with lax.scan over time and vmap over the sample axis for
batched generation. ETDRK4 coefficients are precomputed on the host in NumPy
(pdeforge.solvers.semilinear.etdrk4_coeffs) and shipped to the device once.

Everything returned to callers is host NumPy.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from pdeforge.solvers.ops_jax import JAX_OPS  # noqa: F401  (enables x64)
from pdeforge.solvers.semilinear import etdrk4_coeffs, etdrk4_step


def _device_coeffs(model):
    """Precompute ETDRK4 coefficients on host, move to device."""
    L = np.asarray(model.linear_symbol())
    C = etdrk4_coeffs(L, model.effective_dt())
    return {k: jnp.asarray(v) for k, v in C.items()}


def _make_step(model, C):
    """Build the jit-able single-substep function v -> v'."""
    ops = JAX_OPS

    def u_of(v):
        return model._ifft(v, ops)

    def N_of(v, u):
        return model.nonlinear_hat(v, u, ops)

    # Detect purely linear problems once, on a dummy state.
    probe = jnp.zeros(model.field_shape, dtype=jnp.complex128)
    linear_only = model.nonlinear_hat(probe, model._ifft(probe, ops), ops) is None

    if linear_only:

        def step(v, _):
            return C["E"] * v, None

    else:

        def step(v, _):
            return etdrk4_step(v, u_of, N_of, C, ops), None

    return step


def solve_jax(model, ic, return_full=False):
    """Solve one trajectory on the JAX engine. Mirrors _solve_numpy exactly."""
    C = _device_coeffs(model)
    step = _make_step(model, C)

    n_substeps = model._n_substeps()
    n_t = model.n_t
    output_interval = max(1, n_substeps // max(1, n_t - 1))

    ops = JAX_OPS
    v = model._fft(jnp.asarray(ic), ops)

    run_segment = jax.jit(
        lambda v0, length: lax.scan(step, v0, None, length=length)[0],
        static_argnums=1,
    )

    frames = [np.asarray(ic, dtype=float).copy()]
    done = 0
    while done < n_substeps:
        length = min(output_interval, n_substeps - done)
        v = run_segment(v, length)
        done += length
        if done % output_interval == 0 and len(frames) < n_t:
            frames.append(ops.to_numpy(model._ifft(v, ops)))

    while len(frames) < n_t:
        frames.append(ops.to_numpy(model._ifft(v, ops)))
    frames = frames[:n_t]
    frames[-1] = ops.to_numpy(model._ifft(v, ops))

    if return_full:
        return np.stack(frames, axis=0)
    return frames[-1]


def solve_batch_final(model, ics):
    """
    Batched final-state solve: vmap over the sample axis.

    ics: (B, *field_shape) host array. Returns (B, *field_shape) host array.
    This is the fast path used by generate_dataset on the JAX backend.
    """
    C = _device_coeffs(model)
    step = _make_step(model, C)
    n_substeps = model._n_substeps()
    ops = JAX_OPS

    def solve_one(ic):
        v = model._fft(ic, ops)
        v = lax.scan(step, v, None, length=n_substeps)[0]
        return model._ifft(v, ops)

    solve_all = jax.jit(jax.vmap(solve_one))
    out = solve_all(jnp.asarray(ics))
    return np.asarray(jax.device_get(out))
