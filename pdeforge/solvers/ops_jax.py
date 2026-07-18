"""
JAX implementation of the solver ops surface.

Import-time side effect: enables float64 so that the JAX engine matches the
NumPy engine to solver tolerance (see the backend design note — never let the
silent float32 default into a convergence study). Datasets can still be saved
as float32 at IO time.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np


class JaxOps:
    """JAX implementation of the solver ops surface."""

    name = "jax"

    @staticmethod
    def asarray(a):
        return jnp.asarray(a)

    @staticmethod
    def fftn(a, axes):
        return jnp.fft.fftn(a, axes=axes)

    @staticmethod
    def ifftn(a, axes):
        return jnp.fft.ifftn(a, axes=axes)

    @staticmethod
    def real(a):
        return jnp.real(a)

    @staticmethod
    def abs(a):
        return jnp.abs(a)

    @staticmethod
    def exp(a):
        return jnp.exp(a)

    @staticmethod
    def where(cond, a, b):
        return jnp.where(cond, a, b)

    @staticmethod
    def stack(arrays, axis=0):
        return jnp.stack(arrays, axis=axis)

    @staticmethod
    def to_numpy(a):
        return np.asarray(jax.device_get(a))


JAX_OPS = JaxOps()
