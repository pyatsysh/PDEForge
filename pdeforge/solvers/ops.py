"""
Backend array-ops namespaces for the spectral solver seam.

A model's solver spec (linear symbol + nonlinear term) is written against this
tiny ops surface, so the same spec runs on the NumPy engine today and the JAX
engine when requested. Everything crossing the PDEDataset boundary remains host
NumPy — backends are engines, never surface types.
"""

import numpy as np


class NumpyOps:
    """NumPy implementation of the solver ops surface."""

    name = "numpy"

    @staticmethod
    def asarray(a):
        return np.asarray(a)

    @staticmethod
    def fftn(a, axes):
        return np.fft.fftn(a, axes=axes)

    @staticmethod
    def ifftn(a, axes):
        return np.fft.ifftn(a, axes=axes)

    @staticmethod
    def real(a):
        return np.real(a)

    @staticmethod
    def abs(a):
        return np.abs(a)

    @staticmethod
    def exp(a):
        return np.exp(a)

    @staticmethod
    def where(cond, a, b):
        return np.where(cond, a, b)

    @staticmethod
    def stack(arrays, axis=0):
        return np.stack(arrays, axis=axis)

    @staticmethod
    def to_numpy(a):
        return np.asarray(a)


_NUMPY_OPS = NumpyOps()


def jax_available():
    """True if jax is importable."""
    try:
        import jax  # noqa: F401

        return True
    except ImportError:
        return False


def get_ops(backend):
    """Return the ops namespace for a backend name."""
    if backend == "numpy":
        return _NUMPY_OPS
    if backend == "jax":
        from pdeforge.solvers.ops_jax import JAX_OPS

        return JAX_OPS
    raise ValueError(f"Unknown backend: {backend!r}")


def resolve_backend(model_cls, requested="auto"):
    """
    Resolve a requested backend against a model class's declared BACKENDS.

    Rules (see the project design note):
    - FEM models always run on "fenicsx"; requesting numpy/jax on them errors.
    - "auto" resolves to "numpy" for spectral models — deterministic default;
      JAX is strictly opt-in (backend="jax").
    """
    supported = getattr(model_cls, "BACKENDS", {"numpy"})

    if "fenicsx" in supported:
        if requested in ("auto", "fenicsx"):
            return "fenicsx"
        raise ValueError(
            f"{model_cls.__name__} is a FEM model and only runs on the "
            f"'fenicsx' backend (got backend={requested!r})."
        )

    if requested == "auto":
        return "numpy"
    if requested == "numpy":
        return "numpy"
    if requested == "jax":
        if "jax" not in supported:
            raise ValueError(
                f"{model_cls.__name__} does not support the JAX backend "
                f"(supported: {sorted(supported)})."
            )
        if not jax_available():
            raise ImportError(
                "backend='jax' requested but jax is not installed. "
                "Install with: pip install pdeforge[jax]"
            )
        return "jax"
    raise ValueError(f"Unknown backend: {requested!r}")
