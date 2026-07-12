"""
Base class for PDE models.

Models expose physical params (viscosity, Re, etc) but hide solver internals.
"""

from abc import ABC, abstractmethod

import numpy as np
from tqdm import tqdm

from pdeforge.core.params import ParamSpec, ParamType, describe_params
from pdeforge.core.types import Domain, GridSpec, PDEDataset


class PDEModel(ABC):
    """
    Abstract base for PDE models. Subclasses implement solve() and generate_ic().
    """

    NDIM = None
    DEFAULT_PARAMS = {}
    INPUT_NAMES = ["input"]
    OUTPUT_NAMES = ["output"]
    USER_PARAMS = []

    @classmethod
    def describe(cls):
        """Get model description and configurable params."""
        ndim_str = f"{cls.NDIM}D" if cls.NDIM else "2D / 3D"
        lines = [
            f"Model: {getattr(cls, '_registered_name', cls.__name__)}",
            f"Dimensions: {ndim_str}",
            "",
            cls.__doc__ or "No description available.",
            "",
            "Input/Output:",
            f"  Inputs:  {cls.INPUT_NAMES}",
            f"  Outputs: {cls.OUTPUT_NAMES}",
        ]

        if cls.USER_PARAMS:
            lines.append(describe_params(cls.USER_PARAMS))
        else:
            lines.append("\nParameters: See DEFAULT_PARAMS")

        return "\n".join(lines)

    @classmethod
    def get_user_params(cls):
        """Get dict of user-facing parameters."""
        return {p.name: p for p in cls.USER_PARAMS}

    def __init__(self, resolution, domain=None, **params):
        """
        resolution: grid points per dim, e.g. {"x": 256}
        domain: bounds per dim, defaults to unit domain
        """
        if domain == None:
            domain = {k: (0.0, 1.0) for k in resolution.keys()}

        self.domain = Domain(bounds=domain)
        self.grid_spec = GridSpec(resolution=resolution, domain=self.domain)
        self.params = {**self.DEFAULT_PARAMS, **params}
        self.resolution = resolution
        self._setup_grids()

    def _setup_grids(self):
        self.grids = {}
        for dim in self.grid_spec.resolution.keys():
            self.grids[dim] = self.grid_spec.get_grid(dim, endpoint=False)

    @abstractmethod
    def solve(self, ic, **kwargs):
        pass

    @abstractmethod
    def generate_ic(self, generator="fourier", generator_params=None, seed=None):
        pass

    def validate_solution(self, ic, solution, tol=1e-6):
        is_valid = not np.isnan(solution).any() and not np.isinf(solution).any()
        return {"valid": is_valid}

    def generate_sample(
        self,
        generator="fourier",
        generator_params=None,
        seed=None,
        validate=True,
        max_attempts=10,
    ):
        """
        Generate single (input, output) sample.
        Returns (ic, solution, info) tuple.
        """
        if generator_params is None:
            generator_params = {}

        for attempt in range(max_attempts):
            current_seed = seed + attempt if seed is not None else None

            ic = self.generate_ic(
                generator=generator,
                generator_params=generator_params,
                seed=current_seed,
            )

            solution = self.solve(ic)

            if validate:
                validation = self.validate_solution(ic, solution)
                if validation["valid"]:
                    return ic, solution, validation
            else:
                return ic, solution, {"valid": True}

        raise RuntimeError("sample generation failed")

    def generate_dataset(
        self,
        n_samples,
        ic_generator="fourier",
        ic_params=None,
        seed=None,
        validate=True,
        n_jobs=1,
        verbose=True,
    ):
        """
        Generate dataset of (input, output) pairs.

        n_samples: number of samples
        ic_generator: type of IC generator or callable
        seed: random seed for reproducability
        n_jobs: parallel workers (not fully implemented yet)
        """
        if ic_params == None:
            ic_params = {}

        inputs = []
        outputs = []

        # setup random seeds
        if seed is not None:
            np.random.seed(seed)
        seeds = [np.random.randint(0, 2**31) for _ in range(n_samples)]

        iterator = tqdm(
            range(n_samples), disable=not verbose, desc="Generating samples"
        )

        # sequential for now, parallel TODO
        for i in iterator:
            ic, solution, _ = self.generate_sample(
                generator=ic_generator,
                generator_params=ic_params,
                seed=seeds[i],
                validate=validate,
            )
            inputs.append(ic)
            outputs.append(solution)

        inputs = np.stack(inputs, axis=0)
        outputs = np.stack(outputs, axis=0)

        metadata = {
            "model": getattr(self, "_registered_name", self.__class__.__name__),
            "n_samples": n_samples,
            "resolution": dict(self.grid_spec.resolution),
            "domain": {k: list(v) for k, v in self.domain.bounds.items()},
            "params": self.params,
            "ic_generator": ic_generator if isinstance(ic_generator, str) else "custom",
            "ic_params": ic_params,
            "seed": seed,
        }

        return PDEDataset(
            inputs=inputs,
            outputs=outputs,
            grid=self.grids.copy(),
            metadata=metadata,
            input_names=self.INPUT_NAMES,
            output_names=self.OUTPUT_NAMES,
        )

    def preview(self, n_samples=3, seed=42):
        """Generate and visualise a few samples."""
        from pdeforge.visualization.interactive import preview_samples

        dataset = self.generate_dataset(
            n_samples=n_samples,
            seed=seed,
            verbose=False,
        )
        return preview_samples(dataset)
