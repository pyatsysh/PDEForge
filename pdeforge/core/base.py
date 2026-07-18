"""
Base class for PDE models.

Models expose physical params (viscosity, Re, etc) but hide solver internals.
"""

import datetime
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from tqdm import tqdm

from pdeforge._version import __version__
from pdeforge.core.params import ParamSpec, ParamType, describe_params
from pdeforge.core.types import Domain, GridSpec, PDEDataset


def _git_sha():
    """Best-effort short git SHA of the source tree (None for installed copies)."""
    try:
        repo_dir = Path(__file__).resolve().parents[2]
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return sha.stdout.strip() or None
    except Exception:
        return None


def _seed_sequence(seed):
    """Coerce an int / None / SeedSequence into a SeedSequence."""
    if isinstance(seed, np.random.SeedSequence):
        return seed
    return np.random.SeedSequence(seed)


def _sample_worker(
    model, generator, generator_params, seed, validate, return_full=False
):
    """Module-level worker for process pools (must be picklable)."""
    extra = {"return_full": True} if return_full else {}
    ic, solution, _ = model.generate_sample(
        generator=generator,
        generator_params=generator_params,
        seed=seed,
        validate=validate,
        **extra,
    )
    return ic, solution


def _legacy_seed(seq):
    """Derive a 31-bit int seed for legacy np.random.seed consumers."""
    return int(seq.generate_state(1)[0] % (2**31))


class PDEModel(ABC):
    """
    Abstract base for PDE models. Subclasses implement solve() and generate_ic().
    """

    NDIM = None
    DEFAULT_PARAMS = {}
    INPUT_NAMES = ["input"]
    OUTPUT_NAMES = ["output"]
    USER_PARAMS = []
    # Process-parallel sample generation. FEM models override to False:
    # dolfinx/PETSc objects are not picklable and MPI does not mix with pools.
    PARALLEL_SAFE = True
    # Steady-state models (elliptic solves) override to False; requesting
    # outputs="trajectory" on them is an error.
    TIME_DEPENDENT = True

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

    # Which solver backends this model supports (see solvers/ops.py).
    BACKENDS = {"numpy"}

    def __init__(self, resolution, domain=None, backend="numpy", **params):
        """
        resolution: grid points per dim, e.g. {"x": 256}
        domain: bounds per dim, defaults to unit domain
        backend: solver backend for this instance ("numpy", "jax", "fenicsx")
        """
        if domain == None:
            domain = {k: (0.0, 1.0) for k in resolution.keys()}

        self.domain = Domain(bounds=domain)
        self.grid_spec = GridSpec(resolution=resolution, domain=self.domain)
        self.params = {**self.DEFAULT_PARAMS, **params}
        self.resolution = resolution
        self.backend = backend
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
        return_full=False,
    ):
        """
        Generate single (input, output) sample.
        Returns (ic, solution, info) tuple.
        """
        if generator_params is None:
            generator_params = {}

        # A SeedSequence per attempt: retries get independent, collision-free
        # streams instead of the old seed+attempt overlap with neighbours.
        attempt_seqs = _seed_sequence(seed).spawn(max_attempts)

        for attempt in range(max_attempts):
            current_seed = (
                _legacy_seed(attempt_seqs[attempt]) if seed is not None else None
            )

            ic = self.generate_ic(
                generator=generator,
                generator_params=generator_params,
                seed=current_seed,
            )

            # Forward return_full only when set, so models whose solve()
            # signature predates trajectory support keep working untouched.
            if return_full:
                try:
                    solution = self.solve(ic, return_full=True)
                except TypeError as e:
                    raise ValueError(
                        f"{self.__class__.__name__} does not support "
                        "outputs='trajectory' yet."
                    ) from e
            else:
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
        outputs="final",
    ):
        """
        Generate dataset of (input, output) pairs.

        n_samples: number of samples
        ic_generator: type of IC generator or callable
        seed: random seed for reproducability
        n_jobs: parallel worker processes (1 = sequential, -1 = all cores).
            FEM models ignore this and run sequentially.
        outputs: "final" for the end-time snapshot (default), "trajectory"
            for full rollouts shaped (n_samples, n_t, *spatial). Emulator
            benchmarks want trajectories; operator-learning IO maps want final.
        """
        if ic_params == None:
            ic_params = {}

        if outputs not in ("final", "trajectory"):
            raise ValueError(
                f"outputs must be 'final' or 'trajectory', got {outputs!r}"
            )
        if outputs == "trajectory" and not self.TIME_DEPENDENT:
            raise ValueError(
                f"{self.__class__.__name__} is a steady-state model; "
                "outputs='trajectory' is not available."
            )

        # Per-sample SeedSequences spawned from the root seed: collision-free,
        # order-independent, and safe to hand to parallel workers.
        sample_seqs = _seed_sequence(seed).spawn(n_samples)

        ins, outs = self._generate_samples(
            n_samples=n_samples,
            sample_seqs=sample_seqs,
            ic_generator=ic_generator,
            ic_params=ic_params,
            seed=seed,
            validate=validate,
            n_jobs=n_jobs,
            verbose=verbose,
            return_full=(outputs == "trajectory"),
        )

        inputs = np.stack(ins, axis=0)
        outs = np.stack(outs, axis=0)

        return PDEDataset(
            inputs=inputs,
            outputs=outs,
            grid=self.dataset_grid(outputs),
            metadata=self.dataset_metadata(
                n_samples, ic_generator, ic_params, seed, outputs
            ),
            input_names=self.INPUT_NAMES,
            output_names=self.OUTPUT_NAMES,
        )

    def dataset_grid(self, outputs="final"):
        """Grid dict for a dataset, including "t" for trajectory outputs."""
        grid = self.grids.copy()
        if outputs == "trajectory" and hasattr(self, "T") and hasattr(self, "n_t"):
            grid["t"] = np.linspace(0.0, self.T, self.n_t)
        return grid

    def dataset_metadata(self, n_samples, ic_generator, ic_params, seed, outputs):
        """Provenance metadata block (shared by in-memory and streaming paths)."""
        return {
            "model": getattr(self, "_registered_name", self.__class__.__name__),
            "n_samples": n_samples,
            "resolution": dict(self.grid_spec.resolution),
            "domain": {k: list(v) for k, v in self.domain.bounds.items()},
            "params": self.params,
            "ic_generator": ic_generator if isinstance(ic_generator, str) else "custom",
            "ic_params": ic_params,
            "seed": seed,
            "backend": self.backend,
            "outputs": outputs,
            "pdeforge_version": __version__,
            "git_sha": _git_sha(),
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    def _generate_samples(
        self,
        n_samples,
        sample_seqs,
        ic_generator,
        ic_params,
        seed,
        validate,
        n_jobs,
        verbose,
        return_full=False,
    ):
        """Run the per-sample generation loop, sequentially or in parallel."""
        seeds = [_legacy_seed(seq) if seed is not None else None for seq in sample_seqs]

        if n_jobs != 1 and not self.PARALLEL_SAFE:
            import warnings

            warnings.warn(
                f"{self.__class__.__name__} does not support process-parallel "
                "generation (FEM/MPI solver); falling back to n_jobs=1.",
                stacklevel=3,
            )
            n_jobs = 1

        # JAX fast path (final-state only): host ICs, one vmapped batch solve.
        if self.backend == "jax" and hasattr(self, "_solve_numpy") and not return_full:
            return self._generate_samples_jax_batch(
                n_samples, seeds, ic_generator, ic_params, validate, verbose
            )

        # Forward return_full only when set — overridden generate_sample
        # signatures (stochastic, FEM) predate the kwarg.
        extra = {"return_full": True} if return_full else {}

        if n_jobs == 1 or self.backend == "jax":
            inputs, outputs = [], []
            iterator = tqdm(
                range(n_samples), disable=not verbose, desc="Generating samples"
            )
            for i in iterator:
                ic, solution, _ = self.generate_sample(
                    generator=ic_generator,
                    generator_params=ic_params,
                    seed=seeds[i],
                    validate=validate,
                    **extra,
                )
                inputs.append(ic)
                outputs.append(solution)
            return inputs, outputs

        # Parallel path: each worker re-runs generate_sample with its own seed.
        from concurrent.futures import ProcessPoolExecutor, as_completed

        max_workers = None if n_jobs in (-1, 0) else n_jobs
        results = [None] * n_samples
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _sample_worker,
                    self,
                    ic_generator,
                    ic_params,
                    seeds[i],
                    validate,
                    return_full,
                ): i
                for i in range(n_samples)
            }  # _sample_worker forwards return_full only when True
            iterator = tqdm(
                as_completed(futures),
                total=n_samples,
                disable=not verbose,
                desc=f"Generating samples ({n_jobs} workers)",
            )
            for fut in iterator:
                i = futures[fut]
                results[i] = fut.result()

        inputs = [r[0] for r in results]
        outputs = [r[1] for r in results]
        return inputs, outputs

    def _generate_samples_jax_batch(
        self, n_samples, seeds, ic_generator, ic_params, validate, verbose
    ):
        """Batched generation on the JAX engine: host ICs, one vmapped solve."""
        from pdeforge.solvers.engine_jax import solve_batch_final

        if verbose:
            print(f"Generating {n_samples} samples (JAX batched solve)...")

        # Mirror generate_sample()'s derivation (attempt-0 spawned child) so
        # ICs are bit-identical across the sequential and batched paths.
        eff_seeds = [
            _legacy_seed(_seed_sequence(s).spawn(1)[0]) if s is not None else None
            for s in seeds
        ]
        ics = [
            self.generate_ic(generator=ic_generator, generator_params=ic_params, seed=s)
            for s in eff_seeds
        ]
        ics = np.stack(ics, axis=0)
        outs = solve_batch_final(self, ics)

        if validate:
            bad = [
                i
                for i in range(n_samples)
                if not self.validate_solution(ics[i], outs[i])["valid"]
            ]
            if bad:
                # Fall back to the sequential retry machinery for failures.
                for i in bad:
                    ic, sol, _ = self.generate_sample(
                        generator=ic_generator,
                        generator_params=ic_params,
                        seed=seeds[i],
                        validate=True,
                    )
                    ics[i], outs[i] = ic, sol

        return list(ics), list(outs)

    def preview(self, n_samples=3, seed=42):
        """Generate and visualise a few samples."""
        from pdeforge.visualization.interactive import preview_samples

        dataset = self.generate_dataset(
            n_samples=n_samples,
            seed=seed,
            verbose=False,
        )
        return preview_samples(dataset)
