"""
Chunked-to-disk generation: remove the RAM ceiling for large datasets.

Samples are generated in chunks and written incrementally — to the directory
format (preallocated .npy memmaps; loadable lazily with mmap=True) or to a
resizable HDF5 file. Per-sample seeds are spawned from the SAME root
SeedSequence as the in-memory path, so sample i is bit-identical to what
generate_dataset() without `to=` would have produced.
"""

import json
from pathlib import Path

import numpy as np

from pdeforge.core.base import _seed_sequence


def generate_streaming(
    model,
    n_samples,
    path,
    ic_generator="fourier",
    ic_params=None,
    seed=None,
    validate=True,
    n_jobs=1,
    verbose=True,
    outputs="final",
    chunk_size=256,
):
    """
    Generate a dataset chunk-by-chunk, writing each chunk to disk.

    path ending in .h5/.hdf5 -> single resizable HDF5 file;
    anything else -> directory format with memmapped .npy arrays.
    Returns the path (load with pdeforge.load_dataset; the directory format
    supports mmap=True for lazy, larger-than-RAM reads).
    """
    if ic_params is None:
        ic_params = {}
    path = Path(path)

    sample_seqs = _seed_sequence(seed).spawn(n_samples)
    return_full = outputs == "trajectory"

    # Probe one sample for shapes (its result is written, not discarded).
    probe_in, probe_out = model._generate_samples(
        n_samples=1,
        sample_seqs=sample_seqs[:1],
        ic_generator=ic_generator,
        ic_params=ic_params,
        seed=seed,
        validate=validate,
        n_jobs=1,
        verbose=False,
        return_full=return_full,
    )
    in_shape = np.asarray(probe_in[0]).shape
    out_shape = np.asarray(probe_out[0]).shape
    in_dtype = np.asarray(probe_in[0]).dtype
    out_dtype = np.asarray(probe_out[0]).dtype

    metadata = model.dataset_metadata(n_samples, ic_generator, ic_params, seed, outputs)
    grid = model.dataset_grid(outputs)
    meta_full = {
        **metadata,
        "input_names": model.INPUT_NAMES,
        "output_names": model.OUTPUT_NAMES,
        "grid_dims": list(grid.keys()),
    }

    is_hdf5 = path.suffix in (".h5", ".hdf5")

    if is_hdf5:
        import h5py

        f = h5py.File(path, "w")
        d_in = f.create_dataset(
            "inputs", shape=(n_samples, *in_shape), dtype=in_dtype, compression="gzip"
        )
        d_out = f.create_dataset(
            "outputs",
            shape=(n_samples, *out_shape),
            dtype=out_dtype,
            compression="gzip",
        )
        grid_grp = f.create_group("grid")
        for dim, coords in grid.items():
            grid_grp.create_dataset(dim, data=coords)
        f.attrs["input_names"] = json.dumps(model.INPUT_NAMES)
        f.attrs["output_names"] = json.dumps(model.OUTPUT_NAMES)
        f.attrs["metadata"] = json.dumps(metadata, default=str)

        def write(lo, hi, ins, outs):
            d_in[lo:hi] = np.stack(ins, axis=0)
            d_out[lo:hi] = np.stack(outs, axis=0)

        def close():
            f.close()

    else:
        path.mkdir(parents=True, exist_ok=True)
        d_in = np.lib.format.open_memmap(
            path / "inputs.npy", mode="w+", dtype=in_dtype, shape=(n_samples, *in_shape)
        )
        d_out = np.lib.format.open_memmap(
            path / "outputs.npy",
            mode="w+",
            dtype=out_dtype,
            shape=(n_samples, *out_shape),
        )
        for dim, coords in grid.items():
            np.save(path / f"grid_{dim}.npy", coords)
        with open(path / "metadata.json", "w") as fh:
            json.dump(meta_full, fh, indent=2, default=str)

        def write(lo, hi, ins, outs):
            d_in[lo:hi] = np.stack(ins, axis=0)
            d_out[lo:hi] = np.stack(outs, axis=0)

        def close():
            d_in.flush()
            d_out.flush()

    try:
        write(0, 1, probe_in, probe_out)
        done = 1
        while done < n_samples:
            hi = min(done + chunk_size, n_samples)
            ins, outs = model._generate_samples(
                n_samples=hi - done,
                sample_seqs=sample_seqs[done:hi],
                ic_generator=ic_generator,
                ic_params=ic_params,
                seed=seed,
                validate=validate,
                n_jobs=n_jobs,
                verbose=verbose,
                return_full=return_full,
            )
            write(done, hi, ins, outs)
            done = hi
            if verbose:
                print(f"  wrote {done}/{n_samples} samples -> {path}")
    finally:
        close()

    return path
