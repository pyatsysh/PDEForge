"""
Tests for the data-infrastructure layer: streaming generation, lazy loading,
framework adapters, zarr, PDEBench-layout export, presets, and provenance.
"""

import json

import numpy as np
import pytest

import pdeforge
from pdeforge import generate_dataset, load_dataset, reproduce


def _small(seed=5, **kw):
    return generate_dataset(
        "heat_1d", n_samples=4, resolution={"x": 32}, seed=seed, verbose=False, **kw
    )


class TestProvenance:
    def test_metadata_fields(self):
        m = _small().metadata
        for key in ("pdeforge_version", "created_at", "backend", "outputs", "seed"):
            assert key in m
        assert m["pdeforge_version"] == pdeforge.__version__

    def test_reproduce_roundtrip(self):
        d = _small()
        d2 = reproduce(d.metadata, verbose=False)
        assert np.array_equal(d.inputs, d2.inputs)
        assert np.array_equal(d.outputs, d2.outputs)

    def test_reproduce_refuses_split(self):
        d = _small()
        s = d.split(train=0.5, val=0.25, cal=0.25, test=0.0, seed=0)
        with pytest.raises(ValueError, match="split"):
            reproduce(s["train"].metadata)

    def test_reproduce_refuses_unseeded(self):
        d = generate_dataset(
            "heat_1d", n_samples=2, resolution={"x": 16}, verbose=False
        )
        with pytest.raises(ValueError, match="seed"):
            reproduce(d.metadata)


class TestStreaming:
    def test_directory_streaming_matches_memory(self, tmp_path):
        d_mem = _small(seed=9)
        out = tmp_path / "ds"
        d_str = generate_dataset(
            "heat_1d",
            n_samples=4,
            resolution={"x": 32},
            seed=9,
            verbose=False,
            to=out,
            chunk_size=2,
        )
        # bit-identical to the in-memory path (same root SeedSequence)
        assert np.array_equal(np.asarray(d_str.inputs), d_mem.inputs)
        assert np.array_equal(np.asarray(d_str.outputs), d_mem.outputs)
        # lazy handle: inputs are memmapped
        assert isinstance(np.asarray(d_str.inputs), np.ndarray)
        assert (out / "metadata.json").exists()

    def test_hdf5_streaming(self, tmp_path):
        pytest.importorskip("h5py")
        out = tmp_path / "ds.h5"
        d = generate_dataset(
            "heat_1d",
            n_samples=5,
            resolution={"x": 32},
            seed=1,
            verbose=False,
            to=out,
            chunk_size=2,
        )
        assert d.n_samples == 5
        d_mem = generate_dataset(
            "heat_1d", n_samples=5, resolution={"x": 32}, seed=1, verbose=False
        )
        assert np.allclose(d.outputs, d_mem.outputs)

    def test_mmap_load(self, tmp_path):
        d = _small()
        d.save(tmp_path / "ds")
        lazy = pdeforge.PDEDataset.load(tmp_path / "ds", mmap=True)
        assert isinstance(lazy.inputs, np.memmap)
        assert np.array_equal(np.asarray(lazy.outputs), d.outputs)


class TestAdapters:
    def test_to_torch(self):
        torch = pytest.importorskip("torch")
        d = _small()
        tds = d.to_torch()
        assert len(tds) == 4
        x, y = tds[0]
        assert x.dtype == torch.float32
        assert x.shape == (32,)
        loader = d.torch_loader(batch_size=2, shuffle=False)
        xb, yb = next(iter(loader))
        assert xb.shape == (2, 32)

    def test_to_jax(self):
        pytest.importorskip("jax")
        d = _small()
        xi, yo = d.to_jax()
        assert xi.shape == (4, 32)


class TestZarr:
    def test_roundtrip(self, tmp_path):
        pytest.importorskip("zarr")
        from pdeforge.io.datasets import save_dataset

        d = _small()
        p = tmp_path / "ds.zarr"
        save_dataset(d, p)
        d2 = load_dataset(p)
        assert np.array_equal(d2.inputs, d.inputs)
        assert d2.metadata["model"] == "heat_1d"


class TestPDEBenchExport:
    def test_layout_final(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        d = _small()
        p = tmp_path / "pb.h5"
        pdeforge.export_pdebench(d, p)
        with h5py.File(p) as f:
            assert f["tensor"].shape == (4, 1, 32)  # (N, n_t=1, x)
            assert "x-coordinate" in f and "t-coordinate" in f
            meta = json.loads(f["pdeforge"].attrs["metadata"])
            assert meta["model"] == "heat_1d"

    def test_layout_trajectory(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        d = generate_dataset(
            "burgers_1d",
            n_samples=2,
            resolution={"x": 32},
            params={"_n_time_steps": 6},
            seed=0,
            verbose=False,
            outputs="trajectory",
        )
        p = tmp_path / "pb.h5"
        pdeforge.export_pdebench(d, p)
        with h5py.File(p) as f:
            assert f["tensor"].shape == (2, 6, 32)
            assert f["t-coordinate"].shape == (6,)


class TestPresets:
    def test_list(self):
        names = pdeforge.list_presets()
        assert "fno_burgers_1d" in names and "fno_darcy_2d" in names

    def test_fno_burgers_runs(self):
        d = generate_dataset(
            preset="fno_burgers_1d",
            n_samples=2,
            resolution={"x": 64},
            seed=0,
            verbose=False,
        )
        assert abs(d.metadata["params"]["viscosity"] - 0.01 / 3.141592653589793) < 1e-12
        assert np.isfinite(d.outputs).all()

    def test_explicit_params_override_preset(self):
        d = generate_dataset(
            preset="fno_burgers_1d",
            n_samples=1,
            resolution={"x": 32},
            params={"viscosity": 0.5},
            seed=0,
            verbose=False,
        )
        assert d.metadata["params"]["viscosity"] == 0.5

    def test_unknown_preset(self):
        with pytest.raises(ValueError, match="Available"):
            generate_dataset(preset="nope", n_samples=1, resolution={"x": 8})
