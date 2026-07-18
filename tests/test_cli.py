"""
CLI tests: the appliance surface. Everything runs main() in-process.
"""

import json

import numpy as np
import pytest

from pdeforge import load_dataset
from pdeforge.cli import main


class TestGenerate:
    def test_generate_dir(self, tmp_path):
        out = tmp_path / "ds"
        rc = main(
            [
                "generate",
                "--model",
                "heat_1d",
                "--n",
                "4",
                "--resolution",
                "x=32",
                "--seed",
                "0",
                "--out",
                str(out),
                "--quiet",
            ]
        )
        assert rc == 0
        d = load_dataset(out)
        assert d.n_samples == 4
        assert d.metadata["model"] == "heat_1d"
        assert d.metadata["seed"] == 0

    def test_generate_with_params_and_preset(self, tmp_path):
        out = tmp_path / "darcy"
        rc = main(
            [
                "generate",
                "--preset",
                "fno_darcy_2d",
                "--n",
                "2",
                "--resolution",
                "x=41",
                "y=41",
                "--param",
                "sigma=0.2",
                "--seed",
                "1",
                "--out",
                str(out),
                "--quiet",
            ]
        )
        assert rc == 0
        d = load_dataset(out)
        assert d.metadata["params"]["sigma"] == 0.2  # override beat the preset
        assert d.outputs.shape == (2, 41, 41)

    def test_generate_h5(self, tmp_path):
        pytest.importorskip("h5py")
        out = tmp_path / "ds"
        main(
            [
                "generate",
                "--model",
                "heat_1d",
                "--n",
                "2",
                "--resolution",
                "x=16",
                "--seed",
                "0",
                "--out",
                str(out),
                "--format",
                "h5",
                "--quiet",
            ]
        )
        d = load_dataset(str(out) + ".h5")
        assert d.n_samples == 2

    def test_generate_pdebench_layout(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        out = tmp_path / "pb.h5"
        main(
            [
                "generate",
                "--model",
                "heat_1d",
                "--n",
                "2",
                "--resolution",
                "x=16",
                "--seed",
                "0",
                "--out",
                str(out),
                "--format",
                "pdebench",
                "--quiet",
            ]
        )
        with h5py.File(out) as f:
            assert f["tensor"].shape == (2, 1, 16)

    def test_generate_chunked(self, tmp_path):
        out = tmp_path / "big"
        main(
            [
                "generate",
                "--model",
                "heat_1d",
                "--n",
                "6",
                "--resolution",
                "x=16",
                "--seed",
                "3",
                "--out",
                str(out),
                "--chunk-size",
                "2",
                "--quiet",
            ]
        )
        d = load_dataset(out)
        assert d.n_samples == 6

    def test_bad_kv_errors(self, tmp_path):
        with pytest.raises(SystemExit):
            main(
                [
                    "generate",
                    "--model",
                    "heat_1d",
                    "--n",
                    "1",
                    "--resolution",
                    "x:16",  # not KEY=VALUE
                    "--out",
                    str(tmp_path / "x"),
                ]
            )


class TestReproduce:
    def test_roundtrip(self, tmp_path):
        src = tmp_path / "src"
        main(
            [
                "generate",
                "--model",
                "burgers_1d",
                "--n",
                "2",
                "--resolution",
                "x=32",
                "--seed",
                "7",
                "--out",
                str(src),
                "--quiet",
            ]
        )
        dst = tmp_path / "dst"
        rc = main(
            ["reproduce", str(src / "metadata.json"), "--out", str(dst), "--quiet"]
        )
        assert rc == 0
        a, b = load_dataset(src), load_dataset(dst)
        assert np.array_equal(np.asarray(a.outputs), np.asarray(b.outputs))


class TestListings:
    def test_models(self, capsys):
        assert main(["models"]) == 0
        out = capsys.readouterr().out
        assert "burgers_1d" in out and "darcy_fno_3d" in out

    def test_presets(self, capsys):
        assert main(["presets"]) == 0
        out = capsys.readouterr().out
        assert "fno_darcy_2d" in out and "burgers_rough_1d" in out

    def test_describe(self, capsys):
        assert main(["describe", "heat_1d"]) == 0
        assert "diffusivity" in capsys.readouterr().out
