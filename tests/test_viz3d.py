"""
3D visualization tests: PyVista wrap + mesh extraction (compute only — no
rendering, so they run headless), and the matplotlib fallback.
"""

import numpy as np
import pytest

from pdeforge import generate_dataset


@pytest.fixture(scope="module")
def d3():
    return generate_dataset(
        "darcy_fno_3d",
        n_samples=1,
        resolution={"x": 17, "y": 17, "z": 17},
        seed=0,
        verbose=False,
    )


class TestImageDataWrap:
    def test_dims_spacing_and_voxel_roundtrip(self, d3):
        pv = pytest.importorskip("pyvista")
        from pdeforge.visualization import dataset_to_imagedata

        img = dataset_to_imagedata(d3, sample=0, which="output")
        assert img.dimensions == (17, 17, 17)
        # spacing from the boundary-inclusive unit-cube grid: 1/16
        assert np.allclose(img.spacing, (1 / 16, 1 / 16, 1 / 16))
        # voxel value roundtrip at a known (iz, iy, ix)
        field = np.asarray(d3.outputs[0])
        name = img.point_data.keys()[0]
        vals = np.asarray(img.point_data[name]).reshape(17, 17, 17)
        assert np.allclose(vals, field)

    def test_input_field_too(self, d3):
        pytest.importorskip("pyvista")
        from pdeforge.visualization import dataset_to_imagedata

        img = dataset_to_imagedata(d3, sample=0, which="input")
        name = img.point_data.keys()[0]
        assert np.asarray(img.point_data[name]).min() > 0  # permeability > 0

    def test_rejects_2d(self):
        pytest.importorskip("pyvista")
        from pdeforge.visualization import dataset_to_imagedata

        d2 = generate_dataset(
            "heat_2d",
            n_samples=1,
            resolution={"x": 8, "y": 8},
            seed=0,
            verbose=False,
        )
        with pytest.raises(ValueError, match="3D"):
            dataset_to_imagedata(d2)


class TestMeshExtraction:
    def test_isosurface_has_geometry(self, d3):
        pytest.importorskip("pyvista")
        from pdeforge.visualization import dataset_to_imagedata

        img = dataset_to_imagedata(d3)
        name = img.point_data.keys()[0]
        lo, hi = img.get_data_range(name)
        contours = img.contour(isosurfaces=np.linspace(lo, hi, 5)[1:-1], scalars=name)
        assert contours.n_points > 0  # real level sets exist

    def test_orthogonal_slices(self, d3):
        pytest.importorskip("pyvista")
        from pdeforge.visualization import dataset_to_imagedata

        img = dataset_to_imagedata(d3)
        slices = img.slice_orthogonal()
        assert len(slices) == 3
        assert sum(s.n_points for s in slices) > 0


class TestMatplotlibFallback:
    def test_slices_figure_saved(self, d3, tmp_path):
        import matplotlib

        matplotlib.use("Agg")
        from pdeforge.visualization import plot_3d_slices

        out = tmp_path / "slices.png"
        fig = plot_3d_slices(d3, sample=0, save_path=str(out))
        assert out.exists() and out.stat().st_size > 10_000
        assert len(fig.axes) >= 3

    def test_raw_array_accepted(self, tmp_path):
        import matplotlib

        matplotlib.use("Agg")
        from pdeforge.visualization import plot_3d_slices

        field = np.random.default_rng(0).random((9, 9, 9))
        fig = plot_3d_slices(field, save_path=str(tmp_path / "raw.png"))
        assert (tmp_path / "raw.png").exists()
