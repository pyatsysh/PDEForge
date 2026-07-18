"""
3D volumetric visualization for PDEForge datasets.

PyVista-based (optional dependency: pip install pdeforge[viz3d]) with a
matplotlib orthogonal-slices fallback that needs no GPU or display — so
docs, CI, and headless boxes always have a working view.

Field convention: 3D PDEForge fields are (nz, ny, nx) with the grid
coordinates in dataset.grid["x"|"y"|"z"] (C-order ravel of (nz, ny, nx) is
exactly VTK's x-fastest point order, so the wrap is copy-free).

    from pdeforge.visualization import visualize_3d
    p = visualize_3d(dataset, sample=0, mode="isosurface")
    p.show()                       # interactive window / notebook widget
"""

from typing import Optional, Sequence, Union

import numpy as np


def _extract_field(dataset, sample=0, which="output"):
    """(field, name, grid) from a PDEDataset or a raw (nz, ny, nx) array."""
    if hasattr(dataset, "outputs"):
        arr = dataset.outputs if which == "output" else dataset.inputs
        field = np.asarray(arr[sample], dtype=float)
        names = dataset.output_names if which == "output" else dataset.input_names
        name = names[0] if names else which
        grid = dataset.grid
    else:
        field = np.asarray(dataset, dtype=float)
        name, grid = "field", None
    if field.ndim != 3:
        raise ValueError(
            f"visualize_3d needs a 3D field; got shape {field.shape}. "
            "For 1D/2D data use dataset.visualize() (the widget explorer)."
        )
    return field, name, grid


def dataset_to_imagedata(dataset, sample=0, which="output"):
    """
    Wrap a 3D field as a pyvista.ImageData (uniform grid) with physical
    spacing/origin taken from the dataset's grid coordinates.
    """
    try:
        import pyvista as pv
    except ImportError as e:
        raise ImportError(
            "PyVista is required for 3D visualization: pip install pdeforge[viz3d]"
        ) from e

    field, name, grid = _extract_field(dataset, sample, which)
    nz, ny, nx = field.shape

    def axis(gname, n):
        if grid is not None and gname in grid:
            g = np.asarray(grid[gname])
            step = g[1] - g[0] if len(g) > 1 else 1.0
            return float(g[0]), float(step)
        return 0.0, 1.0 / max(n - 1, 1)

    x0, dx = axis("x", nx)
    y0, dy = axis("y", ny)
    z0, dz = axis("z", nz)

    img = pv.ImageData(
        dimensions=(nx, ny, nz), spacing=(dx, dy, dz), origin=(x0, y0, z0)
    )
    # (nz, ny, nx) C-ravel is x-fastest: exactly VTK point order.
    img.point_data[name] = field.ravel()
    return img


def visualize_3d(
    dataset,
    sample: int = 0,
    which: str = "output",
    mode: str = "isosurface",
    isosurfaces: Union[int, Sequence[float]] = 5,
    cmap: str = "viridis",
    opacity: Union[str, float] = "sigmoid",
    off_screen: bool = False,
    show: bool = False,
):
    """
    Render a 3D field with PyVista. Returns the Plotter (call .show(), or
    .screenshot(path) with off_screen=True).

    mode: "isosurface" (contour shells), "volume" (volume rendering), or
    "slices" (three orthogonal planes).
    """
    import pyvista as pv

    img = dataset_to_imagedata(dataset, sample, which)
    name = img.point_data.keys()[0]

    plotter = pv.Plotter(off_screen=off_screen)
    if mode == "isosurface":
        lo, hi = img.get_data_range(name)
        if isinstance(isosurfaces, int):
            levels = np.linspace(lo, hi, isosurfaces + 2)[1:-1]
        else:
            levels = np.asarray(isosurfaces, dtype=float)
        contours = img.contour(isosurfaces=levels, scalars=name)
        plotter.add_mesh(contours, cmap=cmap, opacity=0.6, show_scalar_bar=True)
    elif mode == "volume":
        plotter.add_volume(img, scalars=name, cmap=cmap, opacity=opacity)
    elif mode == "slices":
        plotter.add_mesh(img.slice_orthogonal(), cmap=cmap, show_scalar_bar=True)
    else:
        raise ValueError(f"Unknown mode: {mode!r} (isosurface | volume | slices)")

    plotter.add_axes()
    plotter.add_text(f"{name} (sample {sample})", font_size=10)
    if show:
        plotter.show()
    return plotter


def plot_3d_slices(
    dataset,
    sample: int = 0,
    which: str = "output",
    cmap: str = "viridis",
    save_path: Optional[str] = None,
):
    """
    Matplotlib fallback: the three orthogonal mid-plane slices. Works
    everywhere (no GPU, no display with the Agg backend). Returns the figure.
    """
    import matplotlib.pyplot as plt

    field, name, grid = _extract_field(dataset, sample, which)
    nz, ny, nx = field.shape

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    planes = [
        (field[nz // 2, :, :], "z = mid", "x", "y"),
        (field[:, ny // 2, :], "y = mid", "x", "z"),
        (field[:, :, nx // 2], "x = mid", "y", "z"),
    ]
    vmin, vmax = field.min(), field.max()
    for ax, (plane, title, xl, yl) in zip(axes, planes):
        im = ax.imshow(plane, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"{name}, {title}")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"sample {sample}")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=110)
    return fig
