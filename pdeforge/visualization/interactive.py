"""
Interactive visualization tools using ipywidgets.

This module provides tools for exploring PDE datasets interactively
in Jupyter notebooks.
"""

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    import ipywidgets as widgets
    from IPython.display import display
    from ipywidgets import (
        Dropdown,
        FloatSlider,
        IntSlider,
        fixed,
        interact,
        interactive,
    )

    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False


def _is_binary_field(field: np.ndarray) -> bool:
    """True if the field contains only the values {0, 1} (i.e. a mask)."""
    finite = field[np.isfinite(field)]
    return bool(finite.size) and bool(np.isin(finite, (0.0, 1.0)).all())


def plot_sample_1d(
    inputs: np.ndarray,
    outputs: np.ndarray,
    x: np.ndarray = None,
    input_names: List[str] = None,
    output_names: List[str] = None,
    title: str = None,
    figsize: Tuple[int, int] = (12, 4),
) -> Figure:
    """
    Plot a single 1D sample (input and output).

    Parameters
    ----------
    inputs : np.ndarray
        Input field(s), shape (nx,) or (nx, n_channels)
    outputs : np.ndarray
        Output field(s), shape (nx,) or (nx, n_channels)
    x : np.ndarray, optional
        Spatial coordinates
    input_names : List[str], optional
        Names for input channels
    output_names : List[str], optional
        Names for output channels
    title : str, optional
        Plot title
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    Figure
        Matplotlib figure
    """
    # Handle shapes
    if inputs.ndim == 1:
        inputs = inputs[:, np.newaxis]
    if outputs.ndim == 1:
        outputs = outputs[:, np.newaxis]

    n_inputs = inputs.shape[-1]
    n_outputs = outputs.shape[-1]
    n_total = n_inputs + n_outputs

    if x is None:
        x = np.arange(inputs.shape[0])

    if input_names is None:
        input_names = [f"Input {i+1}" for i in range(n_inputs)]
    if output_names is None:
        output_names = [f"Output {i+1}" for i in range(n_outputs)]

    fig, axes = plt.subplots(1, n_total, figsize=figsize)
    if n_total == 1:
        axes = [axes]

    # Plot inputs
    for i in range(n_inputs):
        axes[i].plot(x, inputs[:, i], "b-", linewidth=2)
        axes[i].set_xlabel("x")
        axes[i].set_ylabel(input_names[i])
        axes[i].set_title(input_names[i])
        axes[i].grid(True, alpha=0.3)

    # Plot outputs
    for i in range(n_outputs):
        axes[n_inputs + i].plot(x, outputs[:, i], "r-", linewidth=2)
        axes[n_inputs + i].set_xlabel("x")
        axes[n_inputs + i].set_ylabel(output_names[i])
        axes[n_inputs + i].set_title(output_names[i])
        axes[n_inputs + i].grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    return fig


def plot_sample_2d(
    inputs: np.ndarray,
    outputs: np.ndarray,
    x: np.ndarray = None,
    y: np.ndarray = None,
    input_names: List[str] = None,
    output_names: List[str] = None,
    title: str = None,
    figsize: Tuple[int, int] = None,
    cmap_input: str = "viridis",
    cmap_output: str = "RdBu_r",
) -> Figure:
    """
    Plot a single 2D sample (input and output fields).

    Parameters
    ----------
    inputs : np.ndarray
        Input field(s), shape (ny, nx) or (ny, nx, n_channels)
    outputs : np.ndarray
        Output field(s), shape (ny, nx) or (ny, nx, n_channels)
    x, y : np.ndarray, optional
        Spatial coordinates
    input_names : List[str], optional
        Names for input channels
    output_names : List[str], optional
        Names for output channels
    title : str, optional
        Plot title
    figsize : Tuple[int, int], optional
        Figure size
    cmap_input, cmap_output : str
        Colormaps for inputs and outputs

    Returns
    -------
    Figure
        Matplotlib figure
    """
    # Handle shapes
    if inputs.ndim == 2:
        inputs = inputs[:, :, np.newaxis]
    if outputs.ndim == 2:
        outputs = outputs[:, :, np.newaxis]

    n_inputs = inputs.shape[-1]
    n_outputs = outputs.shape[-1]
    n_total = n_inputs + n_outputs

    if x is None:
        x = np.arange(inputs.shape[1])
    if y is None:
        y = np.arange(inputs.shape[0])

    if input_names is None:
        input_names = [f"Input {i+1}" for i in range(n_inputs)]
    if output_names is None:
        output_names = [f"Output {i+1}" for i in range(n_outputs)]

    if figsize is None:
        figsize = (4 * n_total, 4)

    fig, axes = plt.subplots(1, n_total, figsize=figsize)
    if n_total == 1:
        axes = [axes]

    # Plot inputs
    for i in range(n_inputs):
        im = axes[i].contourf(x, y, inputs[:, :, i], levels=20, cmap=cmap_input)
        axes[i].set_title(input_names[i])
        axes[i].set_aspect("equal")
        axes[i].set_xlabel("x")
        axes[i].set_ylabel("y")
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)

    # Plot outputs
    for i in range(n_outputs):
        data = outputs[:, :, i]
        if _is_binary_field(data):
            im = axes[n_inputs + i].contourf(
                x, y, data, levels=[-0.5, 0.5, 1.5], cmap="gray"
            )
        else:
            vmax = np.abs(data).max()
            im = axes[n_inputs + i].contourf(
                x, y, data, levels=20, cmap=cmap_output, vmin=-vmax, vmax=vmax
            )
        axes[n_inputs + i].set_title(output_names[i])
        axes[n_inputs + i].set_aspect("equal")
        axes[n_inputs + i].set_xlabel("x")
        axes[n_inputs + i].set_ylabel("y")
        divider = make_axes_locatable(axes[n_inputs + i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    return fig


def plot_sample_3d(
    inputs: np.ndarray,
    outputs: np.ndarray,
    grid: Dict[str, np.ndarray] = None,
    input_names: List[str] = None,
    output_names: List[str] = None,
    title: str = None,
    figsize: Tuple[int, int] = None,
    cmap_input: str = "viridis",
    cmap_output: str = "RdBu_r",
) -> Figure:
    """
    Plot a single 3D sample as three orthogonal centre-slices per field.

    Parameters
    ----------
    inputs : np.ndarray
        Input field(s), shape (nz, ny, nx) or (nz, ny, nx, n_channels)
    outputs : np.ndarray
        Output field(s), shape (nz, ny, nx) or (nz, ny, nx, n_channels)
    grid : Dict[str, np.ndarray], optional
        Grid coordinates (unused; kept for API consistency with the 1D/2D plotters)
    input_names, output_names : List[str], optional
        Names for input / output channels
    title : str, optional
        Plot title
    figsize : Tuple[int, int], optional
        Figure size
    cmap_input, cmap_output : str
        Colormaps for inputs and outputs

    Returns
    -------
    Figure
        Matplotlib figure
    """
    # Handle shapes
    if inputs.ndim == 3:
        inputs = inputs[..., np.newaxis]
    if outputs.ndim == 3:
        outputs = outputs[..., np.newaxis]

    n_inputs = inputs.shape[-1]
    n_outputs = outputs.shape[-1]
    n_total = n_inputs + n_outputs

    if input_names is None:
        input_names = [f"Input {i+1}" for i in range(n_inputs)]
    if output_names is None:
        output_names = [f"Output {i+1}" for i in range(n_outputs)]

    if figsize is None:
        figsize = (12, 3.6 * n_total)

    fig, axes = plt.subplots(n_total, 3, figsize=figsize, squeeze=False)

    def show_field(row, field, name, cmap, symmetric):
        nz, ny, nx = field.shape
        slices = [
            (field[nz // 2, :, :], f"z = {nz // 2}"),
            (field[:, ny // 2, :], f"y = {ny // 2}"),
            (field[:, :, nx // 2], f"x = {nx // 2}"),
        ]
        if _is_binary_field(field):
            kw = dict(cmap="gray", vmin=0.0, vmax=1.0)
        elif symmetric:
            vmax = np.abs(field).max()
            kw = dict(cmap=cmap, vmin=-vmax, vmax=vmax)
        else:
            kw = dict(cmap=cmap)
        for col, (sl, label) in enumerate(slices):
            ax = axes[row, col]
            im = ax.imshow(sl, origin="lower", aspect="equal", **kw)
            ax.set_title(f"{name}  ({label})", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)

    for i in range(n_inputs):
        show_field(i, inputs[..., i], input_names[i], cmap_input, symmetric=False)
    for i in range(n_outputs):
        show_field(
            n_inputs + i, outputs[..., i], output_names[i], cmap_output, symmetric=True
        )

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    return fig


def preview_samples(dataset, n_samples: int = None):
    """
    Display static preview of dataset samples.

    Parameters
    ----------
    dataset : PDEDataset
        Dataset to preview
    n_samples : int, optional
        Number of samples to show (default: all)
    """
    if n_samples is None:
        n_samples = dataset.n_samples
    n_samples = min(n_samples, dataset.n_samples)

    ndim = len(dataset.input_shape)
    if ndim == 1 or (ndim == 2 and dataset.input_shape[-1] == 1):
        # 1D case
        for i in range(n_samples):
            fig = plot_sample_1d(
                dataset.inputs[i],
                dataset.outputs[i],
                x=dataset.grid.get("x"),
                input_names=dataset.input_names,
                output_names=dataset.output_names,
                title=f"Sample {i+1}/{n_samples}",
            )
            plt.show()
    elif ndim == 3 and dataset.input_shape[-1] >= 10:
        # 3D case — scalar volume field (nz, ny, nx)
        for i in range(n_samples):
            fig = plot_sample_3d(
                dataset.inputs[i],
                dataset.outputs[i],
                grid=dataset.grid,
                input_names=dataset.input_names,
                output_names=dataset.output_names,
                title=f"Sample {i+1}/{n_samples}",
            )
            plt.show()
    else:
        # 2D case
        for i in range(n_samples):
            fig = plot_sample_2d(
                dataset.inputs[i],
                dataset.outputs[i],
                x=dataset.grid.get("x"),
                y=dataset.grid.get("y"),
                input_names=dataset.input_names,
                output_names=dataset.output_names,
                title=f"Sample {i+1}/{n_samples}",
            )
            plt.show()


class DatasetExplorer:
    """
    Interactive widget for exploring PDE datasets.

    Provides sliders for browsing samples and viewing slices.

    Parameters
    ----------
    dataset : PDEDataset
        Dataset to explore

    Examples
    --------
    >>> explorer = DatasetExplorer(dataset)
    >>> explorer.show()
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.ndim = self._get_ndim()

    def _get_ndim(self) -> int:
        """Determine if dataset is 1D, 2D, or 3D."""
        shape = self.dataset.input_shape
        if len(shape) == 1:
            return 1
        elif len(shape) == 2:
            # Could be (nx, n_channels) for 1D or (ny, nx) for 2D
            if shape[-1] < 10:  # Probably channels
                return 1
            else:
                return 2
        elif len(shape) == 3:
            # Could be (ny, nx, n_channels) for 2D or (nz, ny, nx) for 3D
            if shape[-1] < 10:  # Probably channels
                return 2
            else:
                return 3
        else:
            return 3

    def show(self):
        """Display the interactive explorer."""
        if not HAS_WIDGETS:
            print("ipywidgets not available. Showing static preview instead.")
            preview_samples(self.dataset, n_samples=3)
            return

        if self.ndim == 1:
            return self._show_1d()
        elif self.ndim == 3:
            return self._show_3d()
        else:
            return self._show_2d()

    def _show_1d(self):
        """Interactive 1D explorer."""
        dataset = self.dataset
        x = dataset.grid.get("x", np.arange(dataset.input_shape[0]))

        def plot_sample(sample_idx):
            inputs = dataset.inputs[sample_idx]
            outputs = dataset.outputs[sample_idx]

            fig = plot_sample_1d(
                inputs,
                outputs,
                x,
                input_names=dataset.input_names,
                output_names=dataset.output_names,
                title=f"Sample {sample_idx + 1}/{dataset.n_samples}",
            )
            plt.show()

            # Print statistics
            print(
                f"Input:  min={inputs.min():.4e}, max={inputs.max():.4e}, mean={inputs.mean():.4e}"
            )
            print(
                f"Output: min={outputs.min():.4e}, max={outputs.max():.4e}, mean={outputs.mean():.4e}"
            )

        sample_slider = IntSlider(
            min=0,
            max=dataset.n_samples - 1,
            step=1,
            value=0,
            description="Sample:",
            continuous_update=False,
        )

        return interactive(plot_sample, sample_idx=sample_slider)

    def _show_2d(self):
        """Interactive 2D explorer with slice views."""
        dataset = self.dataset
        x = dataset.grid.get("x", np.arange(dataset.input_shape[1]))
        y = dataset.grid.get("y", np.arange(dataset.input_shape[0]))

        def plot_sample(sample_idx, slice_y, show_slices):
            inputs = dataset.inputs[sample_idx]
            outputs = dataset.outputs[sample_idx]

            if inputs.ndim == 2:
                inputs = inputs[:, :, np.newaxis]
            if outputs.ndim == 2:
                outputs = outputs[:, :, np.newaxis]

            n_inputs = inputs.shape[-1]
            n_outputs = outputs.shape[-1]
            n_total = n_inputs + n_outputs

            if show_slices:
                n_rows = 2
            else:
                n_rows = 1

            fig, axes = plt.subplots(n_rows, n_total, figsize=(4 * n_total, 4 * n_rows))
            if n_total == 1:
                axes = axes.reshape(n_rows, 1)
            if n_rows == 1:
                axes = axes.reshape(1, -1)

            # Contour plots
            for i in range(n_inputs):
                im = axes[0, i].contourf(
                    x, y, inputs[:, :, i], levels=20, cmap="viridis"
                )
                if show_slices:
                    axes[0, i].axhline(y[slice_y], color="r", linestyle="--", alpha=0.7)
                axes[0, i].set_title(
                    dataset.input_names[i]
                    if i < len(dataset.input_names)
                    else f"Input {i+1}"
                )
                axes[0, i].set_aspect("equal")
                plt.colorbar(im, ax=axes[0, i])

            for i in range(n_outputs):
                data = outputs[:, :, i]
                if _is_binary_field(data):
                    im = axes[0, n_inputs + i].contourf(
                        x, y, data, levels=[-0.5, 0.5, 1.5], cmap="gray"
                    )
                else:
                    vmax = np.abs(data).max()
                    im = axes[0, n_inputs + i].contourf(
                        x, y, data, levels=20, cmap="RdBu_r", vmin=-vmax, vmax=vmax
                    )
                if show_slices:
                    axes[0, n_inputs + i].axhline(
                        y[slice_y], color="r", linestyle="--", alpha=0.7
                    )
                axes[0, n_inputs + i].set_title(
                    dataset.output_names[i]
                    if i < len(dataset.output_names)
                    else f"Output {i+1}"
                )
                axes[0, n_inputs + i].set_aspect("equal")
                plt.colorbar(im, ax=axes[0, n_inputs + i])

            # Slice plots
            if show_slices:
                for i in range(n_inputs):
                    axes[1, i].plot(x, inputs[slice_y, :, i], "b-", linewidth=2)
                    axes[1, i].set_xlabel("x")
                    axes[1, i].set_title(f"Slice at y = {y[slice_y]:.3f}")
                    axes[1, i].grid(True, alpha=0.3)

                for i in range(n_outputs):
                    axes[1, n_inputs + i].plot(
                        x, outputs[slice_y, :, i], "r-", linewidth=2
                    )
                    axes[1, n_inputs + i].set_xlabel("x")
                    axes[1, n_inputs + i].set_title(f"Slice at y = {y[slice_y]:.3f}")
                    axes[1, n_inputs + i].grid(True, alpha=0.3)

            fig.suptitle(f"Sample {sample_idx + 1}/{dataset.n_samples}", fontsize=14)
            plt.tight_layout()
            plt.show()

        ny = len(y)

        sample_slider = IntSlider(
            min=0,
            max=dataset.n_samples - 1,
            step=1,
            value=0,
            description="Sample:",
            continuous_update=False,
        )
        slice_slider = IntSlider(
            min=0,
            max=ny - 1,
            step=1,
            value=ny // 2,
            description="Y slice:",
            continuous_update=False,
        )
        show_slices_checkbox = widgets.Checkbox(value=True, description="Show slices")

        return interactive(
            plot_sample,
            sample_idx=sample_slider,
            slice_y=slice_slider,
            show_slices=show_slices_checkbox,
        )

    def _show_3d(self):
        """Interactive 3D explorer with orthogonal slice views."""
        dataset = self.dataset

        def plot_sample(sample_idx, axis, position):
            inputs = dataset.inputs[sample_idx]
            outputs = dataset.outputs[sample_idx]

            if inputs.ndim == 3:
                inputs = inputs[..., np.newaxis]
            if outputs.ndim == 3:
                outputs = outputs[..., np.newaxis]

            n_inputs = inputs.shape[-1]
            n_outputs = outputs.shape[-1]
            n_total = n_inputs + n_outputs

            axis_idx = {"z": 0, "y": 1, "x": 2}[axis]

            def take_slice(vol):
                n = vol.shape[axis_idx]
                idx = int(round(position * (n - 1)))
                return np.take(vol, idx, axis=axis_idx), idx

            fig, axes = plt.subplots(1, n_total, figsize=(4.5 * n_total, 4.2))
            if n_total == 1:
                axes = [axes]

            for i in range(n_inputs):
                sl, idx = take_slice(inputs[..., i])
                im = axes[i].imshow(
                    sl, origin="lower", aspect="equal", cmap="viridis"
                )
                name = (
                    dataset.input_names[i]
                    if i < len(dataset.input_names)
                    else f"Input {i+1}"
                )
                axes[i].set_title(f"{name}  ({axis} = {idx})")
                axes[i].set_xticks([])
                axes[i].set_yticks([])
                plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

            for i in range(n_outputs):
                sl, idx = take_slice(outputs[..., i])
                if _is_binary_field(sl):
                    im = axes[n_inputs + i].imshow(
                        sl,
                        origin="lower",
                        aspect="equal",
                        cmap="gray",
                        vmin=0.0,
                        vmax=1.0,
                    )
                else:
                    vmax = np.abs(sl).max()
                    im = axes[n_inputs + i].imshow(
                        sl,
                        origin="lower",
                        aspect="equal",
                        cmap="RdBu_r",
                        vmin=-vmax,
                        vmax=vmax,
                    )
                name = (
                    dataset.output_names[i]
                    if i < len(dataset.output_names)
                    else f"Output {i+1}"
                )
                axes[n_inputs + i].set_title(f"{name}  ({axis} = {idx})")
                axes[n_inputs + i].set_xticks([])
                axes[n_inputs + i].set_yticks([])
                plt.colorbar(im, ax=axes[n_inputs + i], fraction=0.046, pad=0.04)

            fig.suptitle(f"Sample {sample_idx + 1}/{dataset.n_samples}", fontsize=14)
            plt.tight_layout()
            plt.show()

            print(
                f"Input:  min={inputs.min():.4e}, max={inputs.max():.4e}, "
                f"mean={inputs.mean():.4e}"
            )
            print(
                f"Output: min={outputs.min():.4e}, max={outputs.max():.4e}, "
                f"mean={outputs.mean():.4e}"
            )

        sample_slider = IntSlider(
            min=0,
            max=dataset.n_samples - 1,
            step=1,
            value=0,
            description="Sample:",
            continuous_update=False,
        )
        axis_dropdown = Dropdown(
            options=["z", "y", "x"],
            value="z",
            description="Slice axis:",
        )
        position_slider = FloatSlider(
            min=0.0,
            max=1.0,
            step=0.02,
            value=0.5,
            description="Position:",
            continuous_update=False,
        )

        return interactive(
            plot_sample,
            sample_idx=sample_slider,
            axis=axis_dropdown,
            position=position_slider,
        )
