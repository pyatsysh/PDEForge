"""
Generate publication-quality figures for PDEForge paper and website.

Usage:
    conda activate pdeforge-fenicsx
    python generate_figures.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
from pathlib import Path
import sys

sys.path.insert(0, '..')

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

try:
    FIGURES_DIR = Path(__file__).parent / 'figures'
except NameError:
    FIGURES_DIR = Path('/Users/pyatsyshin/Documents/GitHub/Project_PDEForge/notebooks/figures')
FIGURES_DIR.mkdir(exist_ok=True)

from pdeforge import generate_dataset, get_model, list_models


def generate_burgers_1d_figure():
    """Generate Burgers 1D shock formation figure."""
    print("Generating Burgers 1D figure...")

    dataset = generate_dataset(
        model="burgers_1d",
        n_samples=6,
        resolution={"x": 256},
        params={"viscosity": 0.01, "advection": 1.0, "time_end": 1.0},
        seed=42,
    )

    x = dataset.grid['x']

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    for i in range(3):
        axes[0, i].plot(x, dataset.inputs[i], 'b-', linewidth=2)
        axes[0, i].set_title(f'Sample {i+1}: $u(x, t=0)$')
        axes[0, i].set_xlabel('$x$')
        axes[0, i].set_ylabel('$u$')
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].set_xlim(0, 1)

        axes[1, i].plot(x, dataset.outputs[i], 'r-', linewidth=2)
        axes[1, i].set_title(f'Sample {i+1}: $u(x, t=T)$')
        axes[1, i].set_xlabel('$x$')
        axes[1, i].set_ylabel('$u$')
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].set_xlim(0, 1)

    fig.suptitle('Burgers Equation: Initial Conditions → Shock Formation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'burgers_1d.png')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'burgers_1d.png'}")


def generate_heat_1d_figure():
    """Generate Heat 1D diffusion figure."""
    print("Generating Heat 1D figure...")

    dataset = generate_dataset(
        model="heat_1d",
        n_samples=6,
        resolution={"x": 128},
        params={"diffusivity": 0.01, "time_end": 0.5},
        seed=42,
    )

    x = dataset.grid['x']

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    for i in range(3):
        axes[0, i].plot(x, dataset.inputs[i], 'b-', linewidth=2)
        axes[0, i].set_title(f'Sample {i+1}: $u(x, t=0)$')
        axes[0, i].set_xlabel('$x$')
        axes[0, i].set_ylabel('$u$')
        axes[0, i].grid(True, alpha=0.3)

        axes[1, i].plot(x, dataset.outputs[i], 'r-', linewidth=2)
        axes[1, i].set_title(f'Sample {i+1}: $u(x, t=T)$')
        axes[1, i].set_xlabel('$x$')
        axes[1, i].set_ylabel('$u$')
        axes[1, i].grid(True, alpha=0.3)

    fig.suptitle('Heat Equation: Diffusion Dynamics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'heat_1d.png')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'heat_1d.png'}")


def generate_wave_1d_figure():
    """Generate Wave 1D propagation figure."""
    print("Generating Wave 1D figure...")

    dataset = generate_dataset(
        model="wave_1d",
        n_samples=6,
        resolution={"x": 128},
        params={"wave_speed": 1.0, "time_end": 0.5},
        seed=42,
    )

    x = dataset.grid['x']

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    for i in range(3):
        axes[0, i].plot(x, dataset.inputs[i], 'b-', linewidth=2)
        axes[0, i].set_title(f'Sample {i+1}: $u(x, t=0)$')
        axes[0, i].set_xlabel('$x$')
        axes[0, i].set_ylabel('$u$')
        axes[0, i].grid(True, alpha=0.3)

        axes[1, i].plot(x, dataset.outputs[i], 'r-', linewidth=2)
        axes[1, i].set_title(f'Sample {i+1}: $u(x, t=T)$')
        axes[1, i].set_xlabel('$x$')
        axes[1, i].set_ylabel('$u$')
        axes[1, i].grid(True, alpha=0.3)

    fig.suptitle('Wave Equation: Wave Propagation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'wave_1d.png')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'wave_1d.png'}")


def generate_allen_cahn_1d_figure():
    """Generate Allen-Cahn 1D phase field figure."""
    print("Generating Allen-Cahn 1D figure...")

    dataset = generate_dataset(
        model="allen_cahn_1d",
        n_samples=6,
        resolution={"x": 128},
        params={"epsilon": 0.01, "time_end": 0.1},
        seed=42,
    )

    x = dataset.grid['x']

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    for i in range(3):
        axes[0, i].plot(x, dataset.inputs[i], 'b-', linewidth=2)
        axes[0, i].set_title(f'Sample {i+1}: $u(x, t=0)$')
        axes[0, i].set_xlabel('$x$')
        axes[0, i].set_ylabel('$u$')
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].set_ylim(-1.5, 1.5)

        axes[1, i].plot(x, dataset.outputs[i], 'r-', linewidth=2)
        axes[1, i].set_title(f'Sample {i+1}: $u(x, t=T)$')
        axes[1, i].set_xlabel('$x$')
        axes[1, i].set_ylabel('$u$')
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].set_ylim(-1.5, 1.5)

    fig.suptitle('Allen-Cahn Equation: Phase Separation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'allen_cahn_1d.png')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'allen_cahn_1d.png'}")


def generate_fitzhugh_nagumo_1d_figure():
    """Generate FitzHugh-Nagumo 1D reaction-diffusion figure."""
    print("Generating FitzHugh-Nagumo 1D figure...")

    dataset = generate_dataset(
        model="fitzhugh_nagumo_1d",
        n_samples=6,
        resolution={"x": 128},
        seed=42,
    )

    x = dataset.grid['x']

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    for i in range(3):
        # Input is 1D initial condition, output has 2 channels: u and v
        axes[0, i].plot(x, dataset.inputs[i], 'b-', linewidth=2)
        axes[0, i].set_title(f'Sample {i+1}: Initial $u$')
        axes[0, i].set_xlabel('$x$')
        axes[0, i].set_ylabel('$u$')
        axes[0, i].grid(True, alpha=0.3)

        axes[1, i].plot(x, dataset.outputs[i, :, 0], 'b-', linewidth=2, label='$u$')
        axes[1, i].plot(x, dataset.outputs[i, :, 1], 'g--', linewidth=2, label='$v$')
        axes[1, i].set_title(f'Sample {i+1}: Final')
        axes[1, i].set_xlabel('$x$')
        axes[1, i].grid(True, alpha=0.3)
        if i == 0:
            axes[1, i].legend()

    fig.suptitle('FitzHugh-Nagumo: Excitable Medium Dynamics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fitzhugh_nagumo_1d.png')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'fitzhugh_nagumo_1d.png'}")


def generate_heat_2d_figure():
    """Generate Heat 2D diffusion figure."""
    print("Generating Heat 2D figure...")

    dataset = generate_dataset(
        model="heat_2d",
        n_samples=4,
        resolution={"x": 64, "y": 64},
        params={"diffusivity": 0.01, "time_end": 0.1},
        seed=42,
    )

    x = dataset.grid['x']
    y = dataset.grid['y']

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))

    for i in range(4):
        im0 = axes[0, i].contourf(x, y, dataset.inputs[i], levels=30, cmap='viridis')
        axes[0, i].set_title(f'Sample {i+1}: $u(t=0)$')
        axes[0, i].set_aspect('equal')
        plt.colorbar(im0, ax=axes[0, i], shrink=0.8)

        im1 = axes[1, i].contourf(x, y, dataset.outputs[i], levels=30, cmap='viridis')
        axes[1, i].set_title(f'Sample {i+1}: $u(t=T)$')
        axes[1, i].set_aspect('equal')
        plt.colorbar(im1, ax=axes[1, i], shrink=0.8)

    fig.suptitle('2D Heat Equation: Diffusion', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'heat_2d.png')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'heat_2d.png'}")


def generate_wave_2d_figure():
    """Generate Wave 2D propagation figure."""
    print("Generating Wave 2D figure...")

    dataset = generate_dataset(
        model="wave_2d",
        n_samples=4,
        resolution={"x": 64, "y": 64},
        params={"wave_speed": 1.0, "time_end": 0.3},
        seed=42,
    )

    x = dataset.grid['x']
    y = dataset.grid['y']

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))

    for i in range(4):
        vmax = max(np.abs(dataset.inputs[i]).max(), np.abs(dataset.outputs[i]).max())

        im0 = axes[0, i].contourf(x, y, dataset.inputs[i], levels=30, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[0, i].set_title(f'Sample {i+1}: $u(t=0)$')
        axes[0, i].set_aspect('equal')
        plt.colorbar(im0, ax=axes[0, i], shrink=0.8)

        im1 = axes[1, i].contourf(x, y, dataset.outputs[i], levels=30, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[1, i].set_title(f'Sample {i+1}: $u(t=T)$')
        axes[1, i].set_aspect('equal')
        plt.colorbar(im1, ax=axes[1, i], shrink=0.8)

    fig.suptitle('2D Wave Equation: Propagation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'wave_2d.png')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'wave_2d.png'}")


def generate_allen_cahn_2d_figure():
    """Generate Allen-Cahn 2D phase field figure."""
    print("Generating Allen-Cahn 2D figure...")

    dataset = generate_dataset(
        model="allen_cahn_2d",
        n_samples=4,
        resolution={"x": 64, "y": 64},
        params={"epsilon": 0.02, "time_end": 0.05},
        seed=42,
    )

    x = dataset.grid['x']
    y = dataset.grid['y']

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))

    for i in range(4):
        im0 = axes[0, i].contourf(x, y, dataset.inputs[i], levels=30, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0, i].set_title(f'Sample {i+1}: $u(t=0)$')
        axes[0, i].set_aspect('equal')
        plt.colorbar(im0, ax=axes[0, i], shrink=0.8)

        im1 = axes[1, i].contourf(x, y, dataset.outputs[i], levels=30, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1, i].set_title(f'Sample {i+1}: $u(t=T)$')
        axes[1, i].set_aspect('equal')
        plt.colorbar(im1, ax=axes[1, i], shrink=0.8)

    fig.suptitle('2D Allen-Cahn: Phase Separation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'allen_cahn_2d.png')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'allen_cahn_2d.png'}")


def generate_darcy_2d_figure():
    """Generate Darcy 2D flow figure."""
    print("Generating Darcy 2D figure...")

    dataset = generate_dataset(
        model="darcy_2d",
        n_samples=4,
        resolution={"x": 64, "y": 64},
        seed=42,
    )

    x = dataset.grid['x']
    y = dataset.grid['y']

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))

    for i in range(4):
        im0 = axes[0, i].contourf(x, y, dataset.inputs[i], levels=30, cmap='viridis')
        axes[0, i].set_title(f'Sample {i+1}: Permeability $k$')
        axes[0, i].set_aspect('equal')
        plt.colorbar(im0, ax=axes[0, i], shrink=0.8)

        im1 = axes[1, i].contourf(x, y, dataset.outputs[i], levels=30, cmap='viridis')
        axes[1, i].set_title(f'Sample {i+1}: Pressure $p$')
        axes[1, i].set_aspect('equal')
        plt.colorbar(im1, ax=axes[1, i], shrink=0.8)

    fig.suptitle('2D Darcy Flow: Permeability → Pressure', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'darcy_2d.png')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'darcy_2d.png'}")


def generate_stokes_2d_figure():
    """Generate Stokes 2D flow figure."""
    print("Generating Stokes 2D figure...")

    dataset = generate_dataset(
        model="stokes_2d",
        n_samples=3,
        resolution={"x": 64, "y": 64},
        params={"viscosity": 1.0, "n_force_modes": 5},
        seed=42,
    )

    x = dataset.grid['x']
    y = dataset.grid['y']

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    for i in range(3):
        # Input: force magnitude
        f_mag = np.sqrt(dataset.inputs[i, :, :, 0]**2 + dataset.inputs[i, :, :, 1]**2)
        im0 = axes[0, i].contourf(x, y, f_mag, levels=30, cmap='viridis')
        axes[0, i].set_title(f'Sample {i+1}: $|\\mathbf{{f}}|$')
        axes[0, i].set_aspect('equal')
        plt.colorbar(im0, ax=axes[0, i], shrink=0.8)

        # Output: velocity magnitude
        v_mag = np.sqrt(dataset.outputs[i, :, :, 0]**2 + dataset.outputs[i, :, :, 1]**2)
        im1 = axes[1, i].contourf(x, y, v_mag, levels=30, cmap='viridis')
        axes[1, i].set_title(f'Sample {i+1}: $|\\mathbf{{u}}|$')
        axes[1, i].set_aspect('equal')
        plt.colorbar(im1, ax=axes[1, i], shrink=0.8)

    fig.suptitle('2D Stokes Flow: Force → Velocity', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'stokes_2d.png')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'stokes_2d.png'}")


def generate_cylinder_flow_steady_figure():
    """Generate steady cylinder flow figure."""
    print("Generating Cylinder Flow (steady) figure...")

    dataset = generate_dataset(
        model="cylinder_flow_2d",
        n_samples=4,
        resolution={"x": 110, "y": 41},
        params={"inlet_velocity": 1.0, "viscosity": 0.001},
        seed=42,
    )

    x = dataset.grid['x']
    y = dataset.grid['y']

    # Cylinder geometry
    cx, cy, r = 0.2, 0.2, 0.05

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        # Velocity magnitude (transpose for matplotlib: output is (nx, ny), need (ny, nx))
        u = dataset.outputs[i, :, :, 0].T
        v = dataset.outputs[i, :, :, 1].T
        vmag = np.sqrt(u**2 + v**2)

        im = ax.contourf(x, y, vmag, levels=30, cmap='viridis')
        ax.streamplot(x, y, u, v, color='white', density=1.5, linewidth=0.5, arrowsize=0.5)

        # Add cylinder
        circle = Circle((cx, cy), r, color='gray', ec='black', lw=2, zorder=10)
        ax.add_patch(circle)

        ax.set_title(f'Inlet scale = {dataset.inputs[i, 0]:.2f}')
        ax.set_xlabel('$x$ (m)')
        ax.set_ylabel('$y$ (m)')
        ax.set_aspect('equal')
        ax.set_xlim(0, 2.2)
        ax.set_ylim(0, 0.41)
        plt.colorbar(im, ax=ax, label='$|\\mathbf{u}|$ (m/s)', shrink=0.8)

    fig.suptitle('Steady Cylinder Flow: Velocity Magnitude with Streamlines', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'cylinder_flow_steady.png')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'cylinder_flow_steady.png'}")


def generate_cylinder_flow_unsteady_figure():
    """Generate unsteady cylinder flow snapshots figure."""
    print("Generating Cylinder Flow (unsteady) figure...")
    print("  (This may take a few minutes...)")

    # Get model
    Model = get_model("cylinder_flow_2d_unsteady")
    model = Model(
        resolution={"x": 110, "y": 41},
        inlet_velocity=1.0,
        viscosity=0.001,
        time_end=8.0,
        _n_time_steps=41,
    )

    # Generate trajectory
    trajectory = model.solve(inlet_scale=1.0, return_full=True)
    x = model.grids['x']
    y = model.grids['y']
    t = np.linspace(0, 8.0, 41)

    # Cylinder geometry
    cx, cy, r = 0.2, 0.2, 0.05

    # Vorticity snapshots
    n_snapshots = 6
    snapshot_indices = np.linspace(0, len(t) - 1, n_snapshots, dtype=int)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()

    for i, t_idx in enumerate(snapshot_indices):
        u = trajectory[t_idx, :, :, 0].T
        v = trajectory[t_idx, :, :, 1].T

        # Compute vorticity
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dvdx = np.gradient(v, dx, axis=1)
        dudy = np.gradient(u, dy, axis=0)
        omega = dvdx - dudy

        vmax = np.abs(omega).max()
        im = axes[i].contourf(x, y, omega, levels=30, cmap='RdBu_r', vmin=-vmax, vmax=vmax)

        # Add cylinder
        circle = Circle((cx, cy), r, color='gray', ec='black', lw=2, zorder=10)
        axes[i].add_patch(circle)

        axes[i].set_title(f't = {t[t_idx]:.2f} s')
        axes[i].set_xlabel('$x$ (m)')
        axes[i].set_ylabel('$y$ (m)')
        axes[i].set_aspect('equal')
        axes[i].set_xlim(0, 2.2)
        axes[i].set_ylim(0, 0.41)
        plt.colorbar(im, ax=axes[i], label='$\\omega$ (1/s)', shrink=0.8)

    fig.suptitle('Unsteady Cylinder Flow: Vorticity (Von Kármán Vortex Street)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'cylinder_flow_unsteady_vorticity.png')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'cylinder_flow_unsteady_vorticity.png'}")

    # Return trajectory for animation
    return trajectory, x, y, t


def generate_cylinder_flow_animation(trajectory, x, y, t):
    """Generate animation of unsteady cylinder flow."""
    print("Generating Cylinder Flow animation...")

    cx, cy, r = 0.2, 0.2, 0.05

    fig, ax = plt.subplots(figsize=(14, 5))

    # Initial frame
    u = trajectory[0, :, :, 0].T
    v = trajectory[0, :, :, 1].T
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dvdx = np.gradient(v, dx, axis=1)
    dudy = np.gradient(u, dy, axis=0)
    omega = dvdx - dudy

    # Find global vmax for consistent colorbar
    vmax_global = 0
    for t_idx in range(len(t)):
        u = trajectory[t_idx, :, :, 0].T
        v = trajectory[t_idx, :, :, 1].T
        dvdx = np.gradient(v, dx, axis=1)
        dudy = np.gradient(u, dy, axis=0)
        omega = dvdx - dudy
        vmax_global = max(vmax_global, np.abs(omega).max())

    levels = np.linspace(-vmax_global, vmax_global, 31)

    contour = ax.contourf(x, y, omega, levels=levels, cmap='RdBu_r')
    circle = Circle((cx, cy), r, color='gray', ec='black', lw=2, zorder=10)
    ax.add_patch(circle)
    ax.set_xlabel('$x$ (m)')
    ax.set_ylabel('$y$ (m)')
    ax.set_aspect('equal')
    ax.set_xlim(0, 2.2)
    ax.set_ylim(0, 0.41)
    cbar = plt.colorbar(contour, ax=ax, label='$\\omega$ (1/s)')
    title = ax.set_title(f'Vortex Shedding: t = {t[0]:.2f} s')

    def update(frame):
        ax.clear()

        u = trajectory[frame, :, :, 0].T
        v = trajectory[frame, :, :, 1].T
        dvdx = np.gradient(v, dx, axis=1)
        dudy = np.gradient(u, dy, axis=0)
        omega = dvdx - dudy

        contour = ax.contourf(x, y, omega, levels=levels, cmap='RdBu_r')
        circle = Circle((cx, cy), r, color='gray', ec='black', lw=2, zorder=10)
        ax.add_patch(circle)
        ax.set_xlabel('$x$ (m)')
        ax.set_ylabel('$y$ (m)')
        ax.set_aspect('equal')
        ax.set_xlim(0, 2.2)
        ax.set_ylim(0, 0.41)
        ax.set_title(f'Vortex Shedding: t = {t[frame]:.2f} s')

        return contour.collections

    anim = animation.FuncAnimation(
        fig, update, frames=len(t), interval=100, blit=False
    )

    # Save as GIF
    anim.save(FIGURES_DIR / 'cylinder_flow_animation.gif', writer='pillow', fps=10)
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'cylinder_flow_animation.gif'}")

    # Also save as MP4 if ffmpeg is available
    try:
        fig, ax = plt.subplots(figsize=(14, 5))
        contour = ax.contourf(x, y, omega, levels=levels, cmap='RdBu_r')
        circle = Circle((cx, cy), r, color='gray', ec='black', lw=2, zorder=10)
        ax.add_patch(circle)
        ax.set_xlabel('$x$ (m)')
        ax.set_ylabel('$y$ (m)')
        ax.set_aspect('equal')
        ax.set_xlim(0, 2.2)
        ax.set_ylim(0, 0.41)
        plt.colorbar(contour, ax=ax, label='$\\omega$ (1/s)')

        anim = animation.FuncAnimation(
            fig, update, frames=len(t), interval=100, blit=False
        )
        anim.save(FIGURES_DIR / 'cylinder_flow_animation.mp4', writer='ffmpeg', fps=10)
        plt.close()
        print(f"  Saved: {FIGURES_DIR / 'cylinder_flow_animation.mp4'}")
    except Exception as e:
        print(f"  Note: Could not save MP4 (ffmpeg may not be installed): {e}")


def generate_turbulent_cylinder_flow_figure():
    """Generate turbulent cylinder flow snapshots figure."""
    print("Generating Turbulent Cylinder Flow figure...")
    print("  (This may take several minutes due to high Reynolds number...)")

    # Get model
    Model = get_model("cylinder_flow_2d_turbulent")
    model = Model(
        resolution={"x": 110, "y": 41},
        inlet_velocity=1.0,
        viscosity=0.0002,  # Re ~ 500
        cylinder_radius=0.05,
        use_les=True,
        smagorinsky_constant=0.1,
        time_end=8.0,
        n_time_steps=41,
        _mesh_resolution=0.015,
    )

    # Generate trajectory
    trajectory = model.solve(inlet_scale=1.0, cx=0.2, cy=0.2)
    x = model.grids['x']
    y = model.grids['y']
    t = np.linspace(0, 8.0, 41)

    # Cylinder geometry
    cx, cy, r = 0.2, 0.2, 0.05

    # Vorticity snapshots
    n_snapshots = 6
    snapshot_indices = np.linspace(0, len(t) - 1, n_snapshots, dtype=int)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()

    for i, t_idx in enumerate(snapshot_indices):
        u = trajectory[t_idx, :, :, 0].T
        v = trajectory[t_idx, :, :, 1].T

        # Compute vorticity
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dvdx = np.gradient(v, dx, axis=1)
        dudy = np.gradient(u, dy, axis=0)
        omega = dvdx - dudy

        vmax = np.abs(omega).max()
        im = axes[i].contourf(x, y, omega, levels=30, cmap='RdBu_r', vmin=-vmax, vmax=vmax)

        # Add cylinder
        circle = Circle((cx, cy), r, color='gray', ec='black', lw=2, zorder=10)
        axes[i].add_patch(circle)

        axes[i].set_title(f't = {t[t_idx]:.2f} s')
        axes[i].set_xlabel('$x$ (m)')
        axes[i].set_ylabel('$y$ (m)')
        axes[i].set_aspect('equal')
        axes[i].set_xlim(0, 2.2)
        axes[i].set_ylim(0, 0.41)
        plt.colorbar(im, ax=axes[i], label='$\\omega$ (1/s)', shrink=0.8)

    fig.suptitle(f'Turbulent Cylinder Flow: Vorticity (Re = {model.Re:.0f}, LES)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'cylinder_flow_turbulent_vorticity.png')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'cylinder_flow_turbulent_vorticity.png'}")

    # Return trajectory for animation
    return trajectory, x, y, t, model.Re


def generate_turbulent_cylinder_animation(trajectory, x, y, t, Re):
    """Generate animation of turbulent cylinder flow."""
    print("Generating Turbulent Cylinder Flow animation...")

    cx, cy, r = 0.2, 0.2, 0.05
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Find global vmax for consistent colorbar
    vmax_global = 0
    for t_idx in range(len(t)):
        u = trajectory[t_idx, :, :, 0].T
        v = trajectory[t_idx, :, :, 1].T
        dvdx = np.gradient(v, dx, axis=1)
        dudy = np.gradient(u, dy, axis=0)
        omega = dvdx - dudy
        vmax_global = max(vmax_global, np.abs(omega).max())

    levels = np.linspace(-vmax_global, vmax_global, 31)

    # Use PIL approach for GIF generation (avoids matplotlib API issues)
    from PIL import Image
    from io import BytesIO
    import matplotlib
    matplotlib.use('Agg')

    frames = []
    for frame in range(len(t)):
        fig, ax = plt.subplots(figsize=(14, 5))

        u = trajectory[frame, :, :, 0].T
        v = trajectory[frame, :, :, 1].T
        dvdx = np.gradient(v, dx, axis=1)
        dudy = np.gradient(u, dy, axis=0)
        omega = dvdx - dudy

        contour = ax.contourf(x, y, omega, levels=levels, cmap='RdBu_r')
        circle = Circle((cx, cy), r, color='gray', ec='black', lw=2, zorder=10)
        ax.add_patch(circle)
        ax.set_xlabel('$x$ (m)')
        ax.set_ylabel('$y$ (m)')
        ax.set_aspect('equal')
        ax.set_xlim(0, 2.2)
        ax.set_ylim(0, 0.41)
        plt.colorbar(contour, ax=ax, label='$\\omega$ (1/s)')
        ax.set_title(f'Turbulent Vortex Shedding (Re={Re:.0f}): t = {t[frame]:.2f} s')

        plt.tight_layout()

        # Save to buffer
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()
        plt.close(fig)

    # Save as GIF
    frames[0].save(
        FIGURES_DIR / 'cylinder_flow_turbulent_animation.gif',
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0
    )
    print(f"  Saved: {FIGURES_DIR / 'cylinder_flow_turbulent_animation.gif'}")


def generate_parameterized_cylinder_figure():
    """Generate figure showing different cylinder positions."""
    print("Generating Parameterized Cylinder Flow figure...")

    Model = get_model("cylinder_flow_2d_parameterized")

    positions = [
        (0.2, 0.2),   # Default
        (0.3, 0.2),   # Downstream
        (0.2, 0.15),  # Lower
        (0.3, 0.25),  # Diagonal
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for i, (cx, cy) in enumerate(positions):
        model = Model(
            resolution={"x": 110, "y": 41},
            inlet_velocity=0.3,
            viscosity=0.001,
            cylinder_radius=0.05,
        )

        solution = model.solve(inlet_scale=1.0, cx=cx, cy=cy)
        x = model.grids['x']
        y = model.grids['y']

        # Velocity magnitude (transpose for matplotlib)
        u = solution[:, :, 0].T
        v = solution[:, :, 1].T
        vmag = np.sqrt(u**2 + v**2)

        im = axes[i].contourf(x, y, vmag, levels=30, cmap='viridis')
        axes[i].streamplot(x, y, u, v, color='white', density=1.5, linewidth=0.5, arrowsize=0.5)

        # Add cylinder at actual position
        circle = Circle((cx, cy), 0.05, color='gray', ec='black', lw=2, zorder=10)
        axes[i].add_patch(circle)

        axes[i].set_title(f'Cylinder at ({cx:.2f}, {cy:.2f})')
        axes[i].set_xlabel('$x$ (m)')
        axes[i].set_ylabel('$y$ (m)')
        axes[i].set_aspect('equal')
        axes[i].set_xlim(0, 2.2)
        axes[i].set_ylim(0, 0.41)
        plt.colorbar(im, ax=axes[i], label='$|\\mathbf{u}|$ (m/s)', shrink=0.8)

    fig.suptitle('Parameterized Cylinder Flow: Effect of Cylinder Position', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'cylinder_flow_parameterized.png')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'cylinder_flow_parameterized.png'}")


def generate_stochastic_heat_figure():
    """Generate stochastic heat equation figure."""
    print("Generating Stochastic Heat 1D figure...")

    dataset = generate_dataset(
        model="stochastic_heat_1d",
        n_samples=3,
        resolution={"x": 128},
        seed=42,
    )

    x = dataset.grid['x']

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    for i in range(3):
        axes[0, i].plot(x, dataset.inputs[i], 'b-', linewidth=2)
        axes[0, i].set_title(f'Sample {i+1}: $u(x, t=0)$')
        axes[0, i].set_xlabel('$x$')
        axes[0, i].set_ylabel('$u$')
        axes[0, i].grid(True, alpha=0.3)

        # Output has shape (n_realizations, nx) - plot mean and individual realizations
        realizations = dataset.outputs[i]  # Shape: (n_realizations, nx)
        mean_sol = realizations.mean(axis=0)
        std_sol = realizations.std(axis=0)

        # Plot a few realizations in light gray
        for j in range(min(5, realizations.shape[0])):
            axes[1, i].plot(x, realizations[j], 'gray', alpha=0.3, linewidth=0.5)

        # Plot mean and uncertainty band
        axes[1, i].fill_between(x, mean_sol - 2*std_sol, mean_sol + 2*std_sol,
                                 alpha=0.3, color='red', label='±2σ')
        axes[1, i].plot(x, mean_sol, 'r-', linewidth=2, label='Mean')
        axes[1, i].set_title(f'Sample {i+1}: $u(x, t=T)$')
        axes[1, i].set_xlabel('$x$')
        axes[1, i].set_ylabel('$u$')
        axes[1, i].grid(True, alpha=0.3)
        if i == 0:
            axes[1, i].legend(fontsize=9)

    fig.suptitle('Stochastic Heat Equation: Monte Carlo Realizations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'stochastic_heat_1d.png')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'stochastic_heat_1d.png'}")


def generate_overview_figure():
    """Generate an overview figure showing all model types."""
    print("Generating overview figure...")

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))

    # Row 1: 1D models
    # Burgers
    dataset = generate_dataset(model="burgers_1d", n_samples=1, resolution={"x": 256},
                               params={"viscosity": 0.01}, seed=42)
    x = dataset.grid['x']
    axes[0, 0].plot(x, dataset.inputs[0], 'b-', lw=2, label='Initial')
    axes[0, 0].plot(x, dataset.outputs[0], 'r-', lw=2, label='Final')
    axes[0, 0].set_title('Burgers 1D')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    # Heat
    dataset = generate_dataset(model="heat_1d", n_samples=1, resolution={"x": 128}, seed=42)
    x = dataset.grid['x']
    axes[0, 1].plot(x, dataset.inputs[0], 'b-', lw=2, label='Initial')
    axes[0, 1].plot(x, dataset.outputs[0], 'r-', lw=2, label='Final')
    axes[0, 1].set_title('Heat 1D')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    # Wave
    dataset = generate_dataset(model="wave_1d", n_samples=1, resolution={"x": 128}, seed=42)
    x = dataset.grid['x']
    axes[0, 2].plot(x, dataset.inputs[0], 'b-', lw=2, label='Initial')
    axes[0, 2].plot(x, dataset.outputs[0], 'r-', lw=2, label='Final')
    axes[0, 2].set_title('Wave 1D')
    axes[0, 2].legend(fontsize=9)
    axes[0, 2].grid(True, alpha=0.3)

    # Allen-Cahn
    dataset = generate_dataset(model="allen_cahn_1d", n_samples=1, resolution={"x": 128}, seed=42)
    x = dataset.grid['x']
    axes[0, 3].plot(x, dataset.inputs[0], 'b-', lw=2, label='Initial')
    axes[0, 3].plot(x, dataset.outputs[0], 'r-', lw=2, label='Final')
    axes[0, 3].set_title('Allen-Cahn 1D')
    axes[0, 3].legend(fontsize=9)
    axes[0, 3].grid(True, alpha=0.3)

    # Row 2: 2D scalar models
    # Heat 2D
    dataset = generate_dataset(model="heat_2d", n_samples=1, resolution={"x": 64, "y": 64}, seed=42)
    x, y = dataset.grid['x'], dataset.grid['y']
    im = axes[1, 0].contourf(x, y, dataset.outputs[0], levels=20, cmap='viridis')
    axes[1, 0].set_title('Heat 2D')
    axes[1, 0].set_aspect('equal')

    # Wave 2D
    dataset = generate_dataset(model="wave_2d", n_samples=1, resolution={"x": 64, "y": 64}, seed=42)
    x, y = dataset.grid['x'], dataset.grid['y']
    vmax = np.abs(dataset.outputs[0]).max()
    im = axes[1, 1].contourf(x, y, dataset.outputs[0], levels=20, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1, 1].set_title('Wave 2D')
    axes[1, 1].set_aspect('equal')

    # Allen-Cahn 2D
    dataset = generate_dataset(model="allen_cahn_2d", n_samples=1, resolution={"x": 64, "y": 64}, seed=42)
    x, y = dataset.grid['x'], dataset.grid['y']
    im = axes[1, 2].contourf(x, y, dataset.outputs[0], levels=20, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1, 2].set_title('Allen-Cahn 2D')
    axes[1, 2].set_aspect('equal')

    # Darcy 2D
    dataset = generate_dataset(model="darcy_2d", n_samples=1, resolution={"x": 64, "y": 64}, seed=42)
    x, y = dataset.grid['x'], dataset.grid['y']
    im = axes[1, 3].contourf(x, y, dataset.outputs[0], levels=20, cmap='viridis')
    axes[1, 3].set_title('Darcy 2D')
    axes[1, 3].set_aspect('equal')

    # Row 3: 2D vector/flow models
    # Stokes 2D
    dataset = generate_dataset(model="stokes_2d", n_samples=1, resolution={"x": 64, "y": 64}, seed=42)
    x, y = dataset.grid['x'], dataset.grid['y']
    vmag = np.sqrt(dataset.outputs[0, :, :, 0]**2 + dataset.outputs[0, :, :, 1]**2)
    im = axes[2, 0].contourf(x, y, vmag, levels=20, cmap='viridis')
    axes[2, 0].set_title('Stokes 2D')
    axes[2, 0].set_aspect('equal')

    # FitzHugh-Nagumo 2D
    dataset = generate_dataset(model="fitzhugh_nagumo_2d", n_samples=1, resolution={"x": 64, "y": 64}, seed=42)
    x, y = dataset.grid['x'], dataset.grid['y']
    im = axes[2, 1].contourf(x, y, dataset.outputs[0, :, :, 0], levels=20, cmap='RdBu_r')
    axes[2, 1].set_title('FitzHugh-Nagumo 2D')
    axes[2, 1].set_aspect('equal')

    # Cylinder Flow steady (transpose for matplotlib)
    dataset = generate_dataset(model="cylinder_flow_2d", n_samples=1, resolution={"x": 110, "y": 41}, seed=42)
    x, y = dataset.grid['x'], dataset.grid['y']
    vmag = np.sqrt(dataset.outputs[0, :, :, 0].T**2 + dataset.outputs[0, :, :, 1].T**2)
    im = axes[2, 2].contourf(x, y, vmag, levels=20, cmap='viridis')
    circle = Circle((0.2, 0.2), 0.05, color='gray', ec='black', lw=1.5, zorder=10)
    axes[2, 2].add_patch(circle)
    axes[2, 2].set_title('Cylinder Flow (Steady)')
    axes[2, 2].set_xlim(0, 2.2)
    axes[2, 2].set_ylim(0, 0.41)

    # Placeholder for unsteady
    axes[2, 3].text(0.5, 0.5, 'Cylinder Flow\n(Unsteady)\nSee Animation',
                    ha='center', va='center', fontsize=12, transform=axes[2, 3].transAxes)
    axes[2, 3].set_title('Cylinder Flow (Unsteady)')
    axes[2, 3].axis('off')

    fig.suptitle('PDEForge: Unified PDE Dataset Generation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'pdeforge_overview.png')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'pdeforge_overview.png'}")


def main():
    print("="*60)
    print("PDEForge Figure Generation")
    print("="*60)
    print(f"Output directory: {FIGURES_DIR}")
    print()

    # Generate overview figure first
    generate_overview_figure()

    # 1D models
    generate_burgers_1d_figure()
    generate_heat_1d_figure()
    generate_wave_1d_figure()
    generate_allen_cahn_1d_figure()
    generate_fitzhugh_nagumo_1d_figure()
    generate_stochastic_heat_figure()

    # 2D models
    generate_heat_2d_figure()
    generate_wave_2d_figure()
    generate_allen_cahn_2d_figure()
    generate_darcy_2d_figure()
    generate_stokes_2d_figure()

    # Cylinder flow (FEniCSx required)
    generate_cylinder_flow_steady_figure()

    # Unsteady cylinder flow and animation
    trajectory, x, y, t = generate_cylinder_flow_unsteady_figure()
    generate_cylinder_flow_animation(trajectory, x, y, t)

    # Parameterized cylinder flow
    generate_parameterized_cylinder_figure()

    # Turbulent cylinder flow and animation
    trajectory, x, y, t, Re = generate_turbulent_cylinder_flow_figure()
    generate_turbulent_cylinder_animation(trajectory, x, y, t, Re)

    print()
    print("="*60)
    print("All figures generated successfully!")
    print(f"Output directory: {FIGURES_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
