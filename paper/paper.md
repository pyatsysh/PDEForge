---
title: 'PDEForge: A Unified Framework for Generating PDE Datasets for Operator Learning'
tags:
  - Python
  - partial differential equations
  - operator learning
  - neural operators
  - scientific machine learning
  - data generation
  - uncertainty quantification
authors:
  - name: Peter Yatsyshin
    orcid: 0000-0002-8844-281X
    corresponding: true
    email: p.yatsyshin@icloud.com
    affiliation: 1
affiliations:
  - name: "Affiliation to be confirmed"
    index: 1
date: 16 January 2025
bibliography: paper.bib
---

# Summary

Operator learning is a practical route to accelerating the numerical solution of partial differential equations (PDEs). A neural operator learns the map between the infinite-dimensional spaces of a problem's data and its solution [@kovachki2023neural]. Once trained, it evaluates in a fraction of the time taken by the classical solver it replaces. This moves the computational burden from solving to training, so that the surrogate is only ever as good as the data behind it: large datasets, varied systematically in resolution and in the physical parameters of the model. Generating that data is itself the bottleneck. Public benchmarks fix the resolution and the parameters at which they were produced. Rolling one's own data means writing and validating a separate solver for every PDE of interest. This is a highly non-trivial task, far removed from the machine learning it is meant to serve.

In the present work we present PDEForge, a Python package that generates PDE datasets behind one uniform interface. We implement 41 models: 30 spectral, 2 finite-difference elliptic, 1 finite-volume and 8 finite-element. They span 1D, 2D and 3D domains, from scalar diffusion to turbulent flow around an obstacle (\autoref{fig:overview}). We treat uncertainty quantification (UQ) as a first-class concern, and supply natively the calibration splits that post-hoc methods such as conformal prediction require.

![Overview of PDE models available in PDEForge. Top row: 1D time-dependent equations (Burgers, Heat, Wave, Allen-Cahn). Middle row: 2D scalar field problems (Heat, Wave, Allen-Cahn, Darcy). Bottom row: 2D vector field and flow problems (Stokes, FitzHugh-Nagumo, steady and unsteady cylinder flow).\label{fig:overview}](pdeforge_overview.png)

# Statement of Need

The literature on operator learning has grown quickly. Architectures such as the Fourier Neural Operator [@li2020fourier], DeepONet [@lu2021learning] and their transformer descendants [@kovachki2023neural] now attain good accuracy on PDE surrogate tasks. Yet the data has not kept pace, and in practice it remains fragmented, for four reasons.

Firstly, public benchmarks such as PDEBench [@takamoto2022pdebench] ship at a fixed resolution and a fixed set of parameters. A study that needs a different discretisation or a wider parameter sweep cannot obtain it from the benchmark. Secondly, bespoke solvers each expose a different interface and write a different format. Comparing one method across several PDEs then becomes an exercise in bookkeeping rather than modelling. Thirdly, uncertainty quantification for neural operators [@zou2023uncertainty] needs a dedicated calibration set, held out from training. The standard benchmarks do not provide one. Fourthly, the most demanding benchmarks are flow problems: flow past an obstacle, vortex shedding, turbulence. These require finite-element discretisation and mesh generation. That effort falls on the machine-learning researcher least equipped to absorb it.

We answer all four through a single entry point. One function, `generate_dataset()`, serves every model and returns data in one format. It carries metadata and, where wanted, train/validation/calibration/test splits.

# Implemented Models

We group the models by dimensionality and by numerical method. The 30 spectral models use FFT-based pseudo-spectral solvers (ETDRK4 exponential integrators for the stiff semi-linear families). They install with NumPy, SciPy and Matplotlib alone:

- **1D, time-dependent (11):** Burgers (with shock formation), linear advection (exactly solvable), heat, wave, Korteweg–de Vries, Kuramoto–Sivashinsky (chaotic), the nonlinear Schrödinger equation, Allen–Cahn, FitzHugh–Nagumo, and stochastic heat and Burgers equations.
- **2D (17):** heat, wave and Allen–Cahn; incompressible Navier–Stokes in vorticity form and its forced Kolmogorov-flow variant; the shallow water system; 2D Burgers; Darcy flow; Stokes flow; the Helmholtz equation; wave propagation through heterogeneous media; Gray–Scott and diffusive Lotka–Volterra kinetics; FitzHugh–Nagumo; stochastic heat and Allen–Cahn; and Cahn–Hilliard, resolving spinodal decomposition in two and three dimensions.
- **3D (2):** heat and Allen–Cahn on the periodic cube.
- **Finite-difference elliptic (2):** the canonical Darcy benchmark on the unit square with Dirichlet conditions, and its extension to the unit cube; both draw coefficients from the standard Gaussian measure of the operator-learning literature, with every hyperparameter exposed.

The remaining 5 models treat flow in complex geometry. Here spectral methods no longer apply, and we turn to finite elements. We build these models on FEniCSx [@baratta2023dolfinx] with gmsh for meshing. They cover steady, unsteady, parameterised and turbulent flow past a cylinder, and steady flow past a parameterised NACA airfoil family, where every sample draws its own geometry and the lift and drag coefficients are recorded alongside the fields. We close the turbulent case with a Smagorinsky large-eddy model. \autoref{fig:cylinder} shows the unsteady model resolving the so-called von Kármán vortex street. This is a prototypical benchmark for fluid dynamics and for neural-operator evaluation alike.

![Time evolution of vorticity in the unsteady cylinder flow model, showing development of the von Kármán vortex street. The simulation captures vortex shedding dynamics at moderate Reynolds number using FEniCSx finite elements.\label{fig:cylinder}](cylinder_flow_unsteady_vorticity.png)

# Key Features

## A single interface

We drive every model through the same call. The interface is deliberately minimalistic. It exposes only the model name, the number of samples, the resolution and the physical parameters:

```python
from pdeforge import generate_dataset

dataset = generate_dataset(
    model="burgers_1d",
    n_samples=1000,
    resolution={"x": 256},
    params={"viscosity": 0.01},
    seed=42,
)
```

Notice that switching from Burgers to Darcy flow, or from 1D to 2D, changes the arguments but never the call.

## UQ-ready data splits

We built PDEForge with uncertainty quantification in mind. A dataset partitions in one line into the four-way split that conformal prediction and related methods need:

```python
splits = dataset.split(train=0.6, val=0.15, cal=0.15, test=0.1)
```

The calibration fold is held out from both training and testing. Coverage guarantees computed on it therefore remain valid.

## Extensibility

New models register through a decorator, so that a contributor adds a PDE without touching the core:

```python
from pdeforge.core.registry import register_model

@register_model("my_custom_pde")
class MyPDE(PDEModel):
    ...
```

# Implementation

PDEForge is written in Python. The spectral models rest on NumPy, SciPy and Matplotlib alone, and use FFT-based pseudo-spectral methods. The complex-geometry models call the FEniCSx finite element library [@baratta2023dolfinx] with gmsh for mesh generation. Throughout, the PDE-specific physics is kept separate from the shared machinery of dataset assembly, input/output and visualisation. The two can therefore evolve independently, and the barrier to a community contribution stays low.

# Availability

We release PDEForge under the MIT licence at [github.com/pyatsysh/PDEForge](https://github.com/pyatsysh/PDEForge). Documentation, installation instructions and Jupyter notebook tutorials are provided.

# References
