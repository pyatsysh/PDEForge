# Changelog

## 0.2.0 (unreleased)

The "data engine" release: the solver seam, the JAX backend, UQ-native data
tooling, streaming IO, and a doubled model catalogue.

### Added
- **Solver seam** (`pdeforge.solvers.semilinear`): semi-linear spectral models
  declare a linear symbol + nonlinear term once; ETDRK4 (Kassam-Trefethen
  coefficients) integrates the stiff linear part exactly. Dimension-agnostic,
  multi-component, 2/3-dealiasing built in.
- **JAX backend** (`backend="jax"`, `pip install pdeforge[jax]`): the same
  model specs jit-compiled with `lax.scan` over time and `vmap` over samples;
  float64 enforced; ~16x CPU speedup measured, GPU-capable. NumPy in, NumPy
  out — JAX never leaks into the data surface. ICs are always generated in
  NumPy, so inputs are bit-identical across backends.
- **16 new models** (catalogue now 34): `ns_vorticity_2d` (+ Taylor-Green
  validation), `kolmogorov_flow_2d`, `ks_1d`, `kdv_1d` (+ soliton test),
  `advection_1d` (exact), `gray_scott_2d`, `lotka_volterra_2d` (validated
  against the LV ODE), `burgers_2d`, `shallow_water_2d` (mass-conserving,
  gravity-wave-speed validated), `schrodinger_1d` (norm-conserving
  split-step), `heterogeneous_wave_2d` (medium -> wavefield), `helmholtz_2d`
  (operator-residual exact), `heat_3d`, `allen_cahn_3d`,
  `stochastic_burgers_1d`, `stochastic_allen_cahn_2d`.
- **`naca_flow_2d`** — parameterized NACA 4-digit airfoil family (FEniCSx): per-sample geometry (thickness, camber, camber position, angle of attack) re-meshed via gmsh, steady laminar NS, SDF input channel, lift/drag coefficients from the surface stress integral recorded per sample. Plus `pdeforge.geometry` (NACA coords, polygon SDF — pure NumPy) and a channel-with-polygon gmsh helper.
- **`darcy_fno_3d`** — the canonical Darcy measure extended to 3D (no frozen 3D canon exists; dimension is the knob): n-d `grf_neumann`, 7-point FD, Jacobi-CG at scale (validated: operator residual 1e-12, triple-sine exact series, CG==LU, shell-averaged spectrum matches the analytic eigenvalues).
- **GPU appliance images** (`:cuda`, `:fenicsx-cuda`): JAX CUDA wheels in the container — spectral models accelerate on any NVIDIA GPU with `--gpus all`; FEM models remain CPU (stock dolfinx/PETSc is CPU-only).
- **CLI + Docker appliance**: `pdeforge generate|reproduce|models|presets|describe` console script; `docker run -v $PWD/data:/data ... pdeforge generate ...` produces datasets with zero installation (slim spectral image + full FEniCSx image, both built by the docker workflow).
- **3D visualization** (`pdeforge[viz3d]`): PyVista volume/isosurface/orthogonal-slice rendering (`dataset.visualize_3d()`, off-screen screenshots verified) + a dependency-free matplotlib slices fallback.
- **Canonical recreations** (`darcy_fno_2d`, `grf_neumann`, `grf_periodic`, 9 canonical presets): the FNO Darcy and Burgers setups regenerable at any resolution with measure knobs — validated against distributed data (0.49% rel-L2 on Darcy421; 2.7e-8 vs an independent ETDRK4 reference).
- **UQ layer** (`pdeforge.uq`): parameter distributions (Uniform, LogUniform,
  Normal, Choice) with LHS/Sobol/Halton designs and per-sample provenance;
  `split_ood` distribution-shift splits; `generate_multifidelity` (same
  realisations across resolutions); `observe` (sensors/subsampling/noise);
  dependency-free split-conformal helpers with a coverage test in CI.
- **Verification harness** (`pdeforge.verify`): convergence studies against
  a fine reference — error bars on the ground truth.
- **Trajectory outputs**: `generate_dataset(..., outputs="trajectory")`
  returns full rollouts with a time coordinate.
- **Streaming generation**: `to="path"` writes chunk-by-chunk (memmapped
  directory or HDF5) — no RAM ceiling; directory datasets load lazily with
  `mmap=True`.
- **Parallel generation**: `n_jobs` now actually parallelises (process pool
  with spawned SeedSequences); FEM models fall back gracefully.
- **Provenance**: datasets record package version, git SHA, timestamp,
  backend, and outputs mode; `pdeforge.reproduce(metadata_or_path)`
  regenerates any seeded dataset from its own metadata. Split metadata
  records the split seed and fractions.
- **IO**: zarr export/import; PDEBench-layout HDF5 export
  (`export_pdebench`); `to_torch()` / `torch_loader()` / `to_jax()` adapters.
- **Presets** (`generate_dataset(preset="fno_burgers_1d", ...)`): classic
  benchmark setups regenerable at any resolution.
- Docs: calibration protocol page, comparison page, measured performance
  numbers. CI: extras lane (jax/torch/zarr), python 3.10-3.13, honest
  FEniCSx smoke tests.

### Changed
- **RNG**: per-sample seeding now derives from `numpy SeedSequence.spawn`
  (collision-free, retry-safe, parallel-safe). Datasets generated with 0.1.0
  seeds do NOT reproduce bit-for-bit under 0.2.0.
- `burgers_1d`, `heat_1d`, `heat_2d`, `allen_cahn_1d` moved to the ETDRK4
  seam (the odeint "excess work" warnings are gone; solutions change within
  solver tolerance).
- `requires-python` floor raised to 3.10.
- Steady models (`darcy_2d`, `stokes_2d`, `helmholtz_2d`) reject
  `outputs="trajectory"` explicitly.

### Fixed
- Time-step snapping: integrators no longer overshoot the target time by up
  to one dt (and never take a spurious extra step from float rounding).
- README placeholder URLs; stale author email in packaging metadata.
