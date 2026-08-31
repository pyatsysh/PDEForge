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
- **`eggshell_droplets_3d`** — coalescence versus Ostwald ripening for two droplets in the active shell of an egg-shell catalyst pellet, the canonical sintering dichotomy at the smallest system that separates the two mechanisms. Variable-mobility Cahn-Hilliard, `du/dt = div(M(x) grad mu)`, with the shell geometry carried by the mobility rather than by boundary conditions: the inert core simply does not conduct. Assembled in flux form, so mass is conserved to machine precision whatever the geometry or the noise, and for constant M the step reduces to the `cahn_hilliard` update exactly (4e-16). Runs classify themselves: the regime is decided by where the smaller droplet's material went — absorbed by its partner, or returned to the matrix — with the survival fraction grading the merger. Real-FFT stepper, 3D, second input channel carrying M(x). Validated: mass, energy monotonicity, Gibbs-Thomson ordering, and both regimes reachable and selected by the documented knobs.
- **`darcy_fno_3d`** — the canonical Darcy measure extended to 3D (no frozen 3D canon exists; dimension is the knob): n-d `grf_neumann`, 7-point FD, Jacobi-CG at scale (validated: operator residual 1e-12, triple-sine exact series, CG==LU, shell-averaged spectrum matches the analytic eigenvalues).
- **GPU appliance images** (`:cuda`, `:fenicsx-cuda`): JAX CUDA wheels in the container — spectral models accelerate on any NVIDIA GPU with `--gpus all`; FEM models remain CPU (stock dolfinx/PETSc is CPU-only).
- **CLI + Docker appliance**: `pdeforge generate|reproduce|models|presets|describe` console script; `docker run -v $PWD/data:/data ... pdeforge generate ...` produces datasets with zero installation (slim spectral image + full FEniCSx image, both built by the docker workflow).
- **3D visualization** (`pdeforge[viz3d]`): PyVista volume/isosurface/orthogonal-slice rendering (`dataset.visualize_3d()`, off-screen screenshots verified) + a dependency-free matplotlib slices fallback.
- **Canonical recreations** (`darcy_fno_2d`, `grf_neumann`, `grf_periodic`, 10 canonical presets): the FNO Darcy and Burgers setups regenerable at any resolution with measure knobs — validated against distributed data (bit-exact on Darcy421, see below; 2.7e-8 vs an independent ETDRK4 reference).
- **Canonical Darcy is now bit-exact.** `darcy_fno_2d` transcribes the published generator (`GRF.m` + `solve_gwf.m`) rather than approximating it, and solving the distributed Darcy421 coefficients returns the distributed solutions with 99.1% of the 177,241 float32 values identical, none more than 2 ulp out — the residue is MATLAB's sparse LU against SciPy's. Two fixes got it there:
  - **The grid convention.** The original solves on the NODE grid (spacing 1/(K-1), zero Dirichlet at the boundary nodes) but stores its input and output on the CELL-CENTRE grid (spacing 1/K), moving between them with a not-a-knot cubic spline. That half-cell resampling is why the published solutions are small but nonzero on the boundary, and assuming a plain node-grid solve cost 0.49% rel-L2. The new `grid` parameter selects `"canonical"` (default, reproduces the published arrays) or `"node"` (a clean solve with an exactly zero boundary — the better choice for new data, and the new `fno_darcy_clean_2d` preset).
  - **The measure normalisation.** `GRF.m` fixes the field scale as `tau^(alpha-1)` with no free constant; the previously hard-coded `sigma = 0.2918` was a fit to that constant at alpha = 2, tau = 3 (exact value 0.292083) and was wrong everywhere else. `sigma=None` now derives it, so alpha and tau are genuine knobs; pass a number to override the contrast deliberately.
- **`pdeforge.read_torch_pt`** — read `torch.save` archives as memory-mapped numpy arrays without PyTorch, so inspecting the distributed 7 GB `.pt` benchmark files does not require a deep-learning framework. Refuses any payload beyond tensors and plain containers rather than unpickling it.
- **`pdeforge.load_darcy_fno`** — the distributed Darcy `.pt` files on the `PDEDataset` surface, on the cell-centre grid they are actually sampled on, with strided low-resolution views validated against the stored grid.
- **`airfoil_euler_2d`** — transonic compressible Euler over parameterized NACA airfoils, the shock-capturing seam the spectral solvers cannot provide. Body-fitted C-grid rebuilt per sample (`pdeforge.geometry.airfoil_c_grid`), cell-centred finite volume with HLLC fluxes, MUSCL/minmod reconstruction and local time stepping to steady state (`pdeforge.solvers.euler_fv`), characteristic far field with the Thomas-Salas vortex correction. Geometry IS the data: the deformed mesh is the input, (rho, u, v, p) on it the output, with per-sample C_l, C_d and residual drop in metadata.
- **AirfRANS interop** (`pdeforge.load_airfrans`) — read, do not rebuild. The distributed AirfRANS RANS data (Bonnet et al., NeurIPS 2022) loads onto the `PDEDataset` surface with its own manifest splits, per-case parameters decoded from the case names, and every airfoil wall node retained by default. Backed by `pdeforge.read_vtk_xml`, a dependency-free VTK XML reader (inline base64, zlib or raw), so no vtk/pyvista/meshio enters the install.
- **KdV regimes unified under one model.** `kdv_1d` gains `advection` / `dispersion` / `dealias` / `scale_jitter` knobs plus a trajectory burn-in (`_n_frames_kept`), which makes every published KdV setup a preset over one solver rather than a model of its own: `kdv_dsw_1d` and `kdv_dsw_epistemic_1d` (undular bore — matched bias / epistemic pair at delta2 = 8e-6 and 4e-5) and `mp_pde_kdv_1d` / `mp_pde_kdv_easy_1d` (Brandstetter et al., arXiv:2202.03376 / 2202.07643 — u_t + u u_x + u_xxx = 0 on L = 128, nx = 256, trajectory outputs, per-sample scale jitter). New IC generators `sine_series` (`TruncatedSineGenerator`, carrying the half-open randint wavenumber convention) and `depression_box`. Presets can now pin `domain`, `resolution`, and `outputs`. Validated: 8e-7 rel-L2 from a converged nx = 1024 reference at T = 100, where the reference generator's own Radau + psdiff sits at 2.9e-4.
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
