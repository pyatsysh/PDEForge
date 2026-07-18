# How PDEForge Compares

An honest feature comparison with the main alternatives, as of mid-2026.
Corrections welcome — open an issue.

|  | PDEForge | APEBench/exponax | PDEBench | The Well | PDEArena | py-pde |
|---|---|---|---|---|---|---|
| Delivery | **generate on demand** | generate on demand | download (+per-PDE scripts) | download (15 TB) | download | DIY solver |
| One-call multi-physics API | **yes** | yes (scenarios) | no | no | no | no |
| Any resolution | **yes** | yes | regeneration is DIY | no | no | yes (DIY) |
| Any parameters | **yes** | yes | DIY | no | partial | yes (DIY) |
| Train/val/**calibration**/test | **yes — native** | no | no | no (3-way) | no (3-way) | no |
| OOD splits by parameter range | **yes** | no | no | no | no | no |
| Multi-fidelity pairs | **yes** | no | no | no | no | no |
| FEM / complex geometry | **yes (FEniCSx)** | no (periodic only) | no | fixed datasets | no | no |
| Stochastic PDEs | **yes** | no | no | some (fixed) | no | yes |
| Framework-agnostic output | **NumPy/HDF5/zarr** | JAX arrays | HDF5 | HDF5/torch | HDF5 | NumPy |
| GPU acceleration | optional JAX backend | JAX-native | n/a | n/a | n/a | numba |
| Base install | NumPy/SciPy (pip) | JAX (pip) | heavy | loader only | medium | pip |
| Convergence-verified data | **yes (`pdeforge.verify`)** | no | no | no | no | no |
| PDE count | 30+ (and growing) | ~46 | 11 families | 16 datasets | 5 families | user-defined |

**When to use something else.** Pretraining a foundation model on tens of
terabytes of heterogeneous physics: The Well. Benchmarking autoregressive
emulator rollouts with differentiable-solver training in JAX: APEBench.
Comparing against the literature's fixed reference numbers: PDEBench's
published datasets. A PDE we don't implement and you want to hand-discretise:
py-pde, Dedalus, or PhiFlow.

**When to use PDEForge.** Controlled studies that need data at YOUR
resolution and YOUR parameters; anything involving calibrated uncertainty
(conformal prediction needs the calibration split we ship natively);
distribution-shift and multi-fidelity experiments; complex-geometry flow
data without writing FEM code; reproducibility — every dataset regenerates
from its own metadata (`pdeforge.reproduce`).
