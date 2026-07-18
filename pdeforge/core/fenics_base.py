"""
Base class for FEniCSx-based models.

Handles complex geometries, unstructured meshes, non-periodic BCs.
FEM solutions are interpolated to regular grids for ML compatability.
"""

import warnings
from abc import abstractmethod

import numpy as np

from pdeforge.core.base import PDEModel
from pdeforge.core.types import Domain, GridSpec, PDEDataset

# check if FEniCSx available
try:
    import basix
    import dolfinx
    import ufl
    from dolfinx import fem, geometry, io
    from dolfinx import mesh as dfx_mesh
    from dolfinx.fem.petsc import LinearProblem
    from mpi4py import MPI

    HAS_FENICSX = True
except ImportError:
    HAS_FENICSX = False
    dolfinx = None


def require_fenicsx(func):
    """Decorator to check FEniCSx is available."""

    def wrapper(*args, **kwargs):
        if not HAS_FENICSX:
            raise ImportError(
                "FEniCSx (dolfinx) required. "
                "Install: conda install -c conda-forge fenics-dolfinx"
            )
        return func(*args, **kwargs)

    return wrapper


class FEniCSModel(PDEModel):
    """
    Base for FEniCSx models. Subclasses implement create_mesh(),
    create_function_spaces(), solve(), generate_ic().
    """

    BACKEND = "fenicsx"
    BACKENDS = {"fenicsx"}
    # dolfinx/PETSc objects are not picklable; MPI does not mix with pools.
    PARALLEL_SAFE = False

    def __init__(self, resolution, domain=None, mesh_refinement=1, **params):
        """
        resolution: output grid resolution for interpolated solutions
        domain: bounds for output grid and mesh
        mesh_refinement: higher = finer mesh
        """
        if not HAS_FENICSX:
            raise ImportError(
                "FEniCSx (dolfinx) required. "
                "Install: conda install -c conda-forge fenics-dolfinx"
            )

        super().__init__(resolution, domain, **params)
        self.backend = "fenicsx"

        self.mesh_refinement = mesh_refinement
        self.output_resolution = resolution

        self.mesh = self.create_mesh()
        self.comm = self.mesh.comm

        self.create_function_spaces()
        self._setup_output_grid()

    @abstractmethod
    def create_mesh(self):
        pass

    @abstractmethod
    def create_function_spaces(self):
        # store as self.V, self.Q etc
        pass

    def _setup_output_grid(self):
        """Setup regular grid for output interpolation."""
        dims = sorted(self.output_resolution.keys())

        if len(dims) == 2:
            x = self.grids["x"]
            y = self.grids["y"]
            X, Y = np.meshgrid(x, y, indexing="ij")
            # dolfinx 0.10+ requires 3D points even for 2D meshes
            Z = np.zeros_like(X)
            self._output_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        elif len(dims) == 3:
            x, y, z = self.grids["x"], self.grids["y"], self.grids["z"]
            X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
            self._output_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        else:
            raise ValueError("FEniCSModel only suports 2D and 3D")

    def interpolate_to_grid(self, fem_function, fill_value=0.0):
        """
        Interpolate FEM function to regular output grid.

        fill_value: used for points outside mesh (e.g. inside obstacles)
        """
        from dolfinx import geometry

        bb_tree = geometry.bb_tree(self.mesh, self.mesh.topology.dim)

        cell_candidates = geometry.compute_collisions_points(
            bb_tree, self._output_points
        )
        cell_collisions = geometry.compute_colliding_cells(
            self.mesh, cell_candidates, self._output_points
        )

        points_on_proc = []
        cells_on_proc = []
        point_indices = []

        for i, point in enumerate(self._output_points):
            cells = cell_collisions.links(i)
            if len(cells) > 0:
                points_on_proc.append(point)
                cells_on_proc.append(cells[0])
                point_indices.append(i)

        points_on_proc = np.array(points_on_proc, dtype=np.float64)

        # init output with fill value
        shape_suffix = fem_function.ufl_shape
        if shape_suffix:
            output_flat = np.full(
                (len(self._output_points), *shape_suffix), fill_value, dtype=np.float64
            )
        else:
            output_flat = np.full(
                len(self._output_points), fill_value, dtype=np.float64
            )

        if len(points_on_proc) > 0:
            # pad to 3D if needed (dolfinx 0.10+ requires 3D points)
            if points_on_proc.shape[1] == 2:
                points_3d = np.column_stack(
                    [points_on_proc, np.zeros(len(points_on_proc))]
                )
            else:
                points_3d = points_on_proc

            values = fem_function.eval(points_3d, cells_on_proc)

            # dolfinx 0.10+ returns (n, 1) for scalars, squeeze if needed
            if values.ndim == 2 and values.shape[1] == 1 and not shape_suffix:
                values = values.squeeze(axis=1)

            for idx, val in zip(point_indices, values):
                output_flat[idx] = val

        # reshape to grid
        dims = sorted(self.output_resolution.keys())
        shape = tuple(self.output_resolution[d] for d in dims)

        if shape_suffix:
            shape = shape + shape_suffix

        return output_flat.reshape(shape)

    def create_mask(self):
        # True where point is inside domain
        from dolfinx import geometry

        bb_tree = geometry.bb_tree(self.mesh, self.mesh.topology.dim)
        cell_candidates = geometry.compute_collisions_points(
            bb_tree, self._output_points
        )
        cell_collisions = geometry.compute_colliding_cells(
            self.mesh, cell_candidates, self._output_points
        )

        mask = np.zeros(len(self._output_points), dtype=bool)
        for i in range(len(self._output_points)):
            if len(cell_collisions.links(i)) > 0:
                mask[i] = True

        dims = sorted(self.output_resolution.keys())
        shape = tuple(self.output_resolution[d] for d in dims)

        return mask.reshape(shape)

    def generate_dataset(
        self,
        n_samples,
        ic_generator="default",
        ic_params=None,
        seed=None,
        validate=True,
        n_jobs=1,
        verbose=True,
    ):
        """Generate dataset, adds domain_mask to metadata."""
        dataset = super().generate_dataset(
            n_samples=n_samples,
            ic_generator=ic_generator,
            ic_params=ic_params,
            seed=seed,
            validate=validate,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        dataset.metadata["domain_mask"] = self.create_mask()
        dataset.metadata["backend"] = "fenicsx"

        return dataset
