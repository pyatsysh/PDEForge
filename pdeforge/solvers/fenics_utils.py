"""
FEniCSx utilities for PDEForge.

This module provides helper functions for mesh generation,
boundary condition handling, and other FEniCSx operations.
"""

import numpy as np
from typing import Tuple, Callable, Optional

# Check if dependencies are available
try:
    import gmsh
    HAS_GMSH = True
except ImportError:
    HAS_GMSH = False

try:
    import dolfinx
    from dolfinx import mesh as dfx_mesh
    # API changed in dolfinx 0.8+
    try:
        from dolfinx.io import gmshio  # older API
    except ImportError:
        from dolfinx.io import gmsh as gmshio  # dolfinx 0.8+
    from mpi4py import MPI
    HAS_FENICSX = True
except ImportError:
    HAS_FENICSX = False


def create_rectangle_with_hole(
    L: float = 2.2,
    H: float = 0.41,
    cx: float = 0.2,
    cy: float = 0.2,
    r: float = 0.05,
    resolution: float = 0.02,
    comm=None,
) -> "dolfinx.mesh.Mesh":
    """
    Create a rectangular mesh with a circular hole (for cylinder flow).
    
    This is the classic benchmark geometry for flow around a cylinder.
    
    Parameters
    ----------
    L : float
        Length of the rectangle (x-direction)
    H : float
        Height of the rectangle (y-direction)
    cx, cy : float
        Center of the cylinder
    r : float
        Radius of the cylinder
    resolution : float
        Characteristic mesh size
    comm : MPI communicator
        MPI communicator (default: COMM_WORLD)
        
    Returns
    -------
    dolfinx.mesh.Mesh
        The mesh with markers:
        - Boundary tag 1: Inlet (left)
        - Boundary tag 2: Outlet (right)
        - Boundary tag 3: Walls (top/bottom)
        - Boundary tag 4: Cylinder surface
    """
    if not HAS_GMSH:
        raise ImportError("gmsh is required for mesh generation. Install with: pip install gmsh")
    if not HAS_FENICSX:
        raise ImportError("FEniCSx is required. Install with: conda install -c conda-forge fenics-dolfinx")
    
    if comm is None:
        comm = MPI.COMM_WORLD
    
    gmsh.initialize()
    gmsh.model.add("cylinder_flow")
    
    # Only create geometry on rank 0
    if comm.rank == 0:
        # Create rectangle
        rect = gmsh.model.occ.addRectangle(0, 0, 0, L, H)
        
        # Create cylinder
        cylinder = gmsh.model.occ.addDisk(cx, cy, 0, r, r)
        
        # Cut cylinder from rectangle
        domain = gmsh.model.occ.cut([(2, rect)], [(2, cylinder)])
        gmsh.model.occ.synchronize()
        
        # Get boundaries
        surfaces = gmsh.model.getEntities(dim=2)
        boundary = gmsh.model.getBoundary(surfaces, oriented=False)
        
        # Mark boundaries
        inlet = []
        outlet = []
        walls = []
        cylinder_boundary = []
        
        for dim, tag in boundary:
            com = gmsh.model.occ.getCenterOfMass(dim, tag)
            if np.isclose(com[0], 0.0):
                inlet.append(tag)
            elif np.isclose(com[0], L):
                outlet.append(tag)
            elif np.isclose(com[1], 0.0) or np.isclose(com[1], H):
                walls.append(tag)
            else:
                # Must be the cylinder
                cylinder_boundary.append(tag)
        
        # Create physical groups
        gmsh.model.addPhysicalGroup(1, inlet, 1)
        gmsh.model.setPhysicalName(1, 1, "inlet")
        
        gmsh.model.addPhysicalGroup(1, outlet, 2)
        gmsh.model.setPhysicalName(1, 2, "outlet")
        
        gmsh.model.addPhysicalGroup(1, walls, 3)
        gmsh.model.setPhysicalName(1, 3, "walls")
        
        gmsh.model.addPhysicalGroup(1, cylinder_boundary, 4)
        gmsh.model.setPhysicalName(1, 4, "cylinder")
        
        # Add fluid domain
        gmsh.model.addPhysicalGroup(2, [s[1] for s in surfaces], 1)
        gmsh.model.setPhysicalName(2, 1, "fluid")
        
        # Set mesh size
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", resolution * 0.5)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", resolution)
        
        # Refine near cylinder
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "CurvesList", cylinder_boundary)
        gmsh.model.mesh.field.setNumber(1, "Sampling", 100)
        
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", resolution * 0.2)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", resolution)
        gmsh.model.mesh.field.setNumber(2, "DistMin", r)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 5 * r)
        
        gmsh.model.mesh.field.setAsBackgroundMesh(2)
        
        # Generate mesh
        gmsh.model.mesh.generate(2)
    
    # Import mesh to dolfinx
    # API changed in dolfinx 0.8+ - now returns MeshData named tuple
    result = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2)

    # Handle both old API (tuple) and new API (MeshData namedtuple)
    if hasattr(result, 'mesh'):
        # New API (dolfinx 0.8+)
        mesh = result.mesh
        cell_tags = result.cell_tags
        facet_tags = result.facet_tags
    else:
        # Old API
        mesh, cell_tags, facet_tags = result

    gmsh.finalize()

    # Store facet tags for boundary conditions
    mesh.facet_tags = facet_tags
    mesh.cell_tags = cell_tags

    return mesh


def create_simple_rectangle(
    L: float = 1.0,
    H: float = 1.0,
    nx: int = 32,
    ny: int = 32,
    comm=None,
) -> "dolfinx.mesh.Mesh":
    """
    Create a simple rectangular mesh without holes.
    
    Useful for problems without obstacles.
    
    Parameters
    ----------
    L, H : float
        Domain dimensions
    nx, ny : int
        Number of cells in each direction
    comm : MPI communicator
        
    Returns
    -------
    dolfinx.mesh.Mesh
        Rectangular mesh
    """
    if not HAS_FENICSX:
        raise ImportError("FEniCSx is required")
    
    if comm is None:
        comm = MPI.COMM_WORLD
    
    return dfx_mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0]), np.array([L, H])],
        [nx, ny],
        dfx_mesh.CellType.triangle,
    )


def mark_boundaries_rectangle(
    mesh: "dolfinx.mesh.Mesh",
    L: float,
    H: float,
    tol: float = 1e-10,
) -> "dolfinx.mesh.MeshTags":
    """
    Mark boundaries of a rectangle.
    
    Tags:
    - 1: Left (inlet)
    - 2: Right (outlet)
    - 3: Bottom
    - 4: Top
    
    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        The mesh
    L, H : float
        Domain dimensions
    tol : float
        Geometric tolerance
        
    Returns
    -------
    dolfinx.mesh.MeshTags
        Facet tags for boundaries
    """
    from dolfinx import mesh as dfx_mesh
    
    def left(x):
        return np.isclose(x[0], 0.0, atol=tol)
    
    def right(x):
        return np.isclose(x[0], L, atol=tol)
    
    def bottom(x):
        return np.isclose(x[1], 0.0, atol=tol)
    
    def top(x):
        return np.isclose(x[1], H, atol=tol)
    
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, mesh.topology.dim)
    
    facets_left = dfx_mesh.locate_entities_boundary(mesh, fdim, left)
    facets_right = dfx_mesh.locate_entities_boundary(mesh, fdim, right)
    facets_bottom = dfx_mesh.locate_entities_boundary(mesh, fdim, bottom)
    facets_top = dfx_mesh.locate_entities_boundary(mesh, fdim, top)
    
    markers = np.hstack([
        np.full_like(facets_left, 1),
        np.full_like(facets_right, 2),
        np.full_like(facets_bottom, 3),
        np.full_like(facets_top, 4),
    ])
    facets = np.hstack([facets_left, facets_right, facets_bottom, facets_top])
    
    sorted_idx = np.argsort(facets)
    return dfx_mesh.meshtags(mesh, fdim, facets[sorted_idx], markers[sorted_idx])
