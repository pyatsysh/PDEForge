"""
NACA airfoil family: geometry tests (pure NumPy, always run) and FEM flow
tests (run when dolfinx is available — locally via the micromamba env, and
in the fenicsx CI lane).
"""

import numpy as np
import pytest

from pdeforge.geometry import naca4_coords, polygon_sdf, rotate_airfoil


class TestNACAGeometry:
    def test_thickness_matches_designation(self):
        """Max thickness of the 0012 is 12% chord (at ~30% chord)."""
        poly = naca4_coords(thickness=0.12, camber=0.0, n_points=400)
        x, y = poly[:, 0], poly[:, 1]
        # symmetric foil: thickness = 2 * max(|y|)
        assert abs(2 * np.abs(y).max() - 0.12) < 0.002

    def test_trailing_edge_closed(self):
        poly = naca4_coords(thickness=0.15, camber=0.04)
        assert np.allclose(poly[0], poly[-1], atol=1e-14)

    def test_symmetric_foil_mirror_symmetry(self):
        poly = naca4_coords(thickness=0.12, camber=0.0, n_points=201)
        upper = poly[poly[:, 1] > 1e-8]
        lower = poly[poly[:, 1] < -1e-8]
        # for every upper point there is a mirrored lower point
        for xu, yu in upper[:: max(1, len(upper) // 12)]:
            d = np.abs(lower[:, 0] - xu) + np.abs(lower[:, 1] + yu)
            assert d.min() < 5e-3

    def test_camber_lifts_the_mean_line(self):
        cambered = naca4_coords(thickness=0.12, camber=0.04, camber_pos=0.4)
        assert cambered[:, 1].mean() > 0.005  # mean surface height > 0

    def test_rotation_is_rigid(self):
        """Same-index pairwise distances are preserved (rigid-body motion)."""
        poly = naca4_coords(thickness=0.12)
        rot = rotate_airfoil(poly, 10.0)
        le = np.argmin(poly[:, 0])
        d0 = np.linalg.norm(poly[0] - poly[le])  # TE-to-LE, fixed indices
        d1 = np.linalg.norm(rot[0] - rot[le])
        assert np.isclose(d0, d1, rtol=1e-12)
        # and the quarter-chord pivot stays put
        piv = np.array([0.25, 0.0])
        k = np.argmin(np.linalg.norm(poly - piv, axis=1))
        assert np.linalg.norm(rot[k] - poly[k]) < np.linalg.norm(poly[k] - piv) * 0.35

    def test_positive_aoa_drops_trailing_edge(self):
        """Nose-up rotation about quarter-chord moves the TE down."""
        poly = naca4_coords(thickness=0.12, camber=0.0)
        rot = rotate_airfoil(poly, 8.0)
        te_y = rot[np.argmax(rot[:, 0]), 1]
        assert te_y < -0.05


class TestPolygonSDF:
    def test_sign_convention(self):
        poly = naca4_coords(thickness=0.12, camber=0.0)
        closed = np.vstack([poly, poly[:1]])
        X, Y = np.meshgrid(np.linspace(-0.5, 1.5, 41), np.linspace(-0.5, 0.5, 21))
        sdf = polygon_sdf(X, Y, closed)
        # centroid region inside (negative), far field outside (positive)
        assert sdf[10, 20] < 0  # near (0.5, 0)
        assert sdf[0, 0] > 0
        assert sdf[-1, -1] > 0

    def test_distance_accuracy_far_field(self):
        """Far from the foil, SDF ~ distance to the nearest surface point."""
        poly = naca4_coords(thickness=0.12, camber=0.0, n_points=300)
        closed = np.vstack([poly, poly[:1]])
        pt = np.array([[2.0]]), np.array([[0.0]])  # 1 chord behind the TE
        sdf = polygon_sdf(pt[0], pt[1], closed)
        d_true = np.min(np.linalg.norm(poly - np.array([2.0, 0.0]), axis=1))
        assert abs(sdf[0, 0] - d_true) < 1e-6

    def test_zero_level_near_surface(self):
        poly = naca4_coords(thickness=0.12, camber=0.0, n_points=300)
        closed = np.vstack([poly, poly[:1]])
        # a point ON the surface (max-thickness upper point)
        k = np.argmax(poly[:, 1])
        sdf = polygon_sdf(poly[k : k + 1, :1], poly[k : k + 1, 1:], closed)
        assert abs(sdf[0, 0]) < 1e-10


# ----------------------------------------------------------------------
# FEM flow tests (dolfinx required)
# ----------------------------------------------------------------------

try:
    import dolfinx  # noqa: F401

    HAS_DOLFINX = True
except ImportError:
    HAS_DOLFINX = False

from pdeforge import generate_dataset, get_model  # noqa: E402

RES = {"x": 48, "y": 24}
COARSE = {"_mesh_resolution": 0.15}


@pytest.mark.skipif(not HAS_DOLFINX, reason="FEniCSx (dolfinx) not installed")
class TestNACAFlow:
    def test_solve_converges_and_is_finite(self):
        m = get_model("naca_flow_2d")(resolution=RES, **COARSE)
        sol = m.solve(thickness=0.12, camber=0.0, camber_pos=0.4, aoa=0.0)
        assert np.isfinite(sol).all()
        assert sol.shape == (48, 24, 3)
        # flow actually moves
        assert np.abs(sol[..., 0]).max() > 0.5

    def test_symmetric_zero_aoa_has_no_lift(self):
        """NACA 0012 at zero incidence: Cl ~ 0 by symmetry, Cd > 0."""
        m = get_model("naca_flow_2d")(resolution=RES, **COARSE)
        m.solve(thickness=0.12, camber=0.0, camber_pos=0.4, aoa=0.0)
        f = m._last_forces
        assert f["Cd"] > 0.0
        assert abs(f["Cl"]) < 0.12 * f["Cd"] + 0.05  # small vs drag scale

    def test_positive_aoa_generates_positive_lift(self):
        m = get_model("naca_flow_2d")(resolution=RES, **COARSE)
        m.solve(thickness=0.12, camber=0.0, camber_pos=0.4, aoa=6.0)
        cl_up = m._last_forces["Cl"]
        m.solve(thickness=0.12, camber=0.0, camber_pos=0.4, aoa=-6.0)
        cl_down = m._last_forces["Cl"]
        assert cl_up > 0.05
        assert cl_down < -0.05
        # antisymmetry of the symmetric foil
        assert abs(cl_up + cl_down) < 0.5 * abs(cl_up)

    def test_camber_generates_lift_at_zero_aoa(self):
        m = get_model("naca_flow_2d")(resolution=RES, **COARSE)
        m.solve(thickness=0.12, camber=0.04, camber_pos=0.4, aoa=0.0)
        assert m._last_forces["Cl"] > 0.02

    def test_interior_masked_to_zero(self):
        m = get_model("naca_flow_2d")(resolution={"x": 64, "y": 32}, **COARSE)
        sol = m.solve(thickness=0.16, camber=0.0, camber_pos=0.4, aoa=0.0)
        sdf = m.sdf_input(0.16, 0.0, 0.4, 0.0)
        interior = sdf < -0.02
        assert interior.any()
        assert np.abs(sol[interior]).max() < 1e-10  # fill_value inside foil

    def test_dataset_with_forces_metadata(self):
        d = generate_dataset(
            "naca_flow_2d",
            n_samples=2,
            resolution=RES,
            params=COARSE,
            seed=0,
            verbose=False,
        )
        assert d.inputs.shape == (2, 48, 24)  # SDF channel
        assert d.outputs.shape == (2, 48, 24, 3)
        assert len(d.metadata["Cl"]) == 2
        assert len(d.metadata["param_samples"]["aoa_deg"]) == 2
        assert d.metadata["backend"] == "fenicsx"
        # two different airfoils -> different SDFs
        assert not np.allclose(d.inputs[0], d.inputs[1])
