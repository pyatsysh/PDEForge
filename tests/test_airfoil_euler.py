"""
Transonic airfoil Euler: mesh, solver, and model.

The expensive grid-converged validation against the published AGARD NACA0012
case is gated behind PDEFORGE_SLOW=1 so the default suite stays quick; the
fast tests below still pin the physics (freestream preservation, d'Alembert,
symmetry, a real supersonic pocket).
"""

import os

import numpy as np
import pytest

from pdeforge import generate_dataset, get_model
from pdeforge.geometry import airfoil_c_grid, quad_areas
from pdeforge.solvers.euler_fv import GAMMA, EulerCGrid, to_primitive

SLOW = os.environ.get("PDEFORGE_SLOW") == "1"


def coarse_mesh(**kw):
    opts = dict(n_surf=81, n_wake=25, n_eta=41, first_cell=2e-3, smooth_iters=50)
    opts.update(kw)
    return airfoil_c_grid(**opts)


class TestCGrid:
    def test_topology_and_validity(self):
        X, Y, n_wall, n_wake = coarse_mesh()
        assert X.shape == (2 * n_wake + n_wall, 41)
        assert n_wall == 81
        # a tangled mesh shows up as mixed-sign areas, so this is the check
        A = quad_areas(X, Y)
        assert (A > 0).all() or (A < 0).all()
        assert np.abs(A).min() > 0.0

    def test_wall_spacing_is_honoured_everywhere(self):
        """
        The body-to-far-field distance varies by more than 10x around the C,
        so a single normalised stretch would leave the wall spacing wrong by
        that factor. It must be uniform to the requested value.
        """
        X, Y, _, _ = coarse_mesh(first_cell=1e-3)
        d = np.hypot(X[:, 1] - X[:, 0], Y[:, 1] - Y[:, 0])
        assert np.allclose(d, 1e-3, rtol=0.05)

    def test_airfoil_is_closed_and_on_chord(self):
        X, Y, n_wall, n_wake = coarse_mesh()
        wall_x = X[n_wake : n_wake + n_wall, 0]
        wall_y = Y[n_wake : n_wake + n_wall, 0]
        assert np.isclose(wall_x.min(), 0.0, atol=1e-6)  # leading edge
        assert np.isclose(wall_x.max(), 1.0, atol=1e-6)  # trailing edge
        # the wall runs trailing edge -> nose -> trailing edge (closed)
        assert np.hypot(wall_x[0] - wall_x[-1], wall_y[0] - wall_y[-1]) < 1e-12

    def test_camber_lifts_the_mean_line(self):
        _, Y0, nw, nk = coarse_mesh(camber=0.0)
        _, Y4, _, _ = coarse_mesh(camber=0.04)
        w = slice(nk, nk + nw)
        assert np.isclose(Y0[w, 0].mean(), 0.0, atol=1e-6)
        assert Y4[w, 0].mean() > 0.005


class TestEulerSolver:
    def test_freestream_preservation_and_a_transparent_wake_cut(self):
        """
        The sharpest test of the geometry. A uniform freestream is an exact
        steady solution wherever the body is not felt, so:

        - away from the wall the residual must be at round-off (the geometric
          conservation law: inconsistent normals or volumes would show up here
          as a spurious source long before they moved a force coefficient);
        - on the WAKE CUT it must also be at round-off, which is what proves
          the partner-index mapping across the cut is transparent rather than
          quietly reflecting;
        - on the WALL it must NOT be, because a solid body reflecting the flow
          is physics, not error.
        """
        X, Y, nw, nk = coarse_mesh()
        s = EulerCGrid(X, Y, nw, nk, mach=0.5, aoa_deg=0.0)
        r = np.abs(s.residual()).max(axis=0)

        assert r[:, 1:].max() < 1e-10
        wake = max(r[:nk, 0].max(), r[nk + nw :, 0].max())
        assert wake < 1e-10
        assert r[nk : nk + nw, 0].max() > 1.0

    def test_subsonic_symmetric_has_no_lift_and_little_drag(self):
        """
        NACA0012 at zero incidence: C_l = 0 by symmetry, and C_d = 0 by
        d'Alembert's paradox for subsonic inviscid flow, so whatever drag
        appears is purely the scheme's dissipation.
        """
        X, Y, nw, nk = coarse_mesh(first_cell=2e-3)
        s = EulerCGrid(X, Y, nw, nk, mach=0.5, aoa_deg=0.0, order=2)
        s.solve(iters=3000, tol=1e-5)
        cl, cd = s.force_coefficients()
        assert abs(cl) < 0.02
        assert abs(cd) < 0.05
        assert s.mach_field().max() < 1.0  # still subsonic everywhere

    def test_stagnation_pressure_matches_compressible_value(self):
        """C_p at stagnation is not 1 in compressible flow: it is
        ((1 + (g-1)/2 M^2)^(g/(g-1)) - 1) / (g/2 M^2) = 1.064 at M = 0.5."""
        X, Y, nw, nk = coarse_mesh(first_cell=2e-3)
        s = EulerCGrid(X, Y, nw, nk, mach=0.5, aoa_deg=0.0, order=2)
        s.solve(iters=3000, tol=1e-5)
        _, _, cp = s.surface_cp()
        m2 = 0.5**2
        exact = ((1 + 0.5 * (GAMMA - 1) * m2) ** (GAMMA / (GAMMA - 1)) - 1) / (
            0.5 * GAMMA * m2
        )
        assert abs(cp.max() - exact) < 0.06

    def test_second_order_beats_first_on_spurious_drag(self):
        """MUSCL must actually buy something: d'Alembert drag is pure
        numerical dissipation, so it should fall sharply with the order."""
        X, Y, nw, nk = coarse_mesh(first_cell=2e-3)

        def drag(order):
            s = EulerCGrid(X, Y, nw, nk, mach=0.5, aoa_deg=0.0, order=order)
            s.solve(iters=3000, tol=1e-6)
            return abs(s.force_coefficients()[1])

        assert drag(2) < 0.5 * drag(1)

    def test_transonic_forms_a_supersonic_pocket(self):
        X, Y, nw, nk = coarse_mesh(n_surf=121, n_wake=33, n_eta=49,
                                   first_cell=1e-3)
        s = EulerCGrid(X, Y, nw, nk, mach=0.8, aoa_deg=1.25, order=2)
        s.solve(iters=4000, tol=1e-5)
        M = s.mach_field()
        assert M.max() > 1.15  # genuine supersonic pocket
        assert (M > 1.0).sum() > 20  # a region, not one rogue cell
        cl, cd = s.force_coefficients()
        assert 0.20 < cl < 0.50  # coarse grid, but the right neighbourhood
        assert 0.005 < cd < 0.06  # wave drag is real and positive

    def test_lift_increases_with_incidence(self):
        X, Y, nw, nk = coarse_mesh(first_cell=2e-3)
        lifts = []
        for aoa in (0.0, 2.0):
            s = EulerCGrid(X, Y, nw, nk, mach=0.6, aoa_deg=aoa, order=2)
            s.solve(iters=2500, tol=1e-5)
            lifts.append(s.force_coefficients()[0])
        assert lifts[1] > lifts[0] + 0.1


class TestAirfoilEulerModel:
    def test_registered_and_shapes(self):
        d = generate_dataset(
            "airfoil_euler_2d",
            n_samples=2,
            resolution={"xi": 128, "eta": 40},
            seed=0,
            params={"max_iterations": 1200, "residual_tol": 1e-3},
            verbose=False,
        )
        assert d.inputs.shape == (2, 128, 40, 2)  # the deformed mesh
        assert d.outputs.shape == (2, 128, 40, 4)  # rho, u, v, p
        assert np.isfinite(d.outputs).all()
        assert d.outputs[..., 0].min() > 0.0  # positive density
        assert d.outputs[..., 3].min() > 0.0  # positive pressure

    def test_metadata_records_the_sample_identity(self):
        d = generate_dataset(
            "airfoil_euler_2d",
            n_samples=2,
            resolution={"xi": 128, "eta": 40},
            seed=1,
            params={"max_iterations": 1200, "residual_tol": 1e-3},
            verbose=False,
        )
        m = d.metadata
        for k in ("thickness", "camber", "camber_pos", "mach", "aoa_deg"):
            assert len(m["param_samples"][k]) == 2
        for k in ("Cl", "Cd", "residual_drop", "mach_max"):
            assert len(m[k]) == 2
        assert 0.0 <= m["transonic_fraction"] <= 1.0

    def test_input_mesh_wraps_the_airfoil(self):
        """Inputs are the mesh itself, so the wall ring must trace a chord."""
        m = get_model("airfoil_euler_2d")(resolution={"xi": 128, "eta": 40})
        ic = m.generate_ic(seed=3)
        X, Y, n_wall, n_wake = m.build_mesh(*ic[:3])
        wall_x = X[n_wake : n_wake + n_wall, 0]
        assert np.isclose(wall_x.min(), 0.0, atol=1e-6)
        assert np.isclose(wall_x.max(), 1.0, atol=1e-6)

    def test_xi_is_rounded_even(self):
        m = get_model("airfoil_euler_2d")(resolution={"xi": 127, "eta": 40})
        assert m.n_xi == 128
        assert m.n_surf % 2 == 1  # trailing edge lands on a node

    def test_steady_model_refuses_trajectories(self):
        m = get_model("airfoil_euler_2d")(resolution={"xi": 128, "eta": 40})
        assert m.TIME_DEPENDENT is False
        with pytest.raises(ValueError, match="steady"):
            m.generate_dataset(n_samples=1, outputs="trajectory", verbose=False)

    def test_bad_resolution_keys_rejected(self):
        with pytest.raises(ValueError, match="xi"):
            get_model("airfoil_euler_2d")(resolution={"x": 128, "y": 40})


@pytest.mark.skipif(not SLOW, reason="set PDEFORGE_SLOW=1 for the AGARD case")
class TestAGARDValidation:
    """
    The published inviscid NACA0012 benchmark at M = 0.8, alpha = 1.25 deg:
    C_l ~ 0.352, C_d ~ 0.0224, upper-surface shock near x/c = 0.62.

    Measured here (see docs/guide/models.md): the shock station is within 1%
    and holds under refinement, while the forces sit ~10% low in lift and
    7-15% high in drag and improve with the grid. The tolerances below are
    set to what this solver actually achieves, not to what a tuned production
    Euler code would; tightening them is a real piece of work, not a knob.
    """

    def test_agard_naca0012(self):
        # 241x81 is the grid these tolerances were MEASURED on
        # (C_l 0.3213, C_d 0.02404, shock 0.623); ~18 minutes.
        X, Y, nw, nk = airfoil_c_grid(
            thickness=0.12, camber=0.0, n_surf=241, n_wake=65, n_eta=81,
            first_cell=1e-3, smooth_iters=100,
        )
        s = EulerCGrid(X, Y, nw, nk, mach=0.8, aoa_deg=1.25, order=2)
        s.solve(iters=30000, tol=3e-6)
        cl, cd = s.force_coefficients()
        x, y, cp = s.surface_cp()

        assert abs(cl - 0.352) < 0.05
        assert abs(cd - 0.0224) < 0.012

        upper = y > 0
        xs, cs = x[upper], cp[upper]
        o = np.argsort(xs)
        xs, cs = xs[o], cs[o]
        band = (xs > 0.2) & (xs < 0.95)
        slope = np.diff(cs)[band[:-1]] / np.diff(xs)[band[:-1]]
        x_shock = xs[:-1][band[:-1]][np.argmax(slope)]
        assert 0.45 < x_shock < 0.75
