"""
Physics-validation tests for the egg-shell droplet model.

The invariants that matter here are exact ones: the update is a pure flux
divergence, so mass is conserved to round-off whatever the geometry; the free
energy is a Lyapunov functional for any non-negative mobility; and where the
mobility is constant the scheme must collapse onto the plain Cahn-Hilliard
stepper. On top of those sit the two physics checks the model exists for —
that Gibbs-Thomson makes the smaller droplet the one that dissolves, and that
the regime classifier separates the two coarsening mechanisms.

Tests run at 64^3 with a correspondingly coarser interface; the physics is
scale-free in R/eps, so the invariants hold there just as they do at 128^3.
"""

import numpy as np
import pytest

from pdeforge import get_model
from pdeforge.models.eggshell_droplets_3d import (
    COALESCENCE,
    MIXED,
    RIPENING,
    UNRESOLVED,
    classify_trajectory,
)

# A small, self-consistent configuration: eps ~ 1.3 dx at 64^3, droplets that
# fit inside the shell, and a run short enough for CI.
COARSE = dict(
    resolution={"x": 64, "y": 64, "z": 64},
    epsilon=0.020,
    droplet_radius=0.10,
    shell_outer=0.44,
    shell_thickness=0.24,
    time_end=0.02,
    _dt=5e-4,
    _n_time_steps=11,
)


def build(**overrides):
    params = {**COARSE, **overrides}
    resolution = params.pop("resolution")
    return get_model("eggshell_droplets_3d")(resolution=resolution, **params)


class TestGeometryAndIC:
    def test_registered_and_3d_only(self):
        m = build()
        assert m.NDIM == 3
        with pytest.raises(ValueError, match="3D model"):
            get_model("eggshell_droplets_3d")(resolution={"x": 32, "y": 32})

    def test_ic_has_two_channels(self):
        m = build()
        ic = m.generate_ic(seed=0)
        assert ic.shape == (2, 64, 64, 64)
        assert m.INPUT_NAMES == ["u0", "mobility"]

    def test_droplets_sit_on_the_mid_shell_sphere(self):
        m = build()
        c1, c2, R1, R2 = m._droplet_centres()
        assert np.isclose(np.linalg.norm(c1 - m.centre), m.r_mid)
        assert np.isclose(np.linalg.norm(c2 - m.centre), m.r_mid)
        # The requested surface-to-surface gap is realised as a chord.
        chord = np.linalg.norm(c1 - c2)
        assert np.isclose(chord, R1 + R2 + m.gap * m.R_mean)

    def test_asymmetry_sets_the_radii(self):
        m = build(size_asymmetry=0.25)
        _, _, R1, R2 = m._droplet_centres()
        assert np.isclose((R1 - R2) / (R1 + R2), 0.25)

    def test_ic_contains_two_droplets_of_the_right_volume(self):
        m = build(size_asymmetry=0.2)
        ic = m.generate_ic(seed=0)
        d = m._frame_diagnostics(ic[0])
        assert d["n"] == 2
        _, _, R1, R2 = m._droplet_centres()
        expected = [4 / 3 * np.pi * R**3 for R in (R1, R2)]
        # Coarse grid: a few per cent from the voxelised u > 0 volume.
        assert np.allclose(d["volumes"], expected, rtol=0.08)

    def test_mobility_is_a_shell(self):
        m = build()
        ic = m.generate_ic(seed=0)
        M = ic[1]
        r = m.radius
        r_in = m.r_out - m.shell_thickness
        # Active in the shell, inert in the core and outside the pellet.
        mid = np.abs(r - m.r_mid) < 0.02
        core = r < r_in - 3 * m.wall_width
        outside = r > m.r_out + 3 * m.wall_width
        assert M[mid].min() > 0.9 * m.mobility
        assert M[core].max() < 0.01 * m.mobility
        assert M[outside].max() < 0.01 * m.mobility

    def test_droplet_phase_confined_to_the_shell(self):
        """No droplet material is seeded in the inert core or outside."""
        m = build()
        ic = m.generate_ic(seed=0)
        u, M = ic
        frozen = M < 0.01 * m.mobility
        assert np.all(u[frozen] < -0.95)


class TestExactInvariants:
    def test_mass_conserved_to_machine_precision(self):
        """
        The flagship invariant. The right-hand side is assembled in flux form,
        so the k = 0 Fourier mode of every update is identically zero and the
        drift is round-off, not a tolerance.
        """
        m = build()
        ic = m.generate_ic(seed=0)
        uT = m.solve(ic)
        drift = abs(uT.mean() - ic[0].mean())
        assert drift < 1e-14, f"mass drift {drift:.3e}"

    def test_mass_conserved_with_noise_and_roughness(self):
        """
        Conserved noise is also written as a divergence, so it cannot leak.

        The bound is looser than the deterministic one only because summing a
        million-cell mean accumulates round-off at roughly sqrt(N)*eps per
        step, and the noise adds an independent increment every step. It is
        still four orders below anything a real leak would produce.
        """
        m = build(noise_intensity=1e-6, shell_roughness=0.3)
        ic = m.generate_ic(seed=3)
        uT = m.solve(ic, seed=3)
        assert abs(uT.mean() - ic[0].mean()) < 1e-12

    def test_free_energy_decreases_monotonically(self):
        """dF/dt = -integral M |grad mu|^2 <= 0 for any non-negative mobility."""
        m = build()
        ic = m.generate_ic(seed=0)
        traj = m.solve(ic, return_full=True)
        energies = np.array([m.free_energy(u) for u in traj])
        assert np.all(np.diff(energies) <= 1e-12)

    def test_bounded(self):
        m = build()
        ic = m.generate_ic(seed=0)
        uT = m.solve(ic)
        assert np.isfinite(uT).all()
        assert np.abs(uT).max() < 1.2

    def test_reproducible(self):
        a = build().solve(build().generate_ic(seed=7))
        b = build().solve(build().generate_ic(seed=7))
        assert np.array_equal(a, b)


class TestReductionToCahnHilliard:
    def test_constant_mobility_reproduces_the_plain_scheme(self):
        """
        With M constant the flux form telescopes:

            div(M grad mu) = -M k^2 mu_hat

        and the stabilised step becomes exactly the `cahn_hilliard` update,
        numerator and denominator alike. This pins the variable-coefficient
        machinery to a scheme that is already validated, and it is the reason
        the active shell is integrated by the standard stepper rather than an
        approximation to it.

        The identity holds for any field the grid actually resolves. The
        Nyquist mode is excluded by construction — it carries no flux, since
        i*k times a real Nyquist coefficient is not the transform of a real
        field — so the test field is band-limited well below it, as a physical
        phase field with a resolved interface always is.
        """
        m = build()
        rng = np.random.default_rng(0)
        white = rng.standard_normal(m.field_shape)
        w_hat = np.fft.rfftn(white, axes=(0, 1, 2))
        w_hat[m.K2 > 0.02 * m.K2.max()] = 0.0
        smooth = np.fft.irfftn(w_hat, s=m.field_shape, axes=(0, 1, 2))
        u = 0.3 * smooth / smooth.std()
        M = np.full(m.field_shape, m.mobility)

        got = m._step(u, M, None)

        # The reference scheme, written out independently.
        K2 = m.K2
        u_hat = np.fft.rfftn(u, axes=(0, 1, 2))
        u3_hat = np.fft.rfftn(u**3, axes=(0, 1, 2))
        numer = u_hat + m.dt * m.mobility * K2 * ((1.0 + m.stab) * u_hat - u3_hat)
        denom = 1.0 + m.dt * m.mobility * K2 * (m.stab + m.eps**2 * K2)
        want = np.fft.irfftn(numer / denom, s=m.field_shape, axes=(0, 1, 2))

        assert np.allclose(got, want, atol=1e-12, rtol=0)


class TestGibbsThomson:
    def test_smaller_droplet_dissolves_faster(self):
        """
        The Gibbs-Thomson potential at a droplet of radius R is sigma/R, so the
        smaller partner sits at the higher potential and loses material to the
        larger. Its volume must fall faster.
        """
        m = build(size_asymmetry=0.25, droplet_gap=1.5, time_end=0.03)
        ic = m.generate_ic(seed=0)
        m.solve(ic)
        vols = m.last_diagnostics["volumes"]
        alive = vols[:, 1] > 0
        d_large = vols[alive, 0][-1] - vols[0, 0]
        d_small = vols[alive, 1][-1] - vols[0, 1]
        assert d_small < d_large, "the small droplet must lose the most"
        assert d_small < 0, "the small droplet must shrink"

    def test_chemical_potential_obeys_the_sharp_interface_law(self):
        """
        The quantitative form of Gibbs-Thomson, not just its ordering.

        In the sharp-interface limit the chemical potential is constant inside
        a droplet and equal to sigma/R, so mu*R returns the surface tension
        sigma = 2*sqrt(2)*eps/3 for every droplet independently, whatever its
        size. Measured on a well-separated unequal pair this holds to a few per
        cent; the residue is the curvature correction, which is O(eps/R) and so
        largest for the smaller droplet.
        """
        m = build(
            size_asymmetry=0.25,
            droplet_gap=2.5,
            shell_thickness=0.26,
            time_end=0.004,
            _dt=2e-4,
            _n_time_steps=3,
        )
        ic = m.generate_ic(seed=0)
        u = m.solve(ic)

        potentials = m.droplet_potentials(u)
        assert len(potentials) == 2
        for radius, mu in potentials:
            assert mu * radius == pytest.approx(m.surface_tension, rel=0.15)

        # The 1/R law itself: the ratio cancels sigma entirely.
        (r_big, mu_big), (r_small, mu_small) = potentials
        assert mu_small / mu_big == pytest.approx(r_big / r_small, rel=0.10)
        assert mu_small > mu_big, "the smaller droplet sits at the higher potential"

    def test_supersaturated_matrix_preserves_droplets(self):
        """
        A matrix held at u = -1 is undersaturated against any curved interface,
        so both droplets evaporate into it. Starting at the Gibbs-Thomson value
        instead turns the problem into an exchange between the partners: total
        droplet volume must survive far better.
        """
        kw = dict(size_asymmetry=0.0, droplet_gap=2.0, time_end=0.02)
        naive = build(_supersaturation_factor=0.0, **kw)
        ic = naive.generate_ic(seed=0)
        naive.solve(ic)
        v_naive = naive.last_diagnostics["total_droplet_volume"]

        proper = build(**kw)
        ic = proper.generate_ic(seed=0)
        proper.solve(ic)
        v_proper = proper.last_diagnostics["total_droplet_volume"]

        loss_naive = 1 - v_naive[-1] / v_naive[0]
        loss_proper = 1 - v_proper[-1] / v_proper[0]
        assert loss_proper < loss_naive


class TestClassifier:
    """The classifier is a pure function, so it is tested on synthetic input."""

    def _traj(self, n_comp, v_small, merges, sep=None):
        """
        Build a conservation-consistent synthetic trajectory.

        The survivor holds volume 1.0 while the pair lasts. `merges` says
        whether the event was a merger: if it was, the survivor takes on the
        smaller droplet's remaining volume, which is exactly the signal the
        classifier keys on. A "merge" in which the survivor gains nothing is
        not a merge, so the fixtures may not fabricate one.
        """
        n_t = len(n_comp)
        v_small = np.asarray(v_small, dtype=float)
        v_large = np.ones(n_t)
        gone = np.nonzero(np.asarray(n_comp) < 2)[0]
        if gone.size and merges:
            v_large[gone[0] :] = 1.0 + v_small[gone[0] - 1]
        volumes = np.stack([v_large, v_small], axis=1)
        sep = sep if sep is not None else np.full(n_t, 0.5)
        return (np.linspace(0, 1, n_t), np.asarray(n_comp), volumes, np.asarray(sep))

    def test_merge_with_both_alive_is_coalescence(self):
        r = classify_trajectory(
            *self._traj([2, 2, 2, 1, 1], [1.0, 0.98, 0.95, 0, 0], merges=True)
        )
        assert r["regime"] == COALESCENCE
        assert r["survival_fraction"] == pytest.approx(0.95)
        assert r["absorbed_fraction"] == pytest.approx(1.0)

    def test_dissolution_before_contact_is_ripening(self):
        r = classify_trajectory(
            *self._traj([2, 2, 2, 1, 1], [1.0, 0.5, 0.05, 0, 0], merges=False)
        )
        assert r["regime"] == RIPENING
        assert r["survival_fraction"] == pytest.approx(0.05)
        assert r["absorbed_fraction"] == pytest.approx(0.0)

    def test_partial_dissolution_then_merge_is_mixed(self):
        """Merged, but only after the small droplet had largely ripened away."""
        r = classify_trajectory(
            *self._traj([2, 2, 2, 1, 1], [1.0, 0.6, 0.30, 0, 0], merges=True)
        )
        assert r["regime"] == MIXED

    def test_pair_still_intact_is_unresolved(self):
        r = classify_trajectory(
            *self._traj([2, 2, 2, 2], [1.0, 0.9, 0.8, 0.7], merges=False)
        )
        assert r["regime"] == UNRESOLVED
        assert r["survival_fraction"] == pytest.approx(0.7)

    def test_absorption_signal_survives_coarse_frames(self):
        """
        The frame-rate robustness that motivates `absorbed_fraction`. A droplet
        that ripens away vanishes abruptly, so with coarse sampling its last
        recorded volume can still be large. Survival fraction alone would then
        read that as a merger; the absorption signal does not, because the
        survivor never took the material on.
        """
        r = classify_trajectory(*self._traj([2, 2, 1], [1.0, 0.47, 0.0], merges=False))
        assert r["survival_fraction"] == pytest.approx(0.47)
        assert r["regime"] == RIPENING

    def test_remnant_absorption_after_ripening_is_not_a_merger(self):
        """
        The guard on the absorbed fraction. Once the small droplet is down to a
        per cent of its original volume the quantity has no denominator left to
        speak of: the survivor's ordinary ripening growth over a single frame
        is comparable to everything that remains, so absorption reads near 1
        for reasons that have nothing to do with a merger. A pair that ripened
        away to a remnant ripened, whatever swallowed the last of it.
        """
        r = classify_trajectory(*self._traj([2, 2, 1], [1.0, 0.011, 0.0], merges=True))
        assert r["absorbed_fraction"] > 0.5
        assert r["regime"] == RIPENING

    def test_regime_read_before_the_event_not_at_it(self):
        """
        Reading the survival fraction at the event frame would give 0 for every
        run, whichever mechanism ended the pair. It must be read one frame back.
        """
        r = classify_trajectory(*self._traj([2, 2, 1], [1.0, 1.0, 0.0], merges=True))
        assert r["survival_fraction"] == pytest.approx(1.0)
        assert r["regime"] == COALESCENCE

    def test_centroid_approach_is_an_independent_signal(self):
        """Coalescing droplets move together; ripening ones stay put."""
        merging = classify_trajectory(
            *self._traj([2, 2, 1], [1.0, 0.9, 0.0], merges=True, sep=[0.5, 0.25, 0.25])
        )
        ripening = classify_trajectory(
            *self._traj([2, 2, 1], [1.0, 0.05, 0.0], merges=False, sep=[0.5, 0.5, 0.5])
        )
        assert merging["centroid_approach"] > 0.4
        assert ripening["centroid_approach"] == pytest.approx(0.0)


class TestRegimesAreReachable:
    """
    The point of the model: both mechanisms must actually occur, and the knobs
    must select between them. These are the slowest tests in the file.
    """

    def test_touching_pair_with_no_ripening_drive_coalesces(self):
        """
        Zero asymmetry removes the ripening drive entirely, so a pair whose
        diffuse interfaces already overlap can only merge — and must merge with
        both partners still essentially intact.
        """
        m = build(size_asymmetry=0.0, droplet_gap=0.1, time_end=0.01)
        ic = m.generate_ic(seed=0)
        m.solve(ic)
        d = m.last_diagnostics
        assert d["regime"] == COALESCENCE
        assert d["survival_fraction"] > 0.8

    def test_well_separated_asymmetric_pair_ripens(self):
        m = build(size_asymmetry=0.30, droplet_gap=1.5, time_end=0.14)
        ic = m.generate_ic(seed=0)
        m.solve(ic)
        d = m.last_diagnostics
        assert d["regime"] == RIPENING
        assert d["centroid_approach"] < 0.15, "ripening droplets do not migrate"

    def test_validate_solution_reports_the_regime(self):
        m = build(size_asymmetry=0.0, droplet_gap=0.1, time_end=0.01)
        ic = m.generate_ic(seed=0)
        uT = m.solve(ic)
        v = m.validate_solution(ic, uT)
        assert v["valid"]
        assert v["regime"] == COALESCENCE


class TestDatasetIntegration:
    """The two-channel input has to survive the dataset path, not just solve()."""

    FAST = dict(
        epsilon=0.020,
        droplet_radius=0.10,
        shell_outer=0.44,
        shell_thickness=0.24,
        time_end=0.002,
        _dt=5e-4,
        _n_time_steps=3,
    )

    def test_both_channels_reach_the_dataset(self):
        import pdeforge

        d = pdeforge.generate_dataset(
            "eggshell_droplets_3d",
            n_samples=2,
            resolution={"x": 64, "y": 64, "z": 64},
            params=dict(shell_roughness=0.35, **self.FAST),
            seed=11,
            verbose=False,
        )
        assert d.input_names == ["u0", "mobility"]
        assert d.inputs.shape == (2, 2, 64, 64, 64)
        assert d.outputs.shape == (2, 64, 64, 64)
        # Mass is conserved sample by sample, through the dataset path too.
        for i in range(2):
            assert abs(d.outputs[i].mean() - d.inputs[i, 0].mean()) < 1e-14
        # With roughness on, the shell really is a per-sample input rather than
        # a constant the surrogate could absorb into its weights.
        assert not np.allclose(d.inputs[0, 1], d.inputs[1, 1])

    def test_trajectory_output_carries_a_time_grid(self):
        import pdeforge

        d = pdeforge.generate_dataset(
            "eggshell_droplets_3d",
            n_samples=1,
            resolution={"x": 64, "y": 64, "z": 64},
            params=self.FAST,
            seed=1,
            verbose=False,
            outputs="trajectory",
        )
        assert d.outputs.shape == (1, 3, 64, 64, 64)
        assert "t" in d.grid
