"""
Two droplets coarsening inside an egg-shell pellet (3D Cahn-Hilliard).

Coarsening of a dispersed phase proceeds by two competing mechanisms, and
which one wins is the classical question in supported-catalyst sintering:

  * **Ostwald ripening** — the droplets stay put and exchange material by
    diffusion through the matrix. The Gibbs-Thomson effect makes the chemical
    potential at a droplet of radius R equal to sigma/R, so the smaller droplet
    sits at the higher potential, dissolves, and feeds the larger one.
  * **Coalescence** — the droplets touch, the interface between them bridges,
    and the pair merges with both partners still substantially intact.

Both mechanisms are already contained in the Cahn-Hilliard equation; nothing
extra has to be added. This model isolates the competition in its minimal
setting — exactly two droplets — and confines them to the thin active shell of
an egg-shell catalyst pellet:

    du/dt = div( M(x) grad mu ),    mu = u^3 - u - eps^2 laplacian(u)

with the mobility carrying the geometry,

    M(x) = M0 * [ phi0 + (1 - phi0) * psi(x) ],

where psi is a smoothed indicator of the spherical shell r_in(n) <= |x| <=
r_out and phi0 << 1 is the residual mobility of the inert support core. Where
M is negligible there is no flux, so the core and the pellet exterior are
frozen at u = -1 and all transport happens in the shell. That is the
egg-shell geometry: an inert, impermeable core with the active phase in a thin
outer layer.

Two properties of this formulation matter:

  * The right-hand side is a pure divergence, so the k = 0 Fourier mode of the
    update vanishes identically and the total mass is conserved to machine
    precision — no diffuse-domain weighting needed.
  * Inside the shell M equals its constant maximum, so the stabilised IMEX
    step reduces *exactly* to the constant-coefficient scheme used by
    `cahn_hilliard` — for constant M the flux form telescopes to -M k^2 mu_hat
    and both the numerator and the denominator become the standard ones. The
    variable coefficient is only ever felt where nothing is happening, so the
    active shell is integrated by the already-validated stepper rather than by
    an approximation to it.

The shell is a closed domain, so unlike a periodic box there are no
periodic-image artefacts in the diffusion field: the two droplets see each
other around the sphere both ways and the total available volume is finite
and known.

Regimes
-------
`solve` classifies every run automatically. Connected components of {u > 0}
are tracked frame by frame, and the run is labelled at the moment the count
drops from two to one by the *survival fraction* of the smaller droplet,

    rho = V_small(t_event) / V_small(0),

which is a continuous regime coordinate rather than a bare label:

    rho -> 1   both droplets alive when they merge      -> "coalescence"
    rho -> 0   the small one is gone before any contact -> "ripening"

Which mechanism ended the pair is decided geometrically, by asking whether the
droplets were ever in contact: merging requires it, dissolving at a distance
forbids it. The survival fraction then grades the merger. Note that asking
instead where the material went does not separate the two — in ripening the
surviving droplet collects it as well, just gradually and through the matrix.
The diagnostics from the last solve are on `self.last_diagnostics`.

Operator Learning Task:
    (u(x, 0), M(x))  ->  u(x, T)

The shell geometry travels with the sample as a second input channel, so a
surrogate has to learn the effect of the confining geometry and not just the
initial droplet configuration.
"""

from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
from scipy import ndimage

from pdeforge.core.base import PDEModel, _legacy_seed, _seed_sequence
from pdeforge.core.params import ParamSpec, ParamType
from pdeforge.core.registry import register_model

# Regime labels.
COALESCENCE = "coalescence"
RIPENING = "ripening"
MIXED = "mixed"
UNRESOLVED = "unresolved"

# A pair that merges with the small droplet still above `RHO_COALESCENCE` of
# its initial volume has genuinely coalesced; below it, the "merge" is an
# already-dissolving remnant being mopped up, which is a mixture of the two
# mechanisms rather than either one.
RHO_COALESCENCE = 0.50

# Surface-to-surface gap, in units of eps, below which the two droplets count
# as having been in contact when the pair ended. The scale is set by the
# interface thickness, 2*sqrt(2)*eps = 2.83 eps, but the threshold sits above
# it because the radii come from volumes as (3V/4pi)^(1/3): a pair on the point
# of merging has necked towards each other, so the equivalent-sphere radius
# understates their reach and the measured gap reads high. Calibrated on the
# 128^3 sweep, where the two populations are cleanly separated with a wide void
# between them — merged pairs span 1.3 to 3.8 eps, ripened pairs 5.7 to 9.7.
CONTACT_THRESHOLD = 4.75


def classify_trajectory(
    times: np.ndarray,
    n_components: np.ndarray,
    volumes: np.ndarray,
    separations: np.ndarray,
    epsilon: float,
    rho_coalescence: float = RHO_COALESCENCE,
    contact_threshold: float = CONTACT_THRESHOLD,
) -> Dict:
    """
    Label a two-droplet trajectory as coalescence, ripening, or neither.

    Parameters
    ----------
    times : (n_t,) frame times.
    n_components : (n_t,) number of droplets resolved in each frame.
    volumes : (n_t, 2) per-frame [larger, smaller] droplet volume. The smaller
        entry is 0.0 once only one droplet remains.
    separations : (n_t,) centroid separation, NaN when fewer than two droplets.

    Returns
    -------
    dict with `regime`, the continuous `survival_fraction` rho, the
    `absorbed_fraction`, the event time and index, and the centroid approach.

    Notes
    -----
    The decision is geometric: **did the two droplets ever touch?**
    `contact_gap` is the surface-to-surface separation at the last frame where
    both still exist, in units of eps, taking each radius from its measured
    volume. Merging requires contact by definition, and dissolving at a
    distance forbids it, so the question separates the mechanisms exactly.

    It is also the only signal that survives coarse output sampling. Asking
    instead where the material *went* does not work: in Ostwald ripening the
    surviving droplet ends up with it too, which is what ripening is — the
    difference is that it arrives gradually by diffusion rather than all at
    once. So the absorbed fraction is reported, because it is informative, but
    it does not decide anything.

    `survival_fraction` then grades a merger: a pair that touched with the
    smaller partner already mostly gone is a mixture of the two mechanisms
    rather than a clean coalescence.

    Both quantities are read at the frame *before* the component count drops,
    the last frame at which both droplets still exist. Reading rho at the event
    frame itself would always give 0 — the small droplet no longer exists
    there, whichever mechanism removed it.
    """
    times = np.asarray(times)
    n_components = np.asarray(n_components)
    volumes = np.asarray(volumes)
    separations = np.asarray(separations)

    v_small_0 = float(volumes[0, 1])
    if v_small_0 <= 0.0:
        return {
            "regime": UNRESOLVED,
            "survival_fraction": float("nan"),
            "absorbed_fraction": float("nan"),
            "contact_gap": float("nan"),
            "event_time": float("nan"),
            "event_index": -1,
            "centroid_approach": float("nan"),
            "reason": "initial condition did not contain two droplets",
        }

    # First frame at which the pair has stopped being a pair.
    dropped = np.nonzero(n_components[1:] < 2)[0]
    if dropped.size == 0:
        # Still two droplets at T: report how far the ripening got.
        return {
            "regime": UNRESOLVED,
            "survival_fraction": float(volumes[-1, 1] / v_small_0),
            "absorbed_fraction": float("nan"),
            "contact_gap": float("nan"),
            "event_time": float("nan"),
            "event_index": -1,
            "centroid_approach": float(
                (separations[0] - separations[-1]) / separations[0]
            ),
            "reason": "two droplets still present at time_end",
        }

    event = int(dropped[0]) + 1
    last_pair = event - 1
    rho = float(volumes[last_pair, 1] / v_small_0)

    approach = float((separations[0] - separations[last_pair]) / separations[0])

    # Where did the small droplet's material go? Across the event the survivor
    # either takes it on essentially whole, which is what merging means, or it
    # does not, which means the matrix took it. This is a conservation
    # statement across a single event rather than a rate, so unlike rho it does
    # not degrade when the last of the small droplet disappears between two
    # output frames — the discriminating quantity has to survive coarse frame
    # spacing, because the end of a dissolving droplet is always abrupt.
    v_small_last = volumes[last_pair, 1]
    if v_small_last > 0:
        absorbed = float((volumes[event, 0] - volumes[last_pair, 0]) / v_small_last)
    else:
        absorbed = 0.0

    # Surface-to-surface gap when the pair last existed, in units of eps, with
    # each radius taken from its measured volume as (3V/4pi)^(1/3).
    radii = np.cbrt(3.0 * volumes[last_pair] / (4.0 * np.pi))
    contact_gap = float((separations[last_pair] - radii.sum()) / epsilon)

    # Two independent readings, required to agree. Contact is geometric: did
    # the droplets ever touch, which merging requires and dissolving at a
    # distance forbids. Survival is material: was there still a droplet there
    # to merge. On the sweep both separate the runs with a wide margin and they
    # never disagree, so a disagreement means the run sits on the boundary
    # between the mechanisms and is reported as such rather than forced into
    # one of them.
    touched = contact_gap <= contact_threshold
    intact = rho >= rho_coalescence
    if touched and intact:
        regime = COALESCENCE
    elif not touched and not intact:
        regime = RIPENING
    else:
        regime = MIXED

    return {
        "regime": regime,
        "survival_fraction": rho,
        "absorbed_fraction": absorbed,
        "contact_gap": contact_gap,
        "event_time": float(times[event]),
        "event_index": event,
        "centroid_approach": approach,
        "reason": "",
    }


@register_model("eggshell_droplets_3d")
class EggshellDroplets3D(PDEModel):
    """
    Coalescence versus Ostwald ripening for two droplets in a 3D shell.

    A spherical pellet of outer radius `shell_outer` has an inert core; only
    the outer layer of thickness `shell_thickness` conducts. Two droplets of
    mean radius `droplet_radius` are seeded on the mid-shell sphere with a
    controlled size asymmetry and separation, and the conserved Cahn-Hilliard
    dynamics decides which coarsening mechanism carries the pair.

    `droplet_gap` selects the regime, and it does so almost alone. Bridging is
    an interface-relaxation process: once the diffuse interfaces overlap it
    completes on a timescale set by eps and the mobility, far faster than
    anything diffusive. Ripening is diffusive, taking t ~ 2 R^3 / (M sigma).
    The two are separated by orders of magnitude, so wherever bridging is
    possible at all it wins outright, and wherever it is not, ripening is the
    only channel left. Measured at 128^3, the boundary sits between 0.25 and
    0.40 mean radii at every asymmetry from 0 to 0.35.

    `size_asymmetry` sets the ripening drive, sigma*(1/R2 - 1/R1), and so how
    fast that branch runs and how far the survival fraction falls — but it
    moves the boundary itself only in the narrow window where the two
    timescales are comparable. At zero asymmetry there is no ripening drive at
    all, which makes that line an unstable equilibrium rather than a regime.

    `shell_thickness` favours ripening as it tightens, which is the opposite of
    what the confinement argument first suggests. Squeezing the droplets into
    lenses raises the curvature at their rims, and so the Gibbs-Thomson
    potential driving the exchange, and it turns the diffusion field between
    them quasi-two-dimensional, which decays more slowly with distance than the
    three-dimensional 1/r field. Both accelerate ripening: measured at gap
    0.31, the survival fraction falls from 0.922 to 0.853 as the shell closes
    from 0.28 to 0.10. Like asymmetry, the effect is on the rate rather than on
    the branch.

    Examples
    --------
    >>> model = EggshellDroplets3D(
    ...     resolution={"x": 128, "y": 128, "z": 128},
    ...     size_asymmetry=0.25, droplet_gap=2.0,
    ... )
    >>> ic = model.generate_ic(seed=0)
    >>> u_T = model.solve(ic)
    >>> model.last_diagnostics["regime"]
    'ripening'

    >>> # Close pair, no ripening drive -> the other regime.
    >>> model = EggshellDroplets3D(
    ...     resolution={"x": 128, "y": 128, "z": 128},
    ...     size_asymmetry=0.0, droplet_gap=0.3,
    ... )
    """

    NDIM = 3
    INPUT_NAMES = ["u0", "mobility"]
    OUTPUT_NAMES = ["u_T"]
    BACKENDS = {"numpy"}

    USER_PARAMS = [
        ParamSpec(
            name="epsilon",
            description="Interface width parameter",
            default=0.010,
            param_type=ParamType.PHYSICAL,
            bounds=(0.006, 0.03),
            affects=(
                "Interface thickness is 2*sqrt(2)*eps and surface tension is "
                "sigma = 2*sqrt(2)*eps/3. Needs at least ~4 grid points across "
                "the interface: eps >= 1.5*dx. Smaller eps also means a longer "
                "ripening time, since t ~ R^3/(M*sigma)."
            ),
        ),
        ParamSpec(
            name="mobility",
            description="Cahn-Hilliard mobility M0 inside the active shell",
            default=1.0,
            param_type=ParamType.PHYSICAL,
            bounds=(0.01, 10.0),
            affects="Sets the overall rate; both mechanisms scale with it.",
        ),
        ParamSpec(
            name="shell_outer",
            description="Outer radius of the pellet (box units)",
            default=0.44,
            param_type=ParamType.GEOMETRY,
            bounds=(0.15, 0.48),
            affects=(
                "The pellet must fit inside the unit box with a margin, so "
                "values above ~0.45 leave too little frozen exterior."
            ),
        ),
        ParamSpec(
            name="shell_thickness",
            description="Thickness of the active shell (box units)",
            default=0.20,
            param_type=ParamType.GEOMETRY,
            bounds=(0.02, 1.0),
            affects=(
                "The confinement knob. Below one droplet diameter the droplets "
                "flatten into lenses and the diffusion field goes quasi-2D. "
                "Set it larger than shell_outer to fill the pellet solid, which "
                "recovers unconfined Cahn-Hilliard in a sphere."
            ),
        ),
        ParamSpec(
            name="shell_roughness",
            description="Relative amplitude of random shell-thickness variation",
            default=0.0,
            param_type=ParamType.INPUT,
            bounds=(0.0, 0.6),
            affects=(
                "0 gives a perfect, sample-independent shell. Non-zero makes "
                "the shell thickness a random field, so the confining geometry "
                "differs per sample and the mobility channel becomes a genuine "
                "input — the realistic case for an impregnated pellet."
            ),
        ),
        ParamSpec(
            name="shell_corrugation",
            description="Amplitude of the egg-carton corrugation of the shell",
            default=0.0,
            param_type=ParamType.GEOMETRY,
            bounds=(0.0, 0.8),
            affects=(
                "0 leaves a smooth spherical annulus. Non-zero corrugates the "
                "shell into an egg-carton: the layer thickens into a lattice of "
                "dimples separated by thin necks. Droplets sit preferentially "
                "in the dimples, and the thin necks between them throttle the "
                "diffusive path, so corrugation can hold a pair apart that "
                "would otherwise merge."
            ),
        ),
        ParamSpec(
            name="corrugation_modes",
            description="Number of egg-carton bumps around a great circle",
            default=6.0,
            param_type=ParamType.GEOMETRY,
            bounds=(2.0, 14.0),
            affects=(
                "Sets the dimple size, roughly pi*r_mid/modes across. Needs to "
                "stay well above the droplet diameter to be a landscape rather "
                "than surface texture, and well resolved: the pattern must not "
                "approach the grid scale."
            ),
        ),
        ParamSpec(
            name="droplet_radius",
            description="Mean radius of the two droplets (box units)",
            default=0.09,
            param_type=ParamType.INPUT,
            bounds=(0.02, 0.14),
            affects=(
                "Quantitative Gibbs-Thomson needs R/eps >= 5. Larger droplets "
                "ripen more slowly, as t ~ R^3."
            ),
        ),
        ParamSpec(
            name="size_asymmetry",
            description="Radius asymmetry (R1 - R2) / (R1 + R2)",
            default=0.15,
            param_type=ParamType.INPUT,
            bounds=(0.0, 0.5),
            affects=(
                "The Ostwald ripening drive, sigma*(1/R2 - 1/R1). Zero means "
                "identical partners and no drive at all — an unstable "
                "equilibrium, not a regime; larger values dissolve the small "
                "droplet faster and pull the coalescence boundary to smaller "
                "gaps."
            ),
        ),
        ParamSpec(
            name="droplet_gap",
            description="Surface-to-surface gap between droplets, in mean radii",
            default=1.0,
            param_type=ParamType.INPUT,
            bounds=(0.05, 6.0),
            affects=(
                "The coalescence drive. Deterministic bridging needs the diffuse "
                "interfaces to overlap, roughly gap*R <~ 5*eps; beyond that "
                "ripening wins."
            ),
        ),
        ParamSpec(
            name="noise_intensity",
            description="Conserved thermal noise strength (0 = deterministic)",
            default=0.0,
            param_type=ParamType.PHYSICAL,
            bounds=(0.0, 1e-5),
            affects=(
                "Adds div(sqrt(2*M*Theta) * eta) — mass-conserving by "
                "construction. Randomises which droplet wins near zero "
                "asymmetry and turns a sharp regime boundary into a "
                "probability, which is what makes the pair a UQ target. "
                "Theta is an energy, so it must sit far below the droplet free "
                "energy (order 1e-2 in these units): 1e-7 to 1e-6 roughens the "
                "interfaces, 1e-4 destroys them."
            ),
        ),
        ParamSpec(
            name="time_end",
            description="Final simulation time",
            default=0.3,
            param_type=ParamType.PHYSICAL,
            bounds=(0.001, 5.0),
            units="s",
            affects=(
                "Must outlast the event being classified: over the swept "
                "parameter range every run resolved by t = 0.23, and the "
                "default clears that. Running much further does not change the "
                "label, which is read at the event, but it does let the shell "
                "matrix slowly consume the surviving droplet, so the returned "
                "field drifts away from the configuration of interest."
            ),
        ),
    ]

    DEFAULT_PARAMS = {
        "epsilon": 0.010,
        "mobility": 1.0,
        "shell_outer": 0.44,
        "shell_thickness": 0.20,
        "shell_roughness": 0.0,
        "shell_corrugation": 0.0,
        "corrugation_modes": 6.0,
        "droplet_radius": 0.09,
        "size_asymmetry": 0.15,
        "droplet_gap": 1.0,
        "noise_intensity": 0.0,
        "time_end": 0.3,
        "_dt": 5.0e-4,
        "_n_time_steps": 61,
        "_stabilization": 2.0,
        # Residual mobility of the inert core, relative to the shell. Small
        # enough that the core does not conduct, large enough that the frozen
        # region stays numerically damped (see _step).
        "_core_mobility": 2.0e-3,
        # Roughness band limit: shell-thickness variation is band-limited to
        # wavelengths above this many shell thicknesses, so it is essentially
        # constant across the shell and depends on direction alone.
        "_roughness_scale": 4.0,
        # Multiplier on the Gibbs-Thomson supersaturation the shell matrix
        # starts at. 1.0 puts the critical radius at the mean droplet radius;
        # 0.0 recovers the naive u = -1 matrix, against which both droplets
        # dissolve rather than exchanging (kept for the validation test).
        "_supersaturation_factor": 1.0,
    }

    def __init__(
        self,
        resolution: Dict[str, int],
        domain: Dict[str, Tuple[float, float]] = None,
        **params,
    ):
        super().__init__(resolution, domain, **params)

        p = self.params
        self.eps = p["epsilon"]
        self.mobility = p["mobility"]
        self.r_out = p["shell_outer"]
        self.shell_thickness = p["shell_thickness"]
        self.roughness = p["shell_roughness"]
        self.corrugation = p["shell_corrugation"]
        self.corrugation_modes = p["corrugation_modes"]
        self.R_mean = p["droplet_radius"]
        self.asymmetry = p["size_asymmetry"]
        self.gap = p["droplet_gap"]
        self.theta_noise = p["noise_intensity"]
        self.T = p["time_end"]
        self.dt = p["_dt"]
        self.n_t = p["_n_time_steps"]
        self.stab = p["_stabilization"]
        self.core_mobility = p["_core_mobility"]
        self.roughness_scale = p["_roughness_scale"]
        self.supersaturation_factor = p["_supersaturation_factor"]

        # Surface tension of the double-well, sigma = 2*sqrt(2)*eps/3, and the
        # Gibbs-Thomson supersaturation of a matrix in local equilibrium with a
        # droplet of the mean radius: linearising mu = f'(u) about u = -1 gives
        # mu = 2*(u+1), and mu at a sphere of radius R is sigma/R.
        self.surface_tension = 2.0 * np.sqrt(2.0) / 3.0 * self.eps
        self.matrix_supersaturation = (
            self.supersaturation_factor * self.surface_tension / (2.0 * self.R_mean)
        )

        if set(resolution) != {"x", "y", "z"}:
            raise ValueError(
                "eggshell_droplets_3d is a 3D model; resolution needs x, y and z."
            )

        # Array axes in reverse-sorted order: (nz, ny, nx), matching the other
        # spectral models.
        self.dim_order = sorted(resolution.keys())[::-1]
        self.field_shape = tuple(resolution[d] for d in self.dim_order)

        # Physical coordinate grids, and the cell volume for integrals.
        coords = [self.grids[d] for d in self.dim_order]
        self.spacing = tuple(float(c[1] - c[0]) for c in coords)
        self.cell_volume = float(np.prod(self.spacing))
        mesh = np.meshgrid(*coords, indexing="ij")
        self.X = mesh  # (Z, Y, X) in array-axis order

        # Radius from the box centre.
        centre = [
            0.5 * (self.domain.bounds[d][0] + self.domain.bounds[d][1])
            for d in self.dim_order
        ]
        self.centre = np.array(centre)
        self.radius = np.sqrt(sum((m - c) ** 2 for m, c in zip(mesh, centre)))

        # Real-FFT wavenumbers. rfftn halves the last axis, so that one gets
        # rfftfreq and the leading axes get fftfreq.
        ks = []
        for axis, d in enumerate(self.dim_order):
            n = resolution[d]
            dx = self.spacing[axis]
            if axis == len(self.dim_order) - 1:
                ks.append(2 * np.pi * np.fft.rfftfreq(n, d=dx))
            else:
                ks.append(2 * np.pi * np.fft.fftfreq(n, d=dx))
        self.K = np.meshgrid(*ks, indexing="ij")
        self.K2 = sum(Ki**2 for Ki in self.K)
        self.hat_shape = self.K2.shape

        # First-derivative multipliers, with the Nyquist mode removed.
        #
        # On an even grid the Nyquist coefficient of a real field is real, and
        # i*k times it is purely imaginary — not the transform of any real
        # field. `irfftn` silently drops that imaginary part, so a gradient
        # followed by a divergence does not round-trip there and the flux picks
        # up a large spurious Nyquist component. Zeroing the derivative at
        # Nyquist is the standard remedy: the least-resolved mode simply
        # carries no flux, and the implicit eps^2 k^4 term damps it anyway.
        # The k = 0 entry stays zero, so mass conservation is untouched.
        self.iK = []
        for axis, Ki in enumerate(self.K):
            mult = 1j * Ki
            nyquist = np.isclose(np.abs(Ki), np.pi / self.spacing[axis])
            mult = np.where(nyquist, 0.0, mult)
            self.iK.append(mult)

        # Implicit denominator of the stabilised IMEX step, at the constant
        # maximum mobility. Identical in form to `cahn_hilliard`.
        self.M_bar = self.mobility
        self._denom = 1.0 + self.dt * self.M_bar * self.K2 * (
            self.stab + self.eps**2 * self.K2
        )

        # Interface half-width; also the width of the mobility transition, so
        # the geometry is resolved on the same scale as the phase interfaces.
        self.wall_width = np.sqrt(2.0) * self.eps

        # Mid-shell radius: where droplets are seeded.
        self.r_mid = self.r_out - 0.5 * min(self.shell_thickness, self.r_out)

        self.last_diagnostics: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _thickness_field(self, rng: np.random.Generator) -> np.ndarray:
        """
        Shell thickness as a function of position.

        A perfect shell is homogeneous by symmetry and so exerts no force on a
        droplet; a real impregnated pellet has a thickness set by a stochastic
        process. `shell_roughness` turns the thickness into a band-limited
        random field. The band limit keeps the variation on scales well above
        the shell thickness, so across the shell it depends on direction only
        and evaluating it at x rather than on the mid-sphere costs nothing.
        """
        delta = self.shell_thickness
        modulation = self.corrugation * self._corrugation()
        if self.roughness <= 0.0:
            return delta * np.clip(1.0 + modulation, 0.15, None)

        # Band-limited Gaussian field: white noise low-passed at the roughness
        # scale, normalised to unit variance.
        white = rng.standard_normal(self.field_shape)
        k_cut = 2 * np.pi / (self.roughness_scale * delta)
        filt = np.exp(-0.5 * (self.K2 / k_cut**2) ** 2)
        field = np.fft.irfftn(
            np.fft.rfftn(white, axes=(0, 1, 2)) * filt,
            s=self.field_shape,
            axes=(0, 1, 2),
        )
        std = field.std()
        if std > 0:
            field = field / std
        # Keep the thickness positive whatever the draw.
        return delta * np.clip(1.0 + modulation + self.roughness * field, 0.15, None)

    def _corrugation(self) -> np.ndarray:
        """
        Egg-carton modulation of the shell, as a function of direction alone.

        The pattern is the Cartesian egg-crate cos(k x) cos(k y) cos(k z)
        evaluated on the mid-shell sphere rather than at the point itself, so
        it varies with direction but not with depth through the layer. Taking
        it on the sphere also avoids the pole singularity that a cos(m theta)
        cos(m phi) form would carry, and needs no spherical harmonics: the
        function is smooth everywhere on the sphere by construction.

        `corrugation_modes` counts bumps around a great circle, so the
        wavenumber is k = modes / r_mid and the argument reduces to modes
        times the direction cosine.
        """
        if self.corrugation <= 0.0:
            return np.zeros(self.field_shape)

        r = np.maximum(self.radius, 1e-12)
        n = self.corrugation_modes
        pattern = np.ones(self.field_shape)
        for axis in range(3):
            pattern = pattern * np.cos(n * (self.X[axis] - self.centre[axis]) / r)
        return pattern

    def _mobility_field(self, rng: np.random.Generator) -> np.ndarray:
        """
        M(x) for the egg-shell pellet: active layer, inert core, frozen outside.

        psi is the product of two smoothed step functions, giving 1 in the
        shell and 0 elsewhere with a transition of width `wall_width`. The
        mobility floor keeps the frozen regions numerically well behaved
        without letting them conduct.
        """
        delta = self._thickness_field(rng)
        r_in = self.r_out - delta

        w = self.wall_width
        inner = 0.5 * (1.0 + np.tanh((self.radius - r_in) / w))
        outer = 0.5 * (1.0 + np.tanh((self.r_out - self.radius) / w))
        psi = inner * outer

        phi0 = self.core_mobility
        return self.mobility * (phi0 + (1.0 - phi0) * psi)

    def _droplet_centres(self) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Place two droplets on the mid-shell sphere at the requested gap.

        Radii follow from the mean radius and the asymmetry; the centres sit
        symmetrically about the polar axis in the x-z plane, at the angular
        separation that realises the requested surface-to-surface gap.
        """
        a = self.asymmetry
        R1 = self.R_mean * (1.0 + a)
        R2 = self.R_mean * (1.0 - a)

        chord = R1 + R2 + self.gap * self.R_mean
        # Cannot exceed the sphere's diameter: clamp to antipodal.
        sin_half = min(chord / (2.0 * self.r_mid), 1.0)
        half_angle = np.arcsin(sin_half)

        # Array-axis order is (z, y, x); build centres in that order.
        offset_x = self.r_mid * np.sin(half_angle)
        offset_z = self.r_mid * np.cos(half_angle)
        c1 = self.centre + np.array([offset_z, 0.0, +offset_x])
        c2 = self.centre + np.array([offset_z, 0.0, -offset_x])
        return c1, c2, R1, R2

    # ------------------------------------------------------------------
    # Initial condition
    # ------------------------------------------------------------------

    def generate_ic(
        self,
        generator: Union[str, Callable] = "default",
        generator_params: Dict = None,
        seed: int = None,
    ) -> np.ndarray:
        """
        Seed two tanh droplets inside the shell.

        Returns the two input channels stacked on a leading axis: the phase
        field u0 and the mobility field M(x) that encodes the pellet geometry,
        shape (2, nz, ny, nx). The mobility travels with the sample because
        with `shell_roughness` on it differs from sample to sample, and a
        surrogate cannot predict the dynamics without it.
        """
        if generator_params is None:
            generator_params = {}
        rng = np.random.default_rng(seed)

        M = self._mobility_field(rng)
        c1, c2, R1, R2 = self._droplet_centres()

        # tanh profiles; take the maximum so overlapping droplets merge
        # smoothly rather than summing to u > 1.
        width = np.sqrt(2.0) * self.eps
        phi = np.zeros(self.field_shape)
        for c, R in ((c1, R1), (c2, R2)):
            d = np.sqrt(sum((m - ci) ** 2 for m, ci in zip(self.X, c)))
            phi = np.maximum(phi, 0.5 * (1.0 - np.tanh((d - R) / width)))

        # Confine the droplet phase to the active shell: outside it the field
        # starts in the matrix phase and, having no mobility, stays there.
        psi = np.clip(
            (M / self.mobility - self.core_mobility) / (1.0 - self.core_mobility),
            0.0,
            1.0,
        )
        phi = phi * psi

        # The shell matrix starts at the Gibbs-Thomson supersaturation, not at
        # the flat-interface value u = -1. A droplet of radius R is only in
        # local equilibrium with a matrix at -1 + sigma/(2R); against a matrix
        # at exactly -1 both droplets simply evaporate, and with a shell
        # reservoir far larger than the droplets they evaporate completely.
        # Setting the critical radius to the mean radius is what makes this an
        # exchange problem — the larger droplet grows, the smaller dissolves —
        # rather than a dissolution problem.
        # The supersaturation is applied uniformly, NOT weighted by psi. That
        # matters more than it looks. Seeding the shell at -1 + du0 while the
        # inert core and exterior sit at -1 puts a chemical potential step
        # across the shell wall, and the frozen region is around 60% of the box
        # — an enormous sink whose mobility is small but not zero. It slowly
        # drains both the matrix and the droplets: with the step in place a
        # symmetric pair lost 20% of its volume over t = 0.03 and the survivor
        # of a ripening pair *shrank*, hiding the mechanism the model exists to
        # show. Levelling the potential everywhere removes the driving force,
        # cutting the parasitic loss to ~6% and restoring the proper Ostwald
        # signature: the larger droplet grows as the smaller one dissolves.
        # What the frozen region holds is arbitrary anyway — nothing flows
        # there once there is no gradient to drive it.
        du0 = self.matrix_supersaturation
        u = -1.0 + du0 * (1.0 - phi) + 2.0 * phi

        return np.stack([u, M], axis=0)

    def generate_sample(
        self,
        generator: Union[str, Callable] = "fourier",
        generator_params: Dict = None,
        seed: int = None,
        validate: bool = True,
        max_attempts: int = 10,
        return_full: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Base-class contract, plus a seed for the conserved noise.

        The base implementation seeds `generate_ic` but calls `solve` bare, so
        with `noise_intensity > 0` the dataset path would draw an unseeded
        noise stream and `generate_dataset(seed=...)` would not reproduce.
        This override forwards a per-attempt noise seed to `solve`, following
        the stochastic models' convention. The IC seed is derived exactly as
        the base class derives it, so deterministic datasets are unchanged;
        the noise seed comes from a spawned child of the same SeedSequence, so
        the two streams are independent rather than identical.
        """
        if generator_params is None:
            generator_params = {}

        attempt_seqs = _seed_sequence(seed).spawn(max_attempts)
        for attempt in range(max_attempts):
            seq = attempt_seqs[attempt]
            ic_seed = _legacy_seed(seq) if seed is not None else None
            noise_seed = _legacy_seed(seq.spawn(1)[0]) if seed is not None else None

            ic = self.generate_ic(
                generator=generator,
                generator_params=generator_params,
                seed=ic_seed,
            )
            solution = self.solve(ic, return_full=return_full, seed=noise_seed)

            if not validate:
                return ic, solution, {"valid": True}
            validation = self.validate_solution(ic, solution)
            if validation["valid"]:
                return ic, solution, validation

        raise RuntimeError("sample generation failed")

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------

    def _rfft(self, a: np.ndarray) -> np.ndarray:
        return np.fft.rfftn(a, axes=(0, 1, 2))

    def _irfft(self, a_hat: np.ndarray) -> np.ndarray:
        return np.fft.irfftn(a_hat, s=self.field_shape, axes=(0, 1, 2))

    def _flux_divergence(
        self, u: np.ndarray, M: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Spectral div( M grad mu ) with mu = u^3 - u - eps^2 laplacian(u).

        Computed in flux form rather than expanded, so the k = 0 mode of the
        result is identically zero and mass is conserved exactly. Costs eight
        transforms: two forward for mu, three inverse for grad mu, three
        forward for the flux. Returns the divergence and the transform of u,
        which the caller reuses for the implicit part.
        """
        u_hat = self._rfft(u)
        cube_hat = self._rfft(u**3)
        mu_hat = cube_hat - u_hat + (self.eps**2) * self.K2 * u_hat

        div_hat = np.zeros(self.hat_shape, dtype=complex)
        for axis in range(3):
            grad = self._irfft(self.iK[axis] * mu_hat)
            div_hat += self.iK[axis] * self._rfft(M * grad)
        return div_hat, u_hat

    def _noise_increment(self, M: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Conserved thermal noise: div( sqrt(2 M Theta) eta ).

        Written in divergence form so, like the deterministic flux, it cannot
        touch the k = 0 mode — the noise moves material about but never creates
        it. The per-cell variance is dt / cell_volume, the discrete stand-in
        for space-time white noise.
        """
        amp = np.sqrt(2.0 * self.theta_noise * M)
        scale = np.sqrt(self.dt / self.cell_volume)
        div_hat = np.zeros(self.hat_shape, dtype=complex)
        for axis in range(3):
            xi = rng.standard_normal(self.field_shape)
            div_hat += self.iK[axis] * self._rfft(amp * scale * xi)
        return div_hat

    def _step(
        self, u: np.ndarray, M: np.ndarray, rng: Optional[np.random.Generator]
    ) -> np.ndarray:
        """
        One stabilised IMEX step of the variable-mobility equation.

        The implicit part is the constant-coefficient operator at the shell
        mobility M_bar, which is FFT-diagonal; the remainder is explicit:

            (I - dt L) u^{n+1} = u^n + dt [ R(u^n) - L u^n ]

        with L = M_bar grad^2 ( -eps^2 grad^2 + A ) and R the full variable-
        coefficient flux divergence. Where M = M_bar — that is, everywhere in
        the active shell — the correction R - L u reduces to the plain cubic
        term and the step is identical to the constant-mobility scheme. The
        variable coefficient is only felt in the frozen regions, where the
        frozen-coefficient amplification factor tends to (M_bar - M)/M_bar < 1,
        so those regions are damped rather than driven.
        """
        div_hat, u_hat = self._flux_divergence(u, M)

        # L in Fourier: -M_bar k^2 (eps^2 k^2 + A).
        L_hat = -self.M_bar * self.K2 * (self.eps**2 * self.K2 + self.stab)
        numer = u_hat + self.dt * (div_hat - L_hat * u_hat)

        if rng is not None and self.theta_noise > 0.0:
            numer = numer + self._noise_increment(M, rng)

        return self._irfft(numer / self._denom)

    def solve(
        self,
        ic: np.ndarray,
        return_full: bool = False,
        seed: int = None,
    ) -> np.ndarray:
        """
        Integrate the pair and classify the coarsening regime.

        `ic` is the stacked (u0, M) pair from `generate_ic`. Droplet
        diagnostics are accumulated at every output frame and the regime label
        is left on `self.last_diagnostics`; the returned array is the phase
        field alone, so the model stays a standard field-to-field map.
        """
        ic = np.asarray(ic)
        if ic.shape[0] != 2:
            raise ValueError(
                "eggshell_droplets_3d expects the stacked (u0, mobility) "
                f"initial condition of shape (2, nz, ny, nx), got {ic.shape}."
            )
        u = ic[0].copy()
        M = ic[1]

        rng = np.random.default_rng(seed) if self.theta_noise > 0.0 else None

        n_substeps = int(np.ceil(self.T / self.dt))
        output_interval = max(1, n_substeps // max(1, self.n_t - 1))

        # Diagnostics are cheap and always collected; the fields themselves are
        # only kept when the caller asked for the trajectory. Holding all of
        # them regardless would cost n_t full 3D arrays — a gigabyte at 128^3
        # and 61 frames, for a function that usually returns one snapshot.
        solutions = [u.copy()] if return_full else []
        frames = [self._frame_diagnostics(u)]
        times = [0.0]

        for step in range(n_substeps):
            u = self._step(u, M, rng)
            if (step + 1) % output_interval == 0 and len(frames) < self.n_t:
                if return_full:
                    solutions.append(u.copy())
                frames.append(self._frame_diagnostics(u))
                times.append((step + 1) * self.dt)

        # Pad so the frame count is always exactly n_t, whatever dt and T do.
        while len(frames) < self.n_t:
            if return_full:
                solutions.append(u.copy())
            frames.append(self._frame_diagnostics(u))
            times.append(self.T)

        self.last_diagnostics = self._assemble_diagnostics(
            times[: self.n_t], frames[: self.n_t]
        )

        if return_full:
            return np.stack(solutions[: self.n_t], axis=0)
        return u

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _frame_diagnostics(self, u: np.ndarray) -> Dict:
        """
        Resolve the droplets present in one frame.

        Connected components of {u > 0} with 6-connectivity; components below
        the volume of a sphere of radius eps are sub-interface debris and are
        discarded rather than counted as droplets.
        """
        labels, n_labels = ndimage.label(u > 0.0)
        if n_labels == 0:
            return {"n": 0, "volumes": [], "centroids": []}

        counts = np.bincount(labels.ravel())[1:]
        volumes = counts * self.cell_volume
        min_volume = (4.0 / 3.0) * np.pi * self.eps**3
        keep = np.nonzero(volumes >= min_volume)[0]
        if keep.size == 0:
            return {"n": 0, "volumes": [], "centroids": []}

        order = keep[np.argsort(volumes[keep])[::-1]]
        centroids = ndimage.center_of_mass(u > 0.0, labels, [int(i) + 1 for i in order])
        centroids = [
            np.array([c[a] * self.spacing[a] for a in range(3)]) for c in centroids
        ]
        return {
            "n": int(order.size),
            "volumes": [float(volumes[i]) for i in order],
            "centroids": centroids,
        }

    def _assemble_diagnostics(self, times, frames) -> Dict:
        """Stack per-frame droplet data and classify the trajectory."""
        n_t = len(frames)
        n_components = np.array([f["n"] for f in frames])
        volumes = np.zeros((n_t, 2))
        separations = np.full(n_t, np.nan)

        for i, f in enumerate(frames):
            vols = f["volumes"]
            volumes[i, 0] = vols[0] if len(vols) >= 1 else 0.0
            volumes[i, 1] = vols[1] if len(vols) >= 2 else 0.0
            if f["n"] >= 2:
                separations[i] = float(
                    np.linalg.norm(f["centroids"][0] - f["centroids"][1])
                )

        # Carry the last known separation forward so the classifier always has
        # a finite reference once the pair has broken up.
        for i in range(1, n_t):
            if np.isnan(separations[i]):
                separations[i] = separations[i - 1]

        result = classify_trajectory(
            np.array(times), n_components, volumes, separations, self.eps
        )
        result.update(
            {
                "times": np.array(times),
                "n_components": n_components,
                "volumes": volumes,
                "separations": separations,
                "total_droplet_volume": volumes.sum(axis=1),
            }
        )
        return result

    def free_energy(self, u: np.ndarray) -> float:
        """
        Cahn-Hilliard free energy, integral of (u^2-1)^2/4 + eps^2|grad u|^2/2.

        Decreases monotonically under the dynamics for any non-negative
        mobility field, since dF/dt = -integral M |grad mu|^2. Used as a
        validation invariant.
        """
        u_hat = self._rfft(u)
        grad_sq = np.zeros(self.field_shape)
        for axis in range(3):
            grad_sq += self._irfft(self.iK[axis] * u_hat) ** 2
        bulk = 0.25 * (u**2 - 1.0) ** 2
        return float((bulk + 0.5 * self.eps**2 * grad_sq).sum() * self.cell_volume)

    def chemical_potential(self, u: np.ndarray) -> np.ndarray:
        """
        mu = u^3 - u - eps^2 laplacian(u), the quantity the dynamics transports.

        In the sharp-interface limit mu is constant inside a droplet and equal
        to the Gibbs-Thomson value sigma/R, which is what makes the smaller
        partner the one that dissolves. Exposed because that is checkable:
        mu*R should return sigma for each droplet independently.
        """
        u_hat = self._rfft(u)
        laplacian = self._irfft(-self.K2 * u_hat)
        return u**3 - u - self.eps**2 * laplacian

    def droplet_potentials(self, u: np.ndarray, core_level: float = 0.8):
        """
        Mean chemical potential and equivalent radius of each droplet.

        The potential is averaged over the droplet interior only, `u >
        core_level`, so the interface itself — where mu varies steeply and the
        sharp-interface description does not apply — is excluded. Radii come
        from the measured volumes as (3V/4pi)^(1/3).

        Returns a list of (radius, mean mu) ordered largest droplet first.
        """
        mu = self.chemical_potential(u)
        labels, n_labels = ndimage.label(u > 0.0)
        if n_labels == 0:
            return []

        volumes = np.bincount(labels.ravel())[1:] * self.cell_volume
        min_volume = (4.0 / 3.0) * np.pi * self.eps**3
        keep = np.nonzero(volumes >= min_volume)[0]
        order = keep[np.argsort(volumes[keep])[::-1]]

        out = []
        for i in order:
            interior = (labels == int(i) + 1) & (u > core_level)
            if not interior.any():
                continue
            radius = (3.0 * volumes[i] / (4.0 * np.pi)) ** (1.0 / 3.0)
            out.append((float(radius), float(mu[interior].mean())))
        return out

    def validate_solution(
        self,
        ic: np.ndarray,
        solution: np.ndarray,
        tol: float = 1e-6,
    ) -> Dict:
        """
        Check the invariants: finiteness, boundedness, and exact conservation.

        Mass conservation is the sharp one. The update is a pure divergence, so
        the k = 0 mode is untouched and the drift should sit at round-off
        regardless of the geometry or the noise.
        """
        u0 = ic[0]
        finite = bool(np.isfinite(solution).all())
        mass_drift = float(np.abs(solution.mean() - u0.mean()))
        max_value = float(np.abs(solution).max())

        diag = self.last_diagnostics or {}
        return {
            "valid": finite and max_value < 1.5 and mass_drift < 1e-9,
            "mass_drift": mass_drift,
            "max_value": max_value,
            "regime": diag.get("regime", UNRESOLVED),
            "survival_fraction": diag.get("survival_fraction", float("nan")),
            "absorbed_fraction": diag.get("absorbed_fraction", float("nan")),
            "contact_gap": diag.get("contact_gap", float("nan")),
        }
