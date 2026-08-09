"""
AirfRANS interop — read, do not rebuild.

AirfRANS (Bonnet et al., NeurIPS 2022 Datasets & Benchmarks) is 1000 RANS
solutions of incompressible flow over 2D airfoils: NACA 4- and 5-digit shapes
with continuously sampled digits, Reynolds numbers 2e6 to 6e6, angles of
attack -5 to +15 degrees, each a ~180k-node unstructured point cloud solved
with OpenFOAM's Spalart-Allmaras / k-omega SST closure.

PDEForge deliberately does NOT regenerate it. A faithful recreation means
shipping a RANS solver with a turbulence closure and wall treatment, which is
a different class of machinery from the spectral and FEM generators here, and
every other canonical recreation in this package carries a measured error
against the real thing. This one could not. So the verdict is interop: bring
their data onto the same ``PDEDataset`` surface as generated data, so the
splits, calibration and observation-operator machinery applies to it, and let
the citation point at them.

For airfoil data WITH knobs, see ``naca_flow_2d`` (laminar incompressible FEM,
geometry as a distribution) — inspired-by, not a recreation.

Reference
---------
Bonnet, Mazari, Cinnella, Gallinari (2022). "AirfRANS: High Fidelity
Computational Fluid Dynamics Dataset for Approximating Reynolds-Averaged
Navier-Stokes Solutions." NeurIPS Datasets and Benchmarks.
https://airfrans.readthedocs.io
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from pdeforge.core.types import PDEDataset
from pdeforge.io.vtk_xml import read_vtk_xml

# Kinematic viscosity of air used by the AirfRANS campaign (m^2/s); chord is
# 1 m, so Re = U_inf / NU_AIR.
NU_AIR = 1.56e-5

INPUT_NAMES = ["x", "y", "u_inf_x", "u_inf_y", "sdf", "n_x", "n_y", "surface"]
OUTPUT_NAMES = ["u", "v", "p", "nu_t"]

SPLITS = (
    "full_train",
    "full_test",
    "scarce_train",
    "reynolds_train",
    "reynolds_test",
    "aoa_train",
    "aoa_test",
)


def parse_case_name(name: str) -> Dict:
    """
    Decode an AirfRANS case name.

        airFoil2D_<turbulence>_<U_inf>_<AoA_deg>_<shape params...>

    The shape parameters are the continuously-sampled NACA digits: three of
    them for the 4-digit series, four for the 5-digit series. Reynolds number
    is derived, not stored: Re = U_inf * chord / nu with chord = 1 m.
    """
    parts = name.split("_")
    if len(parts) < 4 or parts[0] != "airFoil2D":
        raise ValueError(f"not an AirfRANS case name: {name!r}")
    nums = [float(x) for x in parts[2:]]
    return {
        "name": name,
        "turbulence": parts[1],
        "inlet_velocity_m_s": nums[0],
        "angle_of_attack_deg": nums[1],
        "naca_params": nums[2:],
        "naca_series": 4 if len(nums[2:]) == 3 else 5,
        "reynolds": nums[0] / NU_AIR,
    }


def read_case(root, name: str) -> Dict[str, np.ndarray]:
    """
    Read one AirfRANS case into named per-point arrays.

    The internal mesh already contains the airfoil wall nodes (they carry
    ``implicit_distance == 0`` and zero velocity), so the point cloud is just
    the .vtu; the .vtp is opened only for the surface normals, which the .vtu
    does not store. Verified on the distributed data: every .vtp point
    coincides with a .vtu point to 0.0 distance.
    """
    root = Path(root)
    case = root / name
    internal = read_vtk_xml(case / f"{name}_internal.vtu")
    aerofoil = read_vtk_xml(case / f"{name}_aerofoil.vtp")

    position = np.asarray(internal["points"][:, :2], dtype=np.float64)
    distance = np.abs(np.asarray(internal["implicit_distance"], dtype=np.float64))
    surface = distance == 0.0

    normals = np.zeros_like(position)
    if surface.any():
        from scipy.spatial import cKDTree

        surf_xy = aerofoil["points"][:, :2].astype(np.float64)
        # Map each wall node to its .vtp twin; they are coincident, so this is
        # a lookup, not an interpolation. Guard the assumption rather than
        # trusting it silently.
        dist, idx = cKDTree(surf_xy).query(position[surface])
        if dist.max() > 1e-9:
            raise ValueError(
                f"{name}: wall nodes do not coincide with the aerofoil file "
                f"(max gap {dist.max():.2e}); normals cannot be mapped."
            )
        normals[surface] = aerofoil["Normals"][idx, :2]

    params = parse_case_name(name)
    aoa = np.radians(params["angle_of_attack_deg"])
    u_inf = params["inlet_velocity_m_s"] * np.array([np.cos(aoa), np.sin(aoa)])

    return {
        "position": position,
        "sdf": distance,
        "normals": normals,
        "surface": surface,
        "inlet_velocity": u_inf,
        "velocity": np.asarray(internal["U"][:, :2], dtype=np.float64),
        "pressure": np.asarray(internal["p"], dtype=np.float64),
        "nu_t": np.asarray(internal["nut"], dtype=np.float64),
    }


def _stack_case(case: Dict[str, np.ndarray], idx: np.ndarray):
    u_inf = np.broadcast_to(case["inlet_velocity"], (idx.size, 2))
    inputs = np.column_stack(
        [
            case["position"][idx],
            u_inf,
            case["sdf"][idx],
            case["normals"][idx],
            case["surface"][idx].astype(np.float64),
        ]
    )
    outputs = np.column_stack(
        [
            case["velocity"][idx],
            case["pressure"][idx],
            case["nu_t"][idx],
        ]
    )
    return inputs, outputs


def load_airfrans(
    root,
    split: str = "full_train",
    n_samples: Optional[int] = None,
    n_points: int = 16384,
    keep_surface: bool = True,
    seed: Optional[int] = 0,
    verbose: bool = True,
) -> PDEDataset:
    """
    Load AirfRANS into a PDEDataset.

    root : directory holding the ~1000 case directories and manifest.json.
    split : a manifest key (see SPLITS), or "all" for every case on disk.
    n_samples : take only the first n cases of the split (None = all).
    n_points : nodes kept per case. The meshes are ~180k nodes and differ in
        size per case, so a common count is REQUIRED to stack them into one
        array; this is a subsample of the real solution, not an interpolation
        of it. Pass None only if you know the meshes match.
    keep_surface : keep every airfoil-wall node and subsample only the
        interior. The wall is ~0.6% of the cloud, so a uniform draw would
        leave ~90 of ~994 wall nodes and gut the quantity most aerodynamic
        targets depend on. Set False for a plain uniform draw.
    seed : subsample seed (None = non-deterministic).

    Inputs are the 7 canonical AirfRANS features plus their surface flag:
    (x, y, u_inf_x, u_inf_y, sdf, n_x, n_y, surface). The flag is a mask, not
    a physical feature — use ``inputs[..., :7]`` for the canonical setup.
    Outputs are (u, v, p, nu_t); p and nu_t are kinematic (divided by density).
    """
    root = Path(root)
    manifest_path = root / "manifest.json"

    if split == "all":
        names = sorted(p.name for p in root.iterdir() if p.is_dir())
    else:
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{manifest_path} not found; pass split='all' to use every "
                "case directory instead of a manifest split."
            )
        manifest = json.loads(manifest_path.read_text())
        if split not in manifest:
            raise ValueError(
                f"unknown split {split!r}; manifest has {sorted(manifest)}"
            )
        names = list(manifest[split])

    if n_samples is not None:
        names = names[:n_samples]
    if not names:
        raise ValueError(f"split {split!r} selected no cases")

    rng = np.random.default_rng(seed)
    ins, outs, params = [], [], []

    for name in tqdm(names, disable=not verbose, desc=f"AirfRANS {split}"):
        case = read_case(root, name)
        n_total = case["position"].shape[0]

        if n_points is None:
            idx = np.arange(n_total)
        elif n_points > n_total:
            raise ValueError(
                f"{name}: asked for {n_points} nodes but the case has {n_total}"
            )
        elif keep_surface:
            wall = np.flatnonzero(case["surface"])
            interior = np.flatnonzero(~case["surface"])
            if wall.size > n_points:
                raise ValueError(
                    f"{name}: {wall.size} wall nodes exceed n_points={n_points}; "
                    "raise n_points or pass keep_surface=False"
                )
            pick = rng.choice(interior, n_points - wall.size, replace=False)
            idx = np.sort(np.concatenate([wall, pick]))
        else:
            idx = np.sort(rng.choice(n_total, n_points, replace=False))

        a, b = _stack_case(case, idx)
        ins.append(a)
        outs.append(b)
        params.append(parse_case_name(name))

    inputs = np.stack(ins, axis=0)
    outputs = np.stack(outs, axis=0)

    return PDEDataset(
        inputs=inputs,
        outputs=outputs,
        # A point cloud has no separable axis grid; coordinates travel as the
        # first two input channels, per sample.
        grid={"node": np.arange(inputs.shape[1])},
        metadata={
            "source": "airfrans",
            "split": split,
            "n_samples": len(names),
            "n_points": int(inputs.shape[1]),
            "subsample_seed": seed,
            "keep_surface": keep_surface,
            "case_names": names,
            "case_params": params,
            "nu_air": NU_AIR,
            "units": "p and nu_t are kinematic (per unit density)",
            "reference": (
                "Bonnet, Mazari, Cinnella, Gallinari (2022), AirfRANS, "
                "NeurIPS Datasets and Benchmarks"
            ),
            "note": (
                "Read from the distributed AirfRANS data, not regenerated by "
                "PDEForge. Nodes are a subsample of each ~180k-node mesh."
            ),
        },
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
    )


def surface_pressure(dataset: PDEDataset, index: int = 0) -> Dict[str, np.ndarray]:
    """
    Surface pressure coefficient for one loaded case.

        C_p = (p / rho) / (0.5 |U_inf|^2)

    with the freestream gauge near zero, so C_p approaches +1 at the
    stagnation point and goes negative over the suction peak.
    """
    if dataset.metadata.get("source") != "airfrans":
        raise ValueError("surface_pressure expects a dataset from load_airfrans")

    names = dataset.input_names
    inp, out = dataset.inputs[index], dataset.outputs[index]
    wall = inp[:, names.index("surface")] > 0.5
    if not wall.any():
        raise ValueError("no wall nodes in this sample; reload with keep_surface=True")

    xy = inp[wall][:, [names.index("x"), names.index("y")]]
    u_inf = inp[0, [names.index("u_inf_x"), names.index("u_inf_y")]]
    p = out[wall][:, dataset.output_names.index("p")]

    x = xy[:, 0]
    chord = x.max() - x.min()
    return {
        "x_c": (x - x.min()) / chord,
        "y": xy[:, 1],
        "cp": p / (0.5 * float(u_inf @ u_inf)),
        "u_inf": float(np.hypot(*u_inf)),
    }
