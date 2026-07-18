"""
The pdeforge command-line interface — the data-acquisition appliance.

Generate datasets without writing Python, which is what makes the Docker
image a turnkey factory:

    docker run -v $PWD/data:/data ghcr.io/pyatsysh/pdeforge:fenicsx \\
        pdeforge generate --preset fno_darcy_2d --n 1024 \\
        --resolution x=421 y=421 --seed 0 --out /data/darcy421

Subcommands:
    generate    make a dataset (model or preset; any format)
    reproduce   regenerate a dataset from its metadata.json
    models      list registered models
    presets     list canonical presets
    describe    show a model's parameters
"""

import argparse
import json
import sys


def _parse_kv(pairs, cast_numbers=True):
    """Parse repeated KEY=VALUE args; values go through JSON when possible."""
    out = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"expected KEY=VALUE, got {item!r}")
        key, val = item.split("=", 1)
        if cast_numbers:
            try:
                val = json.loads(val)
            except json.JSONDecodeError:
                pass  # keep as string
        out[key] = val
    return out


def _add_generate_args(p):
    p.add_argument("--model", help="registered model name")
    p.add_argument("--preset", help="canonical preset name (see: pdeforge presets)")
    p.add_argument("--n", type=int, required=True, help="number of samples")
    p.add_argument(
        "--resolution",
        nargs="+",
        required=True,
        metavar="DIM=N",
        help="grid resolution, e.g. --resolution x=256 or x=64 y=64",
    )
    p.add_argument(
        "--param",
        action="append",
        metavar="KEY=VALUE",
        help="model parameter override (repeatable)",
    )
    p.add_argument(
        "--ic-param",
        action="append",
        metavar="KEY=VALUE",
        help="IC-generator parameter override (repeatable)",
    )
    p.add_argument("--seed", type=int, default=None, help="root seed (reproducible)")
    p.add_argument("--out", required=True, help="output path")
    p.add_argument(
        "--format",
        default="dir",
        choices=["dir", "npz", "h5", "zarr", "pdebench"],
        help="output format (dir = directory of .npy + metadata.json)",
    )
    p.add_argument(
        "--outputs",
        default="final",
        choices=["final", "trajectory"],
        help="final snapshots or full rollouts",
    )
    p.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "numpy", "jax"],
        help="solver backend (jax is opt-in)",
    )
    p.add_argument("--n-jobs", type=int, default=1, help="parallel workers")
    p.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="stream to disk in chunks of this size (removes the RAM ceiling; "
        "dir and h5 formats only)",
    )
    p.add_argument("--quiet", action="store_true")


def _cmd_generate(args):
    import pdeforge

    resolution = {
        k: int(v) for k, v in _parse_kv(args.resolution, cast_numbers=True).items()
    }
    params = _parse_kv(args.param)
    ic_params = _parse_kv(args.ic_param)

    kwargs = dict(
        model=args.model,
        preset=args.preset,
        n_samples=args.n,
        resolution=resolution,
        params=params or None,
        ic_params=ic_params or None,
        seed=args.seed,
        outputs=args.outputs,
        backend=args.backend,
        n_jobs=args.n_jobs,
        verbose=not args.quiet,
    )

    if args.chunk_size is not None:
        if args.format not in ("dir", "h5"):
            raise SystemExit("--chunk-size supports --format dir or h5 only")
        out = args.out if args.format == "dir" else args.out
        dataset = pdeforge.generate_dataset(
            to=out, chunk_size=args.chunk_size, **kwargs
        )
        print(f"dataset streamed to {args.out}")
        return 0

    dataset = pdeforge.generate_dataset(**kwargs)

    if args.format == "dir":
        dataset.save(args.out)
    elif args.format == "pdebench":
        pdeforge.export_pdebench(dataset, args.out)
    else:
        suffix = {"npz": ".npz", "h5": ".h5", "zarr": ".zarr"}[args.format]
        out = args.out if args.out.endswith(suffix) else args.out + suffix
        pdeforge.save_dataset(dataset, out)
    return 0


def _cmd_reproduce(args):
    import pdeforge

    dataset = pdeforge.reproduce(args.source, verbose=not args.quiet)
    dataset.save(args.out)
    print(f"reproduced dataset saved to {args.out}")
    return 0


def _cmd_models(args):
    import pdeforge

    for name in pdeforge.list_models():
        print(name)
    return 0


def _cmd_presets(args):
    from pdeforge.presets import PRESETS

    width = max(len(k) for k in PRESETS)
    for name in sorted(PRESETS):
        print(f"{name:<{width}}  {PRESETS[name].get('notes', '')}")
    return 0


def _cmd_describe(args):
    import pdeforge

    print(pdeforge.describe_model(args.model))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="pdeforge",
        description="Generate PDE datasets for operator learning and UQ.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="generate a dataset")
    _add_generate_args(p)
    p.set_defaults(func=_cmd_generate)

    p = sub.add_parser("reproduce", help="regenerate a dataset from its metadata")
    p.add_argument("source", help="metadata.json or a saved-dataset path")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=_cmd_reproduce)

    p = sub.add_parser("models", help="list registered models")
    p.set_defaults(func=_cmd_models)

    p = sub.add_parser("presets", help="list canonical presets")
    p.set_defaults(func=_cmd_presets)

    p = sub.add_parser("describe", help="show a model's parameters")
    p.add_argument("model")
    p.set_defaults(func=_cmd_describe)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
