#!/usr/bin/env python3
"""Export crop-masked July and August Sentinel-2 means over a fixed area around the farm.

Gives the `local` season reading: the same land every year, averaged over the
pixels the AAFC crop inventory labels as each crop. Requires `earthengine-api`
and a saved Earth Engine login; finished years are kept on rerun. A season to be
predicted needs its year exported first.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Also support a source checkout before installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yield_prediction.regional_features import (
    DEFAULT_LOCAL_FEATURES_PATH, DEFAULT_REGION_YEARS, LOCAL_SCALE_M, export_local_features,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_LOCAL_FEATURES_PATH,
                        help="Crop-year feature CSV (appended as it runs).")
    parser.add_argument("--years", type=int, nargs="+", default=list(DEFAULT_REGION_YEARS), help="Years to export.")
    parser.add_argument("--project", help="Earth Engine Cloud project; defaults to the saved one.")
    parser.add_argument("--scale", type=int, default=LOCAL_SCALE_M, help="Reduction scale in metres.")
    parser.add_argument("--overwrite", action="store_true", help="Start a fresh file instead of resuming.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        import ee
    except ImportError:
        print("Local export requires earthengine-api: pip install earthengine-api", file=sys.stderr)
        return 1
    try:
        rows = export_local_features(args.output, years=tuple(args.years), project=args.project,
                                     overwrite=args.overwrite, scale=args.scale,
                                     progress=lambda message: print(message, flush=True))
    except (OSError, ValueError, KeyError, ee.EEException) as exc:
        print(f"Local export failed (rerun to resume): {exc}", file=sys.stderr)
        return 1
    print(f"{len(rows)} crop-years in {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
