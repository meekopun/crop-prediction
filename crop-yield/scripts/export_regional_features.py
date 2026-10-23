#!/usr/bin/env python3
"""Export crop-masked July Sentinel-2 means for every Alberta small-area region and year.

Requires `earthengine-api` and a saved Earth Engine login. Each region-year is a
single ~20 s request at 250 m; finished region-years are kept on rerun.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Also support a source checkout before installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yield_prediction.regional_features import (
    DEFAULT_REGION_YEARS, DEFAULT_REGIONAL_FEATURES_PATH, SCALE_M, export_regional_features,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_REGIONAL_FEATURES_PATH,
                        help="Region-crop-year feature CSV (appended as it runs).")
    parser.add_argument("--years", type=int, nargs="+", default=list(DEFAULT_REGION_YEARS),
                        help="Years to export.")
    parser.add_argument("--project", help="Earth Engine Cloud project; defaults to the saved one.")
    parser.add_argument("--scale", type=int, default=SCALE_M, help="Reduction scale in metres.")
    parser.add_argument("--overwrite", action="store_true", help="Start a fresh file instead of resuming.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        import ee
    except ImportError:
        print("Regional export requires earthengine-api: pip install earthengine-api", file=sys.stderr)
        return 1
    try:
        rows = export_regional_features(args.output, years=tuple(args.years), project=args.project,
                                        overwrite=args.overwrite, scale=args.scale,
                                        progress=lambda message: print(message, flush=True))
    except (OSError, ValueError, KeyError, ee.EEException) as exc:
        print(f"Regional export failed (rerun to resume): {exc}", file=sys.stderr)
        return 1
    print(f"{len(rows)} region-crop-years in {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
