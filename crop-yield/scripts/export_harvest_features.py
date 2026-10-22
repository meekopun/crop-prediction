#!/usr/bin/env python3
"""Export monthly Sentinel-2/Sentinel-1 features for matched harvest quarter sections.

Requires `earthengine-api` and a saved Earth Engine login (`earthengine authenticate`).
Run `scripts/match_harvest_boundaries.py` first.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Also support a source checkout before installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yield_prediction.harvest_features import (
    DEFAULT_BOUNDARIES_PATH, DEFAULT_FEATURE_DIR, DEFAULT_MATCHES_PATH, DEFAULT_YEARS,
    export_harvest_features,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=Path, default=DEFAULT_MATCHES_PATH,
                        help="boundary_matches.csv from match_harvest_boundaries.py.")
    parser.add_argument("--boundaries", type=Path, default=DEFAULT_BOUNDARIES_PATH,
                        help="field_boundaries.geojson from match_harvest_boundaries.py.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FEATURE_DIR,
                        help="Destination for per-year CSVs, the combined table and the manifest.")
    parser.add_argument("--years", type=int, nargs="+", default=list(DEFAULT_YEARS),
                        help="Growing-season years to export.")
    parser.add_argument("--project", help="Earth Engine Cloud project; defaults to the saved one.")
    parser.add_argument("--scale", type=int, default=10, help="Reduction scale in metres.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-export years whose CSV already exists.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        import ee
    except ImportError:
        print("Feature export requires earthengine-api: pip install earthengine-api", file=sys.stderr)
        return 1
    try:
        manifest = export_harvest_features(
            args.matches, args.boundaries, args.output_dir, years=tuple(args.years),
            scale=args.scale, project=args.project, overwrite=args.overwrite,
        )
    except (OSError, ValueError, RuntimeError, ee.EEException) as exc:
        print(f"Feature export failed: {exc}", file=sys.stderr)
        return 1
    print(f"Combined {manifest['unit_count']} quarter-section years into "
          f"{args.output_dir.resolve() / 'harvest_satellite_features.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
