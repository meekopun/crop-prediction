#!/usr/bin/env python3
"""Build one row per harvested field-year: aggregated yield plus satellite features.

Run `scripts/match_harvest_boundaries.py` and `scripts/export_harvest_features.py` first.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Also support a source checkout before installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yield_prediction.field_yield import (
    DEFAULT_FEATURES_PATH, DEFAULT_MATCHES_PATH, DEFAULT_OPERATIONS_PATH, DEFAULT_TABLE_PATH,
    DEFAULT_WEATHER_PATH, build_field_yield_table,
)
from yield_prediction.harvest_features import DEFAULT_BOUNDARIES_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=Path, default=DEFAULT_MATCHES_PATH,
                        help="boundary_matches.csv from match_harvest_boundaries.py.")
    parser.add_argument("--operations", type=Path, default=DEFAULT_OPERATIONS_PATH,
                        help="operations.csv from import_harvest_data.py (harvested totals).")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH,
                        help="harvest_satellite_features.csv from export_harvest_features.py.")
    parser.add_argument("--boundaries", type=Path, default=DEFAULT_BOUNDARIES_PATH,
                        help="field_boundaries.geojson (parcel acres for weighting).")
    parser.add_argument("--output", type=Path, default=DEFAULT_TABLE_PATH,
                        help="Output CSV; a *_summary.json is written beside it.")
    weather = parser.add_mutually_exclusive_group()
    weather.add_argument("--weather", type=Path, default=DEFAULT_WEATHER_PATH,
                         help="harvest_weather.csv from export_harvest_weather.py; used when it exists.")
    weather.add_argument("--no-weather", action="store_true", help="Omit weather columns.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = build_field_yield_table(args.matches, args.operations, args.features,
                                          args.boundaries, args.output,
                                          weather_path=None if args.no_weather else args.weather)
    except (OSError, KeyError, ValueError) as exc:
        print(f"Field yield table failed: {exc}", file=sys.stderr)
        return 1
    print(f"{summary['field_year_count']} field-years from {summary['eligible_operations']} eligible operations")
    print(f"By crop: {summary['field_years_by_crop_unit']}")
    print(f"Feature source: {summary['feature_source']}; dropped groups: {summary['dropped_groups'] or 'none'}")
    print(f"Weather: {summary['weather_path'] or 'not included'}")
    print(f"Wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
