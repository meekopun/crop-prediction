#!/usr/bin/env python3
"""Match staged harvest operations to ATS quarter-section boundaries and write a match report."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Also support a source checkout before installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yield_prediction.harvest import DEFAULT_OUTPUT_DIR
from yield_prediction.harvest_boundaries import (
    DEFAULT_GEOMETRY_PATH, DEFAULT_OPERATIONS_PATH, match_harvest_boundaries,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operations", type=Path, default=DEFAULT_OPERATIONS_PATH,
                        help="operations.csv written by import_harvest_data.py.")
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY_PATH,
                        help="Province-wide ATS quarter-section GeoJSON (streamed, one feature per line).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Destination for field_boundaries.geojson, boundary_matches.csv and the summary.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = match_harvest_boundaries(args.operations, args.geometry, args.output_dir)
    except (OSError, ValueError) as exc:
        print(f"Boundary matching failed: {exc}", file=sys.stderr)
        return 1
    print(f"{summary['candidate_count']} candidates: {summary['candidates_by_match_status']}")
    print(f"Quarter sections found: {summary['quarter_sections_found']} of "
          f"{summary['quarter_sections_requested']} requested")
    print(f"Quarter-year sharing: {summary['candidates_by_quarter_year_status']}")
    print(f"{summary['boundary_pilot_eligible_count']} operations eligible for a boundary pilot; "
          f"parcel-years by crop: {summary['eligible_parcel_years_by_crop_unit']}")
    print(f"Wrote field_boundaries.geojson, boundary_matches.csv and boundary_match_summary.json "
          f"to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
