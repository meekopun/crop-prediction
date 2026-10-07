#!/usr/bin/env python3
"""Import the farm's .xls harvest reports into operation records and review reports."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Also support a source checkout before installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yield_prediction.harvest import (
    DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_REVIEW_PATH, import_harvest_directory,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                        help="Directory containing the .xls harvest reports.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Destination for operation/candidate CSVs, exceptions.csv and summary.json.")
    review_options = parser.add_mutually_exclusive_group()
    review_options.add_argument("--review-file", type=Path, default=DEFAULT_REVIEW_PATH,
                                help="Saved crop confirmations and temporary exclusions.")
    review_options.add_argument("--no-review", action="store_true",
                                help="Import without applying saved farm-specific decisions.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = import_harvest_directory(args.input_dir, args.output_dir,
            review_file=None if args.no_review else args.review_file)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Harvest import failed: {exc}", file=sys.stderr)
        return 1
    print(f"Parsed {summary['operation_count']} operations from {summary['workbook_count']} workbooks.")
    print(f"Yield units: {summary['operations_by_yield_unit']}")
    print(f"{summary['operations_with_errors']} operations have parsing/validation errors; "
          f"{summary['operations_with_flags']} operations have review flags.")
    print(f"{summary['confirmed_crop_mapping_count']} crop mappings confirmed; "
          f"{summary['excluded_operation_count']} operations temporarily excluded; "
          f"{summary['processing_candidate_count']} candidates awaiting boundary review.")
    print(f"Wrote operations.csv, processing_candidates.csv, exceptions.csv and summary.json "
          f"to {args.output_dir.resolve()}")
    print("Field geometry and crop mappings need review before training.")
    # Outputs remain available even when errors require attention.
    return 2 if summary['issues_by_severity'].get('error', 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
