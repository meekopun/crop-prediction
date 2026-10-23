#!/usr/bin/env python3
"""Extract Peace-region crop yields from the Statistics Canada small-area table.

Expects the full-table zip for table 32-10-0002-01 (32100002-eng.zip) in
data/raw/statcan/, or pass --source. Writes data/processed/regional/regional_yields.csv.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

# Also support a source checkout before installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yield_prediction.regional_yields import (
    DEFAULT_REGION, DEFAULT_REGIONAL_PATH, DEFAULT_STATCAN_ZIP, prepare_regional_yields,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_STATCAN_ZIP,
                        help="StatCan full-table zip or extracted CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_REGIONAL_PATH, help="Tidy regional yield CSV.")
    parser.add_argument("--region", default=DEFAULT_REGION, help="Exact GEO name in the table.")
    parser.add_argument("--first-year", type=int, default=2015, help="Earliest year to keep.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        rows = prepare_regional_yields(args.source, args.output, region=args.region, first_year=args.first_year)
    except (OSError, KeyError, ValueError, StopIteration) as exc:
        print(f"Regional yield preparation failed: {exc}", file=sys.stderr)
        return 1
    years = sorted({row["year"] for row in rows})
    print(f"{len(rows)} crop-years for {args.region}, {years[0]}-{years[-1]}: "
          f"{dict(sorted(Counter(row['crop'] for row in rows).items()))}")
    print(f"Wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
