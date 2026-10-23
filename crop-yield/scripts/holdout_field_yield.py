#!/usr/bin/env python3
"""Choose one model per crop on earlier years, then score it once on a held-out year.

Selection uses leave-one-year-out on the years before `--test-year` only; the
mean baseline is a candidate. Run `scripts/build_field_yield_table.py` first.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

# Also support a source checkout before installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yield_prediction.field_benchmark import DEFAULT_BENCHMARK_DIR, holdout_field_yield
from yield_prediction.field_yield import DEFAULT_TABLE_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_TABLE_PATH, help="field_yield_table.csv.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BENCHMARK_DIR,
                        help="Directory for selection, results, predictions and the summary.")
    parser.add_argument("--test-year", type=int, default=2025, help="Held-out year scored once.")
    parser.add_argument("--min-rows", type=int, default=10,
                        help="Minimum field-years (all years) for a crop to be evaluated.")
    parser.add_argument("--min-improvement", type=float, default=0.0,
                        help="Share by which a model must beat the baseline's development MAE to be chosen.")
    return parser


def build_summary(results: pd.DataFrame, test_year: int, min_improvement: float = 0.0) -> str:
    margin = (f" A model replaced the baseline only if it beat the baseline's development error by "
              f"at least {min_improvement:.0%}." if min_improvement else "")
    lines = [f"# Field Yield Holdout: {test_year}", "",
             f"Each crop's configuration was chosen by leave-one-year-out on years before {test_year}",
             f"only (the mean baseline was a candidate).{margin} It was refitted on those years, then scored once on",
             f"{test_year}. Errors are mean absolute error in the crop's unit; `skill` is the share of",
             f"the baseline's {test_year} error removed. Test sets are small, so treat single crops as indicative.", "",
             "| Crop | Dev field-years | Test field-years | Chosen | Dev MAE (baseline) | "
             f"{test_year} MAE | {test_year} baseline MAE | Skill | Bias |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in results.itertuples(index=False):
        if row.status != "scored":
            lines.append(f"| {row.crop} ({row.yield_unit}) | {row.dev_field_years} | {row.test_field_years} | "
                         f"{row.status} | | | | | |")
            continue
        lines.append(f"| {row.crop} ({row.yield_unit}) | {row.dev_field_years} | {row.test_field_years} | "
                     f"{row.chosen_model} / {row.chosen_feature_set} | {row.dev_mae_chosen:.1f} "
                     f"({row.dev_mae_baseline:.1f}) | {row.test_mae_chosen:.1f} | {row.test_mae_baseline:.1f} | "
                     f"{row.test_mae_skill:+.2f} | {row.test_bias_chosen:+.1f} |")
    return "\n".join(lines)


def main() -> int:
    args = build_parser().parse_args()
    try:
        artifact = holdout_field_yield(args.input, test_year=args.test_year, min_rows=args.min_rows,
                                       min_improvement=args.min_improvement)
    except (OSError, KeyError, ValueError) as exc:
        print(f"Field yield holdout failed: {exc}", file=sys.stderr)
        return 1
    if artifact.results.empty:
        print(f"No crop has at least {args.min_rows} field-years.", file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # The margin is part of the name so margin runs never overwrite the plain holdout.
    prefix = f"holdout_{args.test_year}" + (f"_margin{round(args.min_improvement * 100)}"
                                            if args.min_improvement else "")
    artifact.selection.to_csv(args.output_dir / f"{prefix}_selection.csv", index=False)
    artifact.results.to_csv(args.output_dir / f"{prefix}_results.csv", index=False)
    artifact.predictions.to_csv(args.output_dir / f"{prefix}_predictions.csv", index=False)
    summary = build_summary(artifact.results, args.test_year, args.min_improvement)
    (args.output_dir / f"{prefix}_summary.md").write_text(summary + "\n", encoding="utf-8")
    print(summary)
    print(f"Wrote {prefix}_selection.csv, _results.csv, _predictions.csv and _summary.md "
          f"to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
