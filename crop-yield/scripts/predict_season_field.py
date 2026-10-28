#!/usr/bin/env python3
"""Predict field yields for a held-out season with a saved season + field bundle.

Predicts every field-year of `--year` (default: the bundle's test year) in the
input table; observed yields are not needed. With `--score`, also compares the
predictions with the observed yields once and writes the holdout summary. Score
files are not overwritten without `--overwrite`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

# Also support a source checkout before installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yield_prediction.field_yield import DEFAULT_TABLE_PATH
from yield_prediction.season_field_bundle import (
    MODEL_VARIANT, default_bundle_dir, load_bundle, predict_season_field, score_predictions,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=default_bundle_dir(),
                        help="Bundle directory from fit_season_field_bundle.py.")
    parser.add_argument("--input", type=Path, default=DEFAULT_TABLE_PATH,
                        help="Field-year table with the bundle's feature columns.")
    parser.add_argument("--year", type=int, help="Season to predict (default: the bundle's test year).")
    parser.add_argument("--output-dir", type=Path, help="Destination (default: the bundle directory).")
    parser.add_argument("--score", action="store_true",
                        help="Score against observed yields and write score_<year>.csv/.md.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing score files.")
    return parser


def build_summary(scores: pd.DataFrame, predictions: pd.DataFrame, bundle: dict, year: int) -> str:
    lines = [f"# Season + Field Holdout: {year}", "",
             f"Bundle fitted {bundle['created']} on {min(bundle['training_years'])}-"
             f"{max(bundle['training_years'])} with a {bundle['config']['min_improvement']:.0%} margin; "
             f"choices were fixed before {year} yields were seen. Errors are mean absolute error in the crop's "
             f"unit; `skill` is the share of the training mean's error removed by the chosen prediction.", "",
             "| Crop | Field-years | Chosen | MAE | Training-mean MAE | Model MAE | Skill | Bias |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in scores.itertuples(index=False):
        chosen = {"baseline": "training mean", MODEL_VARIANT: "model"}.get(row.chosen_variant, row.chosen_variant)
        lines.append(f"| {row.crop} ({row.yield_unit}) | {row.field_years} | {chosen} | {row.mae:.1f} | "
                     f"{row.baseline_mae:.1f} | {row.model_mae:.1f} | {row.skill:+.2f} | {row.bias:+.1f} |")
    unscored = predictions[predictions["status"] != "ok"]
    if not unscored.empty:
        counts = unscored.groupby(["crop", "status"]).size()
        lines.extend(["", "Not predicted: " + "; ".join(f"{crop} ({count}, {status})"
                                                        for (crop, status), count in counts.items()) + "."])
    return "\n".join(lines)


def main() -> int:
    args = build_parser().parse_args()
    try:
        bundle = load_bundle(args.bundle)
        year = args.year or bundle["test_year"]
        if year in bundle["training_years"]:
            raise ValueError(f"{year} trained this bundle; predicting it would be in-sample.")
        table = pd.read_csv(args.input, dtype={"field_key": str, "section_key": str, "pids": str})
        rows = table[table["year"] == year]
        if rows.empty:
            raise ValueError(f"{args.input} has no {year} rows; import the {year} harvest workbook and "
                             f"rebuild the table first.")
        predictions = predict_season_field(bundle, rows)
    except (OSError, KeyError, ValueError) as exc:
        print(f"Prediction failed: {exc}", file=sys.stderr)
        return 1
    output_dir = args.output_dir or args.bundle
    score_csv, score_md = output_dir / f"score_{year}.csv", output_dir / f"score_{year}.md"
    if args.score and (score_csv.exists() or score_md.exists()) and not args.overwrite:
        print(f"{score_md} already exists; this holdout has been scored. Pass --overwrite to rescore.",
              file=sys.stderr)
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / f"predictions_{year}.csv", index=False)
    print(f"Predicted {int((predictions['status'] == 'ok').sum())} of {len(predictions)} {year} field-years; "
          f"wrote predictions_{year}.csv to {output_dir.resolve()}")
    if not args.score:
        return 0
    try:
        scores = score_predictions(predictions)
    except ValueError as exc:
        print(f"Scoring failed: {exc}", file=sys.stderr)
        return 1
    scores.to_csv(score_csv, index=False)
    summary = build_summary(scores, predictions, bundle, year)
    score_md.write_text(summary + "\n", encoding="utf-8")
    print(summary)
    print(f"Wrote {score_csv.name} and {score_md.name} to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
