#!/usr/bin/env python3
"""Benchmark field-level yield models against a per-crop mean baseline.

Run `scripts/build_field_yield_table.py` first.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

# Also support a source checkout before installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yield_prediction.field_benchmark import DEFAULT_BENCHMARK_DIR, benchmark_field_yield
from yield_prediction.field_yield import DEFAULT_TABLE_PATH

SPLIT_TEXT = {
    "year": "Leave one year out: each season is predicted by models trained on the other seasons.",
    "section": "Leave sections out: whole ATS sections are held out across all years (up to five folds).",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_TABLE_PATH, help="field_yield_table.csv.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BENCHMARK_DIR,
                        help="Directory for metrics, predictions and the summary.")
    parser.add_argument("--min-rows", type=int, default=10,
                        help="Minimum field-years for a crop to be benchmarked.")
    return parser


def build_summary(metrics) -> str:
    lines = ["# Field Yield Benchmark", "",
             "Errors are mean absolute error in each crop's reported unit. `skill` is the share of",
             "the baseline's error removed (0 = no better than predicting the training mean;",
             "negative = worse). Feature sets: `ndvi` (monthly NDVI), `indices` (seven monthly",
             "indices), `all` (bands, indices and radar).", "",
             "The best model is chosen after seeing these scores, so its skill is optimistic;",
             "the skill range across all six model/feature combinations shows how robust it is.", ""]
    for split, split_metrics in metrics.groupby("split", sort=False):
        lines.extend([f"## Split: `{split}`", "", SPLIT_TEXT[split], "",
                      "| Crop | Field-years | Baseline MAE | Best model | Best MAE | Best skill | Skill range | Best R² |",
                      "| --- | --- | --- | --- | --- | --- | --- | --- |"])
        for (crop, unit), crop_metrics in split_metrics.groupby(["crop", "yield_unit"], sort=True):
            baseline = crop_metrics[crop_metrics["model"] == "mean_baseline"].iloc[0]
            candidates = crop_metrics[crop_metrics["model"] != "mean_baseline"]
            best = candidates.sort_values("mae").iloc[0]
            r2 = "NA" if pd.isna(best["r2"]) else f"{best['r2']:.2f}"
            lines.append(f"| {crop} ({unit}) | {int(baseline['field_years'])} | {baseline['mae']:.1f} | "
                         f"{best['model']} / {best['feature_set']} | {best['mae']:.1f} | "
                         f"{best['mae_skill']:+.2f} | {candidates['mae_skill'].min():+.2f} to "
                         f"{candidates['mae_skill'].max():+.2f} | {r2} |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = build_parser().parse_args()
    try:
        artifact = benchmark_field_yield(args.input, min_rows=args.min_rows)
    except (OSError, KeyError, ValueError) as exc:
        print(f"Field yield benchmark failed: {exc}", file=sys.stderr)
        return 1
    if artifact.metrics.empty:
        print(f"No crop has at least {args.min_rows} field-years.", file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact.metrics.to_csv(args.output_dir / "field_yield_metrics.csv", index=False)
    artifact.predictions.to_csv(args.output_dir / "field_yield_predictions.csv", index=False)
    summary = build_summary(artifact.metrics)
    (args.output_dir / "summary.md").write_text(summary + "\n", encoding="utf-8")
    print(summary)
    print(f"Wrote metrics, predictions and summary.md to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
