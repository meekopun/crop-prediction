#!/usr/bin/env python3
"""Train per-crop pixel-level yield models and write metrics plus predictions."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from yield_prediction.modeling import train_pixel_yield_models


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "reference_inputs" / "pixel_level_all_crop_training_features_2021_2023.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "pixel_yield_models_2021_2023"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for metrics and predictions.",
    )
    parser.add_argument(
        "--feature-set",
        choices=["ndvi_only", "satellite", "nonleaky", "all"],
        default="nonleaky",
        help="Feature subset to use for training.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Ridge regularization strength.",
    )
    parser.add_argument(
        "--split",
        choices=["field_year", "year"],
        default="field_year",
        help="Hold out one field-year per fold, or every field from one year.",
    )
    parser.add_argument(
        "--drop-year-constant",
        action="store_true",
        help="Drop features with a single value within every year; they act as year labels.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Also score a baseline that predicts the training-fold mean yield.",
    )
    return parser


def build_summary(
    metrics: pd.DataFrame,
    feature_set: str,
    feature_count: int,
    *,
    split: str = "field_year",
    dropped_columns: tuple[str, ...] = (),
) -> str:
    if split == "year":
        split_text = "Grouped validation holds out every field from one year, so test years are unseen in training."
    else:
        split_text = "Grouped validation is done by `row_id`, so train/test splits do not share the same field-year."
    lines = [
        "# Pixel Yield Model Summary",
        "",
        f"- Feature set: `{feature_set}`",
        f"- Feature count: {feature_count}",
    ]
    if dropped_columns:
        lines.append(f"- Dropped year-constant features: {', '.join(f'`{c}`' for c in dropped_columns)}")
    lines.extend([
        "",
        split_text,
        "`field_mae`/`field_r2` score each field-year's mean held-out prediction against its yield.",
        "",
    ])
    for crop_name, crop_metrics in metrics.groupby("crop", sort=True):
        lines.extend([f"## {crop_name}", ""])
        for row in crop_metrics.itertuples(index=False):
            if row.cv_splits == 0:
                lines.append(
                    f"- `{row.model}`: insufficient field-years for grouped validation "
                    f"(field_year_count={row.field_year_count})"
                )
                continue
            r2_text = "NA" if pd.isna(row.mean_r2) else f"{row.mean_r2:.3f}"
            field_r2_text = "NA" if pd.isna(row.field_r2) else f"{row.field_r2:.3f}"
            lines.append(
                f"- `{row.model}`: mae={row.mean_mae:.3f} bu/ac, "
                f"rmse={row.mean_rmse:.3f} bu/ac, r2={r2_text}, "
                f"field_mae={row.field_mae:.3f} bu/ac, field_r2={field_r2_text}, "
                f"field_years={row.field_year_count}, pixels={row.pixel_count}"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()
    artifact = train_pixel_yield_models(
        args.input,
        feature_set=args.feature_set,
        alpha=args.alpha,
        split=args.split,
        drop_year_constant=args.drop_year_constant,
        include_baseline=args.baseline,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Option suffixes keep default-run filenames unchanged.
    run_name = args.feature_set
    if args.split == "year":
        run_name += "_year_split"
    if args.drop_year_constant:
        run_name += "_no_year_constant"
    if args.baseline:
        run_name += "_baseline"

    metrics_frame = pd.DataFrame(asdict(metric) for metric in artifact.metrics)
    metrics_path = args.output_dir / f"metrics_{run_name}.csv"
    predictions_path = args.output_dir / f"pixel_predictions_{run_name}.csv"
    summary_path = args.output_dir / f"summary_{run_name}.md"

    metrics_frame.to_csv(metrics_path, index=False)
    artifact.predictions.to_csv(predictions_path, index=False)
    summary_path.write_text(
        build_summary(
            metrics_frame,
            args.feature_set,
            len(artifact.feature_columns),
            split=args.split,
            dropped_columns=artifact.dropped_feature_columns,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {metrics_path}")
    print(f"Wrote {predictions_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
