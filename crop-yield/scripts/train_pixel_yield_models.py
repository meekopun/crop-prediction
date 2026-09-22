#!/usr/bin/env python3
"""Train per-crop pixel-level yield models and write metrics plus predictions."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from yield_prediction.modeling import train_pixel_yield_models


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "pixel_level_all_crop_training_features_2021_2023.csv"
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
    return parser


def build_summary(metrics: pd.DataFrame, feature_set: str, feature_count: int) -> str:
    lines = [
        "# Pixel Yield Model Summary",
        "",
        f"- Feature set: `{feature_set}`",
        f"- Feature count: {feature_count}",
        "",
        "Grouped validation is done by `row_id`, so train/test splits do not share the same field-year.",
        "",
    ]
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
            lines.append(
                f"- `{row.model}`: mae={row.mean_mae:.3f} bu/ac, "
                f"rmse={row.mean_rmse:.3f} bu/ac, r2={r2_text}, "
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
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics_frame = pd.DataFrame(asdict(metric) for metric in artifact.metrics)
    metrics_path = args.output_dir / f"metrics_{args.feature_set}.csv"
    predictions_path = args.output_dir / f"pixel_predictions_{args.feature_set}.csv"
    summary_path = args.output_dir / f"summary_{args.feature_set}.md"

    metrics_frame.to_csv(metrics_path, index=False)
    artifact.predictions.to_csv(predictions_path, index=False)
    summary_path.write_text(
        build_summary(metrics_frame, args.feature_set, len(artifact.feature_columns)),
        encoding="utf-8",
    )

    print(f"Wrote {metrics_path}")
    print(f"Wrote {predictions_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
