#!/usr/bin/env python3
"""Train pixel-level crop classifiers and write metrics plus predictions."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from yield_prediction.modeling import train_pixel_crop_classifier


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "pixel_level_all_crop_training_features_2021_2023.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "pixel_crop_classifier_2021_2023"


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
        choices=["satellite", "nonleaky", "all"],
        default="nonleaky",
        help="Feature subset to use for training.",
    )
    return parser


def _fmt(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{value:.3f}"


def build_summary(metrics: pd.DataFrame, feature_set: str, feature_count: int) -> str:
    lines = [
        "# Pixel Crop Classifier Summary",
        "",
        f"- Feature set: `{feature_set}`",
        f"- Feature count: {feature_count}",
        "",
        "Grouped validation is done by `row_id` to avoid sharing the same field-year across train and test.",
        "Some folds may be skipped when a held-out crop class does not exist in the remaining training groups.",
        "",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(f"## {row.model}")
        lines.append("")
        lines.append(
            f"- accuracy={_fmt(row.grouped_accuracy)}, "
            f"balanced_accuracy={_fmt(row.grouped_balanced_accuracy)}, "
            f"macro_f1={_fmt(row.grouped_macro_f1)}"
        )
        lines.append(
            f"- field_years={int(row.field_year_count)}, classes={int(row.class_count)}, "
            f"pixels={int(row.pixel_count)}, evaluated_pixels={int(row.evaluated_pixels)}"
        )
        if row.notes:
            lines.append(f"- note: {row.notes}")
        lines.append("")
    return "\n".join(lines)


def write_per_crop_reports(
    predictions: pd.DataFrame,
    output_dir: Path,
    feature_set: str,
) -> list[Path]:
    written: list[Path] = []
    true_labels = predictions["crop"].astype(str)
    for prediction_column in [column for column in predictions.columns if column.startswith("pred_")]:
        model_name = prediction_column.removeprefix("pred_").removesuffix(f"_{feature_set}")
        evaluated = predictions[prediction_column].notna()
        if not evaluated.any():
            continue
        y_true = true_labels[evaluated]
        y_pred = predictions.loc[evaluated, prediction_column].astype(str)

        report = pd.DataFrame(
            classification_report(
                y_true,
                y_pred,
                output_dict=True,
                zero_division=0,
            )
        ).transpose()
        report_path = output_dir / f"per_crop_metrics_{model_name}_{feature_set}.csv"
        report.to_csv(report_path, index=True)
        written.append(report_path)

        labels = sorted(set(y_true) | set(y_pred))
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        matrix_frame = pd.DataFrame(
            matrix,
            index=[f"true_{label}" for label in labels],
            columns=[f"pred_{label}" for label in labels],
        )
        matrix_path = output_dir / f"confusion_matrix_{model_name}_{feature_set}.csv"
        matrix_frame.to_csv(matrix_path, index=True)
        written.append(matrix_path)
    return written


def main() -> None:
    args = build_parser().parse_args()
    artifact = train_pixel_crop_classifier(args.input, feature_set=args.feature_set)
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
    extra_paths = write_per_crop_reports(
        artifact.predictions,
        args.output_dir,
        args.feature_set,
    )

    print(f"Wrote {metrics_path}")
    print(f"Wrote {predictions_path}")
    print(f"Wrote {summary_path}")
    for path in extra_paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
