#!/usr/bin/env python3
"""Estimate pixel-level yield by redistributing field-average yield with NDVI scores."""

from __future__ import annotations

import argparse
from pathlib import Path

from yield_prediction.pixel_yield import (
    DEFAULT_NDVI_WEIGHTS,
    estimate_pixel_yield_from_ndvi,
    load_pixel_table,
    summarize_pixel_yield_estimates,
    write_pixel_yield_outputs,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "pixel_level_all_crop_training_features_2021_2023.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "pixel_ndvi_yield_estimates"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Pixel feature CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for pixel and field outputs.",
    )
    parser.add_argument(
        "--group-column",
        default="row_id",
        help="Field-year grouping column that shares the same observed yield.",
    )
    parser.add_argument(
        "--score-floor",
        type=float,
        default=0.25,
        help="Minimum relative NDVI score before field normalization.",
    )
    parser.add_argument(
        "--min-factor",
        type=float,
        default=0.5,
        help="Lower bound for pixel yield factor before rebalancing to the field mean.",
    )
    parser.add_argument(
        "--max-factor",
        type=float,
        default=1.5,
        help="Upper bound for pixel yield factor before rebalancing to the field mean.",
    )
    parser.add_argument(
        "--lower-quantile",
        type=float,
        default=0.05,
        help="Lower within-field quantile used to clip NDVI features before scaling.",
    )
    parser.add_argument(
        "--upper-quantile",
        type=float,
        default=0.95,
        help="Upper within-field quantile used to clip NDVI features before scaling.",
    )
    parser.add_argument(
        "--ndvi-max-weight",
        type=float,
        default=DEFAULT_NDVI_WEIGHTS["NDVI_max"],
        help="Weight for NDVI_max in the pixel score.",
    )
    parser.add_argument(
        "--ndvi-integral-weight",
        type=float,
        default=DEFAULT_NDVI_WEIGHTS["NDVI_integral_proxy"],
        help="Weight for NDVI_integral_proxy in the pixel score.",
    )
    parser.add_argument(
        "--ndvi-min-weight",
        type=float,
        default=DEFAULT_NDVI_WEIGHTS["NDVI_min"],
        help="Weight for NDVI_min in the pixel score.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    frame = load_pixel_table(args.input)
    weighted_columns = {
        "NDVI_max": args.ndvi_max_weight,
        "NDVI_integral_proxy": args.ndvi_integral_weight,
        "NDVI_min": args.ndvi_min_weight,
    }
    pixel_estimates = estimate_pixel_yield_from_ndvi(
        frame,
        group_column=args.group_column,
        ndvi_weights=weighted_columns,
        lower_quantile=args.lower_quantile,
        upper_quantile=args.upper_quantile,
        score_floor=args.score_floor,
        min_factor=args.min_factor,
        max_factor=args.max_factor,
    )
    summary = summarize_pixel_yield_estimates(
        pixel_estimates,
        group_column=args.group_column,
    )

    pixel_output = args.output_dir / "pixel_yield_estimates_ndvi.csv"
    summary_output = args.output_dir / "field_yield_estimate_summary_ndvi.csv"
    write_pixel_yield_outputs(
        pixel_estimates,
        summary,
        pixel_output=pixel_output,
        summary_output=summary_output,
    )

    print(f"Wrote {pixel_output}")
    print(f"Wrote {summary_output}")


if __name__ == "__main__":
    main()
