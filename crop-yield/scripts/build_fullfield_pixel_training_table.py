#!/usr/bin/env python3
"""Build a standardized pixel training table from the full-field GEE export."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PIXEL_INPUT = ROOT / "data" / "raw" / "pixel_crop_features_aci_s2_monthly_csv_quarter_sections_fullfield.csv"
DEFAULT_YIELD_INPUT = ROOT / "reference_inputs" / "Crop Yield Data - Crop Data 2021-2023-2.csv"
DEFAULT_LOOKUP_INPUT = ROOT / "reference_inputs" / "pixel_level_all_crop_training_features_2021_2023.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "pixel_level_fullfield_training_features_2021_2023.csv"
MISSING_VALUE = -9999

CROP_NAME_MAP = {
    "Canola": "Canola and Rapeseed",
    "Canola ": "Canola and Rapeseed",
    "Peas Yellow": "Peas",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixel-input", type=Path, default=DEFAULT_PIXEL_INPUT, help="Full-field GEE export CSV.")
    parser.add_argument("--yield-input", type=Path, default=DEFAULT_YIELD_INPUT, help="Observed yield CSV.")
    parser.add_argument(
        "--lookup-input",
        type=Path,
        default=DEFAULT_LOOKUP_INPUT,
        help="Existing pixel-level training CSV used as the authoritative field-year yield lookup.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Standardized output CSV.")
    return parser


def normalize_quarter_id(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    if text.startswith("ALL-"):
        return None
    if text.startswith("W-"):
        text = "SW-" + text[2:]
    if not any(text.startswith(prefix) for prefix in ("NW-", "NE-", "SW-", "SE-")):
        return None
    return text


def load_yield_reference(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.rename(
        columns={
            "quarter section id": "quarter_section",
            "crop type": "crop",
            "yield (bu/ac)": "yield_bu_ac",
        }
    )
    frame["quarter_section"] = frame["quarter_section"].map(normalize_quarter_id)
    frame["crop"] = frame["crop"].astype(str).str.strip().replace(CROP_NAME_MAP)
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    frame["yield_bu_ac"] = pd.to_numeric(frame["yield_bu_ac"], errors="coerce")
    frame = frame.dropna(subset=["quarter_section", "crop", "year", "yield_bu_ac"]).copy()
    frame["year"] = frame["year"].astype(int)
    frame["yield_original"] = frame["yield_bu_ac"]
    frame["yield_original_unit"] = "bu/ac"
    return frame[["quarter_section", "crop", "year", "yield_bu_ac", "yield_original", "yield_original_unit"]]


def replace_missing_sentinel(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = frame.select_dtypes(include=["number"]).columns.tolist()
    replace_columns = [column for column in numeric_columns if column not in {"aci_crop_code", "is_target_crop", "year"}]
    frame.loc[:, replace_columns] = frame.loc[:, replace_columns].replace(MISSING_VALUE, np.nan)
    return frame.copy()


def derive_ndvi_features(frame: pd.DataFrame) -> pd.DataFrame:
    ndvi_monthly = sorted(column for column in frame.columns if column.startswith("ndvi_m"))
    if not ndvi_monthly:
        raise ValueError("Expected monthly NDVI columns like `ndvi_m04` in the pixel export.")

    frame["NDVI_max"] = frame[ndvi_monthly].max(axis=1, skipna=True)
    frame["NDVI_min"] = frame[ndvi_monthly].min(axis=1, skipna=True)
    frame["NDVI_median"] = frame[ndvi_monthly].median(axis=1, skipna=True)
    frame["NDVI_integral_proxy"] = frame["ndvi_integral"]
    return frame


def load_pixel_features(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["quarter_section"] = frame["quarter_section"].map(normalize_quarter_id)
    frame["crop_name"] = frame["crop_name"].astype(str).str.strip().replace(CROP_NAME_MAP)
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["quarter_section", "year", "lat", "lon"]).copy()
    frame["year"] = frame["year"].astype(int)

    frame = replace_missing_sentinel(frame)
    frame = derive_ndvi_features(frame)
    frame = frame.rename(
        columns={
            "lat": "latitude",
            "lon": "longitude",
            "crop_name": "aci_crop_name",
        }
    )
    frame["is_target_crop"] = pd.to_numeric(frame["is_target_crop"], errors="coerce").fillna(0).astype(int)
    return frame


def build_training_table(pixel_frame: pd.DataFrame, yield_frame: pd.DataFrame) -> pd.DataFrame:
    merged = pixel_frame.merge(
        yield_frame,
        on=["quarter_section", "year"],
        how="inner",
    ).copy()
    merged["row_id"] = (
        merged["quarter_section"].astype(str)
        + "_"
        + merged["crop"].astype(str).str.replace(" ", "", regex=False)
        + "_"
        + merged["year"].astype(str)
    )
    return merged


def load_existing_lookup(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = ["quarter_section", "crop", "year", "yield_bu_ac"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Lookup CSV is missing required columns: {missing}")
    keep = [column for column in ["quarter_section", "crop", "year", "yield_bu_ac", "yield_original", "yield_original_unit"] if column in frame.columns]
    lookup = frame[keep].drop_duplicates().copy()
    lookup["quarter_section"] = lookup["quarter_section"].map(normalize_quarter_id)
    lookup["crop"] = lookup["crop"].astype(str).str.strip().replace(CROP_NAME_MAP)
    lookup["year"] = pd.to_numeric(lookup["year"], errors="coerce").astype("Int64")
    lookup["yield_bu_ac"] = pd.to_numeric(lookup["yield_bu_ac"], errors="coerce")
    lookup = lookup.dropna(subset=["quarter_section", "crop", "year", "yield_bu_ac"]).copy()
    lookup["year"] = lookup["year"].astype(int)
    if "yield_original" not in lookup.columns:
        lookup["yield_original"] = lookup["yield_bu_ac"]
    if "yield_original_unit" not in lookup.columns:
        lookup["yield_original_unit"] = "bu/ac"
    return lookup


def main() -> None:
    args = build_parser().parse_args()
    pixel_frame = load_pixel_features(args.pixel_input)
    if args.lookup_input.exists():
        yield_frame = load_existing_lookup(args.lookup_input)
    else:
        yield_frame = load_yield_reference(args.yield_input)
    training = build_training_table(pixel_frame, yield_frame)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    training.to_csv(args.output, index=False)

    print(f"Wrote {args.output}")
    print(f"Rows: {len(training)}")
    print(f"Field-years: {training['row_id'].nunique(dropna=True)}")
    print(f"Quarter sections: {training['quarter_section'].nunique(dropna=True)}")


if __name__ == "__main__":
    main()
