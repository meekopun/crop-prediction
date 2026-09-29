from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import fmean
from typing import Iterable


MISSING_SENTINEL = -9999.0

AAFC_CLASS_NAMES: dict[int, str] = {
    110: "Grassland",
    120: "Agriculture",
    122: "Pasture and Forages",
    131: "Fallow",
    132: "Cereals",
    133: "Barley",
    136: "Oats",
    140: "Wheat",
    145: "Winter Wheat",
    146: "Spring Wheat",
    147: "Corn",
    150: "Oilseeds",
    153: "Canola and Rapeseed",
    155: "Mustard",
    158: "Soybeans",
    160: "Pulses",
    162: "Peas",
    174: "Lentils",
    177: "Potatoes",
    195: "Buckwheat",
    197: "Hemp",
    199: "Other Crops",
}


def parse_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if parsed == MISSING_SENTINEL:
        return None
    return parsed


def parse_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def mean_ignore_missing(values: Iterable[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return fmean(valid)


def max_ignore_missing(values: Iterable[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return max(valid)


def compute_zscores(rows: list[dict[str, object]], field: str) -> None:
    numeric = [
        row[field]
        for row in rows
        if isinstance(row.get(field), (int, float)) and row[field] is not None
    ]
    zfield = f"{field}_z"
    if len(numeric) < 2:
        for row in rows:
            row[zfield] = 0.0
        return

    mean_value = fmean(numeric)
    variance = fmean([(value - mean_value) ** 2 for value in numeric])
    std_value = math.sqrt(variance)
    if std_value == 0:
        for row in rows:
            row[zfield] = 0.0
        return

    for row in rows:
        value = row.get(field)
        if isinstance(value, (int, float)) and value is not None:
            row[zfield] = (value - mean_value) / std_value
        else:
            row[zfield] = 0.0


def infer_crop(row: dict[str, str], threshold: float) -> str | None:
    crop_fields = [
        ("canola", parse_float(row.get("aafc_canola_fraction"))),
        ("spring_wheat", parse_float(row.get("aafc_spring_wheat_fraction"))),
        ("wheat", parse_float(row.get("aafc_wheat_fraction"))),
        ("barley", parse_float(row.get("aafc_barley_fraction"))),
        ("oats", parse_float(row.get("aafc_oats_fraction"))),
        ("peas", parse_float(row.get("aafc_peas_fraction"))),
        ("lentils", parse_float(row.get("aafc_lentils_fraction"))),
    ]
    best_crop, best_fraction = max(crop_fields, key=lambda item: item[1] or 0.0)
    if best_fraction is None or best_fraction < threshold:
        return None
    return best_crop


def derive_row_features(row: dict[str, str], crop_threshold: float) -> dict[str, object] | None:
    crop_guess = infer_crop(row, crop_threshold)
    if crop_guess is None:
        return None

    peak_ndvi = max_ignore_missing(
        parse_float(row.get(name))
        for name in [
            "may_jun_NDVI_mean",
            "late_jun_jul_NDVI_mean",
            "aug_NDVI_mean",
            "sep_NDVI_mean",
        ]
    )
    peak_evi = max_ignore_missing(
        parse_float(row.get(name))
        for name in [
            "may_jun_EVI_mean",
            "late_jun_jul_EVI_mean",
            "aug_EVI_mean",
            "sep_EVI_mean",
        ]
    )
    peak_ndre1 = max_ignore_missing(
        parse_float(row.get(name))
        for name in [
            "may_jun_NDRE1_mean",
            "late_jun_jul_NDRE1_mean",
            "aug_NDRE1_mean",
            "sep_NDRE1_mean",
        ]
    )
    seasonal_ndmi = mean_ignore_missing(
        parse_float(row.get(name))
        for name in [
            "may_jun_NDMI_mean",
            "late_jun_jul_NDMI_mean",
            "aug_NDMI_mean",
            "sep_NDMI_mean",
        ]
    )
    seasonal_msi = mean_ignore_missing(
        parse_float(row.get(name))
        for name in [
            "may_jun_MSI_mean",
            "late_jun_jul_MSI_mean",
            "aug_MSI_mean",
            "sep_MSI_mean",
        ]
    )
    sep_ndvi = parse_float(row.get("sep_NDVI_mean"))
    late_decline = None
    if peak_ndvi is not None and sep_ndvi is not None:
        late_decline = peak_ndvi - sep_ndvi

    mean_smap_moisture = mean_ignore_missing(
        parse_float(row.get(name))
        for name in [
            "may_jun_smap_soil_moisture_am_mean",
            "late_jun_jul_smap_soil_moisture_am_mean",
            "aug_smap_soil_moisture_am_mean",
            "sep_smap_soil_moisture_am_mean",
        ]
    )
    coverage_score = mean_ignore_missing(
        parse_float(row.get(name))
        for name in [
            "may_jun_obs_count",
            "late_jun_jul_obs_count",
            "aug_obs_count",
            "sep_obs_count",
        ]
    )

    return {
        "quarter_id": row.get("quarter_id"),
        "year": parse_int(row.get("year")),
        "crop_guess": crop_guess,
        "aafc_dominant_class": parse_int(row.get("aafc_dominant_class")),
        "aafc_dominant_crop_name": AAFC_CLASS_NAMES.get(
            parse_int(row.get("aafc_dominant_class")) or -1,
            "Unknown",
        ),
        "peak_ndvi": peak_ndvi,
        "peak_evi": peak_evi,
        "peak_ndre1": peak_ndre1,
        "seasonal_ndmi": seasonal_ndmi,
        "seasonal_msi": seasonal_msi,
        "late_decline": late_decline,
        "mean_smap_moisture": mean_smap_moisture,
        "coverage_score": coverage_score,
    }


def assign_index(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    fields = [
        "peak_ndvi",
        "peak_evi",
        "peak_ndre1",
        "seasonal_ndmi",
        "seasonal_msi",
        "late_decline",
        "mean_smap_moisture",
    ]

    within_year_groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["crop_guess"]), int(row["year"]))
        within_year_groups.setdefault(key, []).append(row)

    for group_rows in within_year_groups.values():
        for field in fields:
            compute_zscores(group_rows, field)

        for row in group_rows:
            index = (
                0.25 * float(row["peak_ndvi_z"])
                + 0.20 * float(row["peak_evi_z"])
                + 0.20 * float(row["peak_ndre1_z"])
                + 0.15 * float(row["seasonal_ndmi_z"])
                + 0.10 * float(row["mean_smap_moisture_z"])
                - 0.10 * float(row["seasonal_msi_z"])
            )
            row["yield_potential_index_within_year"] = index

        sorted_group = sorted(group_rows, key=lambda item: float(item["yield_potential_index_within_year"]))
        total = len(sorted_group)
        for rank, row in enumerate(sorted_group, start=1):
            percentile = rank / total if total else 0.0
            row["yield_potential_percentile_within_year"] = percentile
            if percentile <= 0.33:
                row["yield_potential_class_within_year"] = "low"
            elif percentile <= 0.67:
                row["yield_potential_class_within_year"] = "medium"
            else:
                row["yield_potential_class_within_year"] = "high"

    across_year_groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        across_year_groups.setdefault(str(row["crop_guess"]), []).append(row)

    for group_rows in across_year_groups.values():
        for field in fields:
            compute_zscores(group_rows, field)

        for row in group_rows:
            index = (
                0.25 * float(row["peak_ndvi_z"])
                + 0.20 * float(row["peak_evi_z"])
                + 0.20 * float(row["peak_ndre1_z"])
                + 0.15 * float(row["seasonal_ndmi_z"])
                + 0.10 * float(row["mean_smap_moisture_z"])
                - 0.10 * float(row["seasonal_msi_z"])
            )
            row["yield_potential_index_across_years"] = index

        sorted_group = sorted(group_rows, key=lambda item: float(item["yield_potential_index_across_years"]))
        total = len(sorted_group)
        for rank, row in enumerate(sorted_group, start=1):
            percentile = rank / total if total else 0.0
            row["yield_potential_percentile_across_years"] = percentile
            if percentile <= 0.33:
                row["yield_potential_class_across_years"] = "low"
            elif percentile <= 0.67:
                row["yield_potential_class_across_years"] = "medium"
            else:
                row["yield_potential_class_across_years"] = "high"

    return rows


def write_summary(path: Path | str, rows: list[dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["quarter_id"]), str(row["crop_guess"]))
        groups.setdefault(key, []).append(row)

    summary_rows = []
    for (quarter_id, crop_guess), group_rows in sorted(groups.items()):
        percentiles = [
            float(row["yield_potential_percentile_within_year"])
            for row in group_rows
            if row.get("yield_potential_percentile_within_year") is not None
        ]
        across_values = [
            float(row["yield_potential_index_across_years"])
            for row in group_rows
            if row.get("yield_potential_index_across_years") is not None
        ]
        best = max(group_rows, key=lambda row: float(row["yield_potential_percentile_within_year"]))
        worst = min(group_rows, key=lambda row: float(row["yield_potential_percentile_within_year"]))
        summary_rows.append(
            {
                "quarter_id": quarter_id,
                "crop_guess": crop_guess,
                "num_years": len(group_rows),
                "mean_within_year_percentile": fmean(percentiles) if percentiles else "",
                "mean_across_year_index": fmean(across_values) if across_values else "",
                "best_year": best["year"],
                "best_within_year_percentile": best["yield_potential_percentile_within_year"],
                "worst_year": worst["year"],
                "worst_within_year_percentile": worst["yield_potential_percentile_within_year"],
            }
        )

    fieldnames = [
        "quarter_id",
        "crop_guess",
        "num_years",
        "mean_within_year_percentile",
        "mean_across_year_index",
        "best_year",
        "best_within_year_percentile",
        "worst_year",
        "worst_within_year_percentile",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def write_rows(path: Path | str, rows: list[dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "quarter_id",
        "year",
        "crop_guess",
        "aafc_dominant_class",
        "aafc_dominant_crop_name",
        "peak_ndvi",
        "peak_evi",
        "peak_ndre1",
        "seasonal_ndmi",
        "seasonal_msi",
        "late_decline",
        "mean_smap_moisture",
        "coverage_score",
        "yield_potential_index_within_year",
        "yield_potential_percentile_within_year",
        "yield_potential_class_within_year",
        "yield_potential_index_across_years",
        "yield_potential_percentile_across_years",
        "yield_potential_class_across_years",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a relative yield potential index from GEE features.")
    parser.add_argument("input_csv", type=Path, help="Path to the GEE export CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/yield_potential_index.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/processed/yield_potential_summary.csv"),
        help="Quarter-section summary CSV path",
    )
    parser.add_argument(
        "--crop-threshold",
        type=float,
        default=0.6,
        help="Minimum AAFC crop fraction required to assign a crop",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    with args.input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        raw_rows = list(reader)

    derived_rows = []
    for row in raw_rows:
        derived = derive_row_features(row, args.crop_threshold)
        if derived is not None:
            derived_rows.append(derived)

    indexed = assign_index(derived_rows)
    write_rows(args.output, indexed)
    write_summary(args.summary_output, indexed)

    print(f"Wrote {len(indexed)} rows to {args.output}")
    print(f"Wrote summary to {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
