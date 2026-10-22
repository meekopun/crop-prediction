"""Build one row per harvested field-year from harvest records and satellite features.

Pilot-eligible operations of the same crop and year that touch a common quarter
section are one field-year: their yield is total production / total harvested
area, and their quarters' features are combined. ``field_<band>_<mon>`` uses the
crop-inventory-masked mean when enough of the land is labelled as the reported
crop, otherwise the whole-quarter mean.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from .harvest import DEFAULT_OUTPUT_DIR
from .harvest_features import DEFAULT_BOUNDARIES_PATH, DEFAULT_FEATURE_DIR, FEATURE_BANDS, MONTHS
from .harvest_weather import DEFAULT_WEATHER_DIR, weather_columns

DEFAULT_MATCHES_PATH = DEFAULT_OUTPUT_DIR / "boundary_matches.csv"
DEFAULT_OPERATIONS_PATH = DEFAULT_OUTPUT_DIR / "operations.csv"
DEFAULT_FEATURES_PATH = DEFAULT_FEATURE_DIR / "harvest_satellite_features.csv"
DEFAULT_WEATHER_PATH = DEFAULT_WEATHER_DIR / "harvest_weather.csv"
DEFAULT_TABLE_PATH = DEFAULT_OUTPUT_DIR / "field_yield_table.csv"
# Season totals over the main growing months, derived from the monthly weather.
SEASON_MONTHS = ("may", "jun", "jul", "aug")
SEASON_WEATHER_COLUMNS = ["w_precip_may_aug", "w_gdd5_may_aug"]
# Below this share of crop-labelled land, the crop mask is treated as unreliable.
CROP_MASK_MIN_FRACTION = 0.1
TOTAL_UNITS = {"bu/ac": "bu", "lb/ac": "lb"}
ID_COLUMNS = ["field_key", "section_key", "township_key", "year", "crop", "yield_unit", "yield",
              "area_ac", "parcel_acres", "harvested_share", "operation_count", "operation_ids",
              "pids", "legal_descriptions", "aci_crop_fraction", "feature_source"]


def table_columns(include_weather: bool = False) -> list[str]:
    columns = list(ID_COLUMNS)
    for month in MONTHS.values():
        columns.append(f"s2_valid_{month}")
        columns.extend(f"field_{band}_{month}" for band in FEATURE_BANDS)
    if include_weather:
        columns.extend(weather_columns() + SEASON_WEATHER_COLUMNS)
    return columns


def _number(text: str | None) -> float | None:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if value == value else None


def _weighted_mean(pairs: list[tuple[float | None, float]]) -> float | None:
    usable = [(value, weight) for value, weight in pairs if value is not None and weight > 0]
    total = sum(weight for _, weight in usable)
    return sum(value * weight for value, weight in usable) / total if total else None


def group_operations(rows: list[dict]) -> list[list[dict]]:
    """Group same-crop, same-year operations that share any quarter section."""
    groups = []
    by_crop_year: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_crop_year[(row["crop"], row["year"])].append(row)
    for members in by_crop_year.values():
        parent = list(range(len(members)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        owner: dict[str, int] = {}
        for index, row in enumerate(members):
            for pid in row["pids"].split("|"):
                if pid in owner:
                    parent[find(index)] = find(owner[pid])
                else:
                    owner[pid] = index
        components: dict[int, list[dict]] = defaultdict(list)
        for index, row in enumerate(members):
            components[find(index)].append(row)
        groups.extend(components.values())
    return sorted(groups, key=lambda group: (group[0]["year"], group[0]["crop"], group[0]["pids"]))


def build_field_yield_table(matches_path: Path | str = DEFAULT_MATCHES_PATH,
                            operations_path: Path | str = DEFAULT_OPERATIONS_PATH,
                            features_path: Path | str = DEFAULT_FEATURES_PATH,
                            boundaries_path: Path | str = DEFAULT_BOUNDARIES_PATH,
                            output_path: Path | str = DEFAULT_TABLE_PATH,
                            weather_path: Path | str | None = DEFAULT_WEATHER_PATH) -> dict:
    """Write the field-year table and a summary beside it; return the summary.

    Weather columns are added when ``weather_path`` exists; pass None to omit them.
    """
    with Path(matches_path).open(encoding="utf-8", newline="") as handle:
        eligible = [row for row in csv.DictReader(handle) if row["boundary_pilot_eligible"] == "True"]
    with Path(operations_path).open(encoding="utf-8", newline="") as handle:
        totals = {row["operation_id"]: row for row in csv.DictReader(handle)}
    with Path(features_path).open(encoding="utf-8", newline="") as handle:
        features = {(row["pid"], row["year"]): row for row in csv.DictReader(handle)}
    with Path(boundaries_path).open(encoding="utf-8") as handle:
        parcel_acres = {feature["properties"]["pid"]: feature["properties"]["parcel_acres"]
                        for feature in json.load(handle)["features"]}
    weather = None
    if weather_path is not None and Path(weather_path).exists():
        with Path(weather_path).open(encoding="utf-8", newline="") as handle:
            weather = {(row["pid"], row["year"]): row for row in csv.DictReader(handle)}

    rows, dropped = [], Counter()
    for group in group_operations(eligible):
        first = group[0]
        units = {row["yield_unit"] for row in group}
        pids = sorted({pid for row in group for pid in row["pids"].split("|")})
        if len(units) != 1 or first["yield_unit"] not in TOTAL_UNITS:
            dropped["mixed_or_unsupported_units"] += 1
            continue
        if any((pid, first["year"]) not in features for pid in pids):
            dropped["missing_satellite_features"] += 1
            continue
        area = sum(_number(row["area_ac"]) or 0.0 for row in group)
        production = 0.0
        for row in group:
            source = totals.get(row["operation_id"], {})
            total = _number(source.get("total_value"))
            if total is None or source.get("total_unit") != TOTAL_UNITS[first["yield_unit"]]:
                total = (_number(row["area_ac"]) or 0.0) * (_number(row["yield_value"]) or 0.0)
            production += total
        if area <= 0:
            dropped["no_harvested_area"] += 1
            continue
        acres = {pid: float(parcel_acres[pid]) for pid in pids}
        quarters = [features[(pid, first["year"])] for pid in pids]
        crop_weights = [(_number(row["aci_crop_fraction"]) or 0.0) * acres[row["pid"]] for row in quarters]
        fraction = sum(crop_weights) / sum(acres.values())
        use_crop_mask = fraction >= CROP_MASK_MIN_FRACTION
        out = {
            "field_key": "|".join(pids),
            "section_key": "|".join(sorted({pid[:8] for pid in pids})),
            "township_key": "|".join(sorted({pid[:6] for pid in pids})),
            "year": first["year"], "crop": first["crop"], "yield_unit": first["yield_unit"],
            "yield": round(production / area, 4), "area_ac": round(area, 2),
            "parcel_acres": round(sum(acres.values()), 1),
            "harvested_share": round(area / sum(acres.values()), 3),
            "operation_count": len(group),
            "operation_ids": "|".join(row["operation_id"] for row in group),
            "pids": "|".join(pids),
            "legal_descriptions": "|".join(sorted({row["legal_description"] for row in group})),
            "aci_crop_fraction": round(fraction, 4),
            "feature_source": "crop_mask" if use_crop_mask else "quarter",
        }
        for month in MONTHS.values():
            out[f"s2_valid_{month}"] = _weighted_mean(
                [(_number(row[f"s2_valid_{month}"]), acres[row["pid"]]) for row in quarters])
            for band in FEATURE_BANDS:
                quarter_value = _weighted_mean(
                    [(_number(row[f"quarter_{band}_{month}"]), acres[row["pid"]]) for row in quarters])
                crop_value = _weighted_mean(
                    [(_number(row[f"crop_{band}_{month}"]), weight) for row, weight in zip(quarters, crop_weights)])
                # A month without crop-pixel data falls back to the whole quarter.
                out[f"field_{band}_{month}"] = crop_value if use_crop_mask and crop_value is not None else quarter_value
        if weather is not None:
            records = [(weather.get((pid, first["year"])), acres[pid]) for pid in pids]
            for column in weather_columns():
                out[column] = _weighted_mean([(_number(record[column]) if record else None, weight)
                                              for record, weight in records])
            for column, variable in zip(SEASON_WEATHER_COLUMNS, ("precip", "gdd5")):
                values = [out[f"w_{variable}_{month}"] for month in SEASON_MONTHS]
                out[column] = sum(values) if all(value is not None for value in values) else None
        rows.append(out)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=table_columns(weather is not None), restval="")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": 1,
        "table_path": str(destination.resolve()),
        "eligible_operations": len(eligible),
        "field_year_count": len(rows),
        "dropped_groups": dict(sorted(dropped.items())),
        "field_years_by_crop_unit": dict(sorted(Counter(f"{row['crop']} ({row['yield_unit']})" for row in rows).items())),
        "field_years_by_year": dict(sorted(Counter(row["year"] for row in rows).items())),
        "feature_source": dict(sorted(Counter(row["feature_source"] for row in rows).items())),
        "crop_mask_min_fraction": CROP_MASK_MIN_FRACTION,
        "weather_path": str(Path(weather_path).resolve()) if weather is not None else None,
        "notes": [
            "yield = total production / total harvested area over the grouped operations.",
            "field_* uses crop-mask means when aci_crop_fraction >= crop_mask_min_fraction, "
            "else whole-quarter means; months without crop pixels fall back to the quarter.",
            "Units are kept as reported; never pool bu/ac and lb/ac crops.",
        ],
    }
    destination.with_name(destination.stem + "_summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                                         encoding="utf-8")
    return summary
