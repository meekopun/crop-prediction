from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = PACKAGE_ROOT / "data" / "raw" / "yield_df.csv"


@dataclass(frozen=True)
class DatasetSummary:
    rows: int
    columns: tuple[str, ...]
    areas: int
    items: int
    year_min: int
    year_max: int


def load_records(path: Path | str = DEFAULT_DATASET_PATH) -> list[dict[str, str]]:
    dataset_path = Path(path)
    with dataset_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def clean_records(records: Iterable[dict[str, str]]) -> list[dict[str, object]]:
    cleaned: list[dict[str, object]] = []
    for row in records:
        normalized = dict(row)
        unnamed_value = normalized.pop("", None)
        if unnamed_value is None:
            normalized.pop("Unnamed: 0", None)

        cleaned.append(
            {
                "Area": normalized["Area"],
                "Item": normalized["Item"],
                "Year": int(normalized["Year"]),
                "hg/ha_yield": float(normalized["hg/ha_yield"]),
                "average_rain_fall_mm_per_year": float(
                    normalized["average_rain_fall_mm_per_year"]
                ),
                "pesticides_tonnes": float(normalized["pesticides_tonnes"]),
                "avg_temp": float(normalized["avg_temp"]),
            }
        )
    return cleaned


def summarize_records(records: Iterable[dict[str, object]]) -> DatasetSummary:
    rows = list(records)
    years = [int(row["Year"]) for row in rows]
    return DatasetSummary(
        rows=len(rows),
        columns=tuple(rows[0].keys()) if rows else (),
        areas=len({str(row["Area"]) for row in rows}),
        items=len({str(row["Item"]) for row in rows}),
        year_min=min(years) if years else 0,
        year_max=max(years) if years else 0,
    )
