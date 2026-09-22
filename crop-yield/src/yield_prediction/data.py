"""Read the historical CSV without requiring optional ML packages."""

import csv
import math
from pathlib import Path

TARGET = "hg/ha_yield"
NUMERIC_FEATURES = ["Year", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]
CATEGORICAL_FEATURES = ["Area", "Item"]
COLUMNS = [*CATEGORICAL_FEATURES, *NUMERIC_FEATURES, TARGET]


def load_data(path: Path | str) -> list[dict]:
    rows = []
    with Path(path).open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        missing = set(COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        for line, record in enumerate(reader, start=2):
            try:
                row = {name: record[name].strip() for name in CATEGORICAL_FEATURES}
                if not all(row.values()):
                    raise ValueError("Area and Item must be nonempty")
                for name in [*NUMERIC_FEATURES, TARGET]:
                    value = float(record[name])
                    if not math.isfinite(value):
                        raise ValueError(f"{name} must be finite")
                    row[name] = value
                if not row["Year"].is_integer():
                    raise ValueError("Year must be an integer")
                row["Year"] = int(row["Year"])
                rows.append(row)
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError(f"Invalid data on CSV line {line}: {exc}") from exc
    if not rows:
        raise ValueError("Dataset contains no records")
    return rows


def summarize(rows: list[dict]) -> dict:
    return {
        "rows": len(rows),
        "areas": len({row["Area"] for row in rows}),
        "items": len({row["Item"] for row in rows}),
        "year_min": min(row["Year"] for row in rows),
        "year_max": max(row["Year"] for row in rows),
        "duplicate_records": len(rows) - len({tuple(row[name] for name in COLUMNS) for row in rows}),
    }
