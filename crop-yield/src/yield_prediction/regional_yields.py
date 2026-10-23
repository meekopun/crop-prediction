"""Convert Statistics Canada small-area field-crop yields into a tidy regional table.

Source: Statistics Canada table 32-10-0002-01, "Estimated areas, yield and
production of principal field crops by Small Area Data Regions" (full-table CSV
zip from https://www150.statcan.gc.ca/n1/tbl/csv/32100002-eng.zip, Open
Government Licence - Canada). The Beaverlodge harvest fields lie in Alberta Small
Area Data Region 70 (Census Agricultural Region 7, the Peace region).

The zip is read in place and streamed; only matching rows are kept.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
import zipfile

from .harvest import PACKAGE_ROOT

DEFAULT_STATCAN_ZIP = PACKAGE_ROOT / "data/raw/statcan/32100002-eng.zip"
DEFAULT_REGIONAL_PATH = PACKAGE_ROOT / "data/processed/regional/regional_yields.csv"
DEFAULT_REGION = "Small Area Data Region 70 - Alberta"
YIELD_DISPOSITION = "Average yield (bushels per acre)"
# Harvest crop name -> table crop. The farm's wheat is Stettler (hard red spring).
STATCAN_CROPS = {"barley": "Barley", "canola": "Canola", "field_peas": "Peas, dry",
                 "oats": "Oats", "wheat": "Wheat, spring"}
REGIONAL_COLUMNS = ["region", "year", "crop", "regional_yield", "unit", "source", "status"]
SOURCE = "Statistics Canada table 32-10-0002-01"


def _rows(path: Path):
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            name = next(member for member in archive.namelist()
                        if member.endswith(".csv") and "MetaData" not in member)
            with archive.open(name) as raw:
                yield from csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
    else:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            yield from csv.DictReader(handle)


def prepare_regional_yields(source_path: Path | str = DEFAULT_STATCAN_ZIP,
                            output_path: Path | str = DEFAULT_REGIONAL_PATH,
                            *, region: str = DEFAULT_REGION, first_year: int = 2015) -> list[dict]:
    """Write one row per crop and year for ``region``; return the rows.

    Rows without a published value (suppressed or not available) are skipped.
    """
    by_table_crop = {table: crop for crop, table in STATCAN_CROPS.items()}
    rows = []
    for record in _rows(Path(source_path)):
        if (record.get("GEO") != region or record.get("Harvest disposition") != YIELD_DISPOSITION
                or record.get("Type of crop") not in by_table_crop):
            continue
        year = int(str(record["REF_DATE"])[:4])
        value = (record.get("VALUE") or "").strip()
        if year < first_year or not value:
            continue
        rows.append({"region": region, "year": year, "crop": by_table_crop[record["Type of crop"]],
                     "regional_yield": float(value), "unit": "bu/ac", "source": SOURCE,
                     "status": (record.get("STATUS") or "").strip()})
    if not rows:
        raise ValueError(f"No '{YIELD_DISPOSITION}' rows for {region!r} in {source_path}")
    rows.sort(key=lambda row: (row["crop"], row["year"]))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGIONAL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows
