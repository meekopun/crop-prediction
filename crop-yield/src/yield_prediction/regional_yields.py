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
DEFAULT_ALBERTA_YIELDS_PATH = PACKAGE_ROOT / "data/processed/regional/alberta_regional_yields.csv"
DEFAULT_REGION = "Small Area Data Region 70 - Alberta"
# Alberta small-area regions follow Census Agricultural Regions; each is a set of
# census divisions (Statistics Canada CAR/CD maps, 1996, 2001 and 2021).
ALBERTA_REGION_DIVISIONS = {
    "Small Area Data Region 10 - Alberta": (1, 4),           # CAR 1
    "Small Area Data Region 20 - Alberta": (2, 5),           # CAR 2
    "Small Area Data Region 30 - Alberta": (3, 6, 15),       # CAR 3
    "Small Area Data Region 40 - Alberta": (7,),             # CAR 4A
    "Small Area Data Region 41 - Alberta": (10,),            # CAR 4B
    "Small Area Data Region 50 - Alberta": (8, 9, 11),       # CAR 5
    "Small Area Data Region 60 - Alberta": (12, 13, 14, 16),  # CAR 6
    "Small Area Data Region 70 - Alberta": (17, 18, 19),     # CAR 7 (Peace)
}
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
                            *, region: str | tuple[str, ...] = DEFAULT_REGION,
                            first_year: int = 2015) -> list[dict]:
    """Write one row per region, crop and year; return the rows.

    ``region`` is one GEO name or several. Rows without a published value
    (suppressed or not available) are skipped.
    """
    regions = {region} if isinstance(region, str) else set(region)
    by_table_crop = {table: crop for crop, table in STATCAN_CROPS.items()}
    rows = []
    for record in _rows(Path(source_path)):
        if (record.get("GEO") not in regions or record.get("Harvest disposition") != YIELD_DISPOSITION
                or record.get("Type of crop") not in by_table_crop):
            continue
        year = int(str(record["REF_DATE"])[:4])
        value = (record.get("VALUE") or "").strip()
        if year < first_year or not value:
            continue
        rows.append({"region": record["GEO"], "year": year, "crop": by_table_crop[record["Type of crop"]],
                     "regional_yield": float(value), "unit": "bu/ac", "source": SOURCE,
                     "status": (record.get("STATUS") or "").strip()})
    if not rows:
        raise ValueError(f"No '{YIELD_DISPOSITION}' rows for {sorted(regions)} in {source_path}")
    rows.sort(key=lambda row: (row["region"], row["crop"], row["year"]))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGIONAL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows
