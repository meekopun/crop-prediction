import csv
import io
from pathlib import Path
import tempfile
import unittest
import zipfile

from yield_prediction.regional_yields import DEFAULT_REGION, YIELD_DISPOSITION, prepare_regional_yields

COLUMNS = ["REF_DATE", "GEO", "DGUID", "Harvest disposition", "Type of crop", "UOM", "UOM_ID",
           "SCALAR_FACTOR", "SCALAR_ID", "VECTOR", "COORDINATE", "VALUE", "STATUS", "SYMBOL",
           "TERMINATED", "DECIMALS"]


def record(year, geo, disposition, crop, value, status=""):
    return {"REF_DATE": str(year), "GEO": geo, "DGUID": "", "Harvest disposition": disposition,
            "Type of crop": crop, "UOM": "Bushels per acre", "UOM_ID": "", "SCALAR_FACTOR": "units",
            "SCALAR_ID": "0", "VECTOR": "v1", "COORDINATE": "1.1.1", "VALUE": value, "STATUS": status,
            "SYMBOL": "", "TERMINATED": "", "DECIMALS": "1"}


class RegionalYieldTests(unittest.TestCase):
    def test_keeps_only_the_region_yield_rows_for_mapped_crops(self):
        rows = [
            record(2021, DEFAULT_REGION, YIELD_DISPOSITION, "Canola", "21.5"),
            record(2022, DEFAULT_REGION, YIELD_DISPOSITION, "Wheat, spring", "48.0"),
            record(2022, DEFAULT_REGION, YIELD_DISPOSITION, "Wheat, all", "47.0"),     # not mapped
            record(2022, DEFAULT_REGION, YIELD_DISPOSITION, "Peas, dry", "", "x"),      # suppressed
            record(2022, DEFAULT_REGION, "Seeded area (acres)", "Canola", "900000"),   # wrong measure
            record(2022, "Small Area Data Region 10 - Alberta", YIELD_DISPOSITION, "Canola", "40.0"),
            record(2010, DEFAULT_REGION, YIELD_DISPOSITION, "Barley", "60.0"),          # before first year
        ]
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "32100002-eng.zip"
            text = io.StringIO()
            writer = csv.DictWriter(text, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("32100002.csv", "﻿" + text.getvalue())
                archive.writestr("32100002_MetaData.csv", "not data")
            output = Path(directory) / "regional.csv"
            result = prepare_regional_yields(archive_path, output)
            with output.open(newline="", encoding="utf-8") as handle:
                written = list(csv.DictReader(handle))
        self.assertEqual([(row["crop"], row["year"], row["regional_yield"]) for row in result],
                         [("canola", 2021, 21.5), ("wheat", 2022, 48.0)])
        self.assertEqual(len(written), 2)
        self.assertEqual(written[0]["unit"], "bu/ac")

    def test_no_matching_rows_fails_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "table.csv"
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerow(record(2021, "Alberta", YIELD_DISPOSITION, "Canola", "35.0"))
            with self.assertRaises(ValueError):
                prepare_regional_yields(source, Path(directory) / "out.csv")
