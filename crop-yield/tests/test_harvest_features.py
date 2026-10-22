import csv
import json
from pathlib import Path
import tempfile
import unittest

from yield_prediction.harvest_features import (
    CROP_ACI_CODES, FEATURE_BANDS, MONTHS, feature_columns, load_feature_units,
)

MATCH_COLUMNS = ["operation_id", "year", "crop", "pids", "boundary_pilot_eligible"]
SQUARE = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}


class FeatureUnitTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.boundaries = self.root / "field_boundaries.geojson"
        self.boundaries.write_text(json.dumps({"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {"pid": pid}, "geometry": SQUARE}
            for pid in ("61007215800", "61007215700", "60907130500")
        ]}), encoding="utf-8")

    def tearDown(self):
        self.directory.cleanup()

    def write_matches(self, rows):
        path = self.root / "boundary_matches.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MATCH_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_units_split_half_sections_and_group_same_crop_fields(self):
        matches = self.write_matches([
            {"operation_id": "half", "year": "2021", "crop": "canola",
             "pids": "61007215800|61007215700", "boundary_pilot_eligible": "True"},
            {"operation_id": "field-a", "year": "2021", "crop": "barley",
             "pids": "60907130500", "boundary_pilot_eligible": "True"},
            {"operation_id": "field-b", "year": "2021", "crop": "barley",
             "pids": "60907130500", "boundary_pilot_eligible": "True"},
            {"operation_id": "shared", "year": "2022", "crop": "oats",
             "pids": "60907130500", "boundary_pilot_eligible": "False"},
        ])
        units = load_feature_units(matches, self.boundaries)
        self.assertEqual([(unit.pid, unit.year, unit.crop) for unit in units], [
            ("60907130500", 2021, "barley"),
            ("61007215700", 2021, "canola"),
            ("61007215800", 2021, "canola"),
        ])
        self.assertEqual(units[0].operation_ids, ("field-a", "field-b"))
        self.assertEqual(units[0].aci_codes, (133,))
        self.assertEqual(units[0].geometry, SQUARE)

    def test_conflicting_eligible_crops_or_missing_geometry_fail(self):
        conflicting = self.write_matches([
            {"operation_id": "a", "year": "2021", "crop": "barley", "pids": "60907130500",
             "boundary_pilot_eligible": "True"},
            {"operation_id": "b", "year": "2021", "crop": "oats", "pids": "60907130500",
             "boundary_pilot_eligible": "True"},
        ])
        with self.assertRaises(ValueError):
            load_feature_units(conflicting, self.boundaries)
        missing = self.write_matches([
            {"operation_id": "a", "year": "2021", "crop": "barley", "pids": "99999999900",
             "boundary_pilot_eligible": "True"},
        ])
        with self.assertRaises(ValueError):
            load_feature_units(missing, self.boundaries)

    def test_columns_cover_every_month_band_and_mask(self):
        columns = feature_columns()
        self.assertEqual(len(columns), len(set(columns)))
        self.assertEqual(len(columns), 6 + len(MONTHS) * (1 + 2 * len(FEATURE_BANDS)))
        self.assertIn("crop_NDVI_jul", columns)
        self.assertIn("quarter_VV_minus_VH_oct", columns)
        self.assertIn("s2_valid_apr", columns)

    def test_every_harvest_crop_has_inventory_codes(self):
        harvest_crops = {"barley", "canola", "faba_beans", "fescue", "field_peas", "oats", "rye",
                         "ryegrass", "sunflower", "tall_fescue", "triticale", "wheat"}
        self.assertEqual(set(CROP_ACI_CODES), harvest_crops)
