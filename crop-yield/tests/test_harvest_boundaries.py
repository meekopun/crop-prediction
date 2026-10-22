import csv
import json
from pathlib import Path
import tempfile
import unittest

from yield_prediction.harvest_boundaries import (
    find_quarter_sections, match_harvest_boundaries, parse_legal_description, polygon_acres,
)

# About half a mile per side at 55°N, so each square is close to 160 acres.
SIDE_LAT = 0.0072372
SIDE_LON = 0.0126180


def square(lon, lat):
    ring = [[lon, lat], [lon + SIDE_LON, lat], [lon + SIDE_LON, lat + SIDE_LAT],
            [lon, lat + SIDE_LAT], [lon, lat]]
    return {"type": "MultiPolygon", "coordinates": [[ring]]}


def feature(pid, qs, ra=None, offset=0):
    properties = {"m": int(pid[0]), "rge": int(pid[1:3]), "twp": int(pid[3:6]),
                  "sec": int(pid[6:8]), "qs": qs, "ra": ra, "pid": pid,
                  "descriptor": f"QS-{qs} SEC-{pid[6:8]} TWP-{pid[3:6]} RGE-{pid[1:3]} MER-{pid[0]}"}
    return {"type": "Feature", "properties": properties, "geometry": square(-119.0 + offset * 0.02, 55.0)}


def write_gdal_geojson(path, features):
    # GDAL layout: header lines, then one feature per line separated by commas.
    lines = ['{', '"type": "FeatureCollection",', '"name": "quarter_sections",',
             '"crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:EPSG::4269" } },',
             '"features": [']
    body = [json.dumps(item).replace('{"type": "Feature"', '{ "type": "Feature"', 1) for item in features]
    lines.append(",\n".join(body))
    lines.extend([']', '}'])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


OPERATION_COLUMNS = ["operation_id", "source_file", "source_row", "year", "crop", "crop_raw",
                     "field_name_raw", "legal_description_raw", "area_ac", "yield_value",
                     "yield_unit", "processing_status"]


def operation(op_id, legal, year, crop, area, status="pending_boundary_review"):
    return {"operation_id": op_id, "source_file": f"{year}-harvest-data.xls", "source_row": "5",
            "year": str(year), "crop": crop, "crop_raw": crop.title(), "field_name_raw": op_id,
            "legal_description_raw": legal, "area_ac": str(area), "yield_value": "50",
            "yield_unit": "bu/ac", "processing_status": status}


class LegalDescriptionTests(unittest.TestCase):
    def test_parses_quarter_half_and_compact_descriptions(self):
        self.assertEqual(parse_legal_description("SE 30-71-9 W6M").pids, ("60907130500",))
        self.assertEqual(parse_legal_description("NE12-71-10W6M").pids, ("61007112800",))
        self.assertEqual(parse_legal_description("SE13.71-10 W6m").pids, ("61007113500",))
        self.assertEqual(parse_legal_description("SE  31-71-9 W6M").label, "SE 31-71-9 W6M")
        self.assertEqual(parse_legal_description("N1/2 15-72-10 W6M").pids,
                         ("61007215800", "61007215700"))
        self.assertEqual(parse_legal_description("S1/2 19-70-8 W6M").pids,
                         ("60807019500", "60807019600"))

    def test_rejects_names_and_impossible_sections(self):
        self.assertIsNone(parse_legal_description("Youngs"))
        self.assertIsNone(parse_legal_description(""))
        self.assertIsNone(parse_legal_description("SE 40-71-9 W6M"))


class BoundaryMatchingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.geometry = self.root / "quarters.geojson"
        write_gdal_geojson(self.geometry, [
            feature("60907130501", "SE", ra="RS"),  # road allowance next to the quarter
            feature("60907130500", "SE"),
            feature("61007215800", "NE", offset=1),
            feature("61007215700", "NW", offset=2),
            feature("60907131600", "SW", offset=3),
        ])

    def tearDown(self):
        self.directory.cleanup()

    def test_streaming_returns_only_wanted_farmland_quarters(self):
        found, crs = find_quarter_sections(self.geometry, {"60907130500", "60907130501", "99999999999"})
        self.assertEqual(set(found), {"60907130500"})
        self.assertEqual(crs, "urn:ogc:def:crs:EPSG::4269")
        self.assertAlmostEqual(polygon_acres(found["60907130500"]["geometry"]), 160.0, delta=2.0)

    def test_rejects_geojson_that_is_not_one_feature_per_line(self):
        compact = self.root / "compact.geojson"
        compact.write_text(json.dumps({"type": "FeatureCollection",
                                       "features": [feature("60907130500", "SE")]}), encoding="utf-8")
        with self.assertRaises(ValueError):
            find_quarter_sections(compact, {"60907130500"})

    def test_match_report_flags_shared_and_oversized_parcels(self):
        operations = [
            operation("oats-2020", "SE 30-71-9 W6M", 2020, "oats", 50),
            operation("barley-2020", "SE 30-71-9 W6M", 2020, "barley", 82),
            operation("barley-2021", "SE 30-71-9 W6M", 2021, "barley", 200),
            operation("canola-half", "N1/2 15-72-10 W6M", 2021, "canola", 300),
            operation("wheat-missing", "NE 9-72-10 W6M", 2021, "wheat", 120),
            operation("wheat-excluded", "SW 31-71-9 W6M", 2022, "wheat", 40, status="excluded"),
            operation("canola-shared", "SW 31-71-9 W6M", 2022, "canola", 100),
            operation("name-only", "", 2022, "oats", 30, status="excluded"),
        ]
        operations_path = self.root / "operations.csv"
        with operations_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=OPERATION_COLUMNS)
            writer.writeheader()
            writer.writerows(operations)
        output = self.root / "out"
        summary = match_harvest_boundaries(operations_path, self.geometry, output)

        with (output / "boundary_matches.csv").open(newline="", encoding="utf-8") as handle:
            rows = {row["operation_id"]: row for row in csv.DictReader(handle)}
        self.assertEqual(set(rows), {"oats-2020", "barley-2020", "barley-2021", "canola-half",
                                     "wheat-missing", "canola-shared"})
        self.assertEqual(rows["oats-2020"]["quarter_year_status"], "shared_multiple_crops")
        self.assertEqual(rows["oats-2020"]["boundary_pilot_eligible"], "False")
        self.assertIn("area_exceeds_parcel", rows["barley-2021"]["flags"])
        self.assertEqual(rows["barley-2021"]["boundary_pilot_eligible"], "False")
        self.assertEqual(rows["canola-half"]["match_status"], "matched")
        self.assertEqual(rows["canola-half"]["pids"], "61007215800|61007215700")
        self.assertEqual(rows["canola-half"]["boundary_pilot_eligible"], "True")
        self.assertEqual(rows["wheat-missing"]["match_status"], "unmatched")
        # The excluded wheat record still makes this quarter shared.
        self.assertEqual(rows["canola-shared"]["quarter_year_status"], "shared_multiple_crops")
        self.assertEqual(summary["quarter_sections_missing"], ["61007209800"])
        self.assertEqual(summary["boundary_pilot_eligible_count"], 1)

        boundaries = json.loads((output / "field_boundaries.geojson").read_text(encoding="utf-8"))
        self.assertEqual([item["properties"]["pid"] for item in boundaries["features"]],
                         ["60907130500", "60907131600", "61007215700", "61007215800"])
        self.assertTrue(all(item["properties"]["boundary_source"] == "ats_quarter_section"
                            for item in boundaries["features"]))
