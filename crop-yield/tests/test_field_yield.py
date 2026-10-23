import csv
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest

from yield_prediction.field_yield import build_field_yield_table, group_operations, table_columns
from yield_prediction.harvest_features import feature_columns
from yield_prediction.harvest_weather import weather_columns

MATCH_COLUMNS = ["operation_id", "year", "crop", "pids", "legal_description", "area_ac", "yield_value",
                 "yield_unit", "boundary_pilot_eligible"]


def match(op_id, year, crop, pids, area, yield_value, unit="bu/ac", eligible="True"):
    return {"operation_id": op_id, "year": str(year), "crop": crop, "pids": pids,
            "legal_description": f"legal-{pids}", "area_ac": str(area), "yield_value": str(yield_value),
            "yield_unit": unit, "boundary_pilot_eligible": eligible}


def feature_row(pid, year, crop, fraction, ndvi_quarter, ndvi_crop):
    row = {column: "0.5" for column in feature_columns()}
    row.update({"pid": pid, "year": str(year), "crop": crop, "aci_codes": "153", "operation_ids": "x",
                "aci_crop_fraction": str(fraction)})
    for month in ("apr", "may", "jun", "jul", "aug", "sep", "oct"):
        row[f"quarter_NDVI_{month}"] = str(ndvi_quarter)
        row[f"crop_NDVI_{month}"] = "" if ndvi_crop is None else str(ndvi_crop)
    return row


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class GroupingTests(unittest.TestCase):
    def test_same_crop_operations_sharing_a_quarter_join_through_half_sections(self):
        rows = [match("quarter", 2022, "fescue", "A", 100, 900),
                match("half", 2022, "fescue", "A|B", 180, 950),
                match("other-crop", 2022, "wheat", "A", 50, 40),
                match("other-year", 2023, "fescue", "B", 120, 800),
                match("separate", 2022, "fescue", "C", 90, 700)]
        groups = [sorted(row["operation_id"] for row in group) for group in group_operations(rows)]
        self.assertIn(["half", "quarter"], groups)
        self.assertIn(["separate"], groups)
        self.assertIn(["other-crop"], groups)
        self.assertIn(["other-year"], groups)
        self.assertEqual(len(groups), 4)


class TableTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_yield_is_total_over_area_and_features_follow_the_crop_mask(self):
        matches = self.root / "matches.csv"
        write_csv(matches, [
            match("a", 2021, "canola", "P1", 60, 20),
            match("b", 2021, "canola", "P1", 40, 40),
            match("c", 2021, "oats", "P2", 100, 80),
            match("shared", 2021, "barley", "P3", 50, 60, eligible="False"),
        ], MATCH_COLUMNS)
        operations = self.root / "operations.csv"
        write_csv(operations, [
            {"operation_id": "a", "total_value": "1200", "total_unit": "bu"},
            {"operation_id": "b", "total_value": "1600", "total_unit": "bu"},
            {"operation_id": "c", "total_value": "", "total_unit": ""},
        ], ["operation_id", "total_value", "total_unit"])
        features = self.root / "features.csv"
        write_csv(features, [feature_row("P1", 2021, "canola", 0.6, 0.5, 0.8),
                             feature_row("P2", 2021, "oats", 0.0, 0.7, None)], feature_columns())
        boundaries = self.root / "boundaries.geojson"
        boundaries.write_text(json.dumps({"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {"pid": pid, "parcel_acres": 160.0}, "geometry": None}
            for pid in ("P1", "P2", "P3")]}), encoding="utf-8")
        output = self.root / "table.csv"
        summary = build_field_yield_table(matches, operations, features, boundaries, output, weather_path=None)

        with output.open(newline="", encoding="utf-8") as handle:
            rows = {row["crop"]: row for row in csv.DictReader(handle)}
        self.assertEqual(set(rows), {"canola", "oats"})
        self.assertEqual(float(rows["canola"]["yield"]), 28.0)  # (1200 + 1600) / 100 ac
        self.assertEqual(rows["canola"]["operation_count"], "2")
        self.assertEqual(rows["canola"]["feature_source"], "crop_mask")
        self.assertAlmostEqual(float(rows["canola"]["field_NDVI_jul"]), 0.8)
        # No crop-labelled pixels: the whole-quarter value is used, and the
        # missing total falls back to area x yield.
        self.assertEqual(rows["oats"]["feature_source"], "quarter")
        self.assertAlmostEqual(float(rows["oats"]["field_NDVI_jul"]), 0.7)
        self.assertEqual(float(rows["oats"]["yield"]), 80.0)
        self.assertEqual(summary["field_year_count"], 2)
        self.assertEqual(list(rows["canola"]), table_columns())

        weather = self.root / "weather.csv"
        weather_rows = []
        for pid, rain in (("P1", 50.0), ("P2", 70.0)):
            row = {"pid": pid, "year": "2021", **{column: "1.0" for column in weather_columns()}}
            row.update({f"w_precip_{month}": str(rain) for month in ("may", "jun", "jul", "aug")})
            weather_rows.append(row)
        write_csv(weather, weather_rows, ["pid", "year"] + weather_columns())
        build_field_yield_table(matches, operations, features, boundaries, output, weather_path=weather)
        with output.open(newline="", encoding="utf-8") as handle:
            rows = {row["crop"]: row for row in csv.DictReader(handle)}
        self.assertEqual(list(rows["canola"]), table_columns(include_weather=True))
        self.assertEqual(float(rows["canola"]["w_precip_may_aug"]), 200.0)
        self.assertEqual(float(rows["oats"]["w_precip_jul"]), 70.0)


@unittest.skipUnless(importlib.util.find_spec("pandas") and importlib.util.find_spec("sklearn"),
                     "ML extras not installed")
class BenchmarkTests(unittest.TestCase):
    def test_signal_in_ndvi_beats_the_baseline_under_both_splits(self):
        import pandas as pd
        from yield_prediction.field_benchmark import FEATURE_SETS, benchmark_field_yield

        generator = random.Random(0)
        rows = []
        for year in range(2020, 2026):
            for section in range(5):
                ndvi = generator.uniform(0.4, 0.9)
                row = {column: generator.uniform(0, 1) for column in table_columns() if column.startswith("field_")}
                row.update({f"field_NDVI_{month}": ndvi for month in ("jun", "jul", "aug")})
                row.update({"field_key": f"S{section}", "section_key": f"S{section}", "year": year,
                            "crop": "canola", "yield_unit": "bu/ac",
                            "yield": 10 + 50 * ndvi + generator.gauss(0, 1)})
                rows.append(row)
        rows.append({"field_key": "R", "section_key": "R", "year": 2020, "crop": "rye",
                     "yield_unit": "bu/ac", "yield": 40.0})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            artifact = benchmark_field_yield(path, min_rows=10)
        metrics = artifact.metrics
        self.assertEqual(set(metrics["crop"]), {"canola"})
        self.assertEqual(len(metrics), 2 * (1 + 2 * len(FEATURE_SETS)))
        for split in ("year", "section"):
            subset = metrics[metrics["split"] == split].set_index(["model", "feature_set"])
            self.assertEqual(subset.loc[("mean_baseline", "none"), "mae_skill"], 0.0)
            self.assertGreater(subset.loc[("ridge", "ndvi"), "mae_skill"], 0.5)
        self.assertEqual(len(artifact.predictions), 30 * len(metrics))
        self.assertTrue(artifact.predictions["predicted"].notna().all())

    def _table(self, directory, signal):
        import pandas as pd

        generator = random.Random(1)
        rows = []
        for year in range(2020, 2026):
            for section in range(5):
                ndvi = generator.uniform(0.4, 0.9)
                row = {column: generator.uniform(0, 1) for column in table_columns() if column.startswith("field_")}
                row.update({f"field_NDVI_{month}": ndvi for month in ("jun", "jul", "aug")})
                yield_value = 10 + 50 * ndvi if signal else 40.0
                row.update({"field_key": f"S{section}", "section_key": f"S{section}", "year": year,
                            "crop": "canola", "yield_unit": "bu/ac",
                            "yield": yield_value + generator.gauss(0, 1 if signal else 5)})
                rows.append(row)
        path = Path(directory) / "table.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_holdout_selects_on_earlier_years_and_scores_the_test_year_once(self):
        from yield_prediction.field_benchmark import FEATURE_SETS, holdout_field_yield

        with tempfile.TemporaryDirectory() as directory:
            artifact = holdout_field_yield(self._table(directory, signal=True), test_year=2025)
        result = artifact.results.iloc[0]
        self.assertEqual(result["status"], "scored")
        self.assertEqual((result["dev_field_years"], result["dev_years"], result["test_field_years"]), (25, 5, 5))
        self.assertNotEqual(result["chosen_model"], "mean_baseline")
        self.assertGreater(result["test_mae_skill"], 0.5)
        # Every candidate is scored on development years only.
        self.assertEqual(len(artifact.selection), 1 + 2 * len(FEATURE_SETS))
        self.assertEqual(set(artifact.predictions["year"]), {2025})
        self.assertAlmostEqual(result["dev_mae_chosen"], artifact.selection["dev_mae"].min())

    def test_margin_keeps_the_baseline_unless_a_model_clearly_wins(self):
        from yield_prediction.field_benchmark import holdout_field_yield

        with tempfile.TemporaryDirectory() as directory:
            path = self._table(directory, signal=True)
            plain = holdout_field_yield(path, test_year=2025).results.iloc[0]
            strict = holdout_field_yield(path, test_year=2025, min_improvement=0.99).results.iloc[0]
            with self.assertRaises(ValueError):
                holdout_field_yield(path, test_year=2025, min_improvement=1.0)
        self.assertNotEqual(plain["chosen_model"], "mean_baseline")
        # An impossible margin always keeps the baseline, which then scores zero skill.
        self.assertEqual(strict["chosen_model"], "mean_baseline")
        self.assertEqual(strict["test_mae_skill"], 0.0)
        self.assertEqual(strict["best_model"], plain["best_model"])

    def test_holdout_falls_back_to_the_baseline_without_signal(self):
        from yield_prediction.field_benchmark import holdout_field_yield

        with tempfile.TemporaryDirectory() as directory:
            artifact = holdout_field_yield(self._table(directory, signal=False), test_year=2025)
        result = artifact.results.iloc[0]
        selection = artifact.selection.set_index(["model", "feature_set"])["dev_mae"]
        if result["chosen_model"] == "mean_baseline":
            self.assertEqual(result["test_mae_skill"], 0.0)
        else:
            # A model was chosen only because it beat the baseline on earlier years.
            self.assertLess(result["dev_mae_chosen"], selection[("mean_baseline", "none")])
