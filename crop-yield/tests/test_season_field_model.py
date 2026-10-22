import importlib.util
from pathlib import Path
import random
import tempfile
import unittest

from yield_prediction.field_yield import table_columns


@unittest.skipUnless(importlib.util.find_spec("pandas") and importlib.util.find_spec("sklearn"),
                     "ML extras not installed")
class SeasonFieldModelTests(unittest.TestCase):
    def write_table(self, directory, *, with_weather=True):
        import pandas as pd

        generator = random.Random(3)
        rainfall = {2020: 250, 2021: 180, 2022: 210, 2023: 330, 2024: 280, 2025: 240}
        rows = []
        for crop, base in (("canola", 30.0), ("barley", 60.0)):
            for year, rain in rainfall.items():
                ndvi = [generator.uniform(0.4, 0.9) for _ in range(5)]
                season = base * (1 + 0.004 * (rain - 250))
                for index, value in enumerate(ndvi):
                    row = {column: generator.uniform(0, 1) for column in table_columns(include_weather=True)
                           if column.startswith(("field_", "w_"))}
                    row.update({f"field_NDVI_{month}": value for month in ("jun", "jul", "aug")})
                    row.update({"w_precip_may_aug": rain + generator.uniform(-2, 2),
                                "w_gdd5_may_aug": 1200 + generator.uniform(-50, 50),
                                "w_tmax_jul": 23 + generator.uniform(-1, 1)})
                    deviation = 40 * (value - sum(ndvi) / len(ndvi))
                    # 2025 yields are absurd: any leak into development would show.
                    observed = 1e6 if year == 2025 else season + deviation + generator.gauss(0, 0.5)
                    row.update({"field_key": f"{crop}{index}", "section_key": f"S{index}", "year": year,
                                "crop": crop, "yield_unit": "bu/ac", "yield": observed})
                    if not with_weather:
                        row = {key: value for key, value in row.items() if not key.startswith("w_")}
                    rows.append(row)
        path = Path(directory) / "table.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_both_parts_add_skill_and_later_years_are_never_used(self):
        from yield_prediction.season_field_model import VARIANTS, evaluate_season_field_model

        with tempfile.TemporaryDirectory() as directory:
            artifact = evaluate_season_field_model(self.write_table(directory))
        self.assertEqual(set(artifact.predictions["year"]), {2020, 2021, 2022, 2023, 2024})
        self.assertLess(artifact.predictions[[f"pred_{variant}" for variant in VARIANTS]].abs().max().max(), 1000)
        skill = artifact.metrics.set_index(["crop", "variant"])["skill"]
        mae = artifact.metrics.set_index(["crop", "variant"])["mae"]
        for crop in ("canola", "barley"):
            self.assertEqual(skill[(crop, "baseline")], 0.0)
            self.assertGreater(skill[(crop, "season_model")], 0.3)
            # Given the season mean, the field deviation removes much of the remaining error.
            self.assertLess(mae[(crop, "known_season_plus_field")], 0.7 * mae[(crop, "known_season")])
            self.assertGreater(skill[(crop, "season_model_plus_field")],
                               max(skill[(crop, "season_model")], skill[(crop, "baseline_plus_field")]))
        self.assertEqual(len(artifact.seasons), 2 * 5)

    def test_satellite_season_level_and_peak_deviation_features(self):
        from yield_prediction.season_field_model import evaluate_season_field_model

        with tempfile.TemporaryDirectory() as directory:
            # Season yield follows rainfall, not satellite season means, in this table.
            artifact = evaluate_season_field_model(self.write_table(directory), deviation_features="peak",
                                                   season_features="satellite")
        mae = artifact.metrics.set_index(["crop", "variant"])["mae"]
        for crop in ("canola", "barley"):
            self.assertLess(mae[(crop, "known_season_plus_field")], 0.7 * mae[(crop, "known_season")])
        self.assertIn("field_NDVI_jul", artifact.seasons.columns)

    def test_missing_weather_columns_fail_clearly(self):
        from yield_prediction.season_field_model import evaluate_season_field_model

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                evaluate_season_field_model(self.write_table(directory, with_weather=False))
