import importlib.util
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

from test_regional_season_model import farm_table, region_table

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec("pandas") and importlib.util.find_spec("sklearn"),
                     "ML extras not installed")
class SeasonFieldBundleTests(unittest.TestCase):
    def setUp(self):
        generator = random.Random(7)
        self.regions = region_table(generator)
        self.directory = tempfile.TemporaryDirectory()
        self.table_path = farm_table(self.directory.name, generator)

    def tearDown(self):
        self.directory.cleanup()

    def test_bundle_reproduces_the_evaluated_held_out_season_without_yields(self):
        import pandas as pd
        from yield_prediction.season_field_bundle import fit_season_field_bundle, predict_season_field
        from yield_prediction.season_field_model import evaluate_season_field_model

        bundle = fit_season_field_bundle(self.table_path, test_year=2025, region_table=self.regions,
                                         min_improvement=0.0)
        self.assertEqual(bundle["training_years"], [2020, 2021, 2022, 2023, 2024])
        self.assertEqual(bundle["region_training_years"], [2019, 2020, 2021, 2022, 2023, 2024])
        self.assertEqual(bundle["later_rows_at_fit"], {"2025": 8})
        table = pd.read_csv(self.table_path, dtype={"field_key": str})
        test = table[table["year"] == 2025]
        predictions = predict_season_field(bundle, test.drop(columns="yield"))
        self.assertNotIn("yield", predictions.columns)
        evaluated = evaluate_season_field_model(self.table_path, years=(2020, 2021, 2022, 2023, 2024, 2025),
                                                deviation_features="peak", season_features="regional_satellite",
                                                region_table=self.regions).predictions
        merged = evaluated[evaluated["year"] == 2025].merge(predictions, on=["field_key", "year", "crop"],
                                                             suffixes=("_evaluated", ""))
        self.assertEqual(len(merged), len(test))
        for column in ("pred_baseline", "pred_season_model_plus_field", "predicted_deviation"):
            self.assertLess((merged[f"{column}_evaluated"] - merged[column]).abs().max(), 1e-9)
        self.assertTrue((predictions["season_source"] == "regional_satellite").all())

    def test_test_year_yields_never_reach_the_bundle(self):
        import pandas as pd
        from yield_prediction.season_field_bundle import fit_season_field_bundle, predict_season_field

        table = pd.read_csv(self.table_path, dtype={"field_key": str})
        corrupted = table.copy()
        corrupted.loc[corrupted["year"] == 2025, "yield"] = 1e6
        corrupted_path = Path(self.directory.name) / "corrupted.csv"
        corrupted.to_csv(corrupted_path, index=False)
        test = table[table["year"] == 2025]
        clean = predict_season_field(fit_season_field_bundle(self.table_path, test_year=2025,
                                                             region_table=self.regions), test)
        dirty = predict_season_field(fit_season_field_bundle(corrupted_path, test_year=2025,
                                                             region_table=self.regions), test)
        self.assertTrue((clean["predicted"] == dirty["predicted"]).all())

    def test_margin_keeps_the_training_mean_unless_the_model_clearly_wins(self):
        import pandas as pd
        from yield_prediction.season_field_bundle import MODEL_VARIANT, fit_season_field_bundle, predict_season_field

        test = pd.read_csv(self.table_path, dtype={"field_key": str}).query("year == 2025")
        open_bundle = fit_season_field_bundle(self.table_path, test_year=2025, region_table=self.regions,
                                              min_improvement=0.0)
        strict = fit_season_field_bundle(self.table_path, test_year=2025, region_table=self.regions,
                                         min_improvement=0.99)
        for crop in ("canola", "barley"):
            self.assertEqual(open_bundle["crops"][crop]["variant"], MODEL_VARIANT)
            self.assertEqual(strict["crops"][crop]["variant"], "baseline")
        predictions = predict_season_field(strict, test)
        self.assertTrue((predictions["predicted"] == predictions["pred_baseline"]).all())
        with self.assertRaises(ValueError):
            fit_season_field_bundle(self.table_path, test_year=2025, region_table=self.regions, min_improvement=1.0)

    def test_saved_bundle_reloads_identically_and_refuses_changes(self):
        import pandas as pd
        from yield_prediction.season_field_bundle import (
            fit_season_field_bundle, load_bundle, predict_season_field, save_bundle,
        )

        test = pd.read_csv(self.table_path, dtype={"field_key": str}).query("year == 2025")
        bundle = fit_season_field_bundle(self.table_path, test_year=2025, region_table=self.regions)
        directory = Path(self.directory.name) / "bundle"
        manifest = save_bundle(bundle, directory)
        self.assertNotIn("season_model", manifest["crops"]["canola"])
        self.assertTrue(manifest["crops"]["canola"]["has_season_model"])
        reloaded = predict_season_field(load_bundle(directory), test)
        pd.testing.assert_frame_equal(predict_season_field(bundle, test), reloaded)
        with self.assertRaises(ValueError):
            save_bundle(bundle, directory)
        with (directory / "model.joblib").open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaises(ValueError):
            load_bundle(directory)

    def test_unknown_crops_units_and_missing_columns(self):
        import pandas as pd
        from yield_prediction.season_field_bundle import fit_season_field_bundle, predict_season_field

        bundle = fit_season_field_bundle(self.table_path, test_year=2025, region_table=self.regions)
        test = pd.read_csv(self.table_path, dtype={"field_key": str}).query("year == 2025").reset_index(drop=True)
        test.loc[0, "crop"] = "rye"
        test.loc[test["crop"] == "barley", "yield_unit"] = "lb/ac"
        predictions = predict_season_field(bundle, test)
        self.assertEqual(predictions.loc[0, "status"], "no model for this crop")
        self.assertTrue(predictions.loc[predictions["crop"] == "barley", "status"].str.startswith("unit").all())
        self.assertTrue(predictions.loc[predictions["status"] != "ok", "predicted"].isna().all())
        self.assertTrue(predictions.loc[predictions["crop"] == "canola", "predicted"].notna().all())
        with self.assertRaises(ValueError):
            predict_season_field(bundle, test.drop(columns="field_EVI_jul"))

    def test_scripts_fit_predict_and_score_once(self):
        directory = Path(self.directory.name)
        self.regions[["region", "year", "crop", "regional_yield"]].to_csv(directory / "yields.csv", index=False)
        self.regions.drop(columns="regional_yield").to_csv(directory / "features.csv", index=False)
        bundle = directory / "bundle"

        def run(script, *arguments):
            return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, arguments)],
                                  capture_output=True, text=True)

        fitted = run("fit_season_field_bundle.py", "--input", self.table_path, "--test-year", 2025,
                     "--output-dir", bundle, "--regional-yields", directory / "yields.csv",
                     "--regional-features", directory / "features.csv")
        self.assertEqual(fitted.returncode, 0, fitted.stderr)
        self.assertTrue((bundle / "manifest.json").exists() and (bundle / "summary.md").exists())
        self.assertEqual(run("fit_season_field_bundle.py", "--input", self.table_path, "--test-year", 2025,
                             "--output-dir", bundle).returncode, 1)
        scored = run("predict_season_field.py", "--bundle", bundle, "--input", self.table_path, "--score")
        self.assertEqual(scored.returncode, 0, scored.stderr)
        self.assertIn("all bu/ac crops", (bundle / "score_2025.md").read_text())
        self.assertEqual(run("predict_season_field.py", "--bundle", bundle, "--input", self.table_path,
                             "--score").returncode, 1)
        self.assertEqual(run("predict_season_field.py", "--bundle", bundle, "--input", self.table_path,
                             "--year", 2024).returncode, 1)


if __name__ == "__main__":
    unittest.main()
