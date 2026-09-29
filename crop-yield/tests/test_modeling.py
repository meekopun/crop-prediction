import importlib.util
from pathlib import Path
import tempfile
import unittest

from yield_prediction.modeling import _build_pixel_feature_sets, benchmark_models, train_pixel_yield_models


@unittest.skipUnless(importlib.util.find_spec("pandas") and importlib.util.find_spec("sklearn"), "ML extras not installed")
class ModelingTests(unittest.TestCase):
    def test_feature_sets_exclude_targets_and_keep_monthly_satellite_columns(self):
        import pandas as pd
        frame = pd.DataFrame({"crop": ["Barley"], "row_id": ["field-2021"],
                              "yield_bu_ac": [60.0], "yield_original": [60.0],
                              "NDVI_max": [0.8], "jun_NDVI_mean": [0.6],
                              "latitude": [53.0], "year": [2021]})
        sets = _build_pixel_feature_sets(frame)
        for columns in sets.values():
            self.assertNotIn("yield_bu_ac", columns)
            self.assertNotIn("yield_original", columns)
        self.assertIn("jun_NDVI_mean", sets["satellite"])
        self.assertNotIn("year", sets["nonleaky"])

    def test_restored_pixel_yield_training_returns_grouped_predictions(self):
        import pandas as pd
        rows = [{"crop": "Barley", "row_id": f"field-{g}", "year": 2021 + g,
                 "quarter_section": f"q-{g}", "yield_bu_ac": 40.0 + g * 10,
                 "NDVI_max": 0.4 + g * 0.1 + pixel * 0.01}
                for g in range(4) for pixel in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pixels.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            artifact = train_pixel_yield_models(path, feature_set="ndvi_only")
        self.assertEqual(len(artifact.predictions), 12)
        self.assertEqual({metric.model for metric in artifact.metrics}, {"ols", "ridge"})
        self.assertTrue(all(metric.field_year_count == 4 and metric.cv_splits == 4 for metric in artifact.metrics))
        self.assertTrue(artifact.predictions["pred_ridge_ndvi_only"].notna().all())

    def test_restored_historical_benchmark_on_small_dataset(self):
        import pandas as pd
        import math
        rows = [{"Area": "Canada" if i % 2 else "France", "Item": "Wheat",
                 "Year": 2000 + i % 10, "hg/ha_yield": 30000 + i * 100,
                 "average_rain_fall_mm_per_year": 400 + i,
                 "pesticides_tonnes": 10 + i, "avg_temp": 10 + i / 10}
                for i in range(40)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            results = benchmark_models(path, cv_splits=2)
        self.assertTrue({"Linear Regression", "Random Forest", "KNN"}.issubset({r.model for r in results}))
        self.assertTrue(all(math.isfinite(r.mae) for r in results))
