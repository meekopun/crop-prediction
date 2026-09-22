import importlib.util
from pathlib import Path
import tempfile
import unittest

from yield_prediction.modeling import chronological_split, run_benchmark


@unittest.skipUnless(importlib.util.find_spec("pandas") and importlib.util.find_spec("sklearn"), "ML extras not installed")
class ModelingTests(unittest.TestCase):
    def test_holdout_contains_only_later_years(self):
        import pandas as pd

        frame = pd.DataFrame({"Year": [year for year in range(2000, 2010) for _ in range(2)]})
        train, test = chronological_split(frame)
        self.assertEqual(set(test.Year), {2008, 2009})
        self.assertLess(train.Year.max(), test.Year.min())
        with self.assertRaisesRegex(ValueError, "two distinct years"):
            chronological_split(frame[frame.Year == 2000])

    def test_benchmark_writes_predictions_and_removes_duplicates(self):
        import pandas as pd

        rows = [{
            "Area": "Canada" if year < 2009 else "Unseen area",
            "Item": crop, "Year": year, "hg/ha_yield": 30000 + year,
            "average_rain_fall_mm_per_year": 500,
            "pesticides_tonnes": 100, "avg_temp": 15,
        } for year in range(2000, 2010) for crop in ["Wheat", "Maize"]]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "input.csv"
            pd.DataFrame([*rows, rows[-1]]).to_csv(path, index=False)
            result = run_benchmark(path, root / "results")
            self.assertEqual(result["duplicates_removed"], 1)
            self.assertEqual(result["train_rows"], 16)
            predictions = pd.read_csv(root / "results/predictions.csv")
            self.assertEqual(len(predictions), 4)
            self.assertEqual(set(predictions.Year), {2008, 2009})
            self.assertTrue(predictions.pred_random_forest_hg_ha.notna().all())
            self.assertEqual(len(pd.read_csv(root / "results/metrics.csv")), 2)
