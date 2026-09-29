from pathlib import Path
import unittest

from yield_prediction.data import DEFAULT_DATASET_PATH, clean_records, load_records, summarize_records


class DataTests(unittest.TestCase):
    def test_default_dataset_resolves_to_bundled_raw_data(self):
        expected = Path(__file__).resolve().parents[1] / "data/raw/yield_df.csv"
        self.assertEqual(DEFAULT_DATASET_PATH, expected)
        summary = summarize_records(clean_records(load_records()))
        self.assertEqual((summary.rows, summary.areas, summary.items), (28242, 101, 10))
        self.assertEqual((summary.year_min, summary.year_max), (1990, 2013))
        self.assertNotIn("", summary.columns)

    def test_cleaning_preserves_measurements_and_converts_types(self):
        record = {"Unnamed: 0": "7", "Area": "Canada", "Item": "Wheat",
                  "Year": "2021", "hg/ha_yield": "42000",
                  "average_rain_fall_mm_per_year": "500", "pesticides_tonnes": "10",
                  "avg_temp": "12.5"}
        cleaned = clean_records([record])[0]
        self.assertEqual(cleaned["Year"], 2021)
        self.assertEqual(cleaned["hg/ha_yield"], 42000.0)
        self.assertNotIn("Unnamed: 0", cleaned)
        self.assertIn("Unnamed: 0", record)
