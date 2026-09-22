import csv
from pathlib import Path
import tempfile
import unittest

from yield_prediction.data import COLUMNS, load_data, summarize


class DataTests(unittest.TestCase):
    def test_bundled_dataset(self):
        path = Path(__file__).resolve().parents[1] / "archive/yield_df.csv"
        self.assertEqual(summarize(load_data(path)), {
            "rows": 28242, "areas": 101, "items": 10,
            "year_min": 1990, "year_max": 2013, "duplicate_records": 2310,
        })

    def test_rejects_nonfinite_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.csv"
            with path.open("w", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(COLUMNS)
                writer.writerow(["Canada", "Wheat", 2000, 500, 100, "nan", 30000])
            with self.assertRaisesRegex(ValueError, "line 2.*finite"):
                load_data(path)
