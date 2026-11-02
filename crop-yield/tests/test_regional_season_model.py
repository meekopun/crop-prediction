import importlib.util
from pathlib import Path
import random
import tempfile
import unittest

from yield_prediction.field_yield import table_columns


def region_table(generator, years=range(2019, 2026)):
    import pandas as pd

    rows = []
    for region in range(8):
        for crop, base in (("canola", 35.0), ("barley", 65.0)):
            scale = base * generator.uniform(0.8, 1.2)
            for year in years:
                anomaly = generator.uniform(-0.1, 0.1)
                rows.append({"region": f"R{region}", "year": year, "crop": crop,
                             "regional_yield": scale * (1 + 2 * anomaly) + generator.gauss(0, 0.3),
                             "field_NDVI_jul": 0.6 + anomaly, "field_NDRE_jul": 0.4 + 0.5 * anomaly,
                             "field_EVI_jul": 0.5 + 0.8 * anomaly})
    return pd.DataFrame(rows)


def farm_table(directory, generator):
    import pandas as pd

    rows = []
    for crop, base in (("canola", 30.0), ("barley", 60.0)):
        for year in range(2020, 2026):
            anomaly = generator.uniform(-0.1, 0.1)
            for index in range(4):
                field = generator.uniform(-0.03, 0.03)
                row = {column: generator.uniform(0, 1) for column in table_columns() if column.startswith("field_")}
                row.update({"field_NDVI_jul": 0.6 + anomaly + field, "field_NDRE_jul": 0.4 + 0.5 * anomaly,
                            "field_EVI_jul": 0.5 + 0.8 * anomaly, "field_key": f"{crop}{index}",
                            "section_key": f"S{index}", "year": year, "crop": crop, "yield_unit": "bu/ac",
                            "yield": base * (1 + 2 * anomaly) + generator.gauss(0, 0.5)})
                rows.append(row)
    path = Path(directory) / "table.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@unittest.skipUnless(importlib.util.find_spec("pandas") and importlib.util.find_spec("sklearn"),
                     "ML extras not installed")
class RegionalSeasonModelTests(unittest.TestCase):
    def test_relation_learned_on_regions_transfers_to_the_farm(self):
        from yield_prediction.season_field_model import evaluate_regional_season_models, evaluate_season_field_model

        generator = random.Random(5)
        regions = region_table(generator)
        check = evaluate_regional_season_models(regions).set_index("crop")
        self.assertTrue((check["skill"] > 0.8).all())
        with tempfile.TemporaryDirectory() as directory:
            artifact = evaluate_season_field_model(farm_table(directory, generator), deviation_features="peak",
                                                   season_features="regional_satellite", region_table=regions)
        skill = artifact.metrics.set_index(["crop", "variant"])["skill"]
        for crop in ("canola", "barley"):
            self.assertGreater(skill[(crop, "season_model")], 0.5)
        self.assertTrue(artifact.seasons["predicted_relative"].notna().all())
        self.assertEqual(set(artifact.predictions["year"]), {2020, 2021, 2022, 2023, 2024})

    def test_month_files_merge_into_a_multi_month_season_model(self):
        from yield_prediction.season_field_model import (
            SEASON_FEATURE_SETS, evaluate_regional_season_models, evaluate_season_field_model, load_region_table,
        )

        generator = random.Random(8)
        regions = region_table(generator)
        july = ["field_NDVI_jul", "field_NDRE_jul", "field_EVI_jul"]
        august = [column.replace("_jul", "_aug") for column in july]
        for source, target in zip(july, august):
            regions[target] = regions[source] - 0.1 + [generator.gauss(0, 0.005) for _ in range(len(regions))]
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("yields.csv", "jul.csv", "aug.csv")]
            regions[["region", "year", "crop", "regional_yield"]].to_csv(paths[0], index=False)
            for path, columns in zip(paths[1:], (july, august)):
                regions[["region", "year", "crop"] + columns].assign(pixels=100).to_csv(path, index=False)
            loaded = load_region_table(paths[0], paths[1:])
            self.assertEqual(len(loaded), len(regions))
            self.assertEqual(list(loaded.columns).count("pixels"), 1)
            features = SEASON_FEATURE_SETS["regional_satellite_jul_aug"]
            self.assertEqual(features, july + august)
            check = evaluate_regional_season_models(loaded, features).set_index("crop")
            self.assertTrue((check["skill"] > 0.8).all())
            artifact = evaluate_season_field_model(farm_table(directory, generator), deviation_features="peak",
                                                   season_features="regional_satellite_jul_aug",
                                                   region_table=loaded)
        self.assertTrue(artifact.seasons["predicted_relative"].notna().all())

    def test_the_held_out_season_never_trains_the_regional_model(self):
        from yield_prediction.season_field_model import REGIONAL_FEATURES, fit_regional_season_models

        regions = region_table(random.Random(6))
        corrupted = regions.copy()
        corrupted.loc[corrupted["year"] == 2021, "regional_yield"] = 1e6
        clean = fit_regional_season_models(regions, exclude_year=2021)
        dirty = fit_regional_season_models(corrupted, exclude_year=2021)
        probe = regions[REGIONAL_FEATURES].head(3) - regions[REGIONAL_FEATURES].mean()
        for crop in ("canola", "barley"):
            self.assertEqual(clean[crop][1], dirty[crop][1])
            self.assertTrue((clean[crop][0].predict(probe) == dirty[crop][0].predict(probe)).all())
