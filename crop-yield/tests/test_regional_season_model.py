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

    def test_canola_option_adds_august_for_canola_only(self):
        from yield_prediction.season_field_model import SEASON_FEATURE_SETS, fit_regional_season_models

        generator = random.Random(9)
        regions = region_table(generator)
        for column in ("field_NDVI_jul", "field_NDRE_jul", "field_EVI_jul"):
            regions[column.replace("_jul", "_aug")] = regions[column] - 0.1
        option = "regional_satellite_canola_jul_aug"
        self.assertEqual(SEASON_FEATURE_SETS[option], [f"field_{index}_{month}" for month in ("jul", "aug")
                                                       for index in ("NDVI", "NDRE", "EVI")])
        models = fit_regional_season_models(regions, features=option)
        self.assertEqual(len(models["canola"][0].feature_names_in_), 6)
        self.assertEqual(list(models["barley"][0].feature_names_in_), ["field_NDVI_jul", "field_NDRE_jul",
                                                                       "field_EVI_jul"])

    def test_identical_quarter_and_local_readings_match_the_field_reading(self):
        import pandas as pd
        from yield_prediction.season_field_model import evaluate_season_field_model

        generator = random.Random(10)
        regions = region_table(generator)
        with tempfile.TemporaryDirectory() as directory:
            path = farm_table(directory, generator)
            table = pd.read_csv(path)
            fields = [column for column in table.columns if column.startswith("field_") and column != "field_key"]
            table = pd.concat([table, table[fields].rename(columns=lambda column: column.replace("field_", "quarter_", 1))],
                              axis=1)
            table.to_csv(path, index=False)
            local = table.groupby(["crop", "year"])[fields].mean().reset_index()
            runs = {reading: evaluate_season_field_model(path, deviation_features="peak",
                                                         season_features="regional_satellite",
                                                         region_table=regions, season_reading=reading,
                                                         local_table=local).predictions
                    for reading in ("fields", "quarter", "local")}
            with self.assertRaises(ValueError):
                evaluate_season_field_model(path, season_features="satellite", season_reading="quarter")
        for reading in ("quarter", "local"):
            difference = runs[reading]["pred_season_model"] - runs["fields"]["pred_season_model"]
            self.assertLess(difference.abs().max(), 1e-9)

    def test_field_history_leaves_out_the_rows_own_season(self):
        import pandas as pd
        from yield_prediction.season_field_model import _field_history, field_history_means

        source = pd.DataFrame({"field_key": ["A", "B", "A", "B", "C"], "year": [2020, 2020, 2021, 2021, 2021],
                               "crop": ["wheat", "wheat", "canola", "canola", "oats"],
                               "yield": [60.0, 40.0, 30.0, 30.0, 90.0]})
        # A: +0.2 in 2020 and 0.0 in 2021; C is alone in its crop-season, so it has no history.
        history = _field_history(source, source, own_year_out=True)
        self.assertEqual(list(history.round(6)), [0.0, 0.0, 0.2, -0.2, 0.0])
        target = pd.DataFrame({"field_key": ["A", "C", "D"], "year": [2022, 2022, 2022]})
        self.assertEqual(list(_field_history(source, target, own_year_out=False).round(6)), [0.1, 0.0, 0.0])
        self.assertEqual({key: round(value, 6) for key, value in field_history_means(source).items()},
                         {"A": 0.1, "B": -0.1})

    def test_landsat_option_uses_ndvi_evi_and_rescales_farm_anomalies(self):
        import pandas as pd
        from yield_prediction.season_field_model import (
            SEASON_FEATURE_SETS, _drop_low_coverage, cross_sensor_scale, evaluate_season_field_model,
        )

        self.assertEqual(SEASON_FEATURE_SETS["regional_landsat"], ["field_NDVI_jul", "field_EVI_jul"])
        generator = random.Random(11)
        regions = region_table(generator).assign(pixels=1000)
        regions.loc[0, "pixels"] = 100
        self.assertEqual(len(_drop_low_coverage(regions, 0.25)), len(regions) - 1)
        with tempfile.TemporaryDirectory() as directory:
            s2_path, landsat_path = Path(directory) / "s2.csv", Path(directory) / "landsat.csv"
            regions.to_csv(s2_path, index=False)
            landsat = regions.assign(field_NDVI_jul=0.1 + 0.8 * regions["field_NDVI_jul"],
                                     field_EVI_jul=-0.05 + 0.9 * regions["field_EVI_jul"])
            landsat.loc[0, "pixels"] = 1000
            landsat.drop(columns="field_NDRE_jul").to_csv(landsat_path, index=False)
            scale = cross_sensor_scale(landsat_path, s2_path)
            self.assertAlmostEqual(scale["field_NDVI_jul"], 0.8)
            self.assertAlmostEqual(scale["field_EVI_jul"], 0.9)
            farm = farm_table(directory, generator)
            # A Landsat-scale regional table plus the slope must match the Sentinel-2 fit on raw anomalies.
            on_landsat = evaluate_season_field_model(farm, deviation_features="peak", season_features="regional_landsat",
                                                     region_table=landsat, sensor_scale=scale).seasons
            on_s2 = evaluate_season_field_model(farm, deviation_features="peak", season_features="regional_landsat",
                                                region_table=regions, sensor_scale={}).seasons
        difference = on_landsat["predicted_relative"] - on_s2["predicted_relative"]
        self.assertLess(difference.abs().max(), 1e-6)

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
