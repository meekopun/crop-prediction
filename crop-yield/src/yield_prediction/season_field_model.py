"""Two-part yield model: a season level from weather plus each field's deviation.

A field-year's yield is split into its crop's season mean and the field's
deviation from it. The deviation model is a per-crop ridge regression on
satellite features centred on the crop's season mean, so it can only learn which
fields do better or worse than their season. The season model is one ridge
regression, pooled over crops, that predicts the season mean relative to the
crop's long-run mean from season-level features: ERA5-Land weather totals or the
season mean of the crop's peak-season satellite indices. Season features enter
as anomalies from the crop's training-season mean. The ``regional`` option
instead scales the published regional yield by the farm-to-region ratio learned
on the training seasons; crops without published yields keep the training mean.
The ``regional_satellite`` option learns the satellite-to-season-yield relation
from all Alberta region-years (excluding the held-out season) and applies it to
the farm's own July satellite anomaly; ``regional_satellite_<months>`` options
add the June and/or August composites (``_canola_jul_aug``: canola only), and
``regional_landsat`` learns from Landsat NDVI/EVI region-years back to 2013 (the
farm's Sentinel-2 anomalies are rescaled by a cross-sensor slope). The
farm's season reading is the mean over its fields (``fields``), over their whole
quarters (``quarter``) or over a fixed area around the farm (``local``). The
``peak_history`` deviation set adds each field's mean departure from its
crop-season mean in training seasons.

Evaluation holds out one season at a time and uses development years only
(2020-2024 by default); later years are dropped before anything is fitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .field_benchmark import DEFAULT_BENCHMARK_DIR, FEATURE_SETS
from .field_yield import DEFAULT_TABLE_PATH
from .regional_features import (
    DEFAULT_LOCAL_FEATURES_PATH, DEFAULT_REGIONAL_FEATURES_PATH, MONTHS, REGIONAL_FEATURES, month_features,
    regional_features_path,
)
from .regional_yields import ALBERTA_YIELDS_2013_PATH, DEFAULT_ALBERTA_YIELDS_PATH, DEFAULT_REGIONAL_PATH

DEV_YEARS = (2020, 2021, 2022, 2023, 2024)
PEAK_FEATURES = ["field_NDVI_jul", "field_NDRE_jul", "field_EVI_jul"]
# Computed per training split from yields, never read from the table.
HISTORY_FEATURE = "field_history"
DEVIATION_FEATURE_SETS = {"indices": FEATURE_SETS["indices"], "peak": PEAK_FEATURES,
                          "peak_history": PEAK_FEATURES + [HISTORY_FEATURE]}
# Months of regional and farm composites behind each regional_satellite option; "*" is every other crop.
REGIONAL_SEASON_MONTHS = {"regional_satellite": {"*": ("jul",)},
                          "regional_satellite_jun_jul": {"*": ("jun", "jul")},
                          "regional_satellite_jul_aug": {"*": ("jul", "aug")},
                          "regional_satellite_jun_jul_aug": {"*": ("jun", "jul", "aug")},
                          "regional_satellite_canola_jul_aug": {"*": ("jul",), "canola": ("jul", "aug")},
                          "regional_landsat": {"*": ("jul",)}}
# Sensor behind each option's regional features (Sentinel-2 unless listed).
REGIONAL_SEASON_SENSORS = {"regional_landsat": "landsat"}
# Landsat region-crop-years with less clear coverage than this share of the region-crop's median are dropped.
LANDSAT_MIN_PIXEL_SHARE = 0.25
SEASON_READINGS = ("fields", "quarter", "local")


def _option_months(option: str) -> list[str]:
    used = {month for months in REGIONAL_SEASON_MONTHS[option].values() for month in months}
    return [month for month in MONTHS if month in used]


def _option_sensor(option: str) -> str:
    return REGIONAL_SEASON_SENSORS.get(option, "s2")


def season_crop_features(option: str, crop: str) -> list[str]:
    """Season-level feature columns a regional_satellite option uses for one crop."""
    months = REGIONAL_SEASON_MONTHS[option]
    return [column for month in months.get(crop, months["*"])
            for column in month_features(month, _option_sensor(option))]


# "regional" uses published yields from a separate file rather than table columns;
# regional_satellite options train on region-years but apply to the farm's features.
SEASON_FEATURE_SETS = {"weather": ["w_precip_may_aug", "w_gdd5_may_aug", "w_tmax_jul"],
                       "satellite": PEAK_FEATURES, "regional": [],
                       **{name: [column for month in _option_months(name)
                                 for column in month_features(month, _option_sensor(name))]
                          for name in REGIONAL_SEASON_MONTHS}}
assert REGIONAL_FEATURES == PEAK_FEATURES == SEASON_FEATURE_SETS["regional_satellite"], \
    "regional and farm features must share names"
SEASON_ALPHAS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
DEFAULT_OUTPUT_DIR = DEFAULT_BENCHMARK_DIR
VARIANTS = {
    "baseline": "training mean of the crop",
    "season_model": "modelled season mean only",
    "baseline_plus_field": "training mean + field deviation",
    "season_model_plus_field": "modelled season mean + field deviation",
    "known_season": "other fields' actual season mean (ceiling, needs outside data)",
    "known_season_plus_field": "other fields' actual season mean + field deviation (ceiling)",
}


@dataclass(frozen=True)
class SeasonFieldArtifact:
    metrics: object
    seasons: object
    predictions: object


def _deviation_model():
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("model", RidgeCV(alphas=np.logspace(-2, 5, 29)))])


def _centre(frame, columns):
    """Subtract each crop-season's feature mean (from features only, never yields)."""
    return frame[columns] - frame.groupby(["crop", "year"])[columns].transform("mean")


def _fit_deviation_models(train, centred) -> dict:
    """Per-crop ridge from centred features to each field's departure from its season mean."""
    models = {}
    for crop, crop_train in train.groupby("crop"):
        # Seasons with one field carry no within-season contrast.
        crop_train = crop_train[crop_train.groupby("year")["yield"].transform("size") >= 2]
        if len(crop_train) >= 5:
            target = crop_train["yield"] - crop_train.groupby("year")["yield"].transform("mean")
            models[crop] = _deviation_model().fit(crop_train[centred], target)
    return models


def _predict_deviation(models, test, centred):
    """Predicted departure from the season mean; zero for crops without a model."""
    import pandas as pd

    deviation = pd.Series(0.0, index=test.index)
    for crop, crop_test in test.groupby("crop"):
        if crop in models:
            deviation[crop_test.index] = models[crop].predict(crop_test[centred])
    return deviation


def _relative_departures(frame):
    """Each field-year's yield over its crop-season mean, minus one; NaN in one-field seasons."""
    groups = frame.groupby(["crop", "year"])["yield"]
    return (frame["yield"] / groups.transform("mean") - 1).where(groups.transform("size") >= 2)


def field_history_means(source) -> dict:
    """Mean relative departure per field over every season in ``source`` (any crop or unit)."""
    departures = source[["field_key", "year"]].assign(departure=_relative_departures(source)).dropna()
    return departures.groupby(["field_key", "year"])["departure"].mean().groupby("field_key").mean().to_dict()


def _field_history(source, target, *, own_year_out: bool):
    """Field history for ``target`` rows from ``source`` seasons; 0 for fields without history.

    With ``own_year_out`` a row's own season is left out, so a training row's
    feature never contains its own yield.
    """
    import pandas as pd

    departures = source[["field_key", "year"]].assign(departure=_relative_departures(source)).dropna()
    per_year = departures.groupby(["field_key", "year"])["departure"].mean()
    totals = per_year.groupby("field_key").agg(["sum", "count"])
    keyed = target[["field_key"]].join(totals, on="field_key")
    total, count = keyed["sum"].fillna(0.0), keyed["count"].fillna(0)
    if own_year_out:
        own = pd.Series([per_year.get((field, year)) for field, year in zip(target["field_key"], target["year"])],
                        index=target.index, dtype=float)
        total, count = total - own.fillna(0.0), count - own.notna()
    return (total / count.where(count > 0)).fillna(0.0)


def _season_reading(frame, season_reading: str, columns, local_table=None):
    """Season features per crop-year: mean over fields, over whole quarters, or the local area."""
    if season_reading == "fields":
        return frame.groupby(["crop", "year"])[columns].mean()
    if season_reading == "quarter":
        quarter = [column.replace("field_", "quarter_", 1) for column in columns]
        return frame.groupby(["crop", "year"])[quarter].mean().set_axis(columns, axis=1)
    if season_reading == "local":
        keys = frame[["crop", "year"]].drop_duplicates()
        return keys.merge(local_table, on=["crop", "year"], how="left").set_index(["crop", "year"])[columns]
    raise ValueError(f"Unknown season reading {season_reading!r}; choose from {SEASON_READINGS}.")


def load_local_table(path: Path | str = DEFAULT_LOCAL_FEATURES_PATH):
    import pandas as pd

    if not Path(path).exists():
        raise ValueError(f"{path} not found; run scripts/export_local_features.py first.")
    return pd.read_csv(path)


def _regional_satellite_relative(models, reference, season_features) -> dict:
    """Relative season yield per crop from the farm's anomaly against its training seasons.

    ``season_features`` and ``reference`` are indexed by crop; each crop's model
    names the columns it uses. Crops without a regional model or with a missing
    feature are left out, so callers fall back to the training mean.
    """
    relative = {}
    for crop in season_features.index:
        if crop not in models or crop not in reference.index:
            continue
        columns = list(models[crop][0].feature_names_in_)
        anomaly = season_features.loc[[crop], columns] - reference.loc[[crop], columns]
        if anomaly.notna().all(axis=None):
            relative[crop] = models[crop][0].predict(anomaly)[0]
    return relative


def _fit_season_model(seasons, features):
    """Ridge on standardized season anomalies; alpha chosen by leave-one-season-out within training."""
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def make(alpha):
        return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=alpha))])

    X, y, years = seasons[features], seasons["relative_mean"].to_numpy(), seasons["year"]
    best_alpha, best_error = SEASON_ALPHAS[-1], np.inf
    if years.nunique() >= 3:
        for alpha in SEASON_ALPHAS:
            errors = []
            for year in years.unique():
                train, test = years != year, years == year
                errors.extend(np.abs(make(alpha).fit(X[train], y[train]).predict(X[test]) - y[test]))
            if np.mean(errors) < best_error:
                best_alpha, best_error = alpha, float(np.mean(errors))
    return make(best_alpha).fit(X, y), best_alpha


def _load_regional(path: Path | str) -> dict[tuple[str, int], float]:
    import pandas as pd

    if not Path(path).exists():
        raise ValueError(f"{path} not found; run scripts/prepare_regional_yields.py first.")
    frame = pd.read_csv(path)
    return {(row.crop, int(row.year)): float(row.regional_yield) for row in frame.itertuples(index=False)}


def _regional_season_means(season_train, crops, year, regional, crop_means):
    """Scale each crop's published regional yield by its training farm-to-region ratio."""
    import pandas as pd

    means, ratios = {}, {}
    for crop in crops:
        history = season_train[season_train["crop"] == crop]
        published = history["year"].map(lambda season: regional.get((crop, season)))
        usable = published.notna()
        current = regional.get((crop, year))
        if usable.any() and current is not None:
            # Ratio of totals is steadier than the mean of per-season ratios.
            ratios[crop] = float(history.loc[usable, "season_mean"].sum() / published[usable].sum())
            means[crop] = ratios[crop] * current
        else:
            ratios[crop] = None
            means[crop] = crop_means[crop]
    return pd.Series(means, dtype=float), ratios


def regional_feature_paths(season_features: str) -> list[Path]:
    """Default per-month regional feature files for a regional_satellite option."""
    sensor = _option_sensor(season_features)
    return [regional_features_path(month, sensor) for month in _option_months(season_features)]


def regional_yields_path(season_features: str) -> Path:
    """Default regional yields for a regional_satellite option (from 2013 for Landsat)."""
    return ALBERTA_YIELDS_2013_PATH if _option_sensor(season_features) == "landsat" else DEFAULT_ALBERTA_YIELDS_PATH


def _drop_low_coverage(frame, min_pixel_share: float):
    """Drop rows whose pixel count is below ``min_pixel_share`` of their region-crop's median."""
    median = frame.groupby(["region", "crop"])["pixels"].transform("median")
    return frame[frame["pixels"] >= min_pixel_share * median]


def load_option_region_table(season_features: str, yields_path: Path | str | None = None,
                             features_paths: list | None = None):
    """The region table a regional_satellite option trains on, with its default files and coverage rule."""
    table = load_region_table(yields_path or regional_yields_path(season_features),
                              features_paths or regional_feature_paths(season_features))
    if _option_sensor(season_features) == "landsat":
        table = _drop_low_coverage(table, LANDSAT_MIN_PIXEL_SHARE).reset_index(drop=True)
    return table


def cross_sensor_scale(landsat_path: Path | str | None = None,
                       s2_path: Path | str = DEFAULT_REGIONAL_FEATURES_PATH) -> dict[str, float]:
    """Per-index slope of Landsat on Sentinel-2 July regional anomalies over shared region-crop-years.

    Farm features are Sentinel-2; scaling their anomalies by this slope puts them
    on the Landsat scale the regional model learned. Anomalies are against each
    region-crop's mean over the shared years. Uses features only, never yields.
    """
    import pandas as pd

    landsat = _drop_low_coverage(pd.read_csv(landsat_path or regional_features_path("jul", "landsat")),
                                 LANDSAT_MIN_PIXEL_SHARE)
    columns = month_features("jul", "landsat")
    shared =landsat.merge(pd.read_csv(s2_path), on=["region", "year", "crop"], suffixes=("_landsat", "_s2"))
    scale = {}
    for column in columns:
        pair = shared[["region", "crop", f"{column}_landsat", f"{column}_s2"]].dropna()
        values = pair[[f"{column}_landsat", f"{column}_s2"]]
        centred = values - values.groupby([pair["region"], pair["crop"]]).transform("mean")
        x, y = centred[f"{column}_s2"], centred[f"{column}_landsat"]
        scale[column] = float((x * y).sum() / (x * x).sum())
    return scale


def load_region_table(yields_path: Path | str = DEFAULT_ALBERTA_YIELDS_PATH,
                      features_path: Path | str | list = DEFAULT_REGIONAL_FEATURES_PATH):
    """Join regional yields with regional crop-masked features by region, crop and year.

    ``features_path`` is one file or a list of per-month files; each adds its columns.
    """
    import pandas as pd

    paths = [features_path] if isinstance(features_path, (str, Path)) else list(features_path)
    for path, script in [(yields_path, "prepare_regional_yields.py --all-alberta")] + [
            (path, "export_regional_features.py") for path in paths]:
        if not Path(path).exists():
            raise ValueError(f"{path} not found; run scripts/{script} first.")
    table = pd.read_csv(yields_path)[["region", "year", "crop", "regional_yield"]]
    for index, path in enumerate(paths):
        features = pd.read_csv(path)
        # Pixel counts differ by month; keep the first file's.
        table = table.merge(features if index == 0 else features.drop(columns="pixels", errors="ignore"),
                            on=["region", "year", "crop"], how="inner")
    feature_columns = [column for column in table.columns if column.startswith("field_")]
    return table.dropna(subset=["regional_yield"] + feature_columns)


def _region_anomalies(rows, reference_rows, features=REGIONAL_FEATURES):
    """Relative yield and feature anomalies against each region-crop's reference years."""
    reference = reference_rows.groupby(["region", "crop"])[["regional_yield"] + list(features)].mean()
    keyed = rows.join(reference, on=["region", "crop"], rsuffix="_reference")
    out = rows[["region", "crop", "year"]].copy()
    out["relative_mean"] = keyed["regional_yield"] / keyed["regional_yield_reference"]
    for column in features:
        out[column] = keyed[column] - keyed[f"{column}_reference"]
    return out.dropna()


def _features_for(features, crop: str) -> list[str]:
    """``features`` is a column list, or a regional_satellite option name for per-crop columns."""
    return season_crop_features(features, crop) if isinstance(features, str) else list(features)


def fit_regional_season_models(region_table, exclude_year: int | None = None,
                               features=REGIONAL_FEATURES) -> dict:
    """Per-crop ridge from satellite anomalies to relative season yield, over region-years.

    ``features`` is a column list or a regional_satellite option name.
    """
    training = region_table[region_table["year"] != exclude_year]
    models = {}
    for crop, rows in training.groupby("crop"):
        columns = _features_for(features, crop)
        frame = _region_anomalies(rows, rows, columns)
        if frame["year"].nunique() >= 3:
            models[crop] = _fit_season_model(frame, columns)
    return models


def evaluate_regional_season_models(region_table, features=REGIONAL_FEATURES):
    """Leave one year out across all region-years: does the satellite anomaly predict season yield?"""
    import numpy as np
    import pandas as pd

    rows = []
    for crop, frame in region_table.groupby("crop"):
        errors, baseline_errors = [], []
        for year in sorted(frame["year"].unique()):
            training, test = frame[frame["year"] != year], frame[frame["year"] == year]
            models = fit_regional_season_models(training, features=features)
            if crop not in models:
                continue
            columns = _features_for(features, crop)
            anomalies = _region_anomalies(test, training, columns)
            predicted = models[crop][0].predict(anomalies[columns])
            errors.extend(np.abs(predicted - anomalies["relative_mean"]))
            baseline_errors.extend(np.abs(1 - anomalies["relative_mean"]))
        if errors:
            rows.append({"crop": crop, "region_years": len(errors), "relative_mae": float(np.mean(errors)),
                         "baseline_relative_mae": float(np.mean(baseline_errors)),
                         "skill": 1 - float(np.mean(errors)) / float(np.mean(baseline_errors))})
    return pd.DataFrame(rows)


def evaluate_season_field_model(table_path: Path | str = DEFAULT_TABLE_PATH, *,
                                years: tuple[int, ...] = DEV_YEARS, min_rows: int = 10,
                                deviation_features: str = "indices",
                                season_features: str = "weather",
                                regional_path: Path | str = DEFAULT_REGIONAL_PATH,
                                region_table=None, season_reading: str = "fields",
                                local_table=None, sensor_scale: dict | None = None) -> SeasonFieldArtifact:
    """Leave one season out over ``years``; score every variant per crop.

    ``deviation_features`` names a set in DEVIATION_FEATURE_SETS and
    ``season_features`` one in SEASON_FEATURE_SETS; ``regional`` reads
    ``regional_path`` (from scripts/prepare_regional_yields.py) and
    ``regional_satellite`` options use ``region_table`` (default:
    ``load_option_region_table``). ``season_reading`` (regional_satellite only)
    picks the farm's season features; ``local`` uses ``local_table`` (default:
    ``load_local_table()``). ``sensor_scale`` multiplies the farm's season
    features per column (default: ``cross_sensor_scale()`` for Landsat options).
    """
    import numpy as np
    import pandas as pd

    deviation_columns = DEVIATION_FEATURE_SETS[deviation_features]
    satellite_columns = [column for column in deviation_columns if column != HISTORY_FEATURE]
    season_columns = SEASON_FEATURE_SETS[season_features]
    regional = _load_regional(regional_path) if season_features == "regional" else None
    regional_satellite = season_features in REGIONAL_SEASON_MONTHS
    if season_reading not in SEASON_READINGS or (season_reading != "fields" and not regional_satellite):
        raise ValueError(f"Season reading {season_reading!r} needs a regional_satellite option; "
                         f"readings are {SEASON_READINGS}.")
    if regional_satellite and region_table is None:
        region_table = load_option_region_table(season_features)
    if regional_satellite and sensor_scale is None and _option_sensor(season_features) == "landsat":
        sensor_scale = cross_sensor_scale()
    if season_reading == "local" and local_table is None:
        local_table = load_local_table()
    table = pd.read_csv(table_path, dtype={"field_key": str, "section_key": str, "pids": str})
    needed = season_columns + satellite_columns + (
        [column.replace("field_", "quarter_", 1) for column in season_columns] if season_reading == "quarter" else [])
    missing = [column for column in needed if column not in table.columns]
    if missing:
        raise ValueError(f"{table_path} lacks columns {missing}; rebuild it (with weather for 'weather').")
    table = table[table["year"].isin(years)].reset_index(drop=True)
    # Field history draws on every crop's seasons, including crops too rare to model.
    history_source = table
    counts = table.groupby(["crop", "yield_unit"]).size()
    keep = {crop for (crop, _), count in counts.items() if count >= min_rows}
    table = table[table["crop"].isin(keep)].reset_index(drop=True)
    centred = [f"centred_{column}" for column in deviation_columns]
    table = pd.concat([table, _centre(table, satellite_columns).set_axis(
        [f"centred_{column}" for column in satellite_columns], axis=1)], axis=1)
    readings = _season_reading(table, season_reading, season_columns, local_table) if regional_satellite else None
    if readings is not None and sensor_scale:
        # Anomalies are differences, so scaling the readings scales the anomalies.
        readings = readings * pd.Series({column: sensor_scale.get(column, 1.0) for column in readings.columns})

    prediction_frames, season_rows = [], []
    for held_out in sorted(table["year"].unique()):
        train, test = table[table["year"] != held_out].copy(), table[table["year"] == held_out].copy()
        if HISTORY_FEATURE in deviation_columns:
            source = history_source[history_source["year"] != held_out]
            for frame, own_year_out in ((train, True), (test, False)):
                frame[HISTORY_FEATURE] = _field_history(source, frame, own_year_out=own_year_out)
                frame[f"centred_{HISTORY_FEATURE}"] = _centre(frame, [HISTORY_FEATURE])[HISTORY_FEATURE]
        crop_means = train.groupby("crop")["yield"].mean()

        aggregations = {column: (column, "mean") for column in season_columns}
        season_train = (train.groupby(["crop", "year"])
                        .agg(season_mean=("yield", "mean"), fields=("yield", "size"), **aggregations)
                        .reset_index())
        season_train["relative_mean"] = season_train["season_mean"] / season_train["crop"].map(crop_means)
        season_test = test.groupby("crop").agg(season_mean=("yield", "mean"), fields=("yield", "size"),
                                                **aggregations)
        ratios = {}
        if regional is not None:
            modelled_mean, ratios = _regional_season_means(season_train, season_test.index, held_out,
                                                           regional, crop_means)
            alpha = float("nan")
        elif regional_satellite:
            # The held-out season is excluded from the regional training years too.
            models = fit_regional_season_models(region_table, exclude_year=held_out, features=season_features)
            reference = readings.loc[list(zip(season_train["crop"], season_train["year"]))].groupby(level="crop").mean()
            current = readings.xs(held_out, level="year").reindex(season_test.index)
            ratios = _regional_satellite_relative(models, reference, current)
            modelled_mean = crop_means.reindex(season_test.index).copy()
            for crop, relative in ratios.items():
                modelled_mean[crop] = relative * crop_means[crop]
            alpha = float("nan")
        else:
            # Anomalies from each crop's training seasons put crops on a common footing.
            reference = season_train.groupby("crop")[season_columns].mean()
            anomaly_train = season_train.assign(
                **{column: season_train[column] - season_train["crop"].map(reference[column])
                   for column in season_columns})
            anomaly_test = season_test[season_columns] - reference.reindex(season_test.index)
            season_model, alpha = _fit_season_model(anomaly_train, season_columns)
            relative = pd.Series(season_model.predict(anomaly_test), index=season_test.index)
            modelled_mean = relative * crop_means.reindex(relative.index)

        deviation = _predict_deviation(_fit_deviation_models(train, centred), test, centred)

        # Ceiling: the season mean of the crop's other fields, standing in for a
        # published regional yield; a crop's only field that season falls back.
        sums = test.groupby("crop")["yield"].transform("sum")
        sizes = test.groupby("crop")["yield"].transform("size")
        others = ((sums - test["yield"]) / (sizes - 1)).where(sizes > 1)
        baseline = test["crop"].map(crop_means)
        known = others.fillna(baseline)
        modelled = test["crop"].map(modelled_mean)
        test = test.assign(
            pred_baseline=baseline, pred_season_model=modelled,
            pred_baseline_plus_field=baseline + deviation,
            pred_season_model_plus_field=modelled + deviation,
            pred_known_season=known, pred_known_season_plus_field=known + deviation,
            predicted_deviation=deviation, season_alpha=alpha)
        prediction_frames.append(test[["field_key", "year", "crop", "yield_unit", "yield", "predicted_deviation",
                                       "season_alpha"] + [f"pred_{variant}" for variant in VARIANTS]])
        for crop, row in season_test.iterrows():
            season_rows.append({"crop": crop, "year": held_out, "fields": int(row["fields"]),
                                "actual_season_mean": row["season_mean"], "baseline_mean": crop_means[crop],
                                "modelled_mean": modelled_mean[crop], "season_alpha": alpha,
                                **{column: row[column] for column in season_columns},
                                **({"regional_yield": regional.get((crop, held_out)),
                                    "farm_to_region_ratio": ratios.get(crop)} if regional is not None else {}),
                                **({"predicted_relative": ratios.get(crop)} if regional_satellite else {})})

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metric_rows = []
    for (crop, unit), frame in predictions.groupby(["crop", "yield_unit"], sort=True):
        baseline_mae = float(np.mean(np.abs(frame["pred_baseline"] - frame["yield"])))
        for variant in VARIANTS:
            error = frame[f"pred_{variant}"] - frame["yield"]
            mae = float(np.mean(np.abs(error)))
            metric_rows.append({"crop": crop, "yield_unit": unit, "variant": variant,
                                "field_years": len(frame), "seasons": int(frame["year"].nunique()),
                                "mae": mae, "bias": float(np.mean(error)),
                                "skill": 1 - mae / baseline_mae if baseline_mae else None})
    return SeasonFieldArtifact(metrics=pd.DataFrame(metric_rows), seasons=pd.DataFrame(season_rows),
                               predictions=predictions)
