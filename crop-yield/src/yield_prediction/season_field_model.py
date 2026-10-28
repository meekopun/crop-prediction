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
the farm's own satellite anomaly.

Evaluation holds out one season at a time and uses development years only
(2020-2024 by default); later years are dropped before anything is fitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .field_benchmark import DEFAULT_BENCHMARK_DIR, FEATURE_SETS
from .field_yield import DEFAULT_TABLE_PATH
from .regional_features import DEFAULT_REGIONAL_FEATURES_PATH, REGIONAL_FEATURES
from .regional_yields import DEFAULT_ALBERTA_YIELDS_PATH, DEFAULT_REGIONAL_PATH

DEV_YEARS = (2020, 2021, 2022, 2023, 2024)
PEAK_FEATURES = ["field_NDVI_jul", "field_NDRE_jul", "field_EVI_jul"]
DEVIATION_FEATURE_SETS = {"indices": FEATURE_SETS["indices"], "peak": PEAK_FEATURES}
# "regional" uses published yields from a separate file rather than table columns;
# "regional_satellite" trains on region-years but applies to the farm's peak features.
SEASON_FEATURE_SETS = {"weather": ["w_precip_may_aug", "w_gdd5_may_aug", "w_tmax_jul"],
                       "satellite": PEAK_FEATURES, "regional": [],
                       "regional_satellite": PEAK_FEATURES}
assert REGIONAL_FEATURES == PEAK_FEATURES, "regional and farm features must share names"
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


def _regional_satellite_relative(models, reference, season_features) -> dict:
    """Relative season yield per crop from the farm's anomaly against its training seasons.

    ``season_features`` is indexed by crop. Crops without a regional model or with a
    missing feature are left out, so callers fall back to the training mean.
    """
    anomaly = season_features[REGIONAL_FEATURES] - reference.reindex(season_features.index)
    relative = {}
    for crop in season_features.index:
        if crop in models and anomaly.loc[crop].notna().all():
            relative[crop] = models[crop][0].predict(anomaly.loc[[crop]])[0]
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


def load_region_table(yields_path: Path | str = DEFAULT_ALBERTA_YIELDS_PATH,
                      features_path: Path | str = DEFAULT_REGIONAL_FEATURES_PATH):
    """Join regional yields with regional crop-masked features by region, crop and year."""
    import pandas as pd

    for path, script in ((yields_path, "prepare_regional_yields.py --all-alberta"),
                         (features_path, "export_regional_features.py")):
        if not Path(path).exists():
            raise ValueError(f"{path} not found; run scripts/{script} first.")
    yields = pd.read_csv(yields_path)[["region", "year", "crop", "regional_yield"]]
    features = pd.read_csv(features_path)
    return yields.merge(features, on=["region", "year", "crop"], how="inner").dropna(
        subset=["regional_yield"] + REGIONAL_FEATURES)


def _region_anomalies(rows, reference_rows):
    """Relative yield and feature anomalies against each region-crop's reference years."""
    reference = reference_rows.groupby(["region", "crop"])[["regional_yield"] + REGIONAL_FEATURES].mean()
    keyed = rows.join(reference, on=["region", "crop"], rsuffix="_reference")
    out = rows[["region", "crop", "year"]].copy()
    out["relative_mean"] = keyed["regional_yield"] / keyed["regional_yield_reference"]
    for column in REGIONAL_FEATURES:
        out[column] = keyed[column] - keyed[f"{column}_reference"]
    return out.dropna()


def fit_regional_season_models(region_table, exclude_year: int | None = None) -> dict:
    """Per-crop ridge from satellite anomalies to relative season yield, over region-years."""
    training = region_table[region_table["year"] != exclude_year]
    anomalies = _region_anomalies(training, training)
    models = {}
    for crop, frame in anomalies.groupby("crop"):
        if frame["year"].nunique() >= 3:
            models[crop] = _fit_season_model(frame, REGIONAL_FEATURES)
    return models


def evaluate_regional_season_models(region_table):
    """Leave one year out across all region-years: does the satellite anomaly predict season yield?"""
    import numpy as np
    import pandas as pd

    rows = []
    for crop, frame in region_table.groupby("crop"):
        errors, baseline_errors = [], []
        for year in sorted(frame["year"].unique()):
            training, test = frame[frame["year"] != year], frame[frame["year"] == year]
            models = fit_regional_season_models(training)
            if crop not in models:
                continue
            anomalies = _region_anomalies(test, training)
            predicted = models[crop][0].predict(anomalies[REGIONAL_FEATURES])
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
                                region_table=None) -> SeasonFieldArtifact:
    """Leave one season out over ``years``; score every variant per crop.

    ``deviation_features`` names a set in DEVIATION_FEATURE_SETS and
    ``season_features`` one in SEASON_FEATURE_SETS; ``regional`` reads
    ``regional_path`` (from scripts/prepare_regional_yields.py) and
    ``regional_satellite`` uses ``region_table`` (default: ``load_region_table()``).
    """
    import numpy as np
    import pandas as pd

    deviation_columns = DEVIATION_FEATURE_SETS[deviation_features]
    season_columns = SEASON_FEATURE_SETS[season_features]
    regional = _load_regional(regional_path) if season_features == "regional" else None
    if season_features == "regional_satellite" and region_table is None:
        region_table = load_region_table()
    table = pd.read_csv(table_path, dtype={"field_key": str, "section_key": str, "pids": str})
    missing = [column for column in season_columns + deviation_columns if column not in table.columns]
    if missing:
        raise ValueError(f"{table_path} lacks columns {missing}; rebuild it (with weather for 'weather').")
    table = table[table["year"].isin(years)].reset_index(drop=True)
    counts = table.groupby(["crop", "yield_unit"]).size()
    keep = {crop for (crop, _), count in counts.items() if count >= min_rows}
    table = table[table["crop"].isin(keep)].reset_index(drop=True)
    centred = [f"centred_{column}" for column in deviation_columns]
    table = pd.concat([table, _centre(table, deviation_columns).set_axis(centred, axis=1)], axis=1)

    prediction_frames, season_rows = [], []
    for held_out in sorted(table["year"].unique()):
        train, test = table[table["year"] != held_out], table[table["year"] == held_out].copy()
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
        elif season_features == "regional_satellite":
            # The held-out season is excluded from the regional training years too.
            models = fit_regional_season_models(region_table, exclude_year=held_out)
            reference = season_train.groupby("crop")[season_columns].mean()
            ratios = _regional_satellite_relative(models, reference, season_test)
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
                                **({"predicted_relative": ratios.get(crop)}
                                   if season_features == "regional_satellite" else {})})

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
