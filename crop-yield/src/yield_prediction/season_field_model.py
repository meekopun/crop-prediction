"""Two-part yield model: a season level from weather plus each field's deviation.

A field-year's yield is split into its crop's season mean and the field's
deviation from it. The deviation model is a per-crop ridge regression on
satellite features centred on the crop's season mean, so it can only learn which
fields do better or worse than their season. The season model is one ridge
regression, pooled over crops, that predicts the season mean relative to the
crop's long-run mean from season-level features: ERA5-Land weather totals or the
season mean of the crop's peak-season satellite indices. Season features enter
as anomalies from the crop's training-season mean.

Evaluation holds out one season at a time and uses development years only
(2020-2024 by default); later years are dropped before anything is fitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .field_benchmark import DEFAULT_BENCHMARK_DIR, FEATURE_SETS
from .field_yield import DEFAULT_TABLE_PATH

DEV_YEARS = (2020, 2021, 2022, 2023, 2024)
PEAK_FEATURES = ["field_NDVI_jul", "field_NDRE_jul", "field_EVI_jul"]
DEVIATION_FEATURE_SETS = {"indices": FEATURE_SETS["indices"], "peak": PEAK_FEATURES}
SEASON_FEATURE_SETS = {"weather": ["w_precip_may_aug", "w_gdd5_may_aug", "w_tmax_jul"],
                       "satellite": PEAK_FEATURES}
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


def evaluate_season_field_model(table_path: Path | str = DEFAULT_TABLE_PATH, *,
                                years: tuple[int, ...] = DEV_YEARS, min_rows: int = 10,
                                deviation_features: str = "indices",
                                season_features: str = "weather") -> SeasonFieldArtifact:
    """Leave one season out over ``years``; score every variant per crop.

    ``deviation_features`` names a set in DEVIATION_FEATURE_SETS and
    ``season_features`` one in SEASON_FEATURE_SETS.
    """
    import numpy as np
    import pandas as pd

    deviation_columns = DEVIATION_FEATURE_SETS[deviation_features]
    season_columns = SEASON_FEATURE_SETS[season_features]
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
        # Anomalies from each crop's training seasons put crops on a common footing.
        reference = season_train.groupby("crop")[season_columns].mean()
        anomaly_train = season_train.assign(
            **{column: season_train[column] - season_train["crop"].map(reference[column])
               for column in season_columns})
        anomaly_test = season_test[season_columns] - reference.reindex(season_test.index)
        season_model, alpha = _fit_season_model(anomaly_train, season_columns)
        relative = pd.Series(season_model.predict(anomaly_test), index=season_test.index)
        modelled_mean = relative * crop_means.reindex(relative.index)

        deviation = pd.Series(0.0, index=test.index)
        for crop, crop_test in test.groupby("crop"):
            crop_train = train[train["crop"] == crop]
            # Seasons with one field carry no within-season contrast.
            crop_train = crop_train[crop_train.groupby("year")["yield"].transform("size") >= 2]
            if len(crop_train) >= 5:
                target = crop_train["yield"] - crop_train.groupby("year")["yield"].transform("mean")
                deviation[crop_test.index] = _deviation_model().fit(crop_train[centred], target).predict(
                    crop_test[centred])

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
                                **{column: row[column] for column in season_columns}})

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
