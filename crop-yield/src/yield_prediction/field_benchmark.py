"""Benchmark field-level yield models against a per-crop training-mean baseline.

Each crop with enough field-years is modelled on its own, in its reported unit.
Two grouped evaluations are run: ``year`` holds out one whole season at a time
(can it predict a season it has not seen?) and ``section`` holds out whole ATS
sections across all years (can it predict land it has not seen?). Every model
is fitted inside the training folds only. ``holdout_field_yield`` instead picks
one configuration per crop from earlier years and scores it once on a later year.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .field_yield import DEFAULT_OUTPUT_DIR, DEFAULT_TABLE_PATH
from .harvest_features import FEATURE_BANDS, MONTHS, S2_INDICES

DEFAULT_BENCHMARK_DIR = DEFAULT_OUTPUT_DIR / "benchmark"
SPLITS = ("year", "section")
FEATURE_SETS = {
    "ndvi": [f"field_NDVI_{month}" for month in MONTHS.values()],
    "indices": [f"field_{index}_{month}" for month in MONTHS.values() for index in S2_INDICES],
    "all": [f"field_{band}_{month}" for month in MONTHS.values() for band in FEATURE_BANDS],
}


@dataclass(frozen=True)
class FieldBenchmarkArtifact:
    metrics: object
    predictions: object


@dataclass(frozen=True)
class FieldHoldoutArtifact:
    selection: object
    results: object
    predictions: object


def _models(random_state: int) -> dict:
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        # Ridge picks its penalty by leave-one-out within each training fold.
        "ridge": lambda: Pipeline([("impute", SimpleImputer(strategy="median")),
                                   ("scale", StandardScaler()),
                                   ("model", RidgeCV(alphas=np.logspace(-2, 5, 29)))]),
        # Shallow and heavily constrained: training folds hold only 10-40 field-years.
        "gbm": lambda: HistGradientBoostingRegressor(max_depth=2, learning_rate=0.05, max_iter=150,
                                                     min_samples_leaf=3, l2_regularization=1.0,
                                                     random_state=random_state),
    }


def _candidates(frame, models) -> list[tuple]:
    """Baseline first, then every model on every feature set, as (name, feature_set, make, X)."""
    from sklearn.dummy import DummyRegressor

    runs = [("mean_baseline", "none", lambda: DummyRegressor(strategy="mean"), frame[["year"]])]
    runs += [(name, feature_set, make, frame[columns])
             for feature_set, columns in FEATURE_SETS.items() for name, make in models.items()]
    return runs


def _out_of_fold(make_model, X, y, folds):
    import numpy as np

    predicted = np.full(len(y), np.nan)
    for train, test in folds:
        model = make_model()
        model.fit(X.iloc[train], y[train])
        predicted[test] = model.predict(X.iloc[test])
    return predicted


def _scores(y, predicted) -> dict:
    import numpy as np
    from sklearn.metrics import r2_score

    error = predicted - y
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias": float(np.mean(error)),
        "r2": float(r2_score(y, predicted)) if len(set(y)) > 1 else None,
    }


def benchmark_field_yield(table_path: Path | str = DEFAULT_TABLE_PATH, *, min_rows: int = 10,
                          random_state: int = 42) -> FieldBenchmarkArtifact:
    """Score baseline, ridge and gradient boosting per crop under both grouped splits."""
    import pandas as pd
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

    table = pd.read_csv(table_path, dtype={"field_key": str, "section_key": str, "pids": str})
    models = _models(random_state)
    metric_rows, prediction_frames = [], []
    for (crop, unit), frame in table.groupby(["crop", "yield_unit"], sort=True):
        if len(frame) < min_rows:
            continue
        frame = frame.reset_index(drop=True)
        y = frame["yield"].to_numpy(dtype=float)
        for split in SPLITS:
            groups = frame["year"] if split == "year" else frame["section_key"]
            group_count = int(groups.nunique())
            if group_count < 2:
                continue
            splitter = LeaveOneGroupOut() if split == "year" else GroupKFold(n_splits=min(5, group_count))
            folds = list(splitter.split(frame, y, groups))
            baseline_mae = None
            for name, feature_set, make, X in _candidates(frame, models):
                predicted = _out_of_fold(make, X, y, folds)
                scores = _scores(y, predicted)
                if name == "mean_baseline":
                    baseline_mae = scores["mae"]
                metric_rows.append({
                    "crop": crop, "yield_unit": unit, "split": split, "feature_set": feature_set,
                    "model": name, "field_years": len(frame), "years": int(frame["year"].nunique()),
                    "sections": int(frame["section_key"].nunique()), "folds": len(folds), **scores,
                    "mae_skill": 1 - scores["mae"] / baseline_mae if baseline_mae else None,
                })
                prediction_frames.append(frame[["field_key", "year", "crop", "yield_unit", "yield"]].assign(
                    split=split, feature_set=feature_set, model=name, predicted=predicted))
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return FieldBenchmarkArtifact(metrics=metrics, predictions=predictions)


def holdout_field_yield(table_path: Path | str = DEFAULT_TABLE_PATH, *, test_year: int = 2025,
                        min_rows: int = 10, random_state: int = 42) -> FieldHoldoutArtifact:
    """Choose one configuration per crop on earlier years, then score it once on ``test_year``.

    Selection uses leave-one-year-out on years before ``test_year`` only; the
    baseline is a candidate, so a crop keeps it when no model beats it there.
    Ties go to the earlier (simpler) candidate. The chosen configuration and the
    baseline are then refitted on all earlier years and scored on ``test_year``.
    """
    import pandas as pd
    from sklearn.model_selection import LeaveOneGroupOut

    table = pd.read_csv(table_path, dtype={"field_key": str, "section_key": str, "pids": str})
    models = _models(random_state)
    selection_rows, result_rows, prediction_frames = [], [], []
    for (crop, unit), frame in table.groupby(["crop", "yield_unit"], sort=True):
        if len(frame) < min_rows:
            continue
        dev = frame[frame["year"] < test_year].reset_index(drop=True)
        test = frame[frame["year"] == test_year].reset_index(drop=True)
        base = {"crop": crop, "yield_unit": unit, "dev_field_years": len(dev),
                "dev_years": int(dev["year"].nunique()), "test_field_years": len(test)}
        if test.empty or dev["year"].nunique() < 2:
            result_rows.append({**base, "status": "skipped: no test rows or under two development years"})
            continue
        y_dev = dev["yield"].to_numpy(dtype=float)
        y_test = test["yield"].to_numpy(dtype=float)
        folds = list(LeaveOneGroupOut().split(dev, y_dev, dev["year"]))
        dev_candidates = _candidates(dev, models)
        dev_scores = []
        for name, feature_set, make, X in dev_candidates:
            mae = _scores(y_dev, _out_of_fold(make, X, y_dev, folds))["mae"]
            dev_scores.append(mae)
            selection_rows.append({"crop": crop, "yield_unit": unit, "model": name,
                                   "feature_set": feature_set, "dev_mae": mae})
        chosen_index = min(range(len(dev_scores)), key=lambda index: (dev_scores[index], index))
        test_candidates = _candidates(test, models)
        fitted = {}
        for index in (0, chosen_index):
            name, feature_set, make, X_dev = dev_candidates[index]
            model = make()
            model.fit(X_dev, y_dev)
            fitted[index] = model.predict(test_candidates[index][3])
        chosen, baseline = _scores(y_test, fitted[chosen_index]), _scores(y_test, fitted[0])
        result_rows.append({
            **base, "status": "scored",
            "chosen_model": dev_candidates[chosen_index][0],
            "chosen_feature_set": dev_candidates[chosen_index][1],
            "dev_mae_chosen": dev_scores[chosen_index], "dev_mae_baseline": dev_scores[0],
            "test_mae_chosen": chosen["mae"], "test_mae_baseline": baseline["mae"],
            "test_rmse_chosen": chosen["rmse"], "test_bias_chosen": chosen["bias"],
            "test_mae_skill": 1 - chosen["mae"] / baseline["mae"] if baseline["mae"] else None,
        })
        prediction_frames.append(test[["field_key", "year", "crop", "yield_unit", "yield"]].assign(
            chosen_model=dev_candidates[chosen_index][0],
            chosen_feature_set=dev_candidates[chosen_index][1],
            predicted_chosen=fitted[chosen_index], predicted_baseline=fitted[0]))
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return FieldHoldoutArtifact(selection=pd.DataFrame(selection_rows), results=pd.DataFrame(result_rows),
                                predictions=predictions)
