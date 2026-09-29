from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .data import DEFAULT_DATASET_PATH, clean_records, load_records


@dataclass(frozen=True)
class ModelResult:
    model: str
    test_r2: float
    mse: float
    mae: float
    mape: float
    cv_mean_r2: float


@dataclass(frozen=True)
class PixelYieldModelMetrics:
    crop: str
    model: str
    feature_set: str
    pixel_count: int
    field_year_count: int
    feature_count: int
    cv_splits: int
    mean_mae: float
    mean_rmse: float
    mean_r2: float | None


@dataclass(frozen=True)
class PixelYieldTrainingArtifact:
    metrics: list[PixelYieldModelMetrics]
    predictions: object
    feature_columns: list[str]


@dataclass(frozen=True)
class PixelCropClassificationMetrics:
    model: str
    feature_set: str
    pixel_count: int
    field_year_count: int
    class_count: int
    feature_count: int
    cv_splits: int
    grouped_accuracy: float | None
    grouped_balanced_accuracy: float | None
    grouped_macro_f1: float | None
    evaluated_pixels: int
    notes: str


@dataclass(frozen=True)
class PixelCropClassificationArtifact:
    metrics: list[PixelCropClassificationMetrics]
    predictions: object
    feature_columns: list[str]


def _require_ml_dependencies() -> None:
    try:
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Benchmarking requires optional ML dependencies. "
            "Install them with `pip install -e '.[ml]'`."
        ) from exc


def _build_pixel_feature_sets(frame: object) -> dict[str, list[str]]:
    import pandas as pd

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")

    base_excluded = {
        "system:index",
        "crop",
        "province",
        "quarter_section",
        "row_id",
        "yield_bu_ac",
        "yield_original",
        "yield_original_unit",
        ".geo",
    }
    numeric_columns = [
        column
        for column in frame.select_dtypes(include=["number"]).columns
        if column not in base_excluded
    ]
    satellite_prefixes = (
        "CIgreen",
        "CIrededge",
        "EVI",
        "MSI",
        "NDMI",
        "NDRE",
        "NDRE1",
        "NDRE2",
        "NDVI",
        "VH",
        "VV",
    )
    def is_satellite_feature(column: str) -> bool:
        if column.startswith(satellite_prefixes):
            return True
        return any(f"_{prefix}" in column for prefix in satellite_prefixes)

    satellite = [column for column in numeric_columns if is_satellite_feature(column)]
    ndvi_only = [
        column
        for column in numeric_columns
        if column.startswith("NDVI") or "_NDVI_" in column
    ]
    terrain_soil = [
        column
        for column in numeric_columns
        if column
        in {
            "elevation",
            "latitude",
            "longitude",
            "slope",
            "soil_moisture_am_mean",
            "soil_oc",
            "soil_ph",
            "soil_sand",
        }
    ]
    nonleaky = satellite + terrain_soil
    all_features = list(numeric_columns)
    return {
        "ndvi_only": ndvi_only,
        "satellite": satellite,
        "nonleaky": nonleaky,
        "all": all_features,
    }


def train_pixel_yield_models(
    csv_path: Path | str,
    *,
    target_column: str = "yield_bu_ac",
    crop_column: str = "crop",
    group_column: str = "row_id",
    feature_set: str = "nonleaky",
    alpha: float = 1.0,
) -> PixelYieldTrainingArtifact:
    _require_ml_dependencies()

    import numpy as np
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    frame = pd.read_csv(csv_path)
    feature_sets = _build_pixel_feature_sets(frame)
    if feature_set not in feature_sets:
        raise ValueError(
            f"Unknown feature set '{feature_set}'. Expected one of {sorted(feature_sets)}."
        )
    feature_columns = feature_sets[feature_set]

    model_specs = [
        ("ols", LinearRegression()),
        ("ridge", Ridge(alpha=alpha)),
    ]

    metrics: list[PixelYieldModelMetrics] = []
    prediction_frames: list[pd.DataFrame] = []

    for crop_name, crop_frame in frame.groupby(crop_column, sort=True):
        crop_frame = crop_frame.copy()
        groups = crop_frame[group_column]
        unique_groups = int(groups.nunique(dropna=True))
        crop_predictions = crop_frame[
            [crop_column, group_column, "year", "quarter_section", target_column]
        ].copy()

        X = crop_frame[feature_columns]
        y = crop_frame[target_column]

        for model_name, estimator in model_specs:
            prediction_column = f"pred_{model_name}_{feature_set}"
            crop_predictions[prediction_column] = np.nan

            if unique_groups >= 2 and y.nunique(dropna=True) >= 2:
                cv_splits = min(5, unique_groups)
                splitter = GroupKFold(n_splits=cv_splits)
                fold_mae: list[float] = []
                fold_rmse: list[float] = []
                fold_r2: list[float] = []

                for train_idx, test_idx in splitter.split(X, y, groups):
                    pipeline = Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                            ("estimator", estimator),
                        ]
                    )
                    pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
                    fold_predictions = pipeline.predict(X.iloc[test_idx])
                    crop_predictions.iloc[test_idx, crop_predictions.columns.get_loc(prediction_column)] = (
                        fold_predictions
                    )
                    fold_mae.append(
                        float(mean_absolute_error(y.iloc[test_idx], fold_predictions))
                    )
                    fold_rmse.append(
                        float(
                            np.sqrt(mean_squared_error(y.iloc[test_idx], fold_predictions))
                        )
                    )
                    if y.iloc[test_idx].nunique(dropna=True) >= 2:
                        fold_r2.append(
                            float(r2_score(y.iloc[test_idx], fold_predictions))
                        )

                mean_r2 = float(np.mean(fold_r2)) if fold_r2 else None
                metrics.append(
                    PixelYieldModelMetrics(
                        crop=str(crop_name),
                        model=model_name,
                        feature_set=feature_set,
                        pixel_count=len(crop_frame),
                        field_year_count=unique_groups,
                        feature_count=len(feature_columns),
                        cv_splits=cv_splits,
                        mean_mae=float(np.mean(fold_mae)),
                        mean_rmse=float(np.mean(fold_rmse)),
                        mean_r2=mean_r2,
                    )
                )
            else:
                metrics.append(
                    PixelYieldModelMetrics(
                        crop=str(crop_name),
                        model=model_name,
                        feature_set=feature_set,
                        pixel_count=len(crop_frame),
                        field_year_count=unique_groups,
                        feature_count=len(feature_columns),
                        cv_splits=0,
                        mean_mae=float("nan"),
                        mean_rmse=float("nan"),
                        mean_r2=None,
                    )
                )

            final_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("estimator", estimator),
                ]
            )
            final_pipeline.fit(X, y)
            crop_predictions[f"fitted_{model_name}_{feature_set}"] = final_pipeline.predict(X)

        prediction_frames.append(crop_predictions)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    return PixelYieldTrainingArtifact(
        metrics=metrics,
        predictions=predictions,
        feature_columns=feature_columns,
    )


def train_pixel_crop_classifier(
    csv_path: Path | str,
    *,
    target_column: str = "crop",
    group_column: str = "row_id",
    feature_set: str = "nonleaky",
    random_state: int = 42,
) -> PixelCropClassificationArtifact:
    _require_ml_dependencies()

    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    frame = pd.read_csv(csv_path)
    feature_sets = _build_pixel_feature_sets(frame)
    if feature_set not in feature_sets:
        raise ValueError(
            f"Unknown feature set '{feature_set}'. Expected one of {sorted(feature_sets)}."
        )
    feature_columns = feature_sets[feature_set]

    X = frame[feature_columns]
    y = frame[target_column].astype(str)
    groups = frame[group_column].astype(str)
    unique_groups = int(groups.nunique(dropna=True))
    class_count = int(y.nunique(dropna=True))

    model_specs = [
        (
            "logistic",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "estimator",
                        LogisticRegression(
                            max_iter=5000,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        ),
        (
            "random_forest",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "estimator",
                        RandomForestClassifier(
                            n_estimators=300,
                            random_state=random_state,
                            class_weight="balanced",
                        ),
                    ),
                ]
            ),
        ),
    ]

    predictions = frame[
        ["crop", "row_id", "year", "quarter_section", "latitude", "longitude"]
    ].copy()
    metrics: list[PixelCropClassificationMetrics] = []

    can_group_validate = unique_groups >= 2 and class_count >= 2
    splitter = None
    cv_splits = 0
    if can_group_validate:
        cv_splits = min(5, unique_groups)
        splitter = StratifiedGroupKFold(
            n_splits=cv_splits,
            shuffle=True,
            random_state=random_state,
        )

    for model_name, pipeline in model_specs:
        pred_col = f"pred_{model_name}_{feature_set}"
        fit_col = f"fitted_{model_name}_{feature_set}"
        predictions[pred_col] = pd.Series(pd.NA, index=predictions.index, dtype="object")

        evaluated_mask = np.zeros(len(frame), dtype=bool)
        skipped_folds = 0
        if splitter is not None:
            for train_idx, test_idx in splitter.split(X, y, groups):
                train_classes = set(y.iloc[train_idx].unique())
                test_classes = set(y.iloc[test_idx].unique())
                if not test_classes.issubset(train_classes):
                    skipped_folds += 1
                    continue
                pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
                fold_predictions = pipeline.predict(X.iloc[test_idx])
                predictions.iloc[
                    test_idx, predictions.columns.get_loc(pred_col)
                ] = fold_predictions
                evaluated_mask[test_idx] = True

        evaluated_pixels = int(evaluated_mask.sum())
        if evaluated_pixels > 0:
            grouped_accuracy = float(
                accuracy_score(y[evaluated_mask], predictions.loc[evaluated_mask, pred_col])
            )
            grouped_balanced_accuracy = float(
                balanced_accuracy_score(
                    y[evaluated_mask], predictions.loc[evaluated_mask, pred_col]
                )
            )
            grouped_macro_f1 = float(
                f1_score(
                    y[evaluated_mask],
                    predictions.loc[evaluated_mask, pred_col],
                    average="macro",
                )
            )
            notes = ""
            if skipped_folds:
                notes = (
                    f"Skipped {skipped_folds} grouped fold(s) because the held-out crop "
                    "class was absent from training data."
                )
        else:
            grouped_accuracy = None
            grouped_balanced_accuracy = None
            grouped_macro_f1 = None
            if splitter is None:
                notes = "Insufficient grouped class coverage for validation."
            else:
                notes = (
                    "No grouped folds were evaluable because held-out crop classes were "
                    "absent from the corresponding training folds."
                )

        metrics.append(
            PixelCropClassificationMetrics(
                model=model_name,
                feature_set=feature_set,
                pixel_count=len(frame),
                field_year_count=unique_groups,
                class_count=class_count,
                feature_count=len(feature_columns),
                cv_splits=cv_splits,
                grouped_accuracy=grouped_accuracy,
                grouped_balanced_accuracy=grouped_balanced_accuracy,
                grouped_macro_f1=grouped_macro_f1,
                evaluated_pixels=evaluated_pixels,
                notes=notes,
            )
        )

        pipeline.fit(X, y)
        predictions[fit_col] = pipeline.predict(X)

    return PixelCropClassificationArtifact(
        metrics=metrics,
        predictions=predictions,
        feature_columns=feature_columns,
    )


def benchmark_models(
    path: Path | str = DEFAULT_DATASET_PATH,
    *,
    test_size: float = 0.3,
    random_state: int = 42,
    cv_splits: int = 5,
) -> list[ModelResult]:
    _require_ml_dependencies()

    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import (
        BaggingRegressor,
        GradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import (
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_squared_error,
        r2_score,
    )
    from sklearn.model_selection import KFold, cross_val_score, train_test_split
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.tree import DecisionTreeRegressor

    try:
        from xgboost import XGBRegressor
    except ImportError:
        XGBRegressor = None

    records = clean_records(load_records(path))
    frame = pd.DataFrame(records)

    X = frame.drop(columns=["hg/ha_yield"])
    y = frame["hg/ha_yield"]

    categorical = ["Area", "Item"]
    numeric = [column for column in X.columns if column not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            ),
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ]
    )

    model_specs: list[tuple[str, object]] = [
        ("Linear Regression", LinearRegression()),
        ("Random Forest", RandomForestRegressor(random_state=random_state)),
        (
            "Gradient Boost",
            GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=random_state,
            ),
        ),
        ("KNN", KNeighborsRegressor(n_neighbors=5)),
        ("Decision Tree", DecisionTreeRegressor(random_state=random_state)),
        ("Bagging Regressor", BaggingRegressor(n_estimators=150, random_state=random_state)),
    ]
    if XGBRegressor is not None:
        model_specs.append(
            (
                "XGBoost",
                XGBRegressor(
                    random_state=random_state,
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=6,
                    objective="reg:squarederror",
                ),
            )
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    results: list[ModelResult] = []
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    for name, estimator in model_specs:
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("estimator", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="r2")

        results.append(
            ModelResult(
                model=name,
                test_r2=float(r2_score(y_test, predictions)),
                mse=float(mean_squared_error(y_test, predictions)),
                mae=float(mean_absolute_error(y_test, predictions)),
                mape=float(mean_absolute_percentage_error(y_test, predictions)),
                cv_mean_r2=float(np.mean(cv_scores)),
            )
        )

    return sorted(results, key=lambda result: result.test_r2, reverse=True)
