#!/usr/bin/env python3
"""Train one-vs-rest binary crop models from crop classification Sentinel-2 feature batches."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from quarter_section_batch_utils import ROOT, load_config, output_paths, update_tracker


TARGET_COLUMN = "crop_label"
GROUP_COLUMNS = ("quarter_section", "year")
EXCLUDED_COLUMNS = {
    TARGET_COLUMN,
    "label_code",
    "lat",
    "lon",
    "descriptor",
    "meridian",
    "range",
    "township",
    "section",
    "quarter",
    "quarter_section",
    "year",
}


@dataclass(frozen=True)
class CropMetric:
    crop_name: str
    positive_rows: int
    selected_feature_count: int
    evaluated_rows: int
    precision: float | None
    recall: float | None
    f1: float | None
    roc_auc: float | None
    average_precision: float | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "crop_classification_pipeline_config.json",
    )
    parser.add_argument("--input-glob", default="batch_features/*_features.csv")
    parser.add_argument("--batch-ids", nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--top-features-per-crop", type=int, default=32)
    parser.add_argument("--logistic-c", type=float, default=1.0)
    return parser


def load_feature_frame(input_glob: str, batch_ids: list[str] | None) -> pd.DataFrame:
    paths = sorted(ROOT.glob(input_glob))
    if batch_ids:
        wanted = {f"{batch_id}_features.csv" for batch_id in batch_ids}
        paths = [path for path in paths if path.name in wanted]
    if not paths:
        raise ValueError("No feature CSVs matched the requested input selection.")
    frames = [pd.read_csv(path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    numeric = frame.select_dtypes(include=["number", "bool"]).columns
    return [column for column in numeric if column not in EXCLUDED_COLUMNS]


def build_groups(frame: pd.DataFrame) -> pd.Series:
    return frame[list(GROUP_COLUMNS)].astype(str).agg("|".join, axis=1)


def rank_features(frame: pd.DataFrame, crop_name: str, candidate_features: list[str], top_k: int) -> list[str]:
    binary = (frame[TARGET_COLUMN].astype(str) == crop_name).astype(float).to_numpy()
    scores: list[tuple[str, float]] = []
    for column in candidate_features:
        values = frame[column].to_numpy(dtype=float)
        valid = np.isfinite(values)
        if valid.sum() < 2 or np.unique(binary[valid]).size < 2:
            scores.append((column, 0.0))
            continue
        corr = np.corrcoef(values[valid], binary[valid])[0, 1]
        scores.append((column, 0.0 if np.isnan(corr) else float(abs(corr))))
    scores.sort(key=lambda item: item[1], reverse=True)
    return [feature for feature, _score in scores[:top_k]]


def make_model(logistic_c: float) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "estimator",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    solver="lbfgs",
                    C=logistic_c,
                ),
            ),
        ]
    )


def train_crop_models(
    frame: pd.DataFrame,
    *,
    cv_splits: int,
    top_features_per_crop: int,
    logistic_c: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = build_groups(frame)
    candidate_features = feature_columns(frame)
    if not candidate_features:
        raise ValueError("No numeric Sentinel-2 features were found.")

    metrics: list[CropMetric] = []
    prediction_rows: list[pd.DataFrame] = []
    for crop_name in sorted(frame[TARGET_COLUMN].dropna().astype(str).unique()):
        binary = (frame[TARGET_COLUMN].astype(str) == crop_name).astype(int)
        selected = rank_features(frame, crop_name, candidate_features, top_features_per_crop)
        X = frame[selected]
        preds = pd.DataFrame(
            {
                "quarter_section": frame["quarter_section"],
                "year": frame["year"],
                "crop_name": crop_name,
                "is_crop": binary,
                "probability": np.nan,
            }
        )

        splitter = StratifiedGroupKFold(
            n_splits=min(cv_splits, groups.nunique()),
            shuffle=True,
            random_state=42,
        )
        evaluated = np.zeros(len(frame), dtype=bool)
        for train_idx, test_idx in splitter.split(X, binary, groups):
            y_train = binary.iloc[train_idx]
            y_test = binary.iloc[test_idx]
            if y_train.nunique() < 2 or y_test.nunique() < 2:
                continue
            model = make_model(logistic_c)
            model.fit(X.iloc[train_idx], y_train)
            preds.loc[test_idx, "probability"] = model.predict_proba(X.iloc[test_idx])[:, 1]
            evaluated[test_idx] = True

        y_true = preds.loc[evaluated, "is_crop"].astype(int)
        y_prob = preds.loc[evaluated, "probability"].astype(float)
        y_pred = (y_prob >= 0.5).astype(int)
        metrics.append(
            CropMetric(
                crop_name=crop_name,
                positive_rows=int(binary.sum()),
                selected_feature_count=len(selected),
                evaluated_rows=int(evaluated.sum()),
                precision=float(precision_score(y_true, y_pred, zero_division=0)) if evaluated.any() else None,
                recall=float(recall_score(y_true, y_pred, zero_division=0)) if evaluated.any() else None,
                f1=float(f1_score(y_true, y_pred, zero_division=0)) if evaluated.any() else None,
                roc_auc=float(roc_auc_score(y_true, y_prob)) if evaluated.any() and y_true.nunique() == 2 else None,
                average_precision=float(average_precision_score(y_true, y_prob)) if evaluated.any() else None,
            )
        )
        preds["predicted_label"] = np.where(preds["probability"].fillna(0) >= 0.5, "yes", "no")
        prediction_rows.append(preds)

    return (
        pd.DataFrame(asdict(metric) for metric in metrics),
        pd.concat(prediction_rows, ignore_index=True),
    )


def run() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    paths = output_paths(config)
    output_dir = args.output_dir or paths["model_output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_feature_frame(args.input_glob, args.batch_ids)
    metrics, predictions = train_crop_models(
        frame,
        cv_splits=args.cv_splits,
        top_features_per_crop=args.top_features_per_crop,
        logistic_c=args.logistic_c,
    )

    metrics.to_csv(output_dir / "binary_crop_model_metrics.csv", index=False)
    predictions.to_csv(output_dir / "binary_crop_predictions.csv", index=False)

    if args.batch_ids:
        for batch_id in args.batch_ids:
            update_tracker(
                paths["tracker_csv"],
                batch_id,
                {"model_status": "completed"},
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
