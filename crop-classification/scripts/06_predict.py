"""
Step 6: Apply trained models to new pixel or polygon feature tables.

Supports either a single saved model bundle or a directory of one-vs-rest
crop models. If temporal features were used during training, they are
reconstructed automatically at prediction time from the raw monthly columns.

Usage:
    python 06_predict.py --model models/catboost/model.joblib --features data/sentinel2_features_new.csv
    python 06_predict.py --model models/catboost_one_vs_rest --features data/sentinel2_features_new.csv
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

from common import OVR_MANIFEST
from ml_features import add_temporal_features, filter_columns_by_months

MODEL_DIR = "models"
DATA_DIR = "data"

NON_FEATURE_COLS = {"pid", "crop_label", "aafc_code", "system:index", ".geo"}


def prepare_features(df, bundle):
    work = df.copy()
    months = bundle.get("months")
    work = filter_columns_by_months(work, months, exclude_cols=NON_FEATURE_COLS)
    if bundle.get("temporal_features"):
        work = add_temporal_features(work, exclude_cols=NON_FEATURE_COLS, months=months)

    feature_cols = bundle["feature_cols"]
    missing = [c for c in feature_cols if c not in work.columns]
    if missing:
        raise SystemExit(f"Feature CSV is missing columns the model needs: {missing}")

    X = work[feature_cols].apply(pd.to_numeric, errors="coerce")
    non_numeric_cols = [
        c for c in feature_cols if (X[c].isna() & work[c].notna()).any()
    ]
    if non_numeric_cols:
        raise SystemExit(f"Non-numeric values found in feature columns: {non_numeric_cols}")
    missing_rows = int(X.isna().any(axis=1).sum())
    if missing_rows:
        print(f"{missing_rows} of {len(X)} rows have missing features; passing them as NaN.")
    return X


def positive_class_probability(bundle, X):
    clf = bundle["model"]
    probs = clf.predict_proba(X)
    label_name = bundle.get("label_name", "")
    positive_class = label_name.replace("_vs_other", "")
    class_index = list(clf.classes_).index(positive_class)
    return probs[:, class_index]


def predict_single_model(bundle, df):
    clf = bundle["model"]
    X = prepare_features(df, bundle)
    preds = clf.predict(X)
    probs = clf.predict_proba(X)

    out = df[["pid"]].copy() if "pid" in df.columns else pd.DataFrame(index=df.index)
    out["predicted_crop"] = preds
    for i, cls in enumerate(clf.classes_):
        out[f"prob_{cls}"] = probs[:, i]
    return out


def one_vs_rest_crops(model_dir):
    manifest_path = os.path.join(model_dir, OVR_MANIFEST)
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            return json.load(f)["crops"]
    # Directories trained before the manifest existed: use every crop bundle.
    return [
        crop
        for crop in os.listdir(model_dir)
        if os.path.exists(os.path.join(model_dir, crop, "model.joblib"))
    ]


def predict_one_vs_rest(model_dir, df):
    model_paths = sorted(
        os.path.join(model_dir, crop, "model.joblib")
        for crop in one_vs_rest_crops(model_dir)
    )
    if not model_paths:
        raise SystemExit(f"No one-vs-rest model bundles found under {model_dir}")

    bundles = [joblib.load(path) for path in model_paths]
    out = df[["pid"]].copy() if "pid" in df.columns else pd.DataFrame(index=df.index)

    prob_cols = {}
    for bundle in bundles:
        crop = bundle["label_name"].replace("_vs_other", "")
        if crop not in bundle["model"].classes_:
            # Older runs trained crops with no positive examples.
            print(f"Skipping {crop}: its model was trained without any {crop} rows.")
            continue
        X = prepare_features(df, bundle)
        probs = positive_class_probability(bundle, X)
        col = f"prob_{crop}"
        out[col] = probs
        prob_cols[crop] = col

    prob_matrix = out[[prob_cols[crop] for crop in sorted(prob_cols)]].to_numpy()
    crops = np.array(sorted(prob_cols))
    out["predicted_crop"] = crops[np.argmax(prob_matrix, axis=1)]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=f"{MODEL_DIR}/decision_tree/model.joblib")
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", default=f"{DATA_DIR}/predictions.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    if os.path.isdir(args.model):
        out = predict_one_vs_rest(args.model, df)
    else:
        bundle = joblib.load(args.model)
        out = predict_single_model(bundle, df)

    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} predictions to {args.output}")
    print(out["predicted_crop"].value_counts())


if __name__ == "__main__":
    main()
