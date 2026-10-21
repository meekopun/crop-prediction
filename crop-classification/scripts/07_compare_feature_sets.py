"""
Compare wheat/barley performance before and after enabling extra feature
families such as Sentinel-1 C-band and precipitation.

Usage:
    python scripts/07_compare_feature_sets.py \
        --input data/sentinel2_pixel_samples.csv \
        --model-type lightgbm \
        --split-strategy grouped
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_training_module():
    module_path = Path(__file__).with_name("05_train_decision_tree.py")
    spec = importlib.util.spec_from_file_location("train_step", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric_row(experiment, crop, metrics, cv, feature_count, family_counts):
    cv_f1 = cv["row"]["class_f1"].get(crop, {})
    cv_field_f1 = cv["field"]["class_f1"].get(crop, {}) if cv["field"] else {}
    return {
        "experiment": experiment,
        "crop": crop,
        "precision": float(metrics.get("precision", 0.0)),
        "recall": float(metrics.get("recall", 0.0)),
        "f1_score": float(metrics.get("f1-score", 0.0)),
        "support": int(metrics.get("support", 0)),
        "cv_f1_mean": float(cv_f1.get("mean", np.nan)),
        "cv_f1_std": float(cv_f1.get("std", np.nan)),
        "cv_folds_with_crop": int(cv_f1.get("n_folds", 0)),
        "cv_field_f1_mean": float(cv_field_f1.get("mean", np.nan)),
        "feature_count": int(feature_count),
        "feature_family_counts": family_counts,
    }


def run_one_vs_rest_experiment(train_module, crop, args, feature_families):
    X, y, feature_cols, _, groups = train_module.load_training_data(
        args.input,
        args.add_temporal_features,
        args.months_list,
        feature_families,
    )
    if not feature_cols:
        raise SystemExit(
            f"No usable features found for feature families {feature_families}."
        )

    y_binary = pd.Series(np.where(y == crop, crop, "other"), index=y.index)
    clf = train_module.build_model(args)
    evaluation = train_module.evaluate_model(clf, X, y_binary, groups, args)
    family_counts = train_module.summarize_feature_families(feature_cols)
    crop_metrics = evaluation["report_dict"].get(crop, {})
    return metric_row(
        ",".join(feature_families),
        crop,
        crop_metrics,
        evaluation["cv"],
        len(feature_cols),
        family_counts,
    )


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/training_table.csv")
    parser.add_argument(
        "--model-type",
        choices=["decision-tree", "logistic-regression", "catboost", "lightgbm"],
        default="lightgbm",
    )
    parser.add_argument("--add-temporal-features", action="store_true")
    parser.add_argument(
        "--no-temporal-features", action="store_false", dest="add_temporal_features"
    )
    parser.add_argument("--months", default="apr,may,jun,jul,aug,sep,oct")
    parser.add_argument("--baseline-families", default="s2")
    parser.add_argument("--augmented-families", default="s2,s1,precip")
    parser.add_argument("--crops", default="wheat,barley")
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-strategy",
        choices=["random", "grouped"],
        default="grouped",
    )
    parser.add_argument("--group-prefix-len", type=int, default=6)
    parser.add_argument("--catboost-iterations", type=int, default=400)
    parser.add_argument("--catboost-depth", type=int, default=6)
    parser.add_argument("--catboost-learning-rate", type=float, default=0.05)
    parser.add_argument("--lightgbm-estimators", type=int, default=300)
    parser.add_argument("--lightgbm-learning-rate", type=float, default=0.05)
    parser.add_argument("--lightgbm-num-leaves", type=int, default=31)
    parser.add_argument("--lightgbm-max-depth", type=int, default=-1)
    parser.add_argument("--lightgbm-subsample", type=float, default=0.8)
    parser.add_argument("--lightgbm-colsample-bytree", type=float, default=0.8)
    parser.add_argument("--logreg-max-iter", type=int, default=300)
    parser.add_argument("--logreg-c", type=float, default=1.0)
    parser.add_argument("--output", default="data/feature_set_comparison.csv")
    parser.set_defaults(add_temporal_features=True)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.mode = "one-vs-rest"
    args.months_list = [m.strip() for m in args.months.split(",") if m.strip()]
    baseline_families = [
        family.strip() for family in args.baseline_families.split(",") if family.strip()
    ]
    augmented_families = [
        family.strip() for family in args.augmented_families.split(",") if family.strip()
    ]
    crops = [crop.strip() for crop in args.crops.split(",") if crop.strip()]

    train_module = load_training_module()
    rows = []
    for crop in crops:
        rows.append(
            run_one_vs_rest_experiment(train_module, crop, args, baseline_families)
        )
        rows.append(
            run_one_vs_rest_experiment(train_module, crop, args, augmented_families)
        )

    summary = pd.DataFrame(rows)
    baseline_name = ",".join(baseline_families)
    augmented_name = ",".join(augmented_families)
    for crop in crops:
        crop_rows = summary[summary["crop"] == crop]
        if crop_rows["feature_family_counts"].iloc[0] == crop_rows["feature_family_counts"].iloc[-1]:
            print(
                f"\nWarning: {baseline_name} and {augmented_name} select identical "
                f"features for {crop}; the extra families are absent from {args.input}."
            )
    if baseline_name != augmented_name:
        pivot = summary.pivot(index="crop", columns="experiment", values="cv_f1_mean")
        pivot = pivot[[baseline_name, augmented_name]]
        pivot["cv_f1_delta"] = pivot[augmented_name] - pivot[baseline_name]
        print("\nCV F1 comparison (delta = augmented minus baseline):")
        print(pivot)

    summary.to_csv(args.output, index=False)
    print(f"\nSaved comparison table to {args.output}")
    print(summary.to_string(index=False))

    json_path = str(Path(args.output).with_suffix(".json"))
    with open(json_path, "w") as f:
        json.dump(summary.to_dict(orient="records"), f, indent=2)
    print(f"Saved comparison JSON to {json_path}")


if __name__ == "__main__":
    main()
