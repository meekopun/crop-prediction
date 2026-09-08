"""
Step 5: Train and evaluate crop classifiers on the joined feature table.

Supports either a single multiclass model or one-vs-rest binary models per
crop, using either a decision tree, logistic regression, CatBoost, or
LightGBM.

Usage:
    python 05_train_decision_tree.py --model-type decision-tree --mode multiclass
    python 05_train_decision_tree.py --model-type logistic-regression --mode one-vs-rest
    python 05_train_decision_tree.py --model-type catboost --mode one-vs-rest
    python 05_train_decision_tree.py --model-type lightgbm --mode one-vs-rest
"""
import argparse
import json
import os

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from common import TARGET_CROPS
from ml_features import (
    add_temporal_features,
    filter_columns_by_feature_families,
    filter_columns_by_months,
    infer_feature_family,
)

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

DATA_DIR = "data"
MODEL_DIR = "models"

NON_FEATURE_COLS = {
    "pid",
    "crop_label",
    "aafc_code",
    "system:index",
    ".geo",
    "label_purity",
    "aafc_pixel_count",
}


def build_model(args):
    if args.model_type == "decision-tree":
        return DecisionTreeClassifier(
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight="balanced",
            random_state=args.seed,
        )

    if args.model_type == "logistic-regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=args.logreg_max_iter,
                C=args.logreg_c,
                random_state=args.seed,
                solver="lbfgs",
                n_jobs=None,
            ),
        )

    if args.model_type == "catboost":
        if CatBoostClassifier is None:
            raise SystemExit(
                "CatBoost is not installed. Run `pip install -r requirements.txt` "
                "to use `--model-type catboost`."
            )

        return CatBoostClassifier(
            depth=args.catboost_depth,
            learning_rate=args.catboost_learning_rate,
            iterations=args.catboost_iterations,
            loss_function="Logloss" if args.mode == "one-vs-rest" else "MultiClass",
            eval_metric="Accuracy",
            random_seed=args.seed,
            verbose=False,
            auto_class_weights="Balanced",
        )

    if LGBMClassifier is None:
        raise SystemExit(
            "LightGBM is not installed. Run `pip install -r requirements.txt` "
            "to use `--model-type lightgbm`."
        )

    objective = "binary" if args.mode == "one-vs-rest" else "multiclass"
    return LGBMClassifier(
        objective=objective,
        n_estimators=args.lightgbm_estimators,
        learning_rate=args.lightgbm_learning_rate,
        num_leaves=args.lightgbm_num_leaves,
        max_depth=args.lightgbm_max_depth,
        subsample=args.lightgbm_subsample,
        colsample_bytree=args.lightgbm_colsample_bytree,
        class_weight="balanced",
        random_state=args.seed,
        verbosity=-1,
    )


def load_training_data(path, add_temporal, months, feature_families=None):
    df = pd.read_csv(path)
    df = filter_columns_by_months(df, months, exclude_cols=NON_FEATURE_COLS)
    if add_temporal:
        df = add_temporal_features(df, exclude_cols=NON_FEATURE_COLS, months=months)
    df = filter_columns_by_feature_families(
        df, feature_families, exclude_cols=NON_FEATURE_COLS
    )

    drop_cols = [c for c in NON_FEATURE_COLS if c in df.columns]
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    if not feature_cols:
        raise SystemExit(
            "No feature columns remain after month/family filtering. "
            "Check --months and --feature-families against the input table."
        )
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = df["crop_label"]

    bad_feature_cols = [c for c in feature_cols if X[c].isna().any()]
    if bad_feature_cols:
        raise SystemExit(
            "Non-numeric or missing values found in feature columns: "
            f"{bad_feature_cols}"
        )

    groups = None
    if "pid" in df.columns:
        groups = df["pid"].astype(str)

    return X, y, feature_cols, drop_cols, groups


def summarize_feature_families(feature_cols):
    counts = {}
    for col in feature_cols:
        family = infer_feature_family(col)
        counts[family] = counts.get(family, 0) + 1
    return dict(sorted(counts.items()))


def pid_to_group(pid_series, prefix_len):
    return pid_series.astype(str).str.slice(0, prefix_len)


def split_data(X, y, groups, args):
    if args.split_strategy == "random":
        return train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed, stratify=y
        )

    if groups is None:
        raise SystemExit("Grouped splitting requires a 'pid' column in the training table.")

    spatial_groups = pid_to_group(groups, args.group_prefix_len)
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=args.test_size, random_state=args.seed
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=spatial_groups))
    return (
        X.iloc[train_idx],
        X.iloc[test_idx],
        y.iloc[train_idx],
        y.iloc[test_idx],
    )


def cross_validate_model(clf, X, y, groups, args):
    if args.split_strategy == "random":
        return cross_val_score(clf, X, y, cv=args.cv_folds)

    if groups is None:
        raise SystemExit("Grouped cross-validation requires a 'pid' column in the training table.")

    spatial_groups = pid_to_group(groups, args.group_prefix_len)
    unique_groups = spatial_groups.nunique()
    if unique_groups < args.cv_folds:
        raise SystemExit(
            f"Need at least {args.cv_folds} unique spatial groups for grouped CV, "
            f"but found {unique_groups}."
        )
    cv = GroupKFold(n_splits=args.cv_folds)
    return cross_val_score(clf, X, y, cv=cv, groups=spatial_groups)


def evaluate_model(clf, X, y, groups, args):
    X_train, X_test, y_train, y_test = split_data(X, y, groups, args)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1:
        y_pred = y_pred.ravel()
    report_text = classification_report(y_test, y_pred, zero_division=0)
    report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    cv_scores = cross_validate_model(clf, X, y, groups, args)
    return {
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "report_text": report_text,
        "report_dict": report_dict,
        "cv_scores": cv_scores,
    }


def save_tree_artifacts(clf, feature_cols, y, out_dir):
    rules = export_text(clf, feature_names=feature_cols)
    with open(f"{out_dir}/tree_rules.txt", "w") as f:
        f.write(rules)

    fig, ax = plt.subplots(figsize=(24, 12))
    plot_tree(
        clf,
        feature_names=feature_cols,
        class_names=sorted(y.unique()),
        filled=True,
        max_depth=4,
        fontsize=8,
        ax=ax,
    )
    fig.savefig(f"{out_dir}/tree_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_feature_scores(clf, feature_cols):
    if hasattr(clf, "feature_importances_"):
        return pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    # Pipelines such as logistic regression expose coefficients on the final step.
    final_estimator = clf[-1] if hasattr(clf, "__getitem__") else clf
    if hasattr(final_estimator, "coef_"):
        coefs = np.abs(final_estimator.coef_)
        if coefs.ndim == 2:
            coefs = coefs.mean(axis=0)
        return pd.Series(coefs, index=feature_cols).sort_values(ascending=False)

    raise SystemExit(
        f"Model type '{type(clf).__name__}' does not expose feature importances or coefficients."
    )


def save_common_artifacts(clf, X, y, feature_cols, groups, out_dir, args, label_name):
    os.makedirs(out_dir, exist_ok=True)
    feature_family_counts = summarize_feature_families(feature_cols)
    print(
        f"Training on {len(X)} quarter sections, {len(feature_cols)} features, "
        f"{y.nunique()} classes: {sorted(y.unique())}"
    )
    print(f"Feature family counts: {feature_family_counts}")

    evaluation = evaluate_model(clf, X, y, groups, args)
    X_test = evaluation["X_test"]
    y_test = evaluation["y_test"]
    y_pred = evaluation["y_pred"]
    report = evaluation["report_text"]
    report_dict = evaluation["report_dict"]
    cv_scores = evaluation["cv_scores"]
    print(report)

    cv_summary = (
        f"{args.cv_folds}-fold CV accuracy: "
        f"{cv_scores.mean():.3f} +/- {cv_scores.std():.3f}"
    )
    print(cv_summary)

    with open(f"{out_dir}/metrics.txt", "w") as f:
        f.write(report)
        f.write("\n" + cv_summary + "\n")
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(
            {
                "label_name": label_name,
                "feature_count": len(feature_cols),
                "feature_family_counts": feature_family_counts,
                "feature_families": args.feature_families_list,
                "class_metrics": report_dict,
                "cv_scores": cv_scores.tolist(),
                "cv_mean_accuracy": float(cv_scores.mean()),
                "cv_std_accuracy": float(cv_scores.std()),
            },
            f,
            indent=2,
        )

    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, ax=ax, xticks_rotation=45)
    fig.savefig(f"{out_dir}/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    joblib.dump(
        {
            "model": clf,
            "feature_cols": feature_cols,
            "label_name": label_name,
            "model_type": args.model_type,
            "temporal_features": args.add_temporal_features,
            "months": args.months_list,
            "feature_families": args.feature_families_list,
            "split_strategy": args.split_strategy,
            "group_prefix_len": args.group_prefix_len,
        },
        f"{out_dir}/model.joblib",
    )

    importances = compute_feature_scores(clf, feature_cols)
    with open(f"{out_dir}/feature_importances.json", "w") as f:
        json.dump(importances.to_dict(), f, indent=2)

    print("\nTop 10 most important features:")
    print(importances.head(10))

    if args.model_type == "decision-tree":
        save_tree_artifacts(clf, feature_cols, y, out_dir)

    print(f"\nSaved model + reports to {out_dir}/")


def model_dir_name(model_type, mode):
    slug = model_type.replace("-", "_")
    if mode == "multiclass":
        return f"{MODEL_DIR}/{slug}"
    return f"{MODEL_DIR}/{slug}_one_vs_rest"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=f"{DATA_DIR}/training_table.csv")
    parser.add_argument("--mode", choices=["multiclass", "one-vs-rest"], default="multiclass")
    parser.add_argument(
        "--model-type",
        choices=["decision-tree", "logistic-regression", "catboost", "lightgbm"],
        default="decision-tree",
    )
    parser.add_argument("--add-temporal-features", action="store_true")
    parser.add_argument("--no-temporal-features", action="store_false", dest="add_temporal_features")
    parser.add_argument(
        "--months",
        default="apr,may,jun,jul,aug,sep,oct",
        help="Comma-separated month suffixes to keep, e.g. 'jun,jul,aug'",
    )
    parser.add_argument(
        "--feature-families",
        default="all",
        help="Comma-separated feature families to keep: all, s2, s1, precip.",
    )
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-strategy",
        choices=["random", "grouped"],
        default="random",
        help="Use grouped to keep nearby quarter sections together by pid prefix.",
    )
    parser.add_argument(
        "--group-prefix-len",
        type=int,
        default=6,
        help="Characters of pid to treat as one spatial group when using grouped splits.",
    )
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
    parser.set_defaults(add_temporal_features=True)
    args = parser.parse_args()
    args.months_list = [m.strip() for m in args.months.split(",") if m.strip()]
    args.feature_families_list = [
        family.strip() for family in args.feature_families.split(",") if family.strip()
    ]

    X, y, feature_cols, drop_cols, groups = load_training_data(
        args.input,
        args.add_temporal_features,
        args.months_list,
        args.feature_families_list,
    )
    if drop_cols:
        print(f"Ignored non-feature columns: {sorted(drop_cols)}")
    print(f"Using monthly feature subset: {args.months_list}")
    print(f"Using feature families: {args.feature_families_list}")
    if args.add_temporal_features:
        print("Added temporal delta/range features from monthly bands and indices.")
    if args.split_strategy == "grouped":
        print(
            "Using grouped spatial validation with pid prefix length "
            f"{args.group_prefix_len}."
        )

    if args.mode == "multiclass":
        out_dir = model_dir_name(args.model_type, args.mode)
        clf = build_model(args)
        save_common_artifacts(clf, X, y, feature_cols, groups, out_dir, args, "crop_label")
        return

    base_dir = model_dir_name(args.model_type, args.mode)
    os.makedirs(base_dir, exist_ok=True)
    for crop in TARGET_CROPS:
        print(f"\n=== Training {crop} vs other ({args.model_type}) ===")
        y_binary = np.where(y == crop, crop, "other")
        clf = build_model(args)
        save_common_artifacts(
            clf,
            X,
            pd.Series(y_binary, index=y.index),
            feature_cols,
            groups,
            f"{base_dir}/{crop}",
            args,
            f"{crop}_vs_other",
        )


if __name__ == "__main__":
    main()
