"""
Step 5: Train and evaluate crop classifiers on the joined feature table.

Supports either a single multiclass model or one-vs-rest binary models per
crop, using either a decision tree, logistic regression, CatBoost, or
LightGBM.

Evaluation reports a holdout split plus repeated stratified K-fold
cross-validation (grouped by pid prefix by default) with accuracy, macro-F1
and per-class F1. When a pid has several rows (pixel tables), field-level
scores average the row probabilities of each pid.

Usage:
    python 05_train_decision_tree.py --model-type decision-tree --mode multiclass
    python 05_train_decision_tree.py --model-type logistic-regression --mode one-vs-rest
    python 05_train_decision_tree.py --model-type catboost --mode one-vs-rest
    python 05_train_decision_tree.py --model-type lightgbm --mode one-vs-rest \
        --output-dir models/lightgbm_one_vs_rest_v2
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
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from common import OVR_MANIFEST, TARGET_CROPS
from ml_features import (
    add_temporal_features,
    drop_qa_columns,
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
        # The tree models route missing values natively; logistic regression
        # needs imputation, which the pipeline fits on training folds only.
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
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
    raw = pd.read_csv(path)
    df = drop_qa_columns(raw, exclude_cols=NON_FEATURE_COLS)
    qa_cols = [c for c in raw.columns if c not in df.columns]
    df = filter_columns_by_months(df, months, exclude_cols=NON_FEATURE_COLS)
    if add_temporal:
        df = add_temporal_features(df, exclude_cols=NON_FEATURE_COLS, months=months)
    df = filter_columns_by_feature_families(
        df, feature_families, exclude_cols=NON_FEATURE_COLS
    )

    drop_cols = [c for c in NON_FEATURE_COLS if c in df.columns] + qa_cols
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    if not feature_cols:
        raise SystemExit(
            "No feature columns remain after month/family filtering. "
            "Check --months and --feature-families against the input table."
        )
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = df["crop_label"]

    non_numeric_cols = [c for c in feature_cols if (X[c].isna() & df[c].notna()).any()]
    if non_numeric_cols:
        raise SystemExit(f"Non-numeric values found in feature columns: {non_numeric_cols}")
    report_missing_features(X)

    groups = None
    if "pid" in df.columns:
        groups = df["pid"].astype(str)

    return X, y, feature_cols, drop_cols, groups


def report_missing_features(X):
    missing_rows = X.isna().any(axis=1)
    if not missing_rows.any():
        return
    family_cols = {}
    for col in X.columns:
        family_cols.setdefault(infer_feature_family(col), []).append(col)
    rows_by_family = {
        family: int(X[cols].isna().any(axis=1).sum())
        for family, cols in sorted(family_cols.items())
        if X[cols].isna().any().any()
    }
    print(
        f"{int(missing_rows.sum())} of {len(X)} rows have missing features "
        f"(rows affected per family: {rows_by_family}); keeping them as NaN."
    )


def summarize_feature_families(feature_cols):
    counts = {}
    for col in feature_cols:
        family = infer_feature_family(col)
        counts[family] = counts.get(family, 0) + 1
    return dict(sorted(counts.items()))


def pid_to_group(pid_series, prefix_len):
    return pid_series.astype(str).str.slice(0, prefix_len)


def has_repeated_pids(groups):
    """Pixel tables carry many rows per pid; polygon tables carry one."""
    return groups is not None and bool(groups.duplicated().any())


def class_field_counts(y, groups):
    """Distinct pids per class (rows per class when there is no pid)."""
    if groups is None:
        return y.value_counts().to_dict()
    return groups.groupby(y.to_numpy()).nunique().to_dict()


def require_groups(groups):
    if groups is None:
        raise SystemExit(
            "Grouped splitting requires a 'pid' column in the training table. "
            "Pass --split-strategy random only for tables without pid."
        )


def split_data(X, y, groups, args):
    if args.split_strategy == "random":
        return train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed, stratify=y
        )

    require_groups(groups)
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


def predict_labels(clf, X):
    y_pred = clf.predict(X)
    if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1:
        y_pred = y_pred.ravel()
    return y_pred


def field_level_predictions(clf, X, y, pids):
    """
    Average row probabilities per pid and take the top class. A field's true
    label is its majority row label.
    """
    keys = pids.to_numpy()
    probs = pd.DataFrame(clf.predict_proba(X), columns=clf.classes_, index=X.index)
    pred = probs.groupby(keys).mean().idxmax(axis=1)
    truth = y.groupby(keys).agg(lambda s: s.value_counts().idxmax())
    return truth.loc[pred.index], pred


def score_predictions(y_true, y_pred):
    """Macro-F1 averages the classes present in y_true for this fold."""
    labels = sorted(pd.unique(y_true))
    class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(np.mean(class_f1)),
        "class_f1": dict(zip(labels, class_f1.tolist())),
    }


def summarize_fold_scores(fold_scores):
    def mean_std(values):
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "n_folds": len(values),
        }

    class_values = {}
    for scores in fold_scores:
        for cls, value in scores["class_f1"].items():
            class_values.setdefault(cls, []).append(value)
    return {
        "accuracy": mean_std([s["accuracy"] for s in fold_scores]),
        "macro_f1": mean_std([s["macro_f1"] for s in fold_scores]),
        "class_f1": {cls: mean_std(v) for cls, v in sorted(class_values.items())},
    }


def make_cv_splitter(args, repeat):
    seed = args.seed + repeat
    if args.split_strategy == "random":
        return StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=seed)
    return StratifiedGroupKFold(n_splits=args.cv_folds, shuffle=True, random_state=seed)


def describe_cv(args):
    if args.split_strategy == "random":
        splitter = "StratifiedKFold (random rows)"
    else:
        splitter = f"StratifiedGroupKFold on pid[:{args.group_prefix_len}]"
    return (
        f"{args.cv_folds}-fold {splitter} x {args.cv_repeats} repeats "
        f"= {args.cv_folds * args.cv_repeats} folds"
    )


def cross_validate_model(clf, X, y, groups, args):
    """
    Repeated stratified K-fold CV. Folds of the first repeat cover every row
    once, giving out-of-fold predictions for a pooled report and confusion
    matrix.
    """
    cv_groups = None
    if args.split_strategy == "grouped":
        require_groups(groups)
        cv_groups = pid_to_group(groups, args.group_prefix_len)
        unique_groups = cv_groups.nunique()
        if unique_groups < args.cv_folds:
            raise SystemExit(
                f"Need at least {args.cv_folds} unique spatial groups for grouped CV, "
                f"but found {unique_groups}."
            )

    field_level = has_repeated_pids(groups)
    fold_scores, field_scores = [], []
    oof_pred = pd.Series(index=y.index, dtype=object)
    oof_field_truth, oof_field_pred = [], []
    for repeat in range(args.cv_repeats):
        splitter = make_cv_splitter(args, repeat)
        for train_idx, test_idx in splitter.split(X, y, groups=cv_groups):
            model = clone(clf)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
            y_pred = predict_labels(model, X_test)
            fold_scores.append(score_predictions(y_test, y_pred))
            if repeat == 0:
                oof_pred.iloc[test_idx] = y_pred
            if field_level:
                truth, pred = field_level_predictions(
                    model, X_test, y_test, groups.iloc[test_idx]
                )
                field_scores.append(score_predictions(truth, pred))
                if repeat == 0:
                    oof_field_truth.append(truth)
                    oof_field_pred.append(pred)

    result = {
        "scheme": describe_cv(args),
        "fold_accuracy": [s["accuracy"] for s in fold_scores],
        "row": summarize_fold_scores(fold_scores),
        "field": summarize_fold_scores(field_scores) if field_level else None,
        "oof_pred": oof_pred,
        "oof_report_text": classification_report(y, oof_pred, zero_division=0),
        "oof_report_dict": classification_report(
            y, oof_pred, zero_division=0, output_dict=True
        ),
        "oof_field_report_text": None,
        "oof_field_report_dict": None,
    }
    if field_level:
        truth, pred = pd.concat(oof_field_truth), pd.concat(oof_field_pred)
        result["oof_field_report_text"] = classification_report(truth, pred, zero_division=0)
        result["oof_field_report_dict"] = classification_report(
            truth, pred, zero_division=0, output_dict=True
        )
    return result


def evaluate_model(clf, X, y, groups, args):
    X_train, X_test, y_train, y_test = split_data(X, y, groups, args)

    clf.fit(X_train, y_train)

    y_pred = predict_labels(clf, X_test)
    report_text = classification_report(y_test, y_pred, zero_division=0)
    report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    field_holdout = None
    if has_repeated_pids(groups):
        truth, pred = field_level_predictions(clf, X_test, y_test, groups.loc[X_test.index])
        field_holdout = {
            "report_text": classification_report(truth, pred, zero_division=0),
            "report_dict": classification_report(
                truth, pred, zero_division=0, output_dict=True
            ),
        }

    cv = cross_validate_model(clf, X, y, groups, args)
    return {
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "report_text": report_text,
        "report_dict": report_dict,
        "field_holdout": field_holdout,
        "cv": cv,
        "cv_scores": np.array(cv["fold_accuracy"]),
    }


def format_cv_summary(summary, indent="  "):
    lines = [
        f"{indent}accuracy  {summary['accuracy']['mean']:.3f} +/- {summary['accuracy']['std']:.3f}",
        f"{indent}macro-F1  {summary['macro_f1']['mean']:.3f} +/- {summary['macro_f1']['std']:.3f}",
        f"{indent}per-class F1 (mean +/- std over folds containing the class):",
    ]
    for cls, s in summary["class_f1"].items():
        lines.append(
            f"{indent}  {cls:<10} {s['mean']:.3f} +/- {s['std']:.3f}  ({s['n_folds']} folds)"
        )
    return lines


def format_metrics_report(evaluation, args):
    cv = evaluation["cv"]
    lines = [
        f"Holdout ({len(evaluation['y_test'])} rows, {args.split_strategy} split, "
        f"test_size={args.test_size}):",
        evaluation["report_text"],
    ]
    if evaluation["field_holdout"]:
        lines += [
            "Holdout, field level (mean row probability per pid):",
            evaluation["field_holdout"]["report_text"],
        ]
    lines.append(f"Cross-validation: {cv['scheme']}")
    lines += format_cv_summary(cv["row"])
    if cv["field"]:
        lines.append("  Field level (mean row probability per pid):")
        lines += format_cv_summary(cv["field"], indent="    ")
    lines += [
        "",
        "Out-of-fold predictions (CV repeat 1, every row predicted once):",
        cv["oof_report_text"],
    ]
    if cv["oof_field_report_text"]:
        lines += ["Out-of-fold predictions, field level:", cv["oof_field_report_text"]]
    return "\n".join(lines) + "\n"


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


def save_common_artifacts(
    clf, X, y, feature_cols, groups, out_dir, args, label_name, skipped_classes=None
):
    os.makedirs(out_dir, exist_ok=True)
    feature_family_counts = summarize_feature_families(feature_cols)
    n_fields = groups.nunique() if groups is not None else len(X)
    print(
        f"Training on {len(X)} rows from {n_fields} fields, {len(feature_cols)} features, "
        f"{y.nunique()} classes: {sorted(y.unique())}"
    )
    print(f"Feature family counts: {feature_family_counts}")

    evaluation = evaluate_model(clf, X, y, groups, args)
    X_test = evaluation["X_test"]
    y_test = evaluation["y_test"]
    cv = evaluation["cv"]
    cv_scores = evaluation["cv_scores"]
    report = format_metrics_report(evaluation, args)
    print(report)

    with open(f"{out_dir}/metrics.txt", "w") as f:
        f.write(report)
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(
            {
                "label_name": label_name,
                "feature_count": len(feature_cols),
                "feature_family_counts": feature_family_counts,
                "feature_families": args.feature_families_list,
                "class_metrics": evaluation["report_dict"],
                "holdout_field_class_metrics": (
                    evaluation["field_holdout"]["report_dict"]
                    if evaluation["field_holdout"]
                    else None
                ),
                "cv_scores": cv_scores.tolist(),
                "cv_mean_accuracy": float(cv_scores.mean()),
                "cv_std_accuracy": float(cv_scores.std()),
                "cv": {
                    "scheme": cv["scheme"],
                    "row": cv["row"],
                    "field": cv["field"],
                    "oof_class_metrics": cv["oof_report_dict"],
                    "oof_field_class_metrics": cv["oof_field_report_dict"],
                },
                "data": {
                    "input": args.input,
                    "rows": len(X),
                    "fields": int(n_fields),
                    "spatial_groups": (
                        int(pid_to_group(groups, args.group_prefix_len).nunique())
                        if groups is not None
                        else None
                    ),
                    "rows_with_missing_features": int(X.isna().any(axis=1).sum()),
                    "class_rows": y.value_counts().to_dict(),
                    "class_fields": class_field_counts(y, groups),
                },
                "settings": {
                    "model_type": args.model_type,
                    "mode": args.mode,
                    "split_strategy": args.split_strategy,
                    "group_prefix_len": args.group_prefix_len,
                    "months": args.months_list,
                    "temporal_features": args.add_temporal_features,
                    "test_size": args.test_size,
                    "cv_folds": args.cv_folds,
                    "cv_repeats": args.cv_repeats,
                    "seed": args.seed,
                    "min_class_fields": args.min_class_fields,
                },
                "skipped_classes": skipped_classes or {},
            },
            f,
            indent=2,
        )

    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, ax=ax, xticks_rotation=45)
    fig.savefig(f"{out_dir}/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay.from_predictions(y, cv["oof_pred"], ax=ax, xticks_rotation=45)
    ax.set_title("Out-of-fold predictions (CV repeat 1)")
    fig.savefig(f"{out_dir}/cv_confusion_matrix.png", dpi=150, bbox_inches="tight")
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
            "input": args.input,
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


def ensure_new_output_dir(out_dir, overwrite):
    if os.path.isdir(out_dir) and os.listdir(out_dir) and not overwrite:
        raise SystemExit(
            f"{out_dir} already contains a previous run. Pass --output-dir for a new "
            "directory, or --overwrite to replace files in this one."
        )


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
    parser.add_argument(
        "--cv-repeats",
        type=int,
        default=3,
        help="Repeat CV with reshuffled folds; scores are summarized over all folds.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-strategy",
        choices=["random", "grouped"],
        default="grouped",
        help="grouped keeps nearby quarter sections together by pid prefix; "
        "random splits rows and leaks badly for pixel tables.",
    )
    parser.add_argument(
        "--group-prefix-len",
        type=int,
        default=6,
        help="Characters of pid to treat as one spatial group when using grouped splits "
        "(6 = meridian + range + township).",
    )
    parser.add_argument(
        "--min-class-fields",
        type=int,
        default=10,
        help="Crops labeled on fewer distinct pids are skipped (one-vs-rest) or "
        "their rows dropped (multiclass).",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for this run's outputs (default models/<model_type>[_one_vs_rest]).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an output directory that already holds a run.",
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
    out_dir = args.output_dir or model_dir_name(args.model_type, args.mode)
    ensure_new_output_dir(out_dir, args.overwrite)

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

    field_counts = class_field_counts(y, groups)
    print(f"Fields per class: {field_counts}")

    if args.mode == "multiclass":
        rare = {c: n for c, n in field_counts.items() if n < args.min_class_fields}
        if rare:
            print(
                f"Dropping classes with fewer than {args.min_class_fields} fields: {rare}"
            )
            keep = ~y.isin(list(rare))
            X, y = X[keep], y[keep]
            groups = groups[keep] if groups is not None else None
        clf = build_model(args)
        save_common_artifacts(
            clf, X, y, feature_cols, groups, out_dir, args, "crop_label", skipped_classes=rare
        )
        return

    os.makedirs(out_dir, exist_ok=True)
    trained, skipped = [], {}
    for crop in TARGET_CROPS:
        n_fields = field_counts.get(crop, 0)
        if n_fields < args.min_class_fields:
            print(
                f"\n=== Skipping {crop}: {n_fields} labeled fields < "
                f"--min-class-fields {args.min_class_fields} ==="
            )
            skipped[crop] = n_fields
            continue
        print(f"\n=== Training {crop} vs other ({args.model_type}) ===")
        y_binary = np.where(y == crop, crop, "other")
        clf = build_model(args)
        save_common_artifacts(
            clf,
            X,
            pd.Series(y_binary, index=y.index),
            feature_cols,
            groups,
            f"{out_dir}/{crop}",
            args,
            f"{crop}_vs_other",
        )
        trained.append(crop)

    with open(f"{out_dir}/{OVR_MANIFEST}", "w") as f:
        json.dump(
            {
                "crops": trained,
                "skipped_fields": skipped,
                "min_class_fields": args.min_class_fields,
            },
            f,
            indent=2,
        )
    stale = sorted(
        d for d in os.listdir(out_dir)
        if os.path.isdir(f"{out_dir}/{d}") and d not in trained
    )
    if stale:
        print(f"Ignoring older crop directories not in {OVR_MANIFEST}: {stale}")
    print(f"\nTrained {trained}; skipped {skipped}. Manifest: {out_dir}/{OVR_MANIFEST}")


if __name__ == "__main__":
    main()
