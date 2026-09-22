"""Historical benchmark; Alberta pixel model functions are not included."""

import math

from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, load_data


def chronological_split(frame):
    """Hold out the newest 20% of distinct years, keeping years disjoint."""
    years = sorted(frame["Year"].unique())
    if len(years) < 2:
        raise ValueError("Benchmark requires at least two distinct years")
    cutoff = years[-max(1, math.ceil(len(years) * 0.2))]
    return frame[frame["Year"] < cutoff].copy(), frame[frame["Year"] >= cutoff].copy()


def run_benchmark(path, output_dir):
    try:
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.dummy import DummyRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder
    except ImportError as exc:
        raise RuntimeError(
            "Benchmark requires optional ML dependencies. Install with: pip install -e '.[ml]'"
        ) from exc

    frame = pd.DataFrame(load_data(path))
    original_count = len(frame)
    frame = frame.drop_duplicates().reset_index(drop=True)
    train, test = chronological_split(frame)
    if len(test) < 2:
        raise ValueError("Benchmark requires at least two held-out records")
    features = [*CATEGORICAL_FEATURES, *NUMERIC_FEATURES]
    transform = ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ("numbers", "passthrough", NUMERIC_FEATURES),
    ])
    models = {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "random_forest": make_pipeline(transform, RandomForestRegressor(
            n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=1,
        )),
    }
    predictions = test.copy()
    metrics = []
    for name, model in models.items():
        model.fit(train[features], train[TARGET])
        predicted = model.predict(test[features])
        predictions[f"pred_{name}_hg_ha"] = predicted
        metrics.append({
            "model": name,
            "mae_hg_ha": float(mean_absolute_error(test[TARGET], predicted)),
            "rmse_hg_ha": float(math.sqrt(mean_squared_error(test[TARGET], predicted))),
            "r2": float(r2_score(test[TARGET], predicted)),
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(output_dir / "metrics.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    return {
        "duplicates_removed": original_count - len(frame),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_years": f"{train['Year'].min()}–{train['Year'].max()}",
        "test_years": f"{test['Year'].min()}–{test['Year'].max()}",
        "metrics": metrics,
    }
