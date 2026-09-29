from __future__ import annotations

from pathlib import Path


DEFAULT_NDVI_WEIGHTS: dict[str, float] = {
    "NDVI_max": 0.5,
    "NDVI_integral_proxy": 0.3,
    "NDVI_min": 0.2,
}


def _require_pandas() -> None:
    try:
        import pandas  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Pixel-yield estimation requires pandas. Install it with `pip install -e '.[ml]'`."
        ) from exc


def _scale_series_within_group(
    series: object,
    *,
    lower_quantile: float,
    upper_quantile: float,
) -> object:
    import numpy as np
    import pandas as pd

    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")

    valid = series.dropna()
    if valid.empty:
        return pd.Series(0.5, index=series.index, dtype=float)

    lower = float(valid.quantile(lower_quantile))
    upper = float(valid.quantile(upper_quantile))
    clipped = series.clip(lower=lower, upper=upper)
    if np.isclose(lower, upper):
        return pd.Series(0.5, index=series.index, dtype=float)

    scaled = (clipped - lower) / (upper - lower)
    return scaled.fillna(0.5).astype(float)


def estimate_pixel_yield_from_ndvi(
    frame: object,
    *,
    group_column: str = "row_id",
    target_column: str = "yield_bu_ac",
    ndvi_weights: dict[str, float] | None = None,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    score_floor: float = 0.25,
    min_factor: float = 0.5,
    max_factor: float = 1.5,
) -> object:
    import pandas as pd

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")

    weights = dict(ndvi_weights or DEFAULT_NDVI_WEIGHTS)
    required = [group_column, target_column, *weights.keys()]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("Quantiles must satisfy 0 <= lower < upper <= 1.")
    if not 0 <= score_floor <= 1:
        raise ValueError("score_floor must be between 0 and 1.")
    if min_factor <= 0 or max_factor <= 0 or min_factor > max_factor:
        raise ValueError("Expected positive min_factor <= max_factor.")

    result = frame.copy()
    total_weight = float(sum(weights.values()))
    if total_weight <= 0:
        raise ValueError("At least one positive NDVI weight is required.")

    grouped = result.groupby(group_column, sort=False, dropna=False)
    scaled_columns: list[str] = []
    for column in weights:
        scaled_name = f"{column}_field_scaled"
        result[scaled_name] = grouped[column].transform(
            lambda series: _scale_series_within_group(
                series,
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
            )
        )
        scaled_columns.append(scaled_name)

    result["ndvi_score_raw"] = sum(
        float(weights[column]) * result[f"{column}_field_scaled"]
        for column in weights
    ) / total_weight
    result["ndvi_score"] = score_floor + (1.0 - score_floor) * result["ndvi_score_raw"]

    grouped = result.groupby(group_column, sort=False, dropna=False)
    group_mean_score = grouped["ndvi_score"].transform("mean")
    result["pixel_yield_factor"] = result["ndvi_score"] / group_mean_score
    result["pixel_yield_factor"] = result["pixel_yield_factor"].clip(
        lower=min_factor,
        upper=max_factor,
    )
    grouped = result.groupby(group_column, sort=False, dropna=False)
    clipped_group_mean = grouped["pixel_yield_factor"].transform("mean")
    result["pixel_yield_factor"] = result["pixel_yield_factor"] / clipped_group_mean
    result["pixel_yield_est_bu_ac"] = result[target_column] * result["pixel_yield_factor"]

    grouped = result.groupby(group_column, sort=False, dropna=False)
    result["field_mean_pixel_yield_est_bu_ac"] = grouped["pixel_yield_est_bu_ac"].transform("mean")
    result["field_yield_balance_error_bu_ac"] = (
        result["field_mean_pixel_yield_est_bu_ac"] - result[target_column]
    )
    return result


def summarize_pixel_yield_estimates(
    pixel_frame: object,
    *,
    group_column: str = "row_id",
    target_column: str = "yield_bu_ac",
) -> object:
    import pandas as pd

    if not isinstance(pixel_frame, pd.DataFrame):
        raise TypeError("pixel_frame must be a pandas DataFrame")

    summary_columns = [
        column
        for column in ["quarter_section", "crop", "year", group_column, target_column]
        if column in pixel_frame.columns
    ]
    grouped = pixel_frame.groupby(group_column, sort=True, dropna=False)
    summary = grouped.apply(
        lambda group: pd.Series(
            {
                "pixel_count": int(len(group)),
                "field_yield_bu_ac": float(group[target_column].iloc[0]),
                "estimated_mean_pixel_yield_bu_ac": float(group["pixel_yield_est_bu_ac"].mean()),
                "estimated_min_pixel_yield_bu_ac": float(group["pixel_yield_est_bu_ac"].min()),
                "estimated_max_pixel_yield_bu_ac": float(group["pixel_yield_est_bu_ac"].max()),
                "mean_pixel_yield_factor": float(group["pixel_yield_factor"].mean()),
                "min_pixel_yield_factor": float(group["pixel_yield_factor"].min()),
                "max_pixel_yield_factor": float(group["pixel_yield_factor"].max()),
                "mean_ndvi_score": float(group["ndvi_score"].mean()),
                "min_ndvi_score": float(group["ndvi_score"].min()),
                "max_ndvi_score": float(group["ndvi_score"].max()),
                "yield_balance_error_bu_ac": float(
                    group["pixel_yield_est_bu_ac"].mean() - group[target_column].iloc[0]
                ),
            }
        )
    ).reset_index()

    if summary_columns:
        first_rows = pixel_frame[summary_columns].drop_duplicates(subset=[group_column]).copy()
        summary = first_rows.merge(summary, on=group_column, how="right")
    return summary


def load_pixel_table(path: Path | str) -> object:
    _require_pandas()
    import pandas as pd

    return pd.read_csv(path)


def write_pixel_yield_outputs(
    pixel_frame: object,
    summary_frame: object,
    *,
    pixel_output: Path | str,
    summary_output: Path | str,
) -> None:
    pixel_path = Path(pixel_output)
    summary_path = Path(summary_output)
    pixel_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pixel_frame.to_csv(pixel_path, index=False)
    summary_frame.to_csv(summary_path, index=False)
