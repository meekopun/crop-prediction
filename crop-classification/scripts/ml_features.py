"""
Feature engineering helpers shared by training and prediction.
"""
import pandas as pd

from common import GROWING_SEASON_MONTHS, MONTH_NAMES

MONTH_ORDER = [MONTH_NAMES[m] for m in GROWING_SEASON_MONTHS]
S2_PREFIXES = {
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B11",
    "B12",
    "NDVI",
    "NDYI",
    "GNDVI",
    "NDRE",
    "NDWI",
    "EVI",
    "SWIR_RATIO",
}
S1_PREFIXES = {"VV", "VH", "VV_minus_VH"}
PRECIP_PREFIXES = {"PRECIP"}


def filter_columns_by_months(df, months, exclude_cols=None):
    """
    Keep only columns tied to the requested monthly suffixes, plus excluded
    metadata columns and non-monthly derived columns.
    """
    if not months:
        return df.copy()

    exclude_cols = set(exclude_cols or [])
    keep = []
    for col in df.columns:
        if col in exclude_cols or "_" not in col:
            keep.append(col)
            continue
        suffix = col.rsplit("_", 1)[1]
        if suffix in months:
            keep.append(col)
    return df[keep].copy()


def add_temporal_features(df, exclude_cols=None, months=None):
    """
    Add simple time-series features from monthly columns of the form
    <base>_<month>, such as NDVI_jul or B11_sep.
    """
    exclude_cols = set(exclude_cols or [])
    out = df.copy()
    month_order = [m for m in MONTH_ORDER if months is None or m in months]
    derived = {}

    groups = {}
    for col in df.columns:
        if col in exclude_cols or "_" not in col:
            continue
        base, month = col.rsplit("_", 1)
        if month in month_order:
            groups.setdefault(base, {})[month] = col

    for base, month_cols in groups.items():
        ordered = [month_cols[m] for m in month_order if m in month_cols]
        if len(ordered) < 2:
            continue

        # Consecutive month-to-month deltas capture growth and senescence.
        for prev, curr in zip(ordered[:-1], ordered[1:]):
            prev_month = prev.rsplit("_", 1)[1]
            curr_month = curr.rsplit("_", 1)[1]
            derived[f"{base}_delta_{prev_month}_{curr_month}"] = out[curr] - out[prev]

        # Peak and seasonal spread summarize the phenology shape.
        ordered_frame = out[ordered]
        season_max = ordered_frame.max(axis=1)
        season_min = ordered_frame.min(axis=1)
        derived[f"{base}_season_max"] = season_max
        derived[f"{base}_season_min"] = season_min
        derived[f"{base}_season_range"] = season_max - season_min

    if derived:
        out = pd.concat([out, pd.DataFrame(derived, index=out.index)], axis=1)

    return out


def signal_base_name(column):
    """
    Collapse temporal feature names back to their original signal prefix.
    """
    if "_delta_" in column:
        return column.split("_delta_", 1)[0]
    if "_season_" in column:
        return column.split("_season_", 1)[0]
    if "_" in column:
        base, month = column.rsplit("_", 1)
        if month in MONTH_ORDER:
            return base
    return column


def infer_feature_family(column):
    base = signal_base_name(column)
    if base in S2_PREFIXES:
        return "s2"
    if base in S1_PREFIXES:
        return "s1"
    if base in PRECIP_PREFIXES:
        return "precip"
    return "other"


def filter_columns_by_feature_families(df, include_families, exclude_cols=None):
    """
    Keep only metadata columns plus features whose inferred family is included.
    """
    if not include_families or "all" in include_families:
        return df.copy()

    exclude_cols = set(exclude_cols or [])
    include_families = set(include_families)
    keep = []
    for col in df.columns:
        if col in exclude_cols:
            keep.append(col)
            continue
        if infer_feature_family(col) in include_families:
            keep.append(col)
    return df[keep].copy()
