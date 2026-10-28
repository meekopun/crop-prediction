"""Fit, save, load and apply the locked season + field yield model.

The configuration was fixed on 2026-10-28, before any 2026 harvest record existed,
for the 2026 holdout: the season level comes from ``regional_satellite`` (a ridge
per crop trained on Alberta region-years, applied to the farm's July anomaly) and
the field deviation from the three ``peak`` July indices. A crop uses that model
only if its leave-one-season-out error on the development years is at least 10%
below the training mean's (``MIN_IMPROVEMENT``); otherwise it keeps the training
mean. Development years are every table year before the test year, and the
regional models use only region-years before it as well.

Prediction needs no yields. Season-level features are the mean over all of a
crop's fields in a season, and field deviations are centred on that mean, so a
season's fields must be predicted together.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import warnings

from .field_yield import DEFAULT_OUTPUT_DIR, DEFAULT_TABLE_PATH
from .regional_features import DEFAULT_REGIONAL_FEATURES_PATH
from .regional_yields import DEFAULT_ALBERTA_YIELDS_PATH
from .season_field_model import (
    DEVIATION_FEATURE_SETS, SEASON_FEATURE_SETS, _centre, _fit_deviation_models, _predict_deviation,
    _regional_satellite_relative, evaluate_season_field_model, fit_regional_season_models, load_region_table,
)

SCHEMA_VERSION = 1
SEASON_FEATURES = "regional_satellite"
DEVIATION_FEATURES = "peak"
MIN_IMPROVEMENT = 0.1
MIN_ROWS = 10
TEST_YEAR = 2026
MODEL_VARIANT = "season_model_plus_field"
DEFAULT_MODEL_DIR = DEFAULT_OUTPUT_DIR / "models"
FEATURE_CUTOFF = "July Sentinel-2 composites only; a season can be predicted from early August."
ID_COLUMNS = ["field_key", "year", "crop", "yield_unit"]


def default_bundle_dir(test_year: int = TEST_YEAR) -> Path:
    return DEFAULT_MODEL_DIR / f"season_field_{test_year}"


def feature_columns() -> list[str]:
    """Ordered feature schema: season-level columns, then any extra deviation columns."""
    columns = list(SEASON_FEATURE_SETS[SEASON_FEATURES])
    return columns + [column for column in DEVIATION_FEATURE_SETS[DEVIATION_FEATURES] if column not in columns]


def _sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _versions() -> dict:
    import numpy
    import pandas
    import sklearn

    return {"python": platform.python_version(), "numpy": numpy.__version__,
            "pandas": pandas.__version__, "scikit-learn": sklearn.__version__}


def fit_season_field_bundle(table_path: Path | str = DEFAULT_TABLE_PATH, *, test_year: int = TEST_YEAR,
                            regional_yields_path: Path | str = DEFAULT_ALBERTA_YIELDS_PATH,
                            regional_features_path: Path | str = DEFAULT_REGIONAL_FEATURES_PATH,
                            region_table=None, min_improvement: float = MIN_IMPROVEMENT,
                            min_rows: int = MIN_ROWS) -> dict:
    """Choose model or training mean per crop on years before ``test_year``, then refit on all of them.

    ``region_table`` (from ``load_region_table``) overrides the two regional paths.
    """
    if not 0 <= min_improvement < 1:
        raise ValueError("min_improvement must be in [0, 1).")
    import pandas as pd

    table = pd.read_csv(table_path, dtype={"field_key": str, "section_key": str, "pids": str})
    dev_years = tuple(sorted(int(year) for year in table["year"].unique() if year < test_year))
    if len(dev_years) < 3:
        raise ValueError(f"Need at least three development years before {test_year}; found {list(dev_years)}.")
    inputs = {"field_table": {"path": str(Path(table_path).resolve()), "sha256": _sha256(table_path)}}
    if region_table is None:
        region_table = load_region_table(regional_yields_path, regional_features_path)
        inputs.update({name: {"path": str(Path(path).resolve()), "sha256": _sha256(path)}
                       for name, path in (("regional_yields", regional_yields_path),
                                          ("regional_features", regional_features_path))})
    region_dev = region_table[region_table["year"] < test_year]

    # Selection: leave one development season out, exactly as the evaluation script scores it.
    selection = evaluate_season_field_model(table_path, years=dev_years, min_rows=min_rows,
                                            deviation_features=DEVIATION_FEATURES,
                                            season_features=SEASON_FEATURES, region_table=region_dev).metrics
    dev_mae = selection.set_index(["crop", "variant"])["mae"]

    # Refit on every development year.
    dev = table[table["year"].isin(dev_years) & table["crop"].isin(selection["crop"])].reset_index(drop=True)
    columns = feature_columns()
    deviation_columns = DEVIATION_FEATURE_SETS[DEVIATION_FEATURES]
    season_columns = SEASON_FEATURE_SETS[SEASON_FEATURES]
    centred = [f"centred_{column}" for column in deviation_columns]
    dev = pd.concat([dev, _centre(dev, deviation_columns).set_axis(centred, axis=1)], axis=1)
    crop_means = dev.groupby("crop")["yield"].mean()
    reference = dev.groupby(["crop", "year"])[season_columns].mean().groupby("crop").mean()
    season_models = fit_regional_season_models(region_dev)
    deviation_models = _fit_deviation_models(dev, centred)

    crops = {}
    for (crop, unit), frame in dev.groupby(["crop", "yield_unit"], sort=True):
        baseline, model = float(dev_mae[(crop, "baseline")]), float(dev_mae[(crop, MODEL_VARIANT)])
        season_model = season_models.get(crop)
        crops[crop] = {
            "yield_unit": unit,
            "variant": MODEL_VARIANT if model < (1 - min_improvement) * baseline else "baseline",
            "training_mean": float(crop_means[crop]),
            "season_reference": {column: float(reference.loc[crop, column]) for column in season_columns},
            "season_model": season_model[0] if season_model else None,
            "season_alpha": season_model[1] if season_model else None,
            "deviation_model": deviation_models.get(crop),
            "dev_field_years": len(frame), "dev_seasons": int(frame["year"].nunique()),
            "dev_mae_baseline": baseline, "dev_mae_model": model,
            "dev_skill": 1 - model / baseline if baseline else None,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {"season_features": SEASON_FEATURES, "deviation_features": DEVIATION_FEATURES,
                   "min_improvement": min_improvement, "min_rows": min_rows, "model_variant": MODEL_VARIANT},
        "test_year": test_year,
        "training_years": list(dev_years),
        "region_training_years": sorted(int(year) for year in region_dev["year"].unique()),
        # Rows of the test year (or later) that existed when the bundle was fitted; they were not used.
        "later_rows_at_fit": {str(year): int(count) for year, count
                              in table.loc[table["year"] >= test_year, "year"].value_counts().sort_index().items()},
        "feature_columns": columns,
        "season_feature_columns": list(season_columns),
        "deviation_feature_columns": list(deviation_columns),
        "feature_cutoff": FEATURE_CUTOFF,
        "inputs": inputs,
        "versions": _versions(),
        "crops": crops,
    }


def _manifest(bundle: dict) -> dict:
    """The bundle without fitted estimators, safe to write as JSON."""
    manifest = {key: value for key, value in bundle.items() if key != "crops"}
    manifest["crops"] = {crop: {key: value for key, value in entry.items()
                                if key not in ("season_model", "deviation_model")}
                         | {"has_season_model": entry["season_model"] is not None,
                            "has_deviation_model": entry["deviation_model"] is not None}
                         for crop, entry in bundle["crops"].items()}
    return manifest


def save_bundle(bundle: dict, directory: Path | str, *, overwrite: bool = False) -> dict:
    """Write ``model.joblib`` and ``manifest.json`` (with the model's SHA-256); return the manifest."""
    import joblib

    destination = Path(directory)
    model_path = destination / "model.joblib"
    if model_path.exists() and not overwrite:
        raise ValueError(f"{model_path} already exists; pass overwrite to replace a locked bundle.")
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    manifest = _manifest(bundle) | {"model_sha256": _sha256(model_path)}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_bundle(directory: Path | str) -> dict:
    """Load a saved bundle after checking it still matches its manifest."""
    import joblib

    source = Path(directory)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if _sha256(source / "model.joblib") != manifest["model_sha256"]:
        raise ValueError(f"{source / 'model.joblib'} does not match its manifest; it changed after saving.")
    current = _versions()
    changed = {name: (saved, current.get(name)) for name, saved in manifest["versions"].items()
               if name != "python" and current.get(name) != saved}
    if changed:
        warnings.warn(f"Library versions differ from the fitted bundle (saved, current): {changed}", stacklevel=2)
    bundle = joblib.load(source / "model.joblib")
    if bundle.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported bundle schema {bundle.get('schema_version')}.")
    return bundle


def predict_season_field(bundle: dict, table):
    """Predict every row of ``table``; a ``yield`` column is carried through but never used.

    Returns one row per input row with the chosen prediction (``predicted``), both
    candidates, the modelled season mean and a ``status`` explaining missing values.
    """
    import numpy as np
    import pandas as pd

    missing = [column for column in ID_COLUMNS + bundle["feature_columns"] if column not in table.columns]
    if missing:
        raise ValueError(f"Input lacks columns {missing}.")
    table = table.reset_index(drop=True)
    deviation_columns = bundle["deviation_feature_columns"]
    season_columns = bundle["season_feature_columns"]
    centred = [f"centred_{column}" for column in deviation_columns]
    table = pd.concat([table, _centre(table, deviation_columns).set_axis(centred, axis=1)], axis=1)
    crops = bundle["crops"]

    keep = ID_COLUMNS + (["yield"] if "yield" in table.columns else [])
    out = table[keep].assign(status="ok", chosen_variant="", season_source="", season_fields=0,
                             predicted_relative=np.nan, predicted_season_mean=np.nan,
                             predicted_deviation=np.nan, pred_baseline=np.nan,
                             **{f"pred_{MODEL_VARIANT}": np.nan}, predicted=np.nan)
    for (crop, year), season in table.groupby(["crop", "year"]):
        rows = season.index
        entry = crops.get(crop)
        if entry is None:
            out.loc[rows, "status"] = "no model for this crop"
            continue
        units = set(season["yield_unit"])
        if units != {entry["yield_unit"]}:
            out.loc[rows, "status"] = f"unit {sorted(units)} differs from training unit {entry['yield_unit']}"
            continue
        reference = pd.DataFrame([entry["season_reference"]], index=[crop])[season_columns]
        models = ({crop: (entry["season_model"], entry["season_alpha"])}
                  if entry["season_model"] is not None else {})
        relative = _regional_satellite_relative(models, reference,
                                                season.groupby("crop")[season_columns].mean()).get(crop)
        season_mean = entry["training_mean"] if relative is None else relative * entry["training_mean"]
        deviation_models = {crop: entry["deviation_model"]} if entry["deviation_model"] is not None else {}
        deviation = _predict_deviation(deviation_models, season, centred)
        model = season_mean + deviation
        out.loc[rows, "chosen_variant"] = entry["variant"]
        out.loc[rows, "season_source"] = "training mean" if relative is None else SEASON_FEATURES
        out.loc[rows, "season_fields"] = len(rows)
        out.loc[rows, "predicted_relative"] = np.nan if relative is None else relative
        out.loc[rows, "predicted_season_mean"] = season_mean
        out.loc[rows, "predicted_deviation"] = deviation
        out.loc[rows, "pred_baseline"] = entry["training_mean"]
        out.loc[rows, f"pred_{MODEL_VARIANT}"] = model
        out.loc[rows, "predicted"] = model if entry["variant"] == MODEL_VARIANT else entry["training_mean"]
    return out


def score_predictions(predictions):
    """Per-crop error of the chosen prediction against the training mean, plus a pooled bu/ac row."""
    import numpy as np
    import pandas as pd

    if "yield" not in predictions.columns:
        raise ValueError("Scoring needs observed yields; the input has no 'yield' column.")
    scored = predictions[(predictions["status"] == "ok") & predictions["yield"].notna()]
    rows = []
    for (crop, unit), frame in scored.groupby(["crop", "yield_unit"], sort=True):
        error = frame["predicted"] - frame["yield"]
        baseline = float(np.mean(np.abs(frame["pred_baseline"] - frame["yield"])))
        model = float(np.mean(np.abs(frame[f"pred_{MODEL_VARIANT}"] - frame["yield"])))
        mae = float(np.mean(np.abs(error)))
        rows.append({"crop": crop, "yield_unit": unit, "field_years": len(frame),
                     "chosen_variant": frame["chosen_variant"].iloc[0], "mae": mae,
                     "baseline_mae": baseline, "model_mae": model, "bias": float(np.mean(error)),
                     "rmse": float(np.sqrt(np.mean(error ** 2))),
                     "skill": 1 - mae / baseline if baseline else None})
    bushels = scored[scored["yield_unit"] == "bu/ac"]
    if not bushels.empty:
        mae = float(np.mean(np.abs(bushels["predicted"] - bushels["yield"])))
        baseline = float(np.mean(np.abs(bushels["pred_baseline"] - bushels["yield"])))
        rows.append({"crop": "all bu/ac crops", "yield_unit": "bu/ac", "field_years": len(bushels),
                     "chosen_variant": "per crop", "mae": mae, "baseline_mae": baseline,
                     "model_mae": float(np.mean(np.abs(bushels[f"pred_{MODEL_VARIANT}"] - bushels["yield"]))),
                     "bias": float(np.mean(bushels["predicted"] - bushels["yield"])),
                     "rmse": float(np.sqrt(np.mean((bushels["predicted"] - bushels["yield"]) ** 2))),
                     "skill": 1 - mae / baseline if baseline else None})
    return pd.DataFrame(rows)
