# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository shape

Two independent projects with direct roots and separate environments:

- `crop-classification/` — crop type prediction for Alberta quarter sections.
- `crop-yield/` — runnable historical yield benchmark plus incomplete Alberta workflows.

See root `README.md` for the directory map. Classification has one dependency
list at its root and one set of docs in `docs/`. Yield uses a conventional
`src/yield_prediction/` package and `pyproject.toml`; its geospatial dependencies
remain in `requirements.txt`. Keep existing local inputs and model outputs.

## Crop Classification (`crop-classification/`)

### Running

All scripts use **flat imports** (`from common import ...`, `from ml_features import ...`)
and **CWD-relative paths** (`data/`, `models/`). Consequences:

- Run as `python scripts/05_train_decision_tree.py`, never `python -m`.
- Run from `crop-classification/` so `data/` and `models/` resolve.
- `scripts/00_extract_study_area.py` reads `raw_data/quarter_sections.geojson`
  directly, relative to the project directory. It has no CLI arguments.

```bash
pip install -r requirements.txt
earthengine authenticate                 # needed for any 02_* script
```

### Polygon pipeline (steps 0-7)

```bash
python scripts/00_extract_study_area.py                                    # -> data/study_area_quarter_sections.geojson
python scripts/01_select_training_sample.py --n 1500                       # -> data/training_sample.geojson
python scripts/02_gee_sentinel_features.py --year 2024 --project GCP_ID    # -> Google Drive; download to data/sentinel2_features.csv
python scripts/03_download_aafc_labels.py --year 2024 --min-purity 0.7     # -> data/aafc_labels.csv
python scripts/04_join_labels_features.py                                  # -> data/training_table.csv
python scripts/04_join_labels_features.py --keep-missing --output data/training_table_2024_with_missing.csv
python scripts/05_train_decision_tree.py --input data/training_table_2024_with_missing.csv --model-type lightgbm \
    --feature-families s2,s1 --output-dir models/step1/C_polygon423_s2s1
python scripts/06_predict.py --model models/step1/C_polygon423_s2s1/model.joblib --features data/sentinel2_features.csv
python scripts/07_compare_feature_sets.py --input data/training_table_2024_with_missing.csv --model-type lightgbm \
    --baseline-families s2 --augmented-families s2,s1
```

`sentinel2_pixel_samples.csv` holds Sentinel-2 columns only, so comparing
`s2,s1,precip` on it compares identical features (07 warns). CHIRPS
precipitation ends at 50°N: precip is NaN for every quarter section north of
50°N, so with `--keep-missing` its missingness encodes location. Train
recovered tables with `--feature-families s2,s1`.

Step 2 exports to Drive by default and the CSV must be downloaded into `data/`
manually before step 4. `data/training_table.csv` is tracked; `data/sentinel2_pixel_samples.csv` is a
local export. With the required table available, training can start at step 5
without GEE access.

### Key architectural facts

- **`scripts/common.py` is the single source of study-area truth.** `STUDY_BBOX`
  (Lethbridge County) and `AAFC_CODE_TO_CROP` live there. Changing the region
  means editing `STUDY_BBOX`, clearing `data/` and `models/`, and re-running from
  step 0. Wheat codes 140/145/146 are deliberately pooled into one `wheat` class;
  `alfalfa` is a proxy for AAFC code 122 ("Pasture and Forages").
- **Feature columns are named `<signal>_<month>`** (e.g. `NDVI_jul`, `B11_sep`).
  `ml_features.py` parses this convention to do everything: `--months` filtering,
  `--feature-families` filtering (`s2`/`s1`/`precip`), and derived temporal
  features (consecutive-month deltas, `_season_max/min/range`). Any new signal
  must follow the suffix convention and be registered in `S2_PREFIXES` /
  `S1_PREFIXES` / `PRECIP_PREFIXES` or it lands in the `other` family and gets
  silently dropped by family filters.
- **`05_train_decision_tree.py` is the shared training engine.** Despite the name
  it dispatches over `--model-type {decision-tree,logistic-regression,catboost,lightgbm}`
  and `--mode {multiclass,one-vs-rest}`. `07_compare_feature_sets.py` imports it
  by file path via `importlib` (the leading digit blocks a normal import) — keep
  its module-level functions importable.
- **Grouped splits require a `pid` column.** `--split-strategy grouped` (the
  default) groups by `pid[:6]` = meridian + range + township, so whole townships
  are held out. Random splits leak badly for pixel tables. CV is
  `StratifiedGroupKFold` × `--cv-repeats` (default 3), reporting accuracy,
  macro-F1 and per-class F1. Pixel tables (many rows per `pid`) also get
  field-level scores from averaging each pid's probabilities.
- `--min-class-fields` (default 10 distinct pids) skips rare one-vs-rest crops
  or drops their rows in multiclass. Training drops QA signals (`QA_PREFIXES`
  = `SCL`) and keeps missing features as NaN; logistic regression imputes
  medians inside its pipeline.
- Output layout per run: `--output-dir`, defaulting to `models/<slug>/` for
  multiclass and `models/<slug>_one_vs_rest/<crop>/` for one-vs-rest. Each
  contains `model.joblib`, `metrics.{txt,json}`, `confusion_matrix.png`
  (holdout), `cv_confusion_matrix.png` (out-of-fold) and
  `feature_importances.json` (+ tree plots for decision trees). Training refuses
  a non-empty output directory without `--overwrite`. One-vs-rest writes
  `one_vs_rest.json` (`OVR_MANIFEST` in `common.py`) listing trained crops;
  `06_predict.py` uses only those. It accepts either a `model.joblib` path or a
  one-vs-rest directory.
- CatBoost and LightGBM are imported in a `try/except ImportError` and only fail
  when their `--model-type` is selected.

### `preprocessing/` is a fork, not a shared library

`preprocessing/` duplicates `01_select_training_sample.py`, `02_gee_pixel_samples.py`,
`merge_pixel_exports.py`, `concat_training_csvs.py`, and `common.py` as a
self-contained pixel-only bundle. Its copies differ substantively:

- paths are absolute (`ROOT = Path(__file__).resolve().parents[1]`), so they run
  from anywhere, unlike the `scripts/` copies
- `common.py` there drops `STUDY_BBOX` / `TARGET_CROPS` and adds a JSON progress
  tracker (`preprocessing/state/preprocessing_progress.json`) via
  `load_tracker`/`save_tracker`; GEE checkpoints go to `preprocessing/state/` instead of `data/`

A fix to a pixel script usually needs applying in both copies. `scripts/05` and
`scripts/07` are the only training path — `preprocessing/` has no trainer.

### `batch_pipeline/` is a separate, incomplete workflow

Moved here from `crop-yield/` on 2026-10-22. It has its own helpers, config and
trainer and does not use `scripts/05`. Run as `python batch_pipeline/<script>.py`
from `crop-classification/`; outputs resolve relative to `batch_pipeline/`.

- `batch_pipeline/crop_classification_pipeline_config.json`
  is the contract for the batch flow: labels may come from AAFC (local raster
  preferred, GEE fallback) but **all predictor features must come from Copernicus
  Data Space Sentinel-2 L2A**, never GEE. Season Apr 1–Oct 1, 7-day aggregation,
  10 m, 40 batches. `feature_policy.exclude` bars soil moisture, weather, static
  soil, and terrain features. Needs `CDSE_USERNAME` / `CDSE_PASSWORD`.
- Scene discovery works; the runner's direct-CDSE extraction raises
  `NotImplementedError` and was never written, so no feature CSVs exist yet.
- Geometry defaults (`../quarter_sections.geojson`) resolve to
  `crop-classification/quarter_sections.geojson`, a local untracked symlink to
  `raw_data/quarter_sections.geojson`; without it pass/set that path explicitly.
- Batch training (`train_one_vs_rest_crop_models.py`) groups by
  `(quarter_section, year)` under `StratifiedGroupKFold` and ranks features before
  CV — same leakage concern as the classification side.

## Crop Yield (`crop-yield/`)

```bash
python -m pip install -e '.[ml]'
PYTHONPATH=src python -m unittest discover -s tests -v
yield-prediction --data data/raw/yield_df.csv summary
bash scripts/run_basic_pipeline.sh
```

The supplied source restores the historical multi-model/random-split benchmark,
pixel trainers, ATS downloader and relative yield index. Preserve these restored
implementations. The current APIs are `load_records`, `clean_records`,
`summarize_records`, `benchmark_models`, `train_pixel_yield_models` and
`train_pixel_crop_classifier`; earlier replacement APIs are no longer present.
Optional numerical imports are lazy so the historical summary can run without ML.

Source lives in `src/yield_prediction/`, Earth Engine JavaScript in `gee/`, and
local entry points in `scripts/`. Raw geometry and satellite exports live in
`data/raw/`; supplied observed-yield/prepared pixel CSVs live in
`reference_inputs/`. Pixel wrapper defaults target the reference table. Generated
outputs go under `data/processed/`. Existing benchmark CSVs are results from the
previous chronological implementation, not from the restored random-split CLI.

### Remaining incomplete Alberta workflows

The current `docs/PIPELINE_GUIDE.md` records workflows A–G and current availability.
Workflow D (batch crop classification) moved to `crop-classification/batch_pipeline/`;
its letter is kept so E–G references stay stable. No documented source is missing:
`scripts/build_fullfield_pixel_training_table.py` was recovered on 2026-10-22, but
it joins yields by quarter section and year (not crop) and adds no field-years.
Files copied from the old-repository zip use pre-reorganization default paths;
check them against `docs/HANDOFF_NOTES.md` before running. New harvest Excel
import/field matching is separate work; supplied 2021–2023 reference CSVs do not
imply that integration is done.

### What the surviving code does

- `src/yield_prediction/pixel_yield.py` — redistributes a field-average yield
  across pixels using NDVI scores (`NDVI_max` 0.5 / `NDVI_integral_proxy` 0.3 /
  `NDVI_min` 0.2), quantile-scaled within `row_id` groups and clamped to
  `[min_factor, max_factor]` so group means are preserved. pandas is imported
  lazily inside functions so the module stays importable without ML extras.
- `train_pixel_yield_models` has opt-in `split="year"`, `drop_year_constant` and
  `include_baseline` (script flags `--split year --drop-year-constant --baseline`)
  plus field-level `field_mae`/`field_r2`. Defaults reproduce the restored output.
  On the reference table `soil_moisture_am_mean` is one value per year; once it is
  dropped, no model beats the training-mean baseline.
- Harvest flow: `scripts/import_harvest_data.py` → `data/processed/harvest/operations.csv`
  → `scripts/match_harvest_boundaries.py` (`harvest_boundaries.py`) →
  `field_boundaries.geojson` + `boundary_matches.csv`. Legal descriptions map to
  ATS pids; the GeoJSON is streamed one feature per line. Quarter sharing counts
  excluded operations too; `shared_multiple_crops` rows are not pilot-eligible.
  → `scripts/export_harvest_features.py` (`harvest_features.py`, Earth Engine) →
  `data/processed/harvest/satellite/harvest_satellite_features.csv`, one row per
  eligible `(pid, year)` with `quarter_*` and AAFC-crop-masked `crop_*`
  `<band>_<mon>` columns. No precipitation (CHIRPS ends at 50°N).
  → `scripts/build_field_yield_table.py` (`field_yield.py`) → `field_yield_table.csv`
  (144 field-years; yield = total / area over same-crop operations sharing a quarter)
  → `scripts/benchmark_field_yield.py` (`field_benchmark.py`): per-crop mean
  baseline vs RidgeCV vs shallow HistGradientBoosting, `year` and `section` splits.
  Reported "best" skill is optimistic (best of six); check the skill range.
  `scripts/holdout_field_yield.py` picks one config per crop on 2020–2024 and
  scored 2025 once (2026-10-22): only wheat beat the baseline; canola −6.08 skill.
  2025 is now spent as a holdout — develop further on 2020–2024 only.
  `scripts/export_harvest_weather.py` (ERA5-Land, Apr–Sep monthly `w_*`) feeds the
  table; `scripts/evaluate_season_field_model.py` (`season_field_model.py`) splits
  yield into a season mean (ridge pooled over crops on weather or satellite season
  anomalies) plus a per-crop deviation from centred satellite features. Weather
  season model loses to the training mean with five seasons; `satellite`/`peak`
  (chosen post hoc) is the best combination. `--season-features regional` uses
  `scripts/prepare_regional_yields.py` output (StatCan 32-10-0002-01, Small Area
  Data Region 70 = CAR 7; zip in `data/raw/statcan/`, run with `python -I`)
  scaled by a training farm-to-region ratio. Season error 33.0% vs satellite
  31.2% vs training mean 34.2%; the farm diverges from the region in 2021/2023.
- `holdout_field_yield.py --min-improvement 0.1` keeps the baseline unless a
  model beats it by 10% on development years; backtests on 2023/2024 favour 10%.
- `row_id = quarter_section + crop + year` is the join key across the yield
  preprocessing and modeling steps.

## Data conventions shared by both sides

- `pid` is the stable ATS quarter-section identifier from `quarter_sections.geojson`.
- Sentinel missing-value sentinels (`-9999`) must be converted to `NaN` during
  preprocessing.
- `quarter_sections.geojson` is province-wide and ~1.2 GB. Read it streaming;
  do not load, copy, or commit it.
