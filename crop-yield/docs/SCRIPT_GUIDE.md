# Crop yield: script-by-script process

Updated 2026-09-29 for the newly supplied source. Paths are relative to
`crop-yield/`. Run local commands from that directory after editable installation.
The [pipeline guide](PIPELINE_GUIDE.md) is the execution/status map.

## A. Historical workflow

### `src/yield_prediction/data.py`

**Input:** `data/raw/yield_df.csv` by default, resolved from the package location.
`load_records()` reads CSV dictionaries; `clean_records()` removes the saved
index, converts year/numeric values, and selects area, crop, rainfall, pesticide,
temperature and yield fields. `summarize_records()` returns a dataclass containing
row/column counts, distinct areas/crops and year range. No cleaned file is saved.

**Model role:** establishes the historical table's predictors and target
`hg/ha_yield`. The restored cleaner does not perform the earlier replacement's
finite-value validation or duplicate removal. It does not import the harvest
Excel layout. `Area` means geographic category, not harvested acreage.

### `src/yield_prediction/cli.py`

Parses `--data` and `summary`/`benchmark`. Summary prints the data dataclass;
benchmark calls `modeling.benchmark_models()` and prints model names, test R²,
MSE, MAE, MAPE and CV mean R². Missing optional ML dependencies produce a readable
error. It does not accept `--output-dir` or write benchmark CSVs.

**Model role:** orchestration, registered as `yield-prediction` by `pyproject.toml`.

### `src/yield_prediction/modeling.py`: historical benchmark

`benchmark_models()` performs the following:

1. Read/clean the historical CSV. It retains duplicate rows.
2. Use all retained columns except yield as predictors. Impute categorical
   values, one-hot encode area/crop, and impute/standardize numeric predictors.
3. Make a random 70/30 training/test split with seed 42 by default.
4. Fit linear regression, random forest, gradient boost, KNN, decision tree,
   bagging and optional XGBoost through preprocessing pipelines.
5. Evaluate the holdout and separately run shuffled five-fold CV over the input.
6. Return metric dataclasses sorted by test R²; fitted models are not saved.

**Model role:** historical algorithm comparison. This restored implementation is
not the earlier two-model chronological benchmark. Previously saved CSVs remain
historical output and must not be presented as results of this implementation.
Random splits with repeated fields/years or duplicates do not establish forecasting
performance on new seasons.

### `src/yield_prediction/__init__.py`

Exports data-summary/cleaning and pixel-redistribution functions. It defines the
public import surface and performs no training. Optional numerical imports are
kept inside the relevant functions so summary works without ML packages.

### `scripts/run_basic_pipeline.sh`

Locates the project root, selects Python through optional `PYTHON_BIN` and data
through optional `DATASET_PATH`, runs summary, then runs the benchmark when
pandas/scikit-learn are available. It sets `PYTHONPATH=src`. This launcher does
not orchestrate satellite extraction or harvest import.

## B. Provincial geometry

### `src/yield_prediction/alberta_ats.py`

**Input:** ArcGIS `--where` filter and page size (default 1,000). Constructs query
URLs to the configured Alberta ATS layer, fetches pages while a full page is
returned, and accumulates features in memory. `save_geojson()` writes a feature
collection; geopandas optionally writes a GeoPackage. Defaults are
`data/raw/alberta_quarter_sections.geojson` and `.gpkg`.

**Model role:** supplies candidate land boundaries/IDs for matching labels and
satellite observations. It fits no model and does not resolve farm subfields.

**Limits:** network/service behavior has not been validated in this organization.
The GeoPackage helper assigns EPSG:3400 without inspecting returned coordinates;
verify CRS before using that output. Existing local geometry need not be downloaded
again. Full-province collection is in memory, not streamed to disk.

## C. Seasonal satellite export

### `gee/sentinel2_quarter_sections.js`

**Input:** uploaded quarter-section GEE asset and configured ID property, year
range, observation thresholds and optional feature limit.

**Process:** filter polygons by coverage diagnostics across configured years;
form seasonal Sentinel-2 collections, cloud mask and derive spectral indices;
reduce seasonal features per polygon; add SMAP moisture, ERA5-Land weather,
static terrain/soil statistics and AAFC crop fractions; build one feature per
polygon/year; export CSV to Drive. Default seasonal windows are May–mid-June,
late-June/July, August and September. Output names include window prefixes and
statistics, along with `quarter_id`/`year`.

**Model role:** produces predictors and crop reference context. It neither trains
a model nor supplies measured yield. Unlike the monthly full-field export, this
is a seasonal polygon table and is the input schema for `yield_index.py`.

**Limits:** asset placeholders, identifier case and years must be configured.
Cloud masking and band scaling need scientific validation; a local syntax check
does not execute Earth Engine. See [gee/README.md](../gee/README.md).

## F. Restored pixel learning functions and wrappers

### `src/yield_prediction/modeling.py`: pixel feature sets

`_build_pixel_feature_sets()` selects numeric columns while excluding crop,
IDs, observed yield and original-yield metadata. `ndvi_only` selects NDVI
columns; `satellite` includes recognized vegetation/radar prefixes; `nonleaky`
adds selected terrain/soil/location columns; `all` includes the remaining numeric
columns. These names are implementation choices, not guarantees of scientific
independence. Review new input schemas and forecast cutoffs before reuse.

### `src/yield_prediction/modeling.py`: pixel yield training

`train_pixel_yield_models()` reads a prepared table, selects predictors and loops
over crops. It evaluates OLS and ridge regression with median imputation and
standardization inside GroupKFold, grouping by `row_id`, up to five folds.
Validation needs at least two groups and two observed yields. It reports mean
fold MAE/RMSE, R² where supported, field-year counts, and returns held-out
`pred_*` values alongside full-data `fitted_*` values. Groups without enough
support have no valid held-out score. No deployable model bundle is returned.

**Model role:** learns relationships between pixel features and the supplied
field-yield target. Pixels sharing a field yield are correlated, so eight
field/crop/year groups are not 749 independent harvest observations.

### `scripts/train_pixel_yield_models.py`

Defaults to `reference_inputs/pixel_level_all_crop_training_features_2021_2023.csv`.
Passes feature set and ridge alpha to the restored training function, writes
metric/prediction CSVs and a Markdown summary under
`data/processed/pixel_yield_models_2021_2023/`. `--input` and `--output-dir`
override locations. This wrapper's formerly missing function is now present.

### `src/yield_prediction/modeling.py`: pixel crop classification

`train_pixel_crop_classifier()` selects features and compares standardized
logistic regression with a 300-tree balanced random forest, using median
imputation. StratifiedGroupKFold groups by `row_id`; folds are skipped if a test
crop is absent from training. It reports evaluated pixels, grouped accuracy,
balanced accuracy and macro-F1, returns out-of-fold predictions, then fits on
all rows to return separate `fitted_*` predictions. The latter are in-sample.

**Model role:** predicts crop identity, not yield. Same-field different-year
groups can still appear in different folds; this is not an entire-field holdout.

### `scripts/train_pixel_crop_classifier.py`

Uses the same default reference input, calls the restored classifier, and writes
metrics, predictions, a Markdown summary, per-crop reports and confusion-matrix
CSVs under `data/processed/pixel_crop_classifier_2021_2023/`. Per-crop reports use
evaluable `pred_*` columns rather than full-data fitted predictions. With few
independent groups, skipped folds and evaluation counts matter more than a
headline score. This function is now present and importable.

## NDVI redistribution: existing calculation, prepared data required

### `src/yield_prediction/pixel_yield.py`

**Input:** pixel table containing `row_id`, `yield_bu_ac`, `NDVI_max`, `NDVI_integral_proxy` and `NDVI_min` by default. The group is intended to identify a field-year with one known yield repeated across its pixels.

**Process:**

1. Check required columns and basic parameter bounds.
2. Within each group, clip each NDVI statistic to its 5th/95th percentiles and scale to 0–1. Missing/constant groups use a neutral value of 0.5.
3. Compute a weighted score: maximum NDVI 0.5, integral proxy 0.3, minimum NDVI 0.2 by default.
4. Apply a score floor of 0.25, then divide each score by the group's mean score.
5. Clip relative yield factors to 0.5–1.5 by default; renormalize those factors to have group mean 1.
6. Multiply the known field yield by the pixel factor and compute group mean/balance error.
7. Additional helpers load CSVs, summarize pixel estimates by group and write pixel/summary tables.

**Output:** estimated pixel yield, scores, factors and field-level summary values. The final factor bounds can move after renormalization.

**Model role:** a fixed allocation heuristic. It learns no parameters from harvest outcomes. For equal-weight pixel rows and a consistent yield within each group, the mean estimate reproduces the known field yield by construction; a near-zero balance error is not an independent accuracy test.

**Limits:** input consistency must be checked before use: constant field yield within a group, valid target units, and the intended pixel support. The code uses a simple row mean, not area-weighted partial-pixel accounting. It does not derive NDVI summaries from imagery, reconstruct geometries, or validate estimates against spatial harvest-monitor observations.

### `scripts/estimate_pixel_yield_from_ndvi.py`

**Input:** prepared pixel CSV, output directory, group column, score/quantile/factor settings and NDVI weights. Default input is the supplied `reference_inputs/pixel_level_all_crop_training_features_2021_2023.csv`.

**Process:** load table; assemble weights; call the redistribution function; summarize by field group; write both output tables.

**Output:** `pixel_yield_estimates_ndvi.csv` and `field_yield_estimate_summary_ndvi.csv` under `data/processed/pixel_ndvi_yield_estimates/` by default.

**Model role:** CLI wrapper around the fixed NDVI calculation, not a trained yield predictor.

**Harvest limitation:** supplying Excel harvest reports alone is insufficient. Prepare matching pixel NDVI statistics, verified group IDs and a known field yield. The wrapper expects `yield_bu_ac`; `lb/ac` observations must not be relabeled as bushels. The classification export's monthly `NDVI_*` columns do not directly supply all required summary column names.

## G. Relative yield-potential index

### `src/yield_prediction/yield_index.py`

**Input:** seasonal GEE polygon CSV with `quarter_id`, year, AAFC crop fractions
and seasonal vegetation/moisture statistics. The positional CSV input
is required; outputs default under `data/processed/`.

**Process:** parse values and treat `-9999` as missing; infer crop with a default
0.6 fraction threshold (rows without a qualifying crop fraction are dropped); derive peak/seasonal signals;
standardize within crop/year and across years within crop; calculate a fixed
weighted index using NDVI, EVI, NDRE1, NDMI, moisture and a negative MSI term;
rank and label low/medium/high; write per-year results and quarter-section summary.

**Model role:** relative ranking heuristic. It is not fitted to observed harvest
yield and does not output calibrated bushels/acre. Rankings depend on the supplied
comparison population. The monthly full-field pixel CSV is not a drop-in input.

## Older batch classification workflow inside the yield project

This folder's name can be confusing: it primarily prepares and classifies crop types. It does not connect harvest workbooks to a working yield regressor.

### `batch_pipeline/prepare_crop_classification_batches.py`

**Input:** source GeoJSON and JSON config. Default geometry is the supplied `data/raw/alberta_quarter_sections.geojson`; the JSON config references the same file.

**Process:** stream features with a JSON decoder; optionally exclude road-allowance features using the `ra` property; count eligible features; divide their ordered indices into configured batches (40 by default); assign configured assignee labels cyclically; initialize progress fields.

**Output:** `quarter_section_batches.csv` with start/end ranges and `quarter_section_tracker.csv` with counts/status/timestamps.

**Model role:** schedules data preparation, without selecting crop labels, extracting features or learning a model.

**Status/limits:** source and config must be supplied correctly. Index ranges depend on unchanged source ordering and filtering. Existing tracker files are overwritten. The parser's buffer-growth loop should be checked before general reuse: an incomplete feature occupying the read threshold can fail to trigger another read. This script is not a verified general-purpose streaming importer.

### `batch_pipeline/run_crop_classification_batch.py`

**Intended input:** batch ID, geometry, batch/config files, optional years (default 2021–2023), AAFC label source and satellite service credentials.

**Visible process:**

1. Load config/batch through `quarter_section_batch_utils`, collect geometry and mark tracker status running.
2. Save the batch GeoJSON. Obtain majority crop labels either by masking a local AAFC raster or calling AAFC/ACI in Earth Engine.
3. Choose the satellite source. The preferred `copernicus_data_space` path raises `NotImplementedError`.
4. The surviving alternative constructs a Sentinel Hub **Statistics API** request, despite the option name `sentinel_hub_process_api`. It fetches optical band statistics in weekly intervals.
5. Flatten weekly means, derive vegetation/red-edge/color/flowering indices, then calculate seasonal means/extremes/ranges, peak week and early/mid/late growth summaries.
6. Join crop labels on quarter section and year, write features, and update tracker state or failure notes.

**Intended output:** batch GeoJSON, `batch_features/<batch-id>_features.csv`, and tracker updates.

**Model role:** creates predictors and crop targets for the following binary classifier. It fits no classifier or yield model.

**Actual blockers:** `quarter_section_batch_utils.py` is absent; preferred CDSE extraction is unimplemented; the referenced scene-manifest builder is absent; local raster path is blank; the supplied config lacks the legacy Statistics endpoint. The GEE path hard-codes project `satellite-analysis-489120`, unlike the main classification exporters' saved-default behavior.

**Review before restoration:** local raster geometry is not reprojected to the raster CRS in this function; the shown evalscript exposes SCL but does not use it to mask clouds in its dataMask; all-NaN temporal groups can fail during peak-week computation. Labeling and feature availability need validation, not just restored imports.

### `batch_pipeline/train_one_vs_rest_crop_models.py`

**Intended input:** batch feature CSVs with `crop_label`, `quarter_section`, `year` and numeric predictors.

**Visible process:**

1. Load/concatenate selected batch CSVs; identify numeric features excluding target, location and ATS metadata.
2. Construct group keys from quarter section + year.
3. For each observed crop, create a binary label and rank features by absolute correlation with that label. Keep up to 32 features by default.
4. Use shuffled `StratifiedGroupKFold` (up to five folds), skipping folds without both classes in train or test.
5. Fit median imputation, standardization and balanced logistic regression within each evaluated fold; write out-of-fold positive-class probabilities.
6. Calculate precision, recall, F1, ROC-AUC and average precision at a 0.5 classification threshold; optionally update tracker entries.

**Intended output:** `binary_crop_model_metrics.csv` and `binary_crop_predictions.csv`. No fitted model bundle is saved.

**Model role:** evaluates binary crop identity classifiers, not yield regression.

**Actual status/limits:** missing batch utilities prevent import. Feature ranking currently uses the entire dataset before cross-validation, leaking held-out labels into feature selection; move selection inside each training fold before relying on scores. Quarter-section/year grouping can allow the same quarter section in different years across folds. Unevaluated rows keep missing probabilities but receive a `no` label through a fill operation, so they must not be counted as evaluated negative predictions.

### `batch_pipeline/crop_classification_pipeline_config.json`

Configuration, not an executable script. Describes source choices, geometry path, crop interests, weekly April–October-1 season, 10 m requested resolution, 40 batches and output directories. It prefers local AAFC labels and CDSE optical features and lists weather/soil/terrain as excluded feature families for this older workflow. These are local configuration settings, not instructions to change the user's project goals. Some settings are not enforced directly by surviving code; missing endpoints/paths and unimplemented branches prevent a runnable configuration.

## E. Full-field preprocessing: remaining missing script

`scripts/build_fullfield_pixel_training_table.py` is still absent. The supplied
monthly raw export, observed-yield CSV and prepared pixel lookup are now present
in `data/raw/` and `reference_inputs/`. The documented preprocessor would normalize
IDs/crops/years, replace missing sentinels, derive NDVI summaries, join observed
yield and form grouped rows. Recovering or implementing it is needed to regenerate
prepared training data from the monthly export. Existing prepared data allows F
to run independently of that missing preprocessing step.

## Tests and configuration

| File | Current role |
| --- | --- |
| `tests/test_data.py` | Check default bundled path, dataset coverage and restored record conversions |
| `tests/test_cli.py` | Check summary, default path from another working directory, and missing-ML error |
| `tests/test_modeling.py` | Check feature exclusions/monthly names, grouped yield prediction outputs and a small restored historical benchmark |
| `pyproject.toml` | Package, CLI registration and optional ML dependencies |
| `requirements.txt` | Broader extraction/modeling dependencies |

Eight tests cover local package behavior. They do not validate live downloads,
Earth Engine execution, original harvest import, or missing batch helpers.

## What remains absent

- `scripts/build_fullfield_pixel_training_table.py`
- `batch_pipeline/quarter_section_batch_utils.py`
- `batch_pipeline/copernicus_data_space_utils.py`
- `batch_pipeline/build_copernicus_scene_manifest.py`

The preferred direct-CDSE branch remains unimplemented. Follow
[NEXT_STEPS.md](NEXT_STEPS.md) for the separate harvest-data integration plan.
