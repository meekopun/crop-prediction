# Crop yield: script-by-script process

Updated 2026-10-22: the batch pipeline moved to `crop-classification/` and the
full-field builder was recovered. Paths are relative to
`crop-yield/`. Run local commands from that directory after editable installation.
The [pipeline guide](PIPELINE_GUIDE.md) is the execution/status map.

## Harvest report import

### `src/yield_prediction/harvest.py`

Reads `Sheet1` from the farm's `.xls` report layout using optional `xlrd`.
Tracks field headings across pages, joins continuation cells, parses explicit
numbers/units and harvest dates, recognizes explicit crop names, and retains
source rows and original text. Keeps each operation separate and flags invalid
measurements, date/unit/total conflicts, unknown crops, name-only fields and
multiple operations sharing a field/year. No unit conversion or geometry matching
occurs; every record starts excluded from the pilot. `src/harvest.py` re-exports
the parser functions for compatibility with the original stub location.

### `scripts/import_harvest_data.py`

Accepts `--input-dir`, `--output-dir`, `--review-file` and `--no-review`. Defaults
resolve from the source location, not the working directory. Applies
`config/harvest_review.json` by default: the user's canola confirmation and ten
temporary operation exclusions. All 206 records remain in `operations.csv`;
`processing_candidates.csv` contains 196 pending boundary review. Also writes
`exceptions.csv` and `summary.json` under `data/processed/harvest/`. Re-running
replaces these staging files. Exit codes are 0 on success (warnings allowed),
1 on file/dependency/layout failure, and 2 when written output contains errors.
Install the dependency with `pip install -e '.[harvest]'`.

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

It also reports `field_mae`/`field_r2`, which score each field-year's mean held-out
prediction against its yield (the measurement scale). Opt-in options, all off by
default: `split="year"` holds out every field from one year instead of one
field-year; `drop_year_constant=True` removes features with a single value within
every year (in the supplied table, `soil_moisture_am_mean`, which otherwise lets
OLS fit year-level yields exactly); `include_baseline=True` adds `mean_baseline`,
which predicts the training-fold mean yield.

**Model role:** learns relationships between pixel features and the supplied
field-yield target. Pixels sharing a field yield are correlated, so eight
field/crop/year groups are not 749 independent harvest observations.

### `scripts/train_pixel_yield_models.py`

Defaults to `reference_inputs/pixel_level_all_crop_training_features_2021_2023.csv`.
Passes feature set and ridge alpha to the restored training function, writes
metric/prediction CSVs and a Markdown summary under
`data/processed/pixel_yield_models_2021_2023/`. `--input` and `--output-dir`
override locations. This wrapper's formerly missing function is now present.
`--split year`, `--drop-year-constant` and `--baseline` expose the opt-in
evaluation options and add `_year_split`, `_no_year_constant` and `_baseline` to
output filenames, so default-run files are not overwritten.

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

**Harvest limitation:** supplying Excel harvest reports alone is insufficient. Prepare matching pixel NDVI statistics, verified group IDs and a known field yield. The wrapper expects `yield_bu_ac`; `lb/ac` observations must not be relabeled as bushels. The classification export's monthly `NDVI_`* columns do not directly supply all required summary column names.

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

## D. Batch crop classification: moved

`batch_pipeline/` moved to [`crop-classification/batch_pipeline/`](../../crop-classification/batch_pipeline/README.md)
on 2026-10-22 with unchanged file contents, because it predicts crop identity
rather than yield. Its script-by-script process is now in the
[classification script guide](../../crop-classification/docs/SCRIPT_GUIDE.md#batch-pipeline-batch_pipeline).

## E. Full-field pixel preprocessing

### `scripts/build_fullfield_pixel_training_table.py`

Recovered from the old repository on 2026-10-22; its defaults were re-pointed to
the current layout.

**Inputs and controls:** `--pixel-input` (default
`data/raw/pixel_crop_features_aci_s2_monthly_csv_quarter_sections_fullfield.csv`),
`--lookup-input` (default `reference_inputs/pixel_level_all_crop_training_features_2021_2023.csv`),
`--yield-input` (default `reference_inputs/Crop Yield Data - Crop Data 2021-2023-2.csv`,
used only when the lookup file is absent) and `--output` (default
`data/processed/pixel_level_fullfield_training_features_2021_2023.csv`).

**Process:**

1. Normalize quarter IDs: drop `ALL-` rows, map `W-` to `SW-`, and keep only
  `NE-`/`NW-`/`SE-`/`SW-` IDs. Map crop names `Canola` → `Canola and Rapeseed`
   and `Peas Yellow` → `Peas`.
2. Replace `-9999` with missing values in numeric columns other than
  `aci_crop_code`, `is_target_crop` and `year`.
3. Derive `NDVI_max`, `NDVI_min` and `NDVI_median` from the monthly `ndvi_m*`
  columns and copy `ndvi_integral` to `NDVI_integral_proxy`. Rename `lat`/`lon`
   to `latitude`/`longitude` and `crop_name` to `aci_crop_name`.
4. Inner-join yields on `quarter_section` and `year`, then build `row_id` from
  quarter section, crop (spaces removed) and year.

**Output:** on the supplied inputs, 87,176 pixel rows, eight field-years and three
quarter sections (checked 2026-10-22). It adds no field-years beyond the prepared table.

**Model role:** regenerates a pixel training table from the monthly export. It
fits no model.

**Limits:** the join ignores crop, so every pixel in a quarter is attached to each
crop reported there that year: SW-33 Barley and Canola 2022 share all 10,880
pixels, as do SW-33 Canola and Spring Wheat 2023. `aci_crop_name` and
`is_target_crop` are not used to filter pixels. Mapping `W-` to `SW-` is an
assumption about a half-section label. Crop names differ from the prepared table.
The monthly columns are lowercase (`ndvi_m04`), which F's `ndvi_only`, `satellite`
and `nonleaky` sets do not recognize: they select only the four derived NDVI
columns (plus latitude/longitude for `nonleaky`), while `all` selects 110 numeric
columns, including `aci_crop_code` and `year`.

## Tests and configuration


| File                     | Current role                                                                                                       |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| `tests/test_data.py`     | Check default bundled path, dataset coverage and restored record conversions                                       |
| `tests/test_cli.py`      | Check summary, default path from another working directory, and missing-ML error                                   |
| `tests/test_harvest.py` | Check page breaks, wrapped values, units, validation failures, staging exports and local report reconciliation |
| `tests/test_modeling.py` | Check feature exclusions/monthly names, grouped yield prediction outputs, year-constant feature removal, year split with baseline and a small restored historical benchmark |
| `pyproject.toml`         | Package, CLI registration and optional ML dependencies                                                             |
| `requirements.txt`       | Broader extraction/modeling dependencies                                                                           |


Tests cover local package behavior and harvest import. Actual farm-workbook
checks require the local reports and optional `xlrd`; synthetic parser checks
run without Excel dependencies. Tests do not validate live downloads, Earth
Engine execution or the full-field builder.

## What remains absent

No documented source file is missing. Follow [NEXT_STEPS.md](NEXT_STEPS.md) for
the separate harvest-data integration plan.
