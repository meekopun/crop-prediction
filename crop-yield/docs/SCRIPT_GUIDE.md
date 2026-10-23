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

### `src/yield_prediction/harvest_boundaries.py`

**Input:** `operations.csv` from the importer and the province-wide
`data/raw/alberta_quarter_sections.geojson`. Uses only the standard library.

**Process:**

1. Parse `legal_description_raw` (`SE 30-71-9 W6M`, compact forms such as
  `NE12-71-10W6M`, `.` section separators, and `N1/2`/`S1/2` half sections) into
   ATS `pid` values: meridian, two-digit range, three-digit township, two-digit
   section, quarter code (SE 5, SW 6, NW 7, NE 8) and `00`.
2. Stream the GeoJSON line by line, decoding only lines whose `pid` is wanted and
  keeping the farmland quarter (`ra` null), not its road-allowance strips. The
   file must have one feature per line, as GDAL writes it; otherwise it fails.
3. For each `pending_boundary_review` operation, record whether every quarter was
  found, approximate parcel acres and `area_fraction` (reported acres / parcel acres).
4. Count every operation with a legal description, including excluded ones, per
  quarter and year: `single_operation`, `shared_same_crop` or `shared_multiple_crops`.
5. Mark `boundary_pilot_eligible` when matched, not shared with another crop and
  not above 105% of the parcel. Flags also record `low_parcel_coverage` (under
   half the parcel) and `quarter_year_area_exceeds_parcel`.

**Output:** `field_boundaries.geojson` (one feature per matched quarter, with `pid`,
ATS properties, `parcel_acres` and `boundary_source: ats_quarter_section`),
`boundary_matches.csv` and `boundary_match_summary.json` under
`data/processed/harvest/`. The boundary file's `pid` property suits the
classification project's `scripts/02_gee_sentinel_features.py --input`.

**Model role:** links yield targets to candidate land for satellite extraction. It
fits no model and converts or aggregates no yields.

**Limits:** a quarter section is not a surveyed field outline; fields can cover
part of a quarter or cross into a neighbour. Same-crop operations sharing a
quarter-year still need aggregation (total production / total area). Areas use a
local equirectangular approximation of lon/lat coordinates.

### `scripts/match_harvest_boundaries.py`

Accepts `--operations`, `--geometry` and `--output-dir`, defaulting to the
importer's output, the supplied geometry and `data/processed/harvest/`. Prints
match, sharing and eligibility counts. Exit code 1 on file or format failure.
On 2026-10-22 it matched all 196 candidates to 41 quarter sections in under two
seconds; 156 are pilot-eligible.

### `src/yield_prediction/harvest_features.py`

**Input:** `boundary_matches.csv` and `field_boundaries.geojson`. Needs
`earthengine-api` and a saved Earth Engine login; Earth Engine is imported lazily.

**Process:**

1. Build one unit per pilot-eligible `(pid, year)`. Eligible rows never share a
  quarter-year with another crop, so each unit has one crop; a half section gives
   a unit per quarter, and same-crop fields keep all their operation IDs.
2. For each year, paint each quarter with its crop and compare against that year's
  AAFC Annual Crop Inventory (`AAFC/ACI` `landcover`), using `CROP_ACI_CODES`
   (wheat 140/145/146, forage-seed crops 122). This gives a per-pixel crop mask.
3. For April–October, build monthly median composites: Sentinel-2 SR harmonized
  (scenes under 60% cloudy pixels, SCL cloud/shadow masked, reflectance / 10000)
   with NDVI, NDYI, GNDVI, NDRE, NDWI, EVI and SWIR ratio; Sentinel-1 IW VV/VH (dB)
   and VV−VH. A fully masked placeholder keeps band names when a month has no scenes.
4. Reduce means at 10 m over each quarter (`quarter_<band>_<mon>`), over its
  crop-labelled pixels (`crop_<band>_<mon>`), plus `s2_valid_<mon>` and
   `aci_crop_fraction`.

**Output:** `harvest_features_<year>.csv` per year (kept on rerun unless
`--overwrite`), combined `harvest_satellite_features.csv` (293 columns) and
`harvest_features_manifest.json` with collections, thresholds and codes, under
`data/processed/harvest/satellite/`.

**Model role:** supplies predictors for the field-yield benchmark. It fits no
model and carries no yield values; join targets by `pid`, `year` and operation IDs.

**Limits:** no precipitation or weather (CHIRPS ends at 50°N). The crop mask is
only as good as the inventory, which mostly disagrees with the farm records for
oats and rye. Monthly medians smooth over within-month timing. Local `getInfo`
suits these few dozen polygons per year, not province-wide runs.

### `scripts/export_harvest_features.py`

Accepts `--matches`, `--boundaries`, `--output-dir`, `--years` (default
2020–2025), `--project`, `--scale` (10) and `--overwrite`. Exit code 1 if
earthengine-api is missing or an export fails. On 2026-10-22 it exported 163
quarter-section years in under a minute.

### `src/yield_prediction/harvest_weather.py` and `scripts/export_harvest_weather.py`

Exports ERA5-Land (`ECMWF/ERA5_LAND/DAILY_AGGR`) monthly weather for the same
pilot-eligible `(pid, year)` units, April–September: `w_precip_<mon>` (mm),
`w_tmean_<mon>` and `w_tmax_<mon>` (°C), `w_gdd5_<mon>` (sum of
max(0, (tmax + tmin) / 2 − 5 °C)) and `w_soilwater_<mon>` (layer 1, m³/m³).
Quarters are sampled at 100 m so each picks up its ~11 km cell. Writes
per-year CSVs (kept unless `--overwrite`), `harvest_weather.csv` and a manifest
to `data/processed/harvest/weather/`. Flags match the feature exporter. On
2026-10-22 it exported 163 units in 10 seconds. Spread between quarters within
a season is a few mm or tenths of a degree, so this is a season-level signal.

### `src/yield_prediction/field_yield.py`

**Input:** `boundary_matches.csv`, `operations.csv` (harvested totals),
`satellite/harvest_satellite_features.csv` and `field_boundaries.geojson`.
Standard library only.

**Process:**

1. Group pilot-eligible operations of the same crop and year that touch a common
  quarter section, so a half section and its quarter join one field-year.
2. Set `yield` = total production / total harvested area, using `total_value`
  when its unit matches (`bu` for `bu/ac`, `lb` for `lb/ac`), else area × rate.
   Groups with mixed units or missing features are dropped and counted.
3. Combine the group's quarters: whole-quarter means weighted by parcel acres,
  crop-mask means weighted by crop-labelled acres.
4. Set `field_<band>_<mon>` to the crop-mask mean when `aci_crop_fraction` ≥ 0.1
  (`feature_source = crop_mask`), else to the whole-quarter mean; a month without
   crop pixels falls back to the quarter.

**Output:** `data/processed/harvest/field_yield_table.csv`, one row per
field-year with `field_key`, `section_key`, `township_key`, crop, unit, yield,
harvested acres and share, operation IDs, `s2_valid_*` and 140 `field_*`
columns, plus `field_yield_table_summary.json`. On 2026-10-22: 144 field-years,
118 using the crop mask, none dropped. When `weather/harvest_weather.csv`
exists (or `--weather PATH`), acre-weighted `w_*` columns plus
`w_precip_may_aug` and `w_gdd5_may_aug` are added; `--no-weather` omits them.

**Limits:** several fields on SE 30-71-9 W6M report more acres than the quarter,
so their land extends beyond it. Same-crop subfields are averaged together.

### `src/yield_prediction/field_benchmark.py`

For each crop with at least `min_rows` (10) field-years, in its own unit, runs
two grouped evaluations: `year` (leave one season out) and `section` (GroupKFold
by ATS section, up to five folds, holding out land across all years). Compares
a training-mean baseline with `RidgeCV` (median imputation, standardization,
penalty chosen by leave-one-out inside each training fold) and a shallow
`HistGradientBoostingRegressor`, on three feature sets: `ndvi` (7 monthly NDVI),
`indices` (49 monthly indices) and `all` (140 columns). Reports MAE, RMSE, bias,
pooled out-of-fold R² and `mae_skill` = 1 − MAE / baseline MAE.

**Limits:** six combinations are scored per crop, so the best one is chosen after
the fact and its skill is optimistic; check the range. Section folds share
seasons with training data, so they test new land, not new years. No model is
saved for inference.

`holdout_field_yield()` is the non-optimistic check. For each crop it scores all
seven candidates (baseline included) by leave-one-year-out on years before
`test_year` only, picks the lowest MAE (ties to the simpler candidate), refits the
chosen candidate and the baseline on those years, and scores both once on
`test_year`. It returns the per-candidate development scores, per-crop results
and test-year predictions. With `min_improvement` (script `--min-improvement`),
the best model replaces the baseline only if its development MAE is below
(1 − margin) × the baseline's; results record the best model either way. Years
after `test_year` are ignored, so `--test-year 2023` or `2024` gives a backtest
on development data. Margin runs add `_margin<pct>` to output names.

### `src/yield_prediction/regional_yields.py` and `scripts/prepare_regional_yields.py`

Streams Statistics Canada table 32-10-0002-01 (full-table zip
`data/raw/statcan/32100002-eng.zip`, or an extracted CSV via `--source`) and keeps
`Average yield (bushels per acre)` rows for `--region` (default `Small Area Data
Region 70 - Alberta`, the Peace region) from `--first-year` (2015). Crops map as
barley, canola, oats, `Peas, dry` → `field_peas` and `Wheat, spring` → `wheat`
(the farm grows Stettler, a hard red spring wheat). Suppressed values are skipped.
Writes `data/processed/regional/regional_yields.csv` (region, year, crop,
regional_yield, unit, source, status). On 2026-10-23 it wrote 55 crop-years
(five crops, 2015–2025) in about a second, with no suppressed values. Region 70
is Census Agricultural Region 7 (census divisions 17–19, including Grande Prairie
County No. 1). Run it with `python -I` because it reads a downloaded archive.

### `src/yield_prediction/season_field_model.py` and `scripts/evaluate_season_field_model.py`

Splits each yield into the crop's season mean plus the field's deviation, and
holds out one season at a time over development years only (2020–2024; later
rows are dropped first).

- **Field deviation:** per crop, `RidgeCV` on satellite features centred on the
  crop-season feature mean (features only, so it is available for a new season),
  trained on seasons with at least two fields. `--deviation-features indices`
  (49) or `peak` (July NDVI, NDRE, EVI).
- **Season level:** one ridge pooled over crops, predicting season mean / crop
  training mean from crop-season anomalies of `--season-features weather` (May–Aug
  rain and GDD5, July mean high) or `satellite` (season means of the peak indices).
  Alpha is chosen by leave-one-season-out inside the training seasons.
  `regional` instead multiplies the held-out season's published regional yield
  by the crop's farm-to-region ratio (sum of training farm season means / sum of
  their regional yields); crops without published yields keep the training mean.
  Reads `--regional` (default `data/processed/regional/regional_yields.csv`).
- **Variants scored:** training mean (baseline), season model, baseline + field,
  season model + field, and a ceiling using the crop's other fields' actual season
  mean (a stand-in for published regional yields), with and without the field part.

Writes `season_field_<season>_<deviation>_{metrics,seasons,predictions}.csv` and
`_summary.md` to `data/processed/harvest/benchmark/`. See NEXT_STEPS section 5 for
the 2026-10-22 results; the `peak` and `satellite` sets were chosen after
inspecting development diagnostics.

### `scripts/build_field_yield_table.py`, `scripts/benchmark_field_yield.py` and `scripts/holdout_field_yield.py`

The builder accepts `--matches`, `--operations`, `--features`, `--boundaries` and
`--output`. The benchmark accepts `--input`, `--output-dir` (default
`data/processed/harvest/benchmark/`) and `--min-rows`, and writes
`field_yield_metrics.csv`, `field_yield_predictions.csv` and `summary.md`
(best model and skill range per crop and split). It takes about 25 seconds.
The holdout script accepts `--input`, `--output-dir`, `--test-year` (2025) and
`--min-rows`, and writes `holdout_<year>_selection.csv`, `_results.csv`,
`_predictions.csv` and `_summary.md`. Run it once per test year: rerunning after
changing models with knowledge of its result makes the holdout optimistic.

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
| `tests/test_harvest_boundaries.py` | Check legal-description parsing, streaming lookup that skips road allowances, format rejection, quarter sharing and eligibility on a synthetic GeoJSON |
| `tests/test_harvest_features.py` | Check feature units (half sections, same-crop grouping, conflicts, missing geometry), output columns and crop-inventory code coverage; Earth Engine calls are not tested |
| `tests/test_field_yield.py` | Check field-year grouping through half sections, total/area yield, crop-mask versus quarter features, weather merging, that the benchmark finds a synthetic NDVI signal under both splits, and that the holdout selects on earlier years only and keeps the baseline unless a model beats it there |
| `tests/test_season_field_model.py` | On a synthetic table with rain-driven seasons and NDVI-driven field deviations: both parts add skill, later years are never used, the satellite/peak options run, the regional option recovers a known farm-to-region ratio and falls back without published yields, and missing inputs fail clearly |
| `tests/test_regional_yields.py` | Check the StatCan converter keeps only the region's yield rows for mapped crops from a zipped table, skips suppressed values and fails without matches |
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
