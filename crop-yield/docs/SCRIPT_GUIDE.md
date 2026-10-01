# Crop yield: script-by-script process

Updated 2026-10-01 for the newly supplied batch helpers and scene-manifest builder. Paths are relative to
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

## D. Batch crop classification inside the yield project

All six Python files in `batch_pipeline/` are now present. This workflow predicts
**crop identity**, not harvested yield. AAFC crop codes are its targets; optical
satellite measurements are its predictors. It does not import the harvest workbooks.

### How the scripts connect


| Stage                                | File                                     | Result for the model                                                       |
| ------------------------------------ | ---------------------------------------- | -------------------------------------------------------------------------- |
| Divide the geometry                  | `prepare_crop_classification_batches.py` | Reproducible lists of parcels to process                                   |
| Select parcels and track work        | `quarter_section_batch_utils.py`         | Shared IDs, batch ranges and progress records                              |
| Describe candidate satellite scenes  | `copernicus_data_space_utils.py`         | Search results and product/download metadata                               |
| Save weekly scene candidates         | `build_copernicus_scene_manifest.py`     | A scene manifest, **not a training feature table**                         |
| Obtain labels and optical predictors | `run_crop_classification_batch.py`       | Intended quarter-section/year feature table; direct CDSE branch unfinished |
| Evaluate crop classifiers            | `train_one_vs_rest_crop_models.py`       | Binary crop probabilities and cross-validation metrics                     |


The intended sequence is preparation → discovery → extraction → training. There
is currently a gap between discovery and extraction: the manifest builder does
not download/process imagery, and the runner does not consume its manifest.
Restoring these files resolves missing imports, but does not finish that gap.

Local checks on 2026-10-01 found `requests` missing in `crop-yield/.venv`.
Preparation and trainer `--help` passed; the manifest builder and combined runner
stop at the `requests` import. All six files passed syntax parsing. This confirms
source coverage and available command-line imports, not completed extraction.

### `batch_pipeline/prepare_crop_classification_batches.py`

**Inputs and controls:** `--geojson`, `--config`, `--batches-out`, `--tracker-out`.
The copied default still points at `crop-yield/quarter_sections.geojson`, which is
absent. The actual geometry is `data/raw/alberta_quarter_sections.geojson`; use
`--geojson data/raw/alberta_quarter_sections.geojson` from `crop-yield/`.
Changing `geojson.path` in the config does not change this script's separate
`--geojson` default.

**Process:**

1. Load batching settings and the road-allowance filter from JSON.
2. Stream the source GeoJSON with a JSON decoder and count eligible features.
  When exclusion is enabled, only lowercase `ra` values of `None`, `""` or
   `"null"` are retained. Uppercase `RA` is not checked.
3. Divide the filtered source order into 40 contiguous batches by default.
  Start/end indices are half-open: `[feature_start, feature_end)`. Ranges are
   approximately equal in feature count, without crop stratification or shuffling.
4. Name batches `batch_01`, `batch_02`, etc. and alternate the configured assignee
  labels (`person_a`, `person_b`). These are bookkeeping labels.
5. Write the plan and initialize tracker statuses to `pending`, counts to zero,
  and timestamps/notes to empty values.

**Outputs:** `batch_pipeline/quarter_section_batches.csv` and
`batch_pipeline/quarter_section_tracker.csv` by default. The latter records
overall, label, satellite and model status, plus counts and notes. Later scripts
update some statuses; the initialized per-parcel counts are not a verified
per-parcel completion ledger.

**Model role:** partitions extraction work. It does not create crop labels,
satellite predictors or fitted models.

**Limits:** rerunning overwrites the tracker and plan. Source ordering and the
filter must stay consistent after planning. Its incremental parser can stall when
an incomplete feature occupies at least the 65,536-character read threshold,
because it stops reading while repeatedly attempting to decode that buffer.
This parser limitation also exists in the shared utility below.

### `batch_pipeline/quarter_section_batch_utils.py`

**Input:** config JSON, batch/tracker CSVs, the source GeoJSON and requested batch
ranges. This is an imported helper module, with no command-line entry point.

**Process and function responsibilities:**

- `BatchSpec` stores batch ID, assignee, start/end indices and feature count.
`load_batches()` reads the CSV into these records; `get_batch()` finds a
requested ID or raises an error.
- `load_config()` reads JSON. `resolve_path()` resolves relative paths against a
supplied directory; `output_paths()` instead uses this module's directory
(`batch_pipeline/`) for configured outputs. Geometry callers also use that
directory, even when `--config` points elsewhere.
- `iter_geojson_features()` incrementally decodes GeoJSON features.
`include_feature()` and `iter_filtered_geojson_features()` apply the same
lowercase `ra` filter as preparation.
- `collect_batch_features()` walks the filtered sequence from its beginning,
skips entries before `feature_start`, and keeps entries before `feature_end`.
These are indices in the **filtered** sequence, not the raw file.
- `feature_identifier()` prefers `pid`, `PID`, `id`, then `ID`; otherwise it
constructs `feature_0000000`-style IDs from the supplied index.
`write_geojson()` saves a selected FeatureCollection.
- `load_tracker()` reads tracker values as strings. `update_tracker()` updates
matching rows and rewrites the CSV. `append_note()` joins notes with `|`;
`tracker_note()` saves the updated notes. `iso_date()` formats a date string.

**Outputs:** returned feature lists, batch records, paths and identifiers;
explicit calls can write subset GeoJSONs and rewrite the tracker.

**Model role:** keeps geometry selection, labels and features tied to the same
parcel/year keys. It performs no learning or satellite extraction.

**Limits:** fallback IDs are batch-local in the runner, so they can repeat across
batches; stable source IDs are needed before joining training data. Each batch
scan starts at the beginning of the large source file. Tracker writes have no
locking, so concurrent processes can overwrite each other's updates. The
streaming-parser limitation described above remains.

### `batch_pipeline/copernicus_data_space_utils.py`

**Input:** geometry, season dates, cloud threshold, configured endpoints and
`CDSE_USERNAME` / `CDSE_PASSWORD`. Imported by the manifest builder; it has no
command-line entry point. It imports `requests`.

**Process and function responsibilities:**

1. `iso_date_range()` constructs a season's dates. `weekly_ranges()` divides
  that interval into numbered seven-day windows, shortening the last window.
2. `bbox_union()` visits Polygon/MultiPolygon coordinates and returns one
  `[min_lon, min_lat, max_lon, max_lat]` rectangle covering the batch.
3. `cdse_client_from_env()` requires the two environment variables and builds
  `CDSEClient` with the endpoints from JSON.
4. `CDSEClient.access_token()` can request a password-grant access token and
  cache it. The manifest builder does **not** call this method, and the search
   and product-lookup methods send no authorization header in this code.
5. `search_sentinel2_l2a()` posts a STAC search for `sentinel-2-l2a`, bounding box,
  date window and maximum `eo:cloud_cover`. It converts the response's features
   into `SceneItem` records containing scene ID, time, cloud cover, collection,
   bounds, geometry and product name.
6. `resolve_product_uuid()` looks up the product name through OData and returns
  its UUID or `None`. `product_download_url()` constructs a `$value` URL string.

**Output:** dates, bounds, scene metadata, product UUIDs and URL strings.
Constructing a download URL does not fetch the raster.

**Model role:** supplies candidate imagery metadata for a future extractor.
It currently produces no band values, vegetation indices or crop predictions.

**Limits:** search reads only the first response page, without following
pagination. The access token has no expiry/refresh handling. Geometry must be a
nonempty set of supported polygons in longitude/latitude coordinates. Discovery
endpoints and credentials have not been validated by the documentation update.

### `batch_pipeline/build_copernicus_scene_manifest.py`

**Inputs and controls:** required `--batch-id`; optional `--config`, `--years`
and `--limit-per-week` (default 8). Years default to 2021–2023 in the code.
Requires the batch/tracker CSVs, matching source geometry, `pandas`, `requests`,
and the environment variables required by the CDSE client factory.

**Process:**

1. Load config/output paths and the requested batch range.
2. Collect that batch's filtered polygons and compute a single union bounding
  box. This rectangle can include land between widely separated parcels.
3. Mark `sentinel2_status` as `running`.
4. For each requested year, use the configured April 1–October 1 season and
  generate seven-day date windows. The helper uses fixed seven-day windows;
   it does not read `season.aggregation_interval`.
5. Search each window using the configured cloud threshold (60 by default) and
  requested result limit. Sort the returned candidates by cloud cover then time,
   treating unknown cloud cover as 999. This ranks only the returned page, not
   every possible scene.
6. Resolve each product's UUID and construct a download URL where available.
7. Save the manifest, set `sentinel2_status` to `manifest_ready`, and append a
  tracker note.

**Output:** `batch_pipeline/scene_manifests/<batch_id>_cdse_scene_manifest.csv`.
Rows contain batch ID, year, week index/start/end, rank, scene ID, product name,
product UUID, download URL, timestamp, cloud cover and collection. A blank URL
means product lookup returned no UUID.

**Model role:** records which imagery could support weekly predictors.
`manifest_ready` means discovery finished; it does not mean features are ready.

**Limits:** no imagery download, cloud masking, polygon statistics, feature CSV
or training occurs here. A failed request can leave the tracker as `running`,
and rows are not checkpointed before completion. An empty search result creates
an empty DataFrame without a defined column schema. The runner does not load
this manifest. The copied config still needs its geometry path corrected.

### `batch_pipeline/run_crop_classification_batch.py`

**Inputs and controls:** required `--batch-id`; `--config`, `--years` (default
2021–2023), `--label-source` and `--sentinel-source`. Both source options default
to `preferred`, which resolves the corresponding JSON choice.

**Process:**

1. Load the batch, mark overall status `running`, collect its polygons and save
  `batch_geojson/<batch_id>.geojson`.
2. Obtain one majority AAFC crop code per quarter-section/year. The local route
  masks a yearly raster and takes the most frequent valid positive code; it
   requires `rasterio` and a nonempty `local_raster_pattern` with `{year}` as needed.
   The alternative calls Earth Engine's AAFC/ACI mode reducer and maps crop codes
   to labels. It uses the hard-coded project `satellite-analysis-489120`.
3. Mark labels `completed`, then select the satellite route. The preferred
  `copernicus_data_space` branch explicitly raises `NotImplementedError`.
4. The alternative option `sentinel_hub_process_api` actually uses the
  **Sentinel Hub Statistics API**. Given valid service settings and credentials,
   it requests seven-day optical-band means for each polygon/year at the
   configured resolution, cloud threshold and aggregation interval.
5. Flatten B02/B03/B04/B05/B06/B07/B08/B8A/B11/B12 means into weekly columns.
  Derive NDVI, NDRE1/2, GNDVI, NDMI, EVI, yellow/blue ratio, VARI, NGRDI,
   flowering contrast and red/green ratio from those band means.
6. Join labels by `quarter_section` and `year`; add index means/min/max/amplitude,
  peak week, early/middle/late-season means, greenup and senescence deltas.
7. Save the feature table and mark overall status `features_ready` and satellite
  status `completed`. On an exception, mark overall status `failed`, append the
   error to tracker notes and re-raise it.

**Intended outputs:** batch GeoJSON, tracker updates, and
`batch_pipeline/batch_features/<batch_id>_features.csv` with parcel/year keys,
location/legal metadata, `crop_label`, `label_code` and numerical predictors.
The feature CSV requires a successful extraction branch.

**Model role:** creates the target/predictor table that the binary trainer needs.
Color ratios and time-window summaries describe spectral differences and crop
seasonality. It does not fit the classifier or predict yield.

**Current blockers and limits:** the preferred CDSE branch is unfinished; the
local raster pattern is blank; the config lacks the alternative route's
`statistics_endpoint`. That route also requires `SENTINEL_HUB_CLIENT_ID` and
`SENTINEL_HUB_CLIENT_SECRET` plus compatible OAuth settings. Setting GEE as a
label source alone does not resolve satellite extraction. The configured label
fallback is not an automatic retry; select it explicitly. Local raster sampling
does not reproject GeoJSON into the raster CRS. The evalscript outputs SCL but
its data mask does not exclude cloudy SCL classes. An all-missing weekly series
can fail at `np.nanargmax()` while calculating peak week. The runner does not
read the CDSE manifest or enforce every configured feature-policy setting.

### `batch_pipeline/train_one_vs_rest_crop_models.py`

**Inputs and controls:** `--input-glob` (default `batch_features/*_features.csv`,
relative to `batch_pipeline/`), optional `--batch-ids`, `--config`, `--output-dir`,
`--cv-splits` (5), `--top-features-per-crop` (32) and `--logistic-c` (1.0).
Requires `pandas`, `numpy`, `scikit-learn`, and completed feature CSVs containing
`crop_label`, `quarter_section`, `year` and numeric predictors.

**Process:**

1. Load/concatenate matching CSVs, optionally restricting filenames to batch IDs.
2. Select numeric/bool columns while excluding target, label code, coordinates,
  legal-description metadata, parcel ID and year. Other numeric columns are not
   checked against a Sentinel-2 allowlist.
3. Group rows by `quarter_section|year` and loop over every observed crop label,
  rather than filtering to JSON `crops_of_interest`.
4. Encode each crop as 1 versus all other crops as 0. Rank predictors by absolute
  Pearson correlation with that binary target and retain up to 32 by default.
5. Build shuffled `StratifiedGroupKFold` splits with seed 42, using at most the
  number of distinct groups. Skip folds whose train or test set lacks both classes.
6. In each evaluated fold, fit median imputation, standard scaling and balanced
  logistic regression (`lbfgs`, up to 5,000 iterations, configured C). Predict
   held-out crop probabilities and apply a 0.5 decision threshold.
7. Calculate precision, recall, F1, ROC AUC and average precision for evaluated
  rows. Save combined metric/prediction tables; when `--batch-ids` is supplied,
   set those batches' model status to `completed`.

**Outputs:** `batch_pipeline/model_outputs/binary_crop_model_metrics.csv` and
`binary_crop_predictions.csv` by default. Metrics include positive-row count,
selected-feature count and evaluated-row count. Predictions include parcel/year,
crop name, binary truth, probability and a `yes`/`no` label. It does not save a
fitted inference model, selected-feature names or a multiclass winner.

**Model role:** evaluates a separate “is this crop?” classifier for each label.
Multiple crops can receive positive decisions; these are not mutually exclusive
multiclass predictions.

**Limits:** feature ranking occurs on the full dataset **before** cross-validation,
so held-out labels influence selection and scores can be optimistic. Move ranking
inside each training fold before relying on results. Parcel/year grouping allows
the same parcel in different years to cross folds; it does not measure strict
new-field or future-year generalization. Fewer than two groups is unsupported.
Unevaluated rows retain missing probabilities but receive `no` via a fill
operation; they must not be counted as evaluated negatives. A `completed` tracker
status does not imply an inference-ready saved model or trustworthy validation.

### `batch_pipeline/crop_classification_pipeline_config.json`

Configuration, not executable code. It describes AAFC label choices, CDSE/legacy
satellite routes, optical feature families, four crop interests, April–October
season, 10 m requested resolution, 60% cloud threshold, 40 batches and outputs.
These settings describe this workflow's intended design; they are not instructions
to change the user's project objectives. Several settings are only partly enforced
by the scripts, as described above.

The copied `geojson.path` is `../quarter_sections.geojson`. Set it to
`../data/raw/alberta_quarter_sections.geojson` before discovery or extraction;
these paths are relative to `batch_pipeline/`. Preparation additionally needs its
explicit `--geojson` override. The local AAFC raster pattern remains empty and
legacy Statistics settings remain absent. See [the batch README](../batch_pipeline/README.md)
for current command locations and prerequisites.

## E. Full-field preprocessing: remaining missing script

`scripts/build_fullfield_pixel_training_table.py` is still absent. The supplied
monthly raw export, observed-yield CSV and prepared pixel lookup are now present
in `data/raw/` and `reference_inputs/`. The documented preprocessor would normalize
IDs/crops/years, replace missing sentinels, derive NDVI summaries, join observed
yield and form grouped rows. Recovering or implementing it is needed to regenerate
prepared training data from the monthly export. Existing prepared data allows F
to run independently of that missing preprocessing step.

## Tests and configuration


| File                     | Current role                                                                                                       |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| `tests/test_data.py`     | Check default bundled path, dataset coverage and restored record conversions                                       |
| `tests/test_cli.py`      | Check summary, default path from another working directory, and missing-ML error                                   |
| `tests/test_modeling.py` | Check feature exclusions/monthly names, grouped yield prediction outputs and a small restored historical benchmark |
| `pyproject.toml`         | Package, CLI registration and optional ML dependencies                                                             |
| `requirements.txt`       | Broader extraction/modeling dependencies                                                                           |


Eight tests cover local package behavior. They do not validate live downloads,
Earth Engine execution, original harvest import, or the incomplete CDSE extraction route.

## What remains absent

- `scripts/build_fullfield_pixel_training_table.py`

The preferred direct-CDSE branch remains unimplemented. Follow
[NEXT_STEPS.md](NEXT_STEPS.md) for the separate harvest-data integration plan.