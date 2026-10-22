# Crop classification: script-by-script process

Source reviewed 2026-09-29; sections for `04`–`07` and `ml_features.py` updated 2026-10-07 for the step-1 evaluation changes; batch pipeline section moved here 2026-10-22. Paths in code spans are relative to `crop-classification/`. Commands normally run from that directory in its Python environment. This guide describes the existing implementation; proposed fixes are in [NEXT_STEPS.md](NEXT_STEPS.md).

## What the model learns

An observation is either a polygon (quarter section) or a sampled satellite pixel. **X** is its numeric satellite/weather feature vector; **y** is its crop label. IDs and label metadata link records but should not tell the model the answer. Data preparation determines whether X and y describe the same land and growing season.

Only training fits a classifier. Exporting imagery, calculating vegetation indices, joining labels and combining CSVs prepare training examples. Prediction applies a fitted classifier without learning new relationships.

## Execution paths

```text
Polygon path:
00 extract area → 01 sample polygons → 02 export polygon features
                                   → 03 obtain AAFC labels
                    features + labels → 04 join → 05 train → 06 predict

Pixel path:
prepared polygons → preprocessing/01 sample → preprocessing/02 export pixels + labels
                  → download CSV parts → merge_pixel_exports → 05 train → 06 predict

Batch path (incomplete):
batch_pipeline/prepare → build_copernicus_scene_manifest → run_crop_classification_batch
                       → train_one_vs_rest_crop_models   (direct-CDSE extraction unfinished)

Optional: concat_training_csvs extends a dataset; 07_compare_feature_sets evaluates inputs.
```

The pixel exporter with `--include-labels` obtains labels itself, so the separate polygon labeling/join steps are unnecessary for that path. Without labels, prepare a defensible label join before training.

## Main scripts

### `scripts/00_extract_study_area.py`

**Input:** `raw_data/quarter_sections.geojson`; `STUDY_BBOX` from `scripts/common.py`.

**Process:**

1. Read a saved byte-offset checkpoint or start a new output.
2. Scan the large source line by line, looking for lines beginning with a particular GeoJSON feature format.
3. Apply a longitude-text prefilter, parse each candidate, and test the first polygon coordinate against the bounding box.
4. Append matching features. After about 30 seconds, save progress and exit; rerun until `ALL DONE` closes the GeoJSON collection.

**Output:** `data/study_area_quarter_sections.geojson`, plus `.extract_study_area.state` and `.extract_study_area.log` under `data/`.

**Model role:** defines the candidate geographic population. It neither labels land nor trains a model.

**Limits:** unfinished output is not a complete GeoJSON. The hard-coded `-113.`/`-112.` prefilter and first-coordinate test make this a specialized Lethbridge extractor, not a general polygon-intersection tool. A new region requires more than editing the bounding box. Preserve outputs and use appropriate new/reset state when changing the source or region.

### `scripts/01_select_training_sample.py`

**Input:** completed study-area GeoJSON; optional existing sample via `--exclude`.

**Process:** load features; remove IDs already in the exclusion sample; choose up to `--n` polygons using a seeded random sample (defaults: 1,500 and seed 42); preserve source CRS metadata.

**Output:** `data/training_sample.geojson` by default, or `--output`.

**Model role:** controls extraction size and training coverage. Sampling is random, not crop-stratified, so rare classes may remain scarce. This is not the later training/test split.

### `scripts/02_gee_sentinel_features.py`

**Input:** sampled polygons with `properties.pid`, target year, Earth Engine access and optional project override.

**Process:**

1. Build an Earth Engine feature collection carrying `pid`.
2. For each April–October month, filter Sentinel-2 surface reflectance observations and mask SCL cloud/shadow classes 3, 8, 9 and 10. The code scales the image by 10,000; despite the surrounding comment mentioning QA60, the actual mask uses SCL.
3. Form monthly median optical composites and derive NDVI, NDYI, GNDVI, NDRE, NDWI, EVI and SWIR ratio. These describe greenness, yellowness, red-edge response and moisture-related spectral variation.
4. Form Sentinel-1 VV/VH monthly medians and their difference. Sum the configured CHIRPS precipitation observations by month.
5. Suffix feature names with months and reduce each band/index to its mean over each polygon (default reduction scale 10 m).
6. Export a CSV to Drive and wait for task completion, or request a small local result through `getInfo()`.

**Output:** one wide row per polygon, typically downloaded as `data/sentinel2_features.csv`.

**Model role:** turns changing crop appearance into predictor columns such as `NDVI_jul` and `NDYI_jul`. It provides no target label and fits no classifier.

**Limits:** feature year is selected by the command but not explicitly added as a year column. Preserve it in filenames/metadata. Drive completion does not download the file locally. Requested spatial scale does not make all source layers have that native resolution. Validate precipitation coverage and empty/masked observations for the target region. SCL itself remains among exported bands and is also scaled by the current code; it is a scene-classification code, not reflectance.

### `scripts/02_gee_pixel_samples.py`

**Input:** polygons, year, Earth Engine access; optional `--include-labels`, `--label-year` and `--keep-other`.

**Process:** build the same monthly feature stack as the polygon exporter, then call `sampleRegions()` inside the polygons at a default 30 m sampling scale. With labels enabled, add AAFC/ACI land-cover codes for the label year, map codes to crop names, and optionally filter out other classes. Drive mode divides polygons into batches (default 250), exports each batch and saves success/failure checkpoints.

**Output:** `data/<out-name>_checkpoint.json` and Drive `<out-name>_partNNN.csv` files; local mode writes `data/<out-name>.csv` directly. Download Drive files before merging.

**Model role:** supplies more detailed spatial observations instead of polygon means. AAFC labels provide pixel targets, not harvest-measured pixel yield.

**Limits:** pixels within a field are correlated. `geometries=False` omits pixel geometry; `pid` alone cannot locate an individual pixel. The checkpoint records configuration but does not reject a changed configuration before skipping completed batches. Use a new output/checkpoint name for a new year or polygon set. Fully masked sample pixels may be omitted.

### `scripts/03_download_aafc_labels.py`

**Input:** sampled polygons, AAFC inventory year, optional minimum purity.

**Process:** convert polygon rings to Web Mercator; request per-polygon AAFC raster histograms from its ImageServer; ignore code 0; find the most frequent code and its share of valid pixels; map it to a configured crop. Exclude no-data labels and, by default, non-target classes. Apply `--min-purity` if provided.

**Output:** `data/aafc_labels.csv` with `pid`, `aafc_code`, `crop_label`, `label_purity`, `aafc_pixel_count`.

**Model role:** supplies supervised crop targets and a way to exclude mixed polygons. For example, a purity threshold of 0.7 requires at least 70% of counted valid pixels to share the dominant class.

**Limits:** the script default is **0.0**, although the runbook suggests 0.7. A dominant quarter-section class can still mislabel a smaller harvested field within that parcel. AAFC forage code 122 is mapped to an alfalfa proxy; it is not species-specific confirmation.

### `scripts/04_join_labels_features.py`

**Input:** polygon feature CSV and label CSV.

**Process:** remove export metadata `system:index`/`.geo` from features; require `pid`; inner-join labels and features on `pid`. Then handle missing values:

- By default, drop rows containing any missing value.
- With `--keep-missing`, keep rows with partly missing features as NaN. Drop only rows missing every feature or any label field.

Either way, print the dropped rows by crop and the class counts.

**Output:** `data/training_table.csv` by default.

**Model role:** creates paired predictor/target examples suitable for training.

**Limits:** does not validate one-to-one keys or preserve/validate year matching. Duplicate keys can multiply observations. It is not ready to combine multi-year harvest records solely by `pid`. Missing CHIRPS precipitation north of 50°N marks location; with `--keep-missing`, exclude the `precip` family when training.

### `scripts/ml_features.py`

**Input:** feature dataframe, requested months/families, and metadata exclusion set supplied by the caller.

**Process:** drop QA signals (`QA_PREFIXES`, currently `SCL`) when the caller asks; select month-suffixed columns; optionally derive differences between consecutive available selected months, seasonal maximum, minimum and range; classify signal names into `s2`, `s1`, `precip` or `other`; filter by requested families.

**Output:** transformed dataframe used by training and prediction; no standalone file or CLI.

**Model role:** exposes seasonal growth and decline patterns. A change in NDVI between June and July may help separate crops even when one month's value is similar.

**Limits:** these transformations are per row; they do not learn fitted parameters. Feature timing still matters for forecasts. Column retention uses naming rules—do not assume arbitrary new metadata survives filtering or is excluded from modeling. Training drops SCL. Models trained before 2026-10-07 used SCL, and prediction still rebuilds it for them.

### `scripts/05_train_decision_tree.py`

Despite its name, this trains four model families.

**Input:** labeled polygon or pixel CSV containing `crop_label`, numeric predictors and `pid` for grouped validation.

**Process:**

1. Drop SCL QA columns. Filter months (default April–October), add temporal features by default, and filter feature families.
2. Remove its explicit non-feature columns: IDs, target, AAFC code, export metadata and AAFC purity/count. Convert remaining columns to numeric; reject non-numeric values. Keep missing values as NaN and print how many rows each feature family affects.
3. Construct a balanced decision tree, logistic-regression pipeline (median imputation, standardization), CatBoost classifier, or LightGBM classifier. Decision trees split feature thresholds; logistic regression combines weighted standardized inputs; boosting models combine many trees. The tree models route NaN natively.
4. Count distinct `pid`s per crop. For `multiclass`, drop rows of crops below `--min-class-fields` (default 10) and learn the crop label directly. For `one-vs-rest`, skip those crops, train each remaining crop against `other`, and write `one_vs_rest.json` listing trained and skipped crops.
5. Split off a holdout (default 25%). The default is now `grouped`: `GroupShuffleSplit` on `pid[:6]` (meridian + range + township). `random` uses stratified row splitting.
6. Fit on the training portion, predict the holdout, and calculate class precision, recall, F1 and accuracy. Separately run repeated cross-validation over the full supplied dataset: `--cv-folds` (5) × `--cv-repeats` (3). Grouped mode uses `StratifiedGroupKFold` on the same township groups; random mode uses `StratifiedKFold`. Each fold reports accuracy, macro-F1 (over classes present in that fold) and per-class F1. The first repeat's out-of-fold predictions give a pooled report and confusion matrix. When a `pid` has several rows (pixel tables), the holdout and every fold also report field-level scores: average each pid's row probabilities, then take the top class. The true field label is the majority pixel label.
7. Save the fitted holdout-training model, preprocessing metadata, reports and diagnostic graphics.

**Output:** `--output-dir`, defaulting to `models/<model_type>/` or `models/<model_type>_one_vs_rest/<crop>/`. Each directory contains `model.joblib`, `metrics.txt`, `metrics.json`, `confusion_matrix.png` (holdout), `cv_confusion_matrix.png` (out-of-fold), and `feature_importances.json`. Decision trees also produce rules and a tree plot. `metrics.json` keeps the earlier keys and adds `cv` (fold summaries and out-of-fold reports), `data` (input, rows, fields, groups, class counts), `settings` and `skipped_classes`.

**Model role:** this is the core learning/evaluation step. Saved bundles include required feature order, months, temporal settings, family selection, group settings and input path so inference can rebuild inputs.

**Limits:** saved models are not refit on all rows after the holdout evaluation. Cross-validation and holdout are separate evaluations; the CV uses all supplied rows. The script refuses to write into a non-empty output directory unless `--overwrite` is given. With `--overwrite`, older crop subdirectories not listed in the manifest stay on disk but prediction ignores them. A large pixel count does not imply many independent fields. Harvest metadata/targets require additional exclusions before they are added to these inputs.

### `scripts/06_predict.py`

**Input:** a saved bundle or a directory of one-vs-rest bundles, plus new feature CSV.

**Process:** load each model; reconstruct month selection and temporal features; select saved feature columns in order; reject non-numeric features and pass missing values as NaN. For a single model, output its class prediction and class probabilities. For one-vs-rest, load the crops listed in `one_vs_rest.json`; older directories have no manifest, so load every crop subdirectory. Skip models that never saw their crop, then choose the crop with the largest positive-class probability.

**Output:** `data/predictions.csv` or `--output`, with `pid` when available, `predicted_crop`, and `prob_<class>` columns.

**Model role:** inference only; it applies what training learned.

**Limits:** no unknown-class threshold or probability calibration. One-vs-rest scores need not sum to one. Output does not retain year, pixel coordinates or other input metadata. Map creation and independent truth comparison are separate steps. Features must have the same meaning and units, not just matching names.

### `scripts/07_compare_feature_sets.py`

**Input:** one labeled table and two feature-family selections, defaults `s2` versus `s2,s1,precip`; default crops wheat/barley and grouped validation.

**Process:** dynamically import the training script; prepare and evaluate one-vs-rest classifiers for each crop/family combination with the selected settings. Extract holdout precision, recall, F1 and support, plus mean ± std cross-validated F1 (and field-level CV F1 for pixel tables) and actual feature counts.

**Output:** `data/feature_set_comparison.csv` and matching `.json` by default. Prints a cross-validated F1 comparison with `cv_f1_delta` = augmented minus baseline.

**Model role:** assesses whether extra inputs help distinguish specific crops. It does not save deployable model bundles.

**Limits:** it filters existing columns; it cannot add radar/rainfall absent from the CSV. It warns when both selections give a crop identical feature families, as with `s2` versus `s2,s1,precip` on the current pixel export. `--cv-repeats` (default 3) multiplies runtime on pixel tables.

### `scripts/merge_pixel_exports.py`

**Input:** a quoted glob of downloaded part CSVs, default `data/sentinel2_pixel_samples_part*.csv`.

**Process:** sort paths, require exactly matching ordered headers, and stream all data rows into one CSV with a single header.

**Output:** `data/sentinel2_pixel_samples.csv` by default.

**Model role:** assembles export partitions into one training input. It does not clean values, verify export completeness or remove duplicate rows.

**Limits:** keep the destination outside the input glob and verify all intended batches were downloaded. A header mismatch can leave a partial destination file.

### `scripts/concat_training_csvs.py`

**Input:** required `--pattern` and `--output`; optional `--dedupe-cols`.

**Process:** stream matching tables with identical ordered headers; optionally keep only the first row for each specified key tuple.

**Output:** the requested combined CSV plus printed counts.

**Model role:** grows the training dataset across compatible exports.

**Limits:** no deduplication by default. `pid` alone would collapse all pixels in a parcel; `system:index` may not be unique across independent exports. Choose a stable observation key including year and pixel identity where appropriate. Avoid including both part files and their already-merged copy or the destination in the glob.

### `scripts/common.py`

**Input/output:** imported constants and simple GeoJSON helpers; no training or standalone output.

**Process and model role:** defines Lethbridge bounds, AAFC-to-crop mapping, target crops, selected optical/radar/precipitation bands, April–October month names, and `properties.pid` ID access. These choices define study population, label vocabulary and feature schema. Wheat codes are pooled. Code 122 becomes the alfalfa proxy.

**Limits:** changing this file alone does not update the separate duplicated constants in `preprocessing/common.py` or remove hard-coded logic in the extractor.

## Preprocessing scripts

These are separate implementations, not aliases. Prefer this folder for tracked pixel preparation and keep its helpers available. Default paths resolve from the project; explicitly supplied relative paths still follow the terminal's working directory.

### `preprocessing/common.py`

Defines the preprocessing crop/band/month constants, GeoJSON/ID helpers, project-relative data paths, `preprocessing/state/`, and tracker load/save functions. The tracker has `quarter_section_sets`, `pixel_exports` and `merged_outputs`. It makes preparation reproducible to inspect but does not train a model. Keep crop mappings synchronized with the main `common.py`.

### `preprocessing/01_select_training_sample.py`

Reads study-area geometry, optionally excludes existing IDs, and performs the same seeded sample as the main sampler. Writes the sample GeoJSON and records its source, count, seed, excluded sample and chosen IDs in `preprocessing/state/preprocessing_progress.json`. Its model role is sample selection and provenance. It still assumes the correct study-area geometry already exists.

### `preprocessing/02_gee_pixel_samples.py`

Uses the monthly pixel export/optional AAFC labeling process described above. Defaults to project-relative input/output paths; stores batch checkpoints under `preprocessing/state/`; also registers export settings and batch outcomes in the global progress tracker. Produces Drive CSV parts or a local table. Its model role is detailed feature/label preparation, not learning. The same geometry omission, source-coverage, correlation and checkpoint-configuration limitations apply.

### `preprocessing/merge_pixel_exports.py`

Reads downloaded part CSVs, enforces matching headers and streams them to a combined table, then registers source pattern/files, row count and merge type in the global tracker. Default paths resolve to the project's `data/`. Its model role is dataset assembly and traceability. It does not deduplicate or establish completeness of the exported dataset.

### `preprocessing/concat_training_csvs.py`

Combines tables using the same strict-header and optional key-based deduplication process as the main concatenator. Its default destination is `data/sentinel2_pixel_samples_all.csv`. Registers source files, output rows, duplicate count and dedupe keys in the tracker. Its model role is controlled dataset expansion. The same observation-key and input-glob cautions apply.

### `preprocessing/progress_tracker.py`

Loads the global tracker and referenced checkpoint files. Reports recorded quarter-section sets, completed/failed/total export batches, downloaded CSV-part counts and line-based row counts for recorded merged outputs. `--json` prints raw tracker state. Its role is operational visibility; it does not verify scientific validity or train/evaluate a model. A completed Drive task and a downloaded file are different states.

## Batch pipeline (`batch_pipeline/`)

Moved here from `crop-yield/batch_pipeline/` on 2026-10-22 with unchanged file
contents. This incomplete workflow predicts **crop identity**, not harvested yield. AAFC crop codes are its targets; optical
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

This project's `requirements.txt` covers `requests`, `pandas`, `numpy`,
`scikit-learn` and `earthengine-api`. Local AAFC raster sampling also needs
`rasterio`, which is not listed. Checks before the move (2026-10-01) covered
syntax and `--help`; extraction has never been validated.

### `batch_pipeline/prepare_crop_classification_batches.py`

**Inputs and controls:** `--geojson`, `--config`, `--batches-out`, `--tracker-out`.
The default is `quarter_sections.geojson` at this project's root. On the current
machine that is a local, untracked symlink to `raw_data/quarter_sections.geojson`;
without it, pass `--geojson raw_data/quarter_sections.geojson`.
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
this manifest.

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

`geojson.path` is `../quarter_sections.geojson`, relative to `batch_pipeline/`,
so it resolves to the same root-level path as preparation's default. Where that
symlink is absent, set it to `../raw_data/quarter_sections.geojson`. The local
AAFC raster pattern remains empty and legacy Statistics settings remain absent.
See [the batch README](../batch_pipeline/README.md) for commands and prerequisites.

## Using the harvest files here

No script above reads the supplied `.xls` reports. First import/review farm labels and geometry, then extract matching observations and build the exact classifier schema. Use the [harvest assessment](../../harvest-data/README.md) and [next-step plan](NEXT_STEPS.md) before substituting farm data into this workflow.
