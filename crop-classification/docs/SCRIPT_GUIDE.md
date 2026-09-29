# Crop classification: script-by-script process

Source reviewed 2026-09-29. Paths in code spans are relative to `crop-classification/`. Commands normally run from that directory in its Python environment. This guide describes the existing implementation; proposed fixes are in [NEXT_STEPS.md](NEXT_STEPS.md).

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

**Process:** remove export metadata `system:index`/`.geo` from features; require `pid`; inner-join labels and features on `pid`; drop rows containing any missing value; print class counts.

**Output:** `data/training_table.csv` by default.

**Model role:** creates paired predictor/target examples suitable for training.

**Limits:** does not validate one-to-one keys, preserve/validate year matching, or report missing-value losses separately. Duplicate keys can multiply observations. It is not ready to combine multi-year harvest records solely by `pid`.

### `scripts/ml_features.py`

**Input:** feature dataframe, requested months/families, and metadata exclusion set supplied by the caller.

**Process:** select month-suffixed columns; optionally derive differences between consecutive available selected months, seasonal maximum, minimum and range; classify signal names into `s2`, `s1`, `precip` or `other`; filter by requested families.

**Output:** transformed dataframe used by training and prediction; no standalone file or CLI.

**Model role:** exposes seasonal growth and decline patterns. A change in NDVI between June and July may help separate crops even when one month's value is similar.

**Limits:** these transformations are per row; they do not learn fitted parameters. Feature timing still matters for forecasts. Column retention uses naming rules—do not assume arbitrary new metadata survives filtering or is excluded from modeling. SCL is categorized as `other`.

### `scripts/05_train_decision_tree.py`

Despite its name, this trains four model families.

**Input:** labeled polygon or pixel CSV containing `crop_label`, numeric predictors and `pid` for grouped validation.

**Process:**

1. Filter months (default April–October), add temporal features by default, and filter feature families.
2. Remove its explicit non-feature columns: IDs, target, AAFC code, export metadata and AAFC purity/count. Convert remaining columns to numeric; reject missing or invalid values.
3. Construct a balanced decision tree, standardized logistic-regression pipeline, CatBoost classifier, or LightGBM classifier. Decision trees split feature thresholds; logistic regression combines weighted standardized inputs; boosting models combine many trees.
4. For `multiclass`, learn the crop label directly. For `one-vs-rest`, repeat training for each configured target crop against `other`.
5. Split off a holdout (default 25%). The default is random stratified splitting. `grouped` uses `GroupShuffleSplit` on the first six characters of `pid` by default.
6. Fit on the training portion, predict the holdout, and calculate class precision, recall, F1 and accuracy. Separately run default five-fold cross-validation over the full supplied dataset; grouped mode uses `GroupKFold`.
7. Save the fitted holdout-training model, preprocessing metadata, reports and diagnostic graphics.

**Output:** `models/<model_type>/` or `models/<model_type>_one_vs_rest/<crop>/`, containing `model.joblib`, `metrics.txt`, `metrics.json`, `confusion_matrix.png`, and `feature_importances.json`. Decision trees also produce rules and a tree plot.

**Model role:** this is the core learning/evaluation step. Saved bundles include required feature order, months, temporal settings, family selection and group settings so inference can rebuild inputs.

**Limits:** saved models are not refit on all rows after the holdout evaluation. Cross-validation and holdout are separate evaluations; the CV uses all supplied rows. Training output directories are fixed and reruns overwrite files. `metrics.json` omits some provenance present in the bundle. One-vs-rest requires positive examples for each intended crop. A large pixel count does not imply many independent fields. Harvest metadata/targets require additional exclusions before they are added to these inputs.

### `scripts/06_predict.py`

**Input:** a saved bundle or a directory of one-vs-rest bundles, plus new feature CSV.

**Process:** load each model; reconstruct month selection and temporal features; select saved feature columns in order; reject missing/non-numeric features. For a single model, output its class prediction and class probabilities. For one-vs-rest, collect each crop's positive-class probability and choose the largest.

**Output:** `data/predictions.csv` or `--output`, with `pid` when available, `predicted_crop`, and `prob_<class>` columns.

**Model role:** inference only; it applies what training learned.

**Limits:** no unknown-class threshold or probability calibration. One-vs-rest scores need not sum to one. Output does not retain year, pixel coordinates or other input metadata. Map creation and independent truth comparison are separate steps. Features must have the same meaning and units, not just matching names.

### `scripts/07_compare_feature_sets.py`

**Input:** one labeled table and two feature-family selections, defaults `s2` versus `s2,s1,precip`; default crops wheat/barley and grouped validation.

**Process:** dynamically import the training script; prepare and evaluate one-vs-rest classifiers for each crop/family combination with the selected settings; extract positive-class precision, recall, F1, support and actual feature counts.

**Output:** `data/feature_set_comparison.csv` and matching `.json` by default. Prints an F1 comparison and difference when there are two experiment columns.

**Model role:** assesses whether extra inputs help distinguish specific crops. It does not save deployable model bundles.

**Limits:** it filters existing columns; it cannot add radar/rainfall absent from the CSV. Check feature counts and identical row/group coverage. The printed difference is based on pivot-column order; inspect the experiment labels when interpreting its sign.

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

## Using the harvest files here

No script above reads the supplied `.xls` reports. First import/review farm labels and geometry, then extract matching observations and build the exact classifier schema. Use the [harvest assessment](../../harvest-data/README.md) and [next-step plan](NEXT_STEPS.md) before substituting farm data into this workflow.
