# Runbook

This runbook gives the shortest reliable path for the common tasks in this
repository.

## 1. Environment setup

From `crop-classification/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If Earth Engine steps are needed:

```bash
earthengine authenticate
earthengine set_project cropclassification-502620
python -c "import ee; ee.Initialize(); print(ee.Number(1).getInfo())"
```

The project is saved in your local Earth Engine settings and persists across
terminal sessions. All three export scripts use it when `--project` is omitted;
pass `--project ANOTHER_PROJECT_ID` to override it for one run.

## 2. Full polygon pipeline

Run these from `crop-classification/`.

If you only need to train or compare models, you can start directly from:

- `data/training_table.csv`

### Step 0: extract study area subset

Place the source at `raw_data/quarter_sections.geojson`. The extractor reads
it directly:

```bash
python scripts/00_extract_study_area.py
```

If the extractor prints `Checkpointed`, repeat the Python command until it
prints `ALL DONE`. Do not sample from an unfinished GeoJSON.

Primary input:

- `raw_data/quarter_sections.geojson`

Primary output:

- `data/study_area_quarter_sections.geojson`

To switch beyond the default Lethbridge County study area, edit `STUDY_BBOX`
in `scripts/common.py` first, then run step 0 again to regenerate the subset.

### Step 1: sample training polygons

```bash
python scripts/01_select_training_sample.py --n 1500
```

Primary output:

- `data/training_sample.geojson`

### Step 2: export polygon features from Earth Engine

```bash
python scripts/02_gee_sentinel_features.py --year 2024
```

Typical output path after download from Drive:

- `data/sentinel2_features.csv`

Notes:

- `--project` is optional when a default is saved with `earthengine set_project`.
- The default export mode is Drive.
- The exported CSV must be downloaded into `data/` before step 4.

### Step 3: download AAFC labels

```bash
python scripts/03_download_aafc_labels.py --year 2024 --min-purity 0.7
```

Primary output:

- `data/aafc_labels.csv`

Notes:

- This step needs network access to the AAFC endpoint.
- Use `--keep-other` if you want to keep non-target classes.

### Step 4: join features and labels

```bash
python scripts/04_join_labels_features.py
```

Primary output:

- `data/training_table.csv`

By default the join drops every row with any missing value. For the 2024
export, that discarded 112 of 423 labeled fields: wheat 43, canola 28,
barley 24, corn 10, peas 6 and alfalfa 1. Keep those rows as NaN instead:

```bash
python scripts/04_join_labels_features.py --keep-missing \
  --output data/training_table_2024_with_missing.csv
```

Most gaps are cloudy April composites. The rest are missing CHIRPS
precipitation: CHIRPS covers only 50°S–50°N, and all 139 sampled quarter
sections north of 50°N lack it. Missing precipitation therefore identifies a
location, not a crop. Train recovered tables with `--feature-families s2,s1`.

### Step 5: train models

Examples:

```bash
python scripts/05_train_decision_tree.py \
  --model-type decision-tree \
  --mode multiclass \
  --max-depth 6 \
  --output-dir models/decision_tree_v2

python scripts/05_train_decision_tree.py \
  --model-type logistic-regression \
  --mode one-vs-rest \
  --output-dir models/logistic_regression_one_vs_rest_v2

python scripts/05_train_decision_tree.py \
  --input data/training_table_2024_with_missing.csv \
  --model-type lightgbm \
  --mode multiclass \
  --feature-families s2,s1 \
  --output-dir models/step1/C_polygon423_s2s1
```

Without `--output-dir`, outputs go to the old fixed directories, for example
`models/lightgbm/` or `models/catboost_one_vs_rest/`. Training refuses to
write into a directory that already holds a run unless you pass `--overwrite`.

Evaluation settings:

- `--split-strategy grouped` is the default. It holds out whole townships
  (`pid[:6]` = meridian + range + township).
- `--cv-repeats 3` (default) runs 5-fold `StratifiedGroupKFold` three times
  with reshuffled folds.
- `--min-class-fields 10` (default) skips one-vs-rest crops labeled on fewer
  than 10 distinct `pid`s. In multiclass mode it drops those crops' rows.
- Sentinel-2 `SCL` columns are cloud-mask codes, so training drops them from
  the inputs.
- Missing feature values are allowed. LightGBM, CatBoost and decision trees
  handle them natively; logistic regression imputes training-fold medians.

Each run directory adds `cv_confusion_matrix.png` from out-of-fold
predictions. `metrics.txt` and `metrics.json` report CV accuracy, macro-F1 and
per-class F1 (mean ± std), plus input, row/field counts and settings. Pixel
tables also get field-level scores, which average each `pid`'s pixel
probabilities. One-vs-rest directories include `one_vs_rest.json`, which lists
the trained and skipped crops.

### Step 6: run prediction

Single model bundle:

```bash
python scripts/06_predict.py \
  --model models/decision_tree/model.joblib \
  --features data/sentinel2_features.csv \
  --output data/predictions.csv
```

One-vs-rest model directory:

```bash
python scripts/06_predict.py \
  --model models/catboost_one_vs_rest \
  --features data/sentinel2_features.csv \
  --output data/predictions.csv
```

Primary output:

- `data/predictions.csv`

For one-vs-rest directories, prediction uses only the crops listed in
`one_vs_rest.json`. Older directories have no manifest, so every crop
subdirectory is used, except models that never saw their crop. One such
example is `alfalfa` in `models/lightgbm_one_vs_rest/`.

## 3. Pixel-level preprocessing path

This path is useful when the collaborator wants pixel-level training data
instead of polygon-averaged features.

If you only need to train or compare pixel-level models, you can start
directly from:

- `data/sentinel2_pixel_samples.csv`

### Sample polygons

```bash
python preprocessing/01_select_training_sample.py --n 1500
```

### Export pixel samples from Earth Engine

```bash
python preprocessing/02_gee_pixel_samples.py \
  --year 2024 \
  --include-labels \
  --polygon-batch-size 100
```

Outputs are batched CSVs such as:

- `data/sentinel2_pixel_samples_part000.csv`
- `data/sentinel2_pixel_samples_part001.csv`

### Merge exported batches

```bash
python preprocessing/merge_pixel_exports.py
```

Primary output:

- `data/sentinel2_pixel_samples.csv`

### Train from pixel samples

```bash
python scripts/05_train_decision_tree.py \
  --input data/sentinel2_pixel_samples.csv \
  --model-type lightgbm \
  --mode one-vs-rest \
  --months jun,jul,aug \
  --output-dir models/step1/D_pixel_ovr_junaug
```

This took about 17 minutes on a laptop with the default three CV repeats. For a quicker
check, pass `--cv-repeats 1`. Pixel scores are reported per pixel and per
field; compare fields, not pixels, across runs.

### Compare feature families

```bash
python scripts/07_compare_feature_sets.py \
  --input data/training_table_2024_with_missing.csv \
  --model-type lightgbm \
  --baseline-families s2 \
  --augmented-families s2,s1 \
  --crops wheat,barley,canola,alfalfa,peas,corn \
  --output models/step1/feature_comparison_polygon423.csv
```

The comparison uses mean cross-validated F1 for each crop. The current
`data/sentinel2_pixel_samples.csv` contains only Sentinel-2 columns. Running
`--augmented-families s2,s1,precip` against it compares identical features,
and the script prints a warning when that happens.

### Batch pipeline (incomplete)

The province-wide batch workflow in `batch_pipeline/` has its own commands,
credentials and current gaps; see [its README](../batch_pipeline/README.md).
Its direct Copernicus Data Space extraction is not implemented, so it cannot yet
produce training features.

## 4. Known runtime requirements and constraints

- `raw_data/quarter_sections.geojson` is large, so step 0 should be run in place rather
  than copied around casually.
- Earth Engine steps are not fully local; they depend on authentication and a
  working GCP project.
- Some workflows assume outputs are downloaded manually from Google Drive into
  `data/`.
- Grouped splits in training depend on a `pid` column being present.
- CHIRPS precipitation covers only up to 50°N. It is missing for the northern
  edge of the Lethbridge box and unavailable for Peace-region fields such as
  Beaverlodge (about 55°N).

## 5. Expanding to a different county or larger study area

Change the bounding box in `scripts/common.py`:

```python
STUDY_BBOX = {
    "lon_min": ...,
    "lon_max": ...,
    "lat_min": ...,
    "lat_max": ...,
}
```

After changing it:

- delete prior outputs in `data/` and `models/` that belong to the old area
- rerun from `scripts/00_extract_study_area.py`
- keep `--year` aligned across steps 2 and 3
- consider increasing `--n` in step 1 for broader regions
