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

### Step 5: train models

Examples:

```bash
python scripts/05_train_decision_tree.py \
  --model-type decision-tree \
  --mode multiclass \
  --max-depth 6 \
  --split-strategy grouped

python scripts/05_train_decision_tree.py \
  --model-type logistic-regression \
  --mode one-vs-rest \
  --split-strategy grouped

python scripts/05_train_decision_tree.py \
  --model-type catboost \
  --mode one-vs-rest \
  --split-strategy grouped

python scripts/05_train_decision_tree.py \
  --model-type lightgbm \
  --mode one-vs-rest \
  --split-strategy grouped
```

Typical outputs:

- `models/decision_tree/...`
- `models/logistic_regression_one_vs_rest/...`
- `models/catboost_one_vs_rest/...`
- `models/lightgbm_one_vs_rest/...`

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
  --split-strategy grouped \
  --months jun,jul,aug
```

### Compare feature families

```bash
python scripts/07_compare_feature_sets.py \
  --input data/sentinel2_pixel_samples.csv \
  --model-type lightgbm \
  --split-strategy grouped \
  --baseline-families s2 \
  --augmented-families s2,s1,precip
```

## 4. Known runtime requirements and constraints

- `raw_data/quarter_sections.geojson` is large, so step 0 should be run in place rather
  than copied around casually.
- Earth Engine steps are not fully local; they depend on authentication and a
  working GCP project.
- Some workflows assume outputs are downloaded manually from Google Drive into
  `data/`.
- Grouped splits in training depend on a `pid` column being present.

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
