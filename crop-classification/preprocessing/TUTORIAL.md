# Pixel Preprocessing Tutorial

This folder contains the pixel-level data-prep workflow. It shares dependencies
and data with the rest of `crop-classification/`.

## Included Steps

- `01_select_training_sample.py`
- `02_gee_pixel_samples.py`
- `merge_pixel_exports.py`
- `concat_training_csvs.py`
- `progress_tracker.py`

## Prerequisites

From `crop-classification/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Google Earth Engine:

1. Register a Google Cloud project for Earth Engine.
2. Run `earthengine authenticate`.
3. Save the default with `earthengine set_project cropclassification-502620`.
   The export script uses this saved default; `--project` can override it.

## Assumption

This folder assumes the study-area quarter sections already exist in:

```bash
data/study_area_quarter_sections.geojson
```

If that file needs to be regenerated, do that before handing this folder to a
collaborator.



## Workflow

### 1. Sample quarter sections for training

From `crop-classification/`:

```bash
python preprocessing/01_select_training_sample.py --n 1500
```

From inside `preprocessing/`:

```bash
python 01_select_training_sample.py --n 1500
```

This writes:

```bash
data/training_sample.geojson
```

If you need more sections later without reusing the current sample:

```bash
python preprocessing/01_select_training_sample.py \
  --n 1000 \
  --exclude data/training_sample.geojson \
  --output data/training_sample_extra.geojson
```

### 2. Export pixel-level Sentinel-2, Sentinel-1 C-band, precipitation features, and AAFC labels

```bash
python preprocessing/02_gee_pixel_samples.py \
  --year 2024 \
  --include-labels \
  --polygon-batch-size 100
```

This creates batched Earth Engine Drive exports such as:

- `sentinel2_pixel_samples_part000.csv`
- `sentinel2_pixel_samples_part001.csv`

Each exported row now includes monthly Sentinel-1 `VV`, `VH`, and
`VV_minus_VH` features plus monthly `PRECIP` totals in addition to the
existing Sentinel-2 bands and indices.

Batch progress is tracked in:

```bash
preprocessing/state/sentinel2_pixel_samples_checkpoint.json
```

Rerun the same command to resume; completed batches will be skipped.

### Progress Tracking

Sampling and export progress are also summarized in:

```bash
preprocessing/state/preprocessing_progress.json
```

To inspect a readable summary:

```bash
python preprocessing/progress_tracker.py
```

### 3. Download the batch CSVs into `data/`

Example filenames:

- `data/sentinel2_pixel_samples_part000.csv`
- `data/sentinel2_pixel_samples_part001.csv`

### 4. Merge the batch CSVs

```bash
python preprocessing/merge_pixel_exports.py
```

This writes:

```bash
data/sentinel2_pixel_samples.csv
```

That file is the main pixel-level output to hand off for training.

## Expanding the Dataset Later

If more data is needed:

1. Sample new quarter sections with `--exclude`.
2. Export a second set of pixel batches with a distinct `--out-name`.
3. Merge those batches.
4. Concatenate the old and new training CSVs.

Example:

```bash
python preprocessing/01_select_training_sample.py \
  --n 1000 \
  --exclude data/training_sample.geojson \
  --output data/training_sample_extra.geojson

python preprocessing/02_gee_pixel_samples.py \
  --input data/training_sample_extra.geojson \
  --year 2024 \
  --include-labels \
  --polygon-batch-size 100 \
  --out-name sentinel2_pixel_samples_extra

python preprocessing/merge_pixel_exports.py \
  --pattern "data/sentinel2_pixel_samples_extra_part*.csv" \
  --output data/sentinel2_pixel_samples_extra.csv

python preprocessing/concat_training_csvs.py \
  --pattern "data/sentinel2_pixel_samples*.csv" \
  --output data/sentinel2_pixel_samples_all.csv \
  --dedupe-cols "system:index"
```

## Expected Hand-Off Files

Give the training collaborator one of:

- `data/sentinel2_pixel_samples.csv`
- `data/sentinel2_pixel_samples_all.csv`

## Notes

- The scripts resolve the shared `data/` directory automatically, so they work
  whether you run them from `crop-classification/` or from inside `preprocessing/`.
- Outputs still go to the shared `data/` directory one level above this folder.
- Tracking and checkpoint files stay inside `preprocessing/state/`.
- For large Earth Engine jobs, keep `--polygon-batch-size` modest, e.g. `100`.
