# Pipeline Guide

> Current checkout: the basic historical CLI (Workflow A) has been reimplemented
> and is runnable; see [the runnable guide](../README.md). Workflows B–F below
> are legacy handoff specifications and reference missing files and input data.
> Their file-inclusion claims do not describe this checkout.

## Overview

This project now has multiple workflow layers. The handoff includes the code and instructions for each layer.

## Workflow A: Basic package CLI

Purpose:
- run a small reproducible summary/benchmark flow against `data/raw/yield_df.csv`

Files:
- `src/yield_prediction/cli.py`
- `src/yield_prediction/data.py`
- `src/yield_prediction/modeling.py`
- `data/raw/yield_df.csv`

Commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
yield-prediction --data data/raw/yield_df.csv summary
```

Optional benchmark:

```bash
pip install -e '.[ml]'
yield-prediction --data data/raw/yield_df.csv benchmark
```

## Workflow B: Alberta quarter-section extraction

Purpose:
- download Alberta ATS quarter sections from the provincial ArcGIS layer

Files:
- `src/yield_prediction/alberta_ats.py`

Command:

```bash
PYTHONPATH=src python3 -m yield_prediction.alberta_ats --geojson-out data/raw/alberta_quarter_sections.geojson
```

Optional spatial filter:

```bash
PYTHONPATH=src python3 -m yield_prediction.alberta_ats --where "M=4 AND RGE=25 AND TWP=50"
```

Outputs:
- GeoJSON always
- GeoPackage also, if `geopandas` is installed

Notes:
- this step requires network access to the Alberta ArcGIS REST endpoint
- this handoff already includes `quarter_sections.geojson` and `data/raw/alberta_quarter_sections.geojson`

## Workflow C: Earth Engine seasonal quarter-section export

Purpose:
- export seasonal quarter-section Sentinel-2 summary features as a flat CSV

Files:
- `gee/sentinel2_quarter_sections.js`
- `gee/README.md`

Inputs:
- quarter-section polygons uploaded to GEE as a table asset
- a stable quarter-section ID field

Key outputs:
- one row per `quarter_id + year`
- ATS metadata
- AAFC crop fraction fields
- seasonal Sentinel-2 band and index summaries

Execution outline:

1. Upload quarter-section GeoJSON to GEE.
2. Open `gee/sentinel2_quarter_sections.js` in the Earth Engine Code Editor.
3. Set `QUARTER_SECTIONS_ASSET`, `QUARTER_ID_FIELD`, `START_YEAR`, and `END_YEAR`.
4. Run a limited pilot export first.
5. Download the CSV from Google Drive.

## Workflow D: Crop classification batch pipeline

Purpose:
- run batch-oriented crop-label and Sentinel-2 feature extraction for quarter sections

Files:
- `batch_pipeline/crop_classification_pipeline_config.json`
- `batch_pipeline/prepare_crop_classification_batches.py`
- `batch_pipeline/build_copernicus_scene_manifest.py`
- `batch_pipeline/run_crop_classification_batch.py`
- `batch_pipeline/train_one_vs_rest_crop_models.py`
- `batch_pipeline/quarter_section_batch_utils.py`
- `batch_pipeline/copernicus_data_space_utils.py`
- `batch_pipeline/README.md`

Task summary:
- split the quarter-section geometry into deterministic processing batches
- build weekly Sentinel-2 scene manifests for each batch
- assign AAFC crop labels from either a local raster or GEE
- compute weekly spectral and vegetation features
- train one-vs-rest crop classifiers from the batch feature outputs

External prerequisites:
- `CDSE_USERNAME`
- `CDSE_PASSWORD`
- Google Earth Engine access if using `gee_aafc_only`
- local AAFC raster if using `local_aafc_raster`

Execution order:

1. Build deterministic batch and tracker CSVs:

```bash
python3 batch_pipeline/prepare_crop_classification_batches.py
```

2. Build a Copernicus Data Space weekly scene manifest for one batch:

```bash
python3 batch_pipeline/build_copernicus_scene_manifest.py --batch-id batch_01
```

3. Run one end-to-end batch:

```bash
python3 batch_pipeline/run_crop_classification_batch.py --batch-id batch_01 --label-source gee_aafc_only
```

4. Train binary crop models from completed batch feature files:

```bash
python3 batch_pipeline/train_one_vs_rest_crop_models.py
```

Preprocessing performed by this flow:
- filters quarter sections, optionally excluding road allowances
- partitions features into deterministic batches
- discovers weekly Sentinel-2 scenes by batch/year/week
- assigns crop labels either from a local AAFC raster or GEE
- computes weekly spectral and derived vegetation features
- writes batch feature tables for downstream crop modeling

## Workflow E: Full-field pixel preprocessing

Purpose:
- standardize a full-field monthly pixel export into a training table

Files:
- `scripts/build_fullfield_pixel_training_table.py`
- `data/pixel_crop_features_aci_s2_monthly_csv_quarter_sections_fullfield.csv`
- `reference_inputs/Crop Yield Data - Crop Data 2021-2023-2.csv`
- `reference_inputs/pixel_level_all_crop_training_features_2021_2023.csv`

Command:

```bash
python3 scripts/build_fullfield_pixel_training_table.py \
  --pixel-input data/pixel_crop_features_aci_s2_monthly_csv_quarter_sections_fullfield.csv \
  --yield-input "reference_inputs/Crop Yield Data - Crop Data 2021-2023-2.csv" \
  --lookup-input reference_inputs/pixel_level_all_crop_training_features_2021_2023.csv
```

Preprocessing performed by this script:
- normalizes quarter-section identifiers
- standardizes crop naming
- coerces year and yield types
- replaces Sentinel missing-value sentinels such as `-9999` with `NaN`
- derives NDVI summary features from monthly columns
- renames location/crop columns into the modeling schema
- merges pixel features with observed yield references
- builds `row_id = quarter_section + crop + year`

Output:
- standardized training CSV for downstream yield modeling

## Workflow F: Downstream Alberta modeling scripts

Files:
- `scripts/train_pixel_yield_models.py`
- `scripts/train_pixel_crop_classifier.py`
- `scripts/estimate_pixel_yield_from_ndvi.py`

These scripts operate on already-prepared feature tables and are downstream of the extraction/preprocessing steps above.

## Dependencies

`requirements.txt` covers every workflow in this bundle: extraction, preprocessing, and modeling. The packaged CLI dependencies in `pyproject.toml` are not sufficient for the geospatial and batch workflows.

The geospatial entries (`geopandas`, `rasterio`, `earthengine-api`) are needed only by the extraction and batch scripts, which import them lazily. Modeling work runs without them installed.
