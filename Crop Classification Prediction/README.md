# Crop Classification Handoff

This folder is the project handoff entry point. It consolidates the minimum
context a new collaborator needs to understand the repository, install
dependencies, and run the main workflows without digging through multiple
folders first.

## What this project does

The repository predicts crop type for Alberta quarter sections from
Sentinel-derived features. The main target classes are:

- alfalfa
- barley
- canola
- corn
- oats
- peas
- wheat

The current study area is Lethbridge County, Alberta. That region is defined
in `scripts/common.py`.

## Main workflows

There are two practical workflows in this repo:

1. Full polygon pipeline
   Uses sampled quarter sections, polygon-aggregated remote sensing features,
   AAFC labels, then trains and applies models.
2. Pixel-level preprocessing and training
   Exports one row per pixel with Sentinel-2, Sentinel-1, precipitation, and
   optional AAFC labels for higher-resolution training.

Use [RUNBOOK.md](./RUNBOOK.md) for the exact commands.

## Environment requirements

- Python 3.10+ recommended
- Internet access for:
  - Google Earth Engine authentication and exports
  - AAFC label download endpoint
- A Google Earth Engine enabled Google Cloud project for steps that call GEE

Install dependencies from either the repo root or this folder:

```bash
pip install -r requirements.txt
```

The requirements in this folder match the project root requirements.

## External setup

### Google Earth Engine

Required for:

- `scripts/02_gee_sentinel_features.py`
- `scripts/02_gee_pixel_samples.py`
- `preprocessing/02_gee_pixel_samples.py`

One-time setup:

```bash
earthengine authenticate
earthengine set_project cropclassification-502620
```

The export scripts use the saved project by default. To override it for one
run, pass `--project ANOTHER_PROJECT_ID`.

### AAFC label download

Required for:

- `scripts/03_download_aafc_labels.py`

This uses a public AAFC endpoint and does not require an account, but it does
require normal outbound network access.

## Where the important files live

- Root overview: [../README.md](../README.md)
- Root dependencies: [../requirements.txt](../requirements.txt)
- Full pipeline scripts: [../scripts](../scripts)
- Pixel preprocessing workflow: [../preprocessing](../preprocessing)
- Generated training and feature tables: [../data](../data)
- Trained model outputs: [../models](../models)

Use [FILE_MANIFEST.md](./FILE_MANIFEST.md) to see what is required for each
handoff scenario.

## What the handoff package includes

The transfer bundle under `handoff/package/` is now intentionally minimal:

- pipeline and preprocessing scripts
- raw quarter-section source data
- exported training-ready datasets
- Python requirements
- run instructions and file manifest

It does not include previously generated model bundles, confusion matrices,
feature importance exports, or other validation outputs. Those are produced
when the pipeline is run.

## Expanding beyond Lethbridge County

The default study area is the Lethbridge County bounding box configured in
`scripts/common.py` as `STUDY_BBOX`.

To use a different county or a larger region:

1. Update `STUDY_BBOX` in `scripts/common.py`
2. Clear prior outputs in `data/` and `models/` for the old region
3. Re-run the pipeline starting at step 0

If the new area is substantially larger, expect longer extraction and Earth
Engine export times, plus a need to increase the training sample size in step
1.

## First files to read

1. [RUNBOOK.md](./RUNBOOK.md)
2. [FILE_MANIFEST.md](./FILE_MANIFEST.md)
3. [../README.md](../README.md)
