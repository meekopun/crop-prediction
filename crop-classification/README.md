# Crop Classification

This project predicts crop type from satellite features. Run commands from
`crop-classification/`; scripts, dependencies, data, and models live directly here.

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

Use [the pipeline map](docs/PIPELINE_GUIDE.md) for stage/folder roles and
[the runbook](docs/RUNBOOK.md) for the exact commands.

For development planning, read [next steps](docs/NEXT_STEPS.md) and the
[detailed script guide](docs/SCRIPT_GUIDE.md). The [harvest-data assessment](../harvest-data/README.md)
explains how the supplied farm crop records can provide additional reference labels.

## Environment requirements

- Python 3.10+ recommended
- Internet access for:
  - Google Earth Engine authentication and exports
  - AAFC label download endpoint
- A Google Earth Engine enabled Google Cloud project for steps that call GEE

Install dependencies from this project directory:

```bash
pip install -r requirements.txt
```

Use this one dependency file for both the polygon and pixel workflows.

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

## Directory guide

- [scripts/](scripts/): numbered polygon pipeline, training, prediction, and feature comparison
- [preprocessing/](preprocessing/): pixel sampling/export with resumable progress tracking
- [raw_data/](raw_data/): local province-wide quarter-section source
- [data/](data/): prepared tables and generated features
- [models/](models/): generated model bundles and evaluation outputs
- [docs/RUNBOOK.md](docs/RUNBOOK.md): commands for both workflows
- [docs/FILE_MANIFEST.md](docs/FILE_MANIFEST.md): files needed to transfer the project
- [requirements.txt](requirements.txt): shared dependencies for this project

The similarly named scripts in `scripts/` and `preprocessing/` are distinct
implementations. The pixel workflow adds progress tracking and resolves default
paths relative to this project. Keep the directories together; use the
[preprocessing tutorial](preprocessing/TUTORIAL.md) for that workflow.

`data/training_table.csv` is included in Git. Large raw inputs, pixel exports,
and trained models stay local and must be generated or supplied separately.

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
