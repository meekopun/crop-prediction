# Crop Prediction

Two independent Python projects for crop type classification and crop yield
modeling. Each project has its own environment, dependencies, and run instructions.

| Project | Start here | Status |
| --- | --- | --- |
| Crop classification | [crop-classification/README.md](crop-classification/README.md) | Polygon and pixel workflows; satellite exports require external services |
| Crop yield | [crop-yield/README.md](crop-yield/README.md) | Restored historical/pixel models and extraction source; batch/preprocessing gaps remain |

```text
crop-classification/
  README.md
  requirements.txt
  scripts/          # Numbered pipeline, training, and prediction
  preprocessing/    # Pixel preparation with progress tracking
  docs/             # Runbook and transfer manifest
  raw_data/         # Large local source geometry
  data/             # Prepared tables and generated features
  models/           # Local trained models and evaluation outputs
crop-yield/
  README.md
  pyproject.toml
  requirements.txt
  src/yield_prediction/  # Historical/pixel models, ATS downloader, relative index
  gee/             # Seasonal Earth Engine JavaScript exporter
  reference_inputs/ # Supplied yield labels and prepared pixel/lookup CSV
  scripts/          # Benchmark launcher and pixel modeling entry points
  tests/
  docs/             # Current workflow map, script guide, next steps, transfer inventory
  batch_pipeline/   # Incomplete Alberta extraction and training workflow
  data/
    raw/            # Historical CSV, supplied ATS geometry and raw pixel export
    processed/      # Generated model/index outputs and previous benchmark results
```

Run commands from the corresponding project directory. See each project's README
for environment setup. Large inputs, generated outputs, and `.venv/` stay local.

See the [classification pipeline map](crop-classification/docs/PIPELINE_GUIDE.md)
and [yield pipeline guide](crop-yield/docs/PIPELINE_GUIDE.md) for stage order and
current availability. The [yield transfer inventory](crop-yield/docs/HANDOFF_NOTES.md)
records where the newly supplied files were moved.

## Harvest data and development guides

The [harvest-data assessment](harvest-data/README.md) explains how the 2020–2025
farm reports support yield targets and crop reference labels, including units,
field matching, and preparation needed before training.

| Project | Next steps | Detailed script process |
| --- | --- | --- |
| Crop classification | [Plan](crop-classification/docs/NEXT_STEPS.md) | [Script guide](crop-classification/docs/SCRIPT_GUIDE.md) |
| Crop yield | [Plan](crop-yield/docs/NEXT_STEPS.md) | [Script guide](crop-yield/docs/SCRIPT_GUIDE.md) |

## Previous locations

- `Crop Classification Prediction/package/` → `crop-classification/`
- `Crop Yield Prediction/basic_pipeline/` → `crop-yield/`
- Classification's duplicate handoff docs → one README and `docs/`
- Yield's `upstream/scripts/` → `scripts/`
- Yield's `upstream/crop_classification_batch_pipeline/` → `batch_pipeline/`
- Yield's `archive/yield_df.csv` → `data/raw/yield_df.csv`

Existing terminals should change into the new project directory and reactivate
its environment. For a fresh checkout, create an environment using the project
README. Update any external shortcuts or jobs that reference the old paths.
