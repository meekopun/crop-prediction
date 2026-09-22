# Crop Prediction

Two independent Python projects for crop type classification and crop yield
modeling. Each project has its own environment, dependencies, and run instructions.

| Project | Start here | Status |
| --- | --- | --- |
| Crop classification | [crop-classification/README.md](crop-classification/README.md) | Polygon and pixel workflows; satellite exports require external services |
| Crop yield | [crop-yield/README.md](crop-yield/README.md) | Local historical benchmark runs; Alberta satellite workflows are incomplete |

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
  src/yield_prediction/
  scripts/          # Benchmark launcher and pixel modeling entry points
  tests/
  docs/             # Historical Alberta workflow specifications
  batch_pipeline/   # Incomplete Alberta extraction and training workflow
  data/
    raw/            # Bundled historical yield CSV
    processed/      # Generated benchmark results
```

Run commands from the corresponding project directory. See each project's README
for environment setup. Large inputs, generated outputs, and `.venv/` stay local.

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
