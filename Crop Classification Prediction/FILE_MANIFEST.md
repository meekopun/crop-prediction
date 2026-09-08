# File Manifest

This manifest describes what to include depending on the handoff goal.

## A. Full pipeline handoff

Minimum files:

- `requirements.txt`
- entire `scripts/` directory
- entire `preprocessing/` directory if pixel preprocessing is in scope
- `quarter_sections.geojson`
- `handoff/` documentation folder

External requirements:

- Google Earth Engine account and authentication
- Google Cloud project ID enabled for Earth Engine
- Network access to AAFC endpoints

Generated at runtime:

- most files under `data/`
- `models/`

Included prepared training data:

- `data/training_table.csv`
- `data/sentinel2_pixel_samples.csv`

## B. Files that are mainly documentation or convenience

- `README.md`
- `preprocessing/TUTORIAL.md`
- `handoff/README.md`
- `handoff/RUNBOOK.md`
- `handoff/FILE_MANIFEST.md`

## C. Large or environment-specific items

- `quarter_sections.geojson` is very large and should only be transferred when
  source-level regeneration is required.
- `.venv/` should not be handed off; recreate the environment locally.
- `catboost_info/` is run output, not required for a clean handoff.
