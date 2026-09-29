# File Manifest

This manifest describes what to include depending on the handoff goal.

## A. Full pipeline handoff

Minimum files:

- `requirements.txt`
- entire `scripts/` directory
- entire `preprocessing/` directory if pixel preprocessing is in scope
- `raw_data/quarter_sections.geojson`
- `docs/` documentation folder

External requirements:

- Google Earth Engine account and authentication
- Google Cloud project ID enabled for Earth Engine
- Network access to AAFC endpoints

Generated at runtime:

- most files under `data/`
- `models/`

Prepared training data:

- `data/training_table.csv` is included in Git.
- `data/sentinel2_pixel_samples.csv` is a local export; supply it separately if needed.

## B. Files that are mainly documentation or convenience

- `README.md`
- `preprocessing/TUTORIAL.md`
- `docs/RUNBOOK.md`
- `docs/PIPELINE_GUIDE.md`
- `docs/SCRIPT_GUIDE.md`
- `docs/NEXT_STEPS.md`
- `docs/FILE_MANIFEST.md`

## C. Large or environment-specific items

- `raw_data/quarter_sections.geojson` is very large and should only be transferred when
  source-level regeneration is required.
- `.venv/` should not be handed off; recreate the environment locally.
- `catboost_info/` is run output, not required for a clean handoff.
