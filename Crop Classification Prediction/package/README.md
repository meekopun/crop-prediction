# Transfer Package

This folder is the actual handoff bundle containing the minimum files needed
to run the pipeline from preprocessing through training, validation, and
prediction.

## Included contents

- `raw_data/`
  - province-wide source input used by `scripts/00_extract_study_area.py`
- `data/`
  - output location for generated datasets and includes training-ready exports
- `scripts/`
  - full main pipeline, training, comparison, and prediction scripts
- `preprocessing/`
  - standalone pixel preprocessing workflow and tutorial
- `models/`
  - output location for generated model bundles and validation artifacts
- `docs/`
  - root project README, requirements, and handoff documentation

## Recommended usage

If the recipient wants to run the full pipeline:

1. Read `docs/RUNBOOK.md`
2. Create a Python environment
3. Install dependencies from `docs/requirements.txt`
4. Start from:
   - `data/training_table.csv` for polygon-level training
   - `data/sentinel2_pixel_samples.csv` for pixel-level training
5. Keep new outputs in `data/` and `models/` as the scripts generate them

## Expanding beyond Lethbridge County

The default study area is defined in `scripts/common.py` as `STUDY_BBOX`.

To run a different county or a larger region:

1. Edit `STUDY_BBOX` in `scripts/common.py`
2. Delete any previously generated files under `data/` that belong to the old
   study area
3. Re-run the workflow starting from `scripts/00_extract_study_area.py`

Practical implications of a larger region:

- step 0 will take longer because more quarter sections will match
- Earth Engine exports will be larger and slower
- training may need a larger sample size in step 1 to cover crop variability

## Notes

- The raw file in `raw_data/quarter_sections.geojson` is large.
- Earth Engine and AAFC-dependent steps still require external access and
  credentials where documented.
