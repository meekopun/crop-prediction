# Handoff Notes

## Scope

This handoff covers two layers:

1. The basic packaged CLI workflow:
   - load and clean `archive/yield_df.csv`
   - summarize the dataset with the CLI
   - optionally run the benchmark command when ML extras are installed
2. The upstream Alberta extraction and preprocessing workflows:
   - Alberta quarter-section download
   - Earth Engine and Copernicus Data Space feature extraction
   - pixel-level preprocessing into training tables
   - downstream model training scripts

The key monthly export and GeoJSON inputs are included in this handoff. Other very large derivative exports are still documented in `docs/PIPELINE_GUIDE.md` when not copied.

## Expected commands

Dataset summary:

```bash
yield-prediction --data archive/yield_df.csv summary
```

Benchmark:

```bash
yield-prediction --data archive/yield_df.csv benchmark
```

Direct module execution without package install:

```bash
PYTHONPATH=src python3 -m yield_prediction.cli --data archive/yield_df.csv summary
```

## Expected summary output

The current dataset should report:

- `rows: 28242`
- `areas: 101`
- `items: 10`
- `year_min: 1990`
- `year_max: 2013`

## File inventory

- `pyproject.toml`: package metadata and dependencies
- `archive/yield_df.csv`: dataset used by the CLI and tests
- `quarter_sections.geojson`: main quarter-section geometry source used by the Alberta workflows
- `data/raw/alberta_quarter_sections.geojson`: raw ATS GeoJSON download
- `data/pixel_crop_features_aci_s2_monthly_csv_quarter_sections_fullfield.csv`: monthly full-field pixel export used by the preprocessing script
- `src/yield_prediction/data.py`: loading, cleaning, summarization
- `src/yield_prediction/modeling.py`: benchmark implementation and feature-set helpers
- `src/yield_prediction/cli.py`: command-line entrypoint
- `tests/test_data.py`: data workflow checks
- `tests/test_cli.py`: CLI smoke tests
- `tests/test_modeling.py`: feature-set regression check
- `gee/README.md`: Earth Engine export instructions
- `gee/sentinel2_quarter_sections.js`: quarter-section Sentinel-2 export script
- `upstream/crop_classification_batch_pipeline/`: batch extraction and feature-engineering workflow files
- `upstream/scripts/build_fullfield_pixel_training_table.py`: standardizes full-field pixel exports into a training table
- `upstream/scripts/train_pixel_yield_models.py`: trains pixel yield baselines
- `upstream/scripts/train_pixel_crop_classifier.py`: trains crop classifiers
- `upstream/scripts/estimate_pixel_yield_from_ndvi.py`: NDVI-based redistribution baseline
- `reference_inputs/Crop Yield Data - Crop Data 2021-2023-2.csv`: observed yield reference used during preprocessing
- `reference_inputs/pixel_level_all_crop_training_features_2021_2023.csv`: lookup table used by preprocessing when available

## Not copied into the handoff

These inputs exist in the parent project and were not duplicated:

- `data/raw/alberta_quarter_sections.zip` at about `241M`
- `pixel_crop_features_aci_s2_daily.csv` at about `948M`
- `pixel_crop_features_aci_s2_weekly-2.csv` at about `234M`

The handoff includes the code and exact input names for those files so they can be regenerated or supplied externally.

## Notes

- `benchmark` requires optional ML dependencies from `pip install -e '.[ml]'`.
- The package can be run either after installation or via `PYTHONPATH=src`.
- Upstream extraction and preprocessing require the geospatial dependencies listed in `requirements.txt`.
- External services used by the upstream workflows include Google Earth Engine and Copernicus Data Space.
