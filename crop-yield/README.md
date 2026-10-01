# Crop Yield

Historical yield benchmarks, Alberta feature extraction, and pixel modeling are
organized around workflows A–G in the [pipeline guide](docs/PIPELINE_GUIDE.md).
The restored source and supplied datasets are now included in the local layout.

## Structure

```text
crop-yield/
  src/yield_prediction/  # Importable Python implementations and module CLIs
    alberta_ats.py       # B: provincial geometry downloader
    data.py, cli.py      # A: historical dataset and CLI
    modeling.py         # A/F: historical benchmark and pixel trainers
    pixel_yield.py      # F: known-yield NDVI redistribution
    yield_index.py      # G: relative yield-potential index
  gee/                  # C: Earth Engine Code Editor JavaScript and instructions
  scripts/              # A/F: local launchers and pixel-model entry points
  batch_pipeline/       # D: separate, still incomplete batch classification route
  data/
    raw/                # Historical CSV, ATS geometry, untouched satellite exports
    processed/          # Generated metrics, predictions and index outputs
  reference_inputs/     # Supplied yield observations and prepared pixel/lookup table
  tests/
  docs/
```

The original harvest workbooks stay at repository-level `harvest-data/` because
both projects can use them. They are not yet imported into this pipeline.

## Setup

From `crop-yield/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[ml]'
```

If an environment already exists, activate it and reinstall the editable package
as needed. For extraction dependencies, also install `requirements.txt`.
`pip install -e .` alone supports the historical summary and standard-library
module commands; model execution requires ML extras. External extraction needs
its service credentials and setup.

## Run the restored historical workflow

```bash
yield-prediction summary
yield-prediction benchmark
# Or run both through the launcher:
bash scripts/run_basic_pipeline.sh
```

The default dataset resolves to `data/raw/yield_df.csv` from the package location.
An explicit `--data` override is also supported. Summary reports 28,242 records,
101 areas, 10 crops and 1990–2013 coverage.

**The restored benchmark differs from the earlier replacement implementation.**
It compares linear regression, random forest, gradient boost, KNN, decision tree,
bagging and optional XGBoost, using a random 70/30 split and shuffled five-fold
cross-validation. It retains duplicates and prints metrics; it does not write
benchmark CSVs or accept `--output-dir`. Existing `data/processed/benchmark/`
CSVs belong to the earlier chronological benchmark and were preserved as prior
results. Do not attribute those scores to the restored implementation.

## Run pixel workflows on the supplied prepared table

After installing the package, all three wrappers default to
`reference_inputs/pixel_level_all_crop_training_features_2021_2023.csv`:

```bash
python scripts/train_pixel_yield_models.py --feature-set nonleaky
python scripts/train_pixel_crop_classifier.py --feature-set nonleaky
python scripts/estimate_pixel_yield_from_ndvi.py
```

They write under `data/processed/` and accept `--input`/`--output-dir` overrides.
The supplied table has 749 rows, eight field/crop/year groups and four crop labels.
That supports local smoke runs but provides limited independent evaluation data.
NDVI redistribution requires known field yield and does not predict an unknown
field yield. See [reference input inventory](reference_inputs/README.md).

## Extraction and remaining gaps

- B: `python -m yield_prediction.alberta_ats --help` shows downloader options.
  Existing local geometry is `data/raw/alberta_quarter_sections.geojson`; do not
  rerun a download merely to reorganize it.
- C: [gee/README.md](gee/README.md) explains the restored Earth Engine exporter.
- D: all six batch scripts are present, including the scene-manifest builder.
  Direct Copernicus feature extraction remains unimplemented; copied geometry
  paths need correction. See [the batch README](batch_pipeline/README.md).
- E: the monthly export and observed-yield reference are present, but
  `scripts/build_fullfield_pixel_training_table.py` is still missing.
- F: both pixel-training implementations and their prepared input are present.
- G: the relative index module is present; it needs the seasonal GEE export
  schema, not the differently named monthly full-field pixel export.

## Guides and checks

- [Pipeline guide and current workflow status](docs/PIPELINE_GUIDE.md)
- [Detailed script behavior](docs/SCRIPT_GUIDE.md)
- [Next steps for the harvest records](docs/NEXT_STEPS.md)
- [Transfer inventory and moved paths](docs/HANDOFF_NOTES.md)
- [Harvest-data assessment](../harvest-data/README.md)

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Tests follow the restored APIs and check input paths, summary behavior, feature
exclusion, grouped pixel predictions, and a small historical benchmark. They do
not validate live Earth Engine or provincial API access.
