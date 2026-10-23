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
  scripts/              # A/E/F: launchers, full-field builder, pixel-model entry points
  data/
    raw/                # Historical CSV, ATS geometry, untouched satellite exports
    processed/          # Generated metrics, predictions and index outputs
  reference_inputs/     # Supplied yield observations and prepared pixel/lookup table
  tests/
  docs/
```

The original harvest workbooks stay at repository-level `harvest-data/` because
both projects can use them. The harvest importer stages their operation records
for review; joining them to field boundaries and satellite predictors is still pending.

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

## Import harvest reports

From `crop-yield/`, with its environment activated:

```bash
python -m pip install -e '.[harvest]'
python scripts/import_harvest_data.py
# Optional overrides:
python scripts/import_harvest_data.py --input-dir ../harvest-data --output-dir data/processed/harvest
```

`src/yield_prediction/harvest.py` parses `Sheet1` in the printed `.xls` reports.
The launcher defaults to the repository's harvest folder regardless of working
directory. It preserves headings across pages, joins wrapped text/units, and
writes `operations.csv`, `processing_candidates.csv`, `exceptions.csv` and
`summary.json`. Re-running replaces these staging outputs. Saved decisions live
separately in `config/harvest_review.json` and are applied automatically by the
launcher. `--review-file PATH` selects another review file; `--no-review` skips
farm-specific decisions, for example when importing unrelated reports. Exclusion
IDs absent from an import cause a failure before writing outputs, so changed
source filenames or rows cannot silently invalidate the saved decisions.

Operations retain source rows, original measurements and explicit yield units.
Crop names are recognized when explicitly present or confirmed in the review
file. On 2026-10-06 the user confirmed `DK 902 TF 2025 (Round up Ready)` as canola.
The six name-only operations and four operations in the two mixed-crop field/year
groups are temporarily excluded by operation ID, with reasons retained in
`operations.csv`. The candidate file contains the remaining 196 observations.
All geometry is unmatched and `included_in_pilot` starts false; candidates still
need boundary review and matching satellite features before training.
The parser flags missing/invalid measurements, inconsistent dates/totals and
multiple operations under one field/year. It does not convert units or aggregate
operations. `qa_flags` is a pipe-separated list in the CSV. Workbook diagnostics
appear in the exceptions file with `source_row=0`; operation rows are one-based.
Operation IDs are stable for unchanged source filenames, sheet names and row
positions. A legal description is a candidate identity, not a verified boundary.

Match the candidates to ATS quarter sections after importing:

```bash
python scripts/match_harvest_boundaries.py
```

It streams `data/raw/alberta_quarter_sections.geojson` and writes
`field_boundaries.geojson` (one polygon per matched quarter, with `pid`),
`boundary_matches.csv` (one row per candidate with match status, acreage share,
quarter-year sharing and flags) and `boundary_match_summary.json` to
`data/processed/harvest/`. Operations sharing a quarter-year with another crop,
or reporting more acres than their quarter, are not pilot-eligible. A quarter
section is a candidate boundary, not a surveyed farm field.

Then export monthly Sentinel-2/Sentinel-1 features for the eligible quarters
(needs `earthengine authenticate` once):

```bash
python scripts/export_harvest_features.py            # 2020–2025 by default
python scripts/export_harvest_features.py --years 2024 --overwrite
```

It writes one CSV per year plus `harvest_satellite_features.csv` and a manifest to
`data/processed/harvest/satellite/`, with whole-quarter (`quarter_*`) and
crop-inventory-masked (`crop_*`) columns. Finished years are kept on rerun.

Build the field-year table and benchmark yield models against a per-crop mean:

```bash
python scripts/build_field_yield_table.py
python scripts/benchmark_field_yield.py
```

The table (`data/processed/harvest/field_yield_table.csv`) has one row per
harvested field-year with its aggregated yield and `field_*` satellite features.
The benchmark writes metrics, predictions and `summary.md` to
`data/processed/harvest/benchmark/`, holding out whole seasons (`year`) and whole
sections (`section`).

For a final, non-optimistic check, choose one model per crop on 2020–2024 and
score it once on 2025:

```bash
python scripts/holdout_field_yield.py --test-year 2025
```

On 2026-10-22 only wheat beat the baseline on 2025; canola, barley and fescue did
worse, so season-to-season yield is not yet predictable from satellite features
alone. See `docs/NEXT_STEPS.md` section 5.

To split yield into a season level and each field's deviation (development years
2020–2024 only), export weather, rebuild the table and evaluate:

```bash
python scripts/export_harvest_weather.py
python scripts/build_field_yield_table.py
python scripts/evaluate_season_field_model.py --season-features satellite --deviation-features peak
```

`--season-features weather` uses ERA5-Land instead; with five seasons it does
worse than the training mean.

For a season level from published Peace-region yields, place Statistics Canada's
`32100002-eng.zip` (table 32-10-0002-01) in `data/raw/statcan/`, then:

```bash
python -I scripts/prepare_regional_yields.py
python scripts/evaluate_season_field_model.py --season-features regional --deviation-features peak
```

On 2020–2024 this beat the satellite season level only for wheat: the farm
departs most from the Peace-region average in the extreme years (2021, 2023).

When choosing models for a final holdout, add `--min-improvement 0.1` so a model
must beat the baseline's development error by 10% to replace it.

The local six-workbook check reconciles 206 operations: 172 `bu/ac`, 34 `lb/ac`.
Exit codes: 0 for a completed import with no errors (review warnings may remain),
1 for a file/dependency/layout failure, 2 for completed output containing errors.
Workbook/layout failures abort before output files are written.

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
`train_pixel_yield_models.py` also accepts `--baseline`, `--drop-year-constant`
and `--split year` for a stricter evaluation against a training-mean baseline.
The supplied table has 749 rows, eight field/crop/year groups and four crop labels.
That supports local smoke runs but provides limited independent evaluation data.
NDVI redistribution requires known field yield and does not predict an unknown
field yield. See [reference input inventory](reference_inputs/README.md).

## Extraction and remaining gaps

- B: `python -m yield_prediction.alberta_ats --help` shows downloader options.
  Existing local geometry is `data/raw/alberta_quarter_sections.geojson`; do not
  rerun a download merely to reorganize it.
- C: [gee/README.md](gee/README.md) explains the restored Earth Engine exporter.
- D: the batch crop-classification route moved to
  [crop-classification/batch_pipeline/](../crop-classification/batch_pipeline/README.md)
  on 2026-10-22.
- E: `python scripts/build_fullfield_pixel_training_table.py` rebuilds a pixel
  table from the monthly export; it still yields only eight field-years.
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
