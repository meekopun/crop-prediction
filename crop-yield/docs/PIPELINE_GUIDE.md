# Crop yield pipeline guide

Updated 2026-10-22: workflow D moved to `crop-classification/batch_pipeline/`, and
E's builder was recovered. This is the current workflow map. Run local commands from `crop-yield/` with its package
installed, or use `PYTHONPATH=src` for Python module/script calls.

## Workflow map

| Workflow | Implementation | Current input/status |
| --- | --- | --- |
| A: historical benchmark | `src/yield_prediction/{data,modeling,cli}.py` | Bundled CSV; local workflow available |
| B: ATS geometry | `src/yield_prediction/alberta_ats.py` | Restored downloader; geometry already supplied locally; live API not verified |
| C: seasonal satellite export | `gee/sentinel2_quarter_sections.js` | Restored; requires GEE asset/project configuration and execution |
| D: batch crop classification | Moved to `../crop-classification/batch_pipeline/` | Crop classification, not yield; documented in that project |
| E: full-field pixel preprocessing | `scripts/build_fullfield_pixel_training_table.py` | Recovered 2026-10-22; runs on supplied inputs, still eight field-years |
| F: pixel models/redistribution | `scripts/` plus package modeling modules | Implementations and prepared 2021–2023 table present |
| G: yield-potential index | `src/yield_prediction/yield_index.py` | Relative index calculation present; requires C-format export |

A is an independent historical example. B supplies geometry for extraction.
C is the seasonal feature route; D, the former batch route, now belongs to the
classification project. E prepares the
monthly full-field export for F; the supplied prepared reference table lets F
run without regenerating E. G consumes C's seasonal polygon table. The separate
`scripts/import_harvest_data.py` stages the 2020–2025 harvest observations;
their integration with these modeling paths still needs field/satellite matching.

## A. Historical dataset and benchmark

Input: `data/raw/yield_df.csv`.

```bash
yield-prediction summary
yield-prediction benchmark
```

`data.py` parses records; `cli.py` prints summary or model results;
`modeling.benchmark_models()` compares six regressors plus optional XGBoost.
The restored implementation uses a random 70/30 holdout and shuffled five-fold
CV, retains duplicate records and prints results. It does not save fitted models
or CSV results. Previous chronological benchmark output under
`data/processed/benchmark/` is preserved historical output, not a result of this
restored command. See the [script guide](SCRIPT_GUIDE.md) for metrics and limits.

## B. Alberta quarter-section geometry

Implementation: `src/yield_prediction/alberta_ats.py`.
Existing supplied source: `data/raw/alberta_quarter_sections.geojson`.

Inspect options without network activity:

```bash
python -m yield_prediction.alberta_ats --help
```

For a future filtered download, use distinct outputs to preserve the supplied source:

```bash
python -m yield_prediction.alberta_ats \
  --where "M=4 AND RGE=25 AND TWP=50" \
  --geojson-out data/raw/ats_pilot.geojson \
  --gpkg-out data/raw/ats_pilot.gpkg
```

The downloader paginates the configured provincial ArcGIS layer and combines
features in memory. It writes GeoJSON and optionally GeoPackage with geopandas.
Verify service behavior and the returned CRS before relying on the GeoPackage's
hard-coded CRS. Full-province downloads can be large; no download is necessary
for the existing supplied geometry.

## C. Earth Engine seasonal quarter-section export

Implementation and setup: [gee/README.md](../gee/README.md).

1. Prepare/upload the intended polygons as a GEE table asset.
2. Open `gee/sentinel2_quarter_sections.js` in the Earth Engine Code Editor.
3. Set asset path, actual identifier property name, years, coverage settings and
   a small feature limit for a pilot.
4. Run and inspect the task; download the CSV into `data/raw/`.

Output has one row per `quarter_id + year`, seasonal optical statistics, AAFC
crop fractions, soil-moisture, weather and static context features. It is a
polygon export, not the supplied monthly pixel table. The broader inputs in
this restored exporter are separate from the classification batch workflow's
optical-only feature policy.

## D. Batch crop classification (moved)

The batch workflow moved to `crop-classification/batch_pipeline/` on 2026-10-22
because it predicts crop identity, not yield. See its
[README](../../crop-classification/batch_pipeline/README.md) for commands and gaps.

## E. Full-field monthly pixel preprocessing

Inputs, all present:

- `data/raw/pixel_crop_features_aci_s2_monthly_csv_quarter_sections_fullfield.csv`
- `reference_inputs/pixel_level_all_crop_training_features_2021_2023.csv` (yield lookup)
- `reference_inputs/Crop Yield Data - Crop Data 2021-2023-2.csv` (used only without the lookup)

```bash
python scripts/build_fullfield_pixel_training_table.py
```

This writes `data/processed/pixel_level_fullfield_training_features_2021_2023.csv`:
87,176 pixel rows but still eight field-years. Yields are joined by quarter section
and year only, so a quarter reporting two crops in one year attaches the same
pixels to both. The monthly export's `ndvi_m04` style columns are mostly not
recognized by F's feature sets; see the
[script guide](SCRIPT_GUIDE.md#e-full-field-pixel-preprocessing). Keep the
supplied reference files unchanged.

## F. Prepared-table modeling

The default input for all wrappers is the supplied table in `reference_inputs/`.
After installing ML extras:

```bash
python scripts/train_pixel_yield_models.py --feature-set nonleaky
python scripts/train_pixel_crop_classifier.py --feature-set nonleaky
python scripts/estimate_pixel_yield_from_ndvi.py
```

Outputs default to separate `data/processed/` subdirectories. Use `--output-dir`
for a separate run. Yield training uses per-crop OLS/ridge and grouped validation;
crop classification uses logistic regression/random forest and grouped validation.
Some groups/classes cannot support held-out evaluation. In-sample `fitted_*`
predictions are not held-out results. Redistribution uses known field yield.

For a stricter yield evaluation, add `--baseline --drop-year-constant` and
optionally `--split year`. On the supplied table no model beats the
training-mean baseline once the year-constant soil-moisture feature is removed;
see the [script guide](SCRIPT_GUIDE.md#srcyield_predictionmodelingpy-pixel-yield-training).

The table contains 749 pixel rows, eight field/crop/year groups and four crops.
It is not an automatically imported version of the six harvest Excel reports.

## G. Relative yield-potential index

Implementation: `src/yield_prediction/yield_index.py`.
After obtaining a seasonal polygon CSV from C, replace the example filename below
with its actual downloaded path:

```bash
python -m yield_prediction.yield_index data/raw/alberta_quarter_sections_s2_features.csv
```

The module infers a crop from AAFC fractions/classes, derives seasonal signals,
standardizes within crop/year and across years within crop, and produces weighted
relative rankings/classes. Default outputs are `data/processed/yield_potential_index.csv`
and `data/processed/yield_potential_summary.csv`. It predicts no calibrated
bushels/acre and learns no relationship to observed harvest yields.

## Shared harvest integration and dependencies

The separate `../harvest-data/` reports remain original inputs for both projects.
`scripts/import_harvest_data.py` now imports operations into `data/processed/harvest/`
with explicit units and an exceptions report. `scripts/match_harvest_boundaries.py`
then matches candidates to ATS quarter sections and writes `field_boundaries.geojson`
plus a match report there. `scripts/export_harvest_features.py` exports monthly
Earth Engine Sentinel-2/Sentinel-1 features for the eligible quarters to
`data/processed/harvest/satellite/`. `scripts/build_field_yield_table.py` then
aggregates same-crop operations into 144 field-years with features, and
`scripts/benchmark_field_yield.py` compares per-crop models with a mean baseline.
`scripts/export_harvest_weather.py` adds ERA5-Land weather, and
`scripts/evaluate_season_field_model.py` separates season-level and field-level yield.
`scripts/fit_season_field_bundle.py` saves the model locked for the 2026 holdout, and
`scripts/predict_season_field.py` predicts and scores a new season. The 2026 scoring
steps and remaining work are in [NEXT_STEPS.md](NEXT_STEPS.md). No unit conversion
is performed.

`pyproject.toml` installs the package and optional ML dependencies;
`requirements.txt` adds geospatial/network tools. Dependencies cannot replace
missing project source. [HANDOFF_NOTES.md](HANDOFF_NOTES.md) lists current assets
and moved paths; [SCRIPT_GUIDE.md](SCRIPT_GUIDE.md) explains implementation details.
