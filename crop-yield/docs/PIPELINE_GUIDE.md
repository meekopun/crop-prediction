# Crop yield pipeline guide

Updated 2026-10-01 after the batch helpers and scene-manifest builder were supplied. This is
the current workflow map. Run local commands from `crop-yield/` with its package
installed, or use `PYTHONPATH=src` for Python module/script calls.

## Workflow map

| Workflow | Implementation | Current input/status |
| --- | --- | --- |
| A: historical benchmark | `src/yield_prediction/{data,modeling,cli}.py` | Bundled CSV; local workflow available |
| B: ATS geometry | `src/yield_prediction/alberta_ats.py` | Restored downloader; geometry already supplied locally; live API not verified |
| C: seasonal satellite export | `gee/sentinel2_quarter_sections.js` | Restored; requires GEE asset/project configuration and execution |
| D: batch crop classification | `batch_pipeline/` | All six scripts present; scene discovery implemented, preferred extraction unfinished |
| E: full-field pixel preprocessing | Expected `scripts/build_fullfield_pixel_training_table.py` | Script still absent; raw export and reference inputs now present |
| F: pixel models/redistribution | `scripts/` plus package modeling modules | Implementations and prepared 2021–2023 table present |
| G: yield-potential index | `src/yield_prediction/yield_index.py` | Relative index calculation present; requires C-format export |

A is an independent historical example. B supplies geometry for extraction.
C and D are alternative feature routes with different schemas. E prepares the
monthly full-field export for F; the supplied prepared reference table lets F
run without regenerating E. G consumes C's seasonal polygon table. None of these
paths currently imports the 2020–2025 harvest Excel workbooks automatically.

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
this restored exporter are separate from D's optical-only feature policy.

## D. Batch crop classification (still incomplete)

Present: config and all six scripts: preparation, shared batch utilities, CDSE
utilities, scene-manifest builder, combined runner and binary classifier.

The intended sequence is prepare batches → discover scenes → extract labels and
features → train binary crop classifiers. The scene builder writes candidate
metadata and download URLs, not raster data or predictors. The runner does not
consume that manifest, and preferred CDSE extraction raises `NotImplementedError`.
The legacy Sentinel Hub Statistics route also lacks required settings in the
supplied config; local AAFC label paths are blank.
The current `.venv` also lacks `requests`, which blocks manifest-builder and
runner imports until installed; it is listed in `requirements.txt`.

The newly copied config and preparation default refer to the former
`quarter_sections.geojson` location. The actual supplied geometry is
`data/raw/alberta_quarter_sections.geojson`. Set config `geojson.path` to
`../data/raw/alberta_quarter_sections.geojson` (relative to `batch_pipeline/`),
and supply preparation's separate `--geojson data/raw/alberta_quarter_sections.geojson`
argument from `crop-yield/`.

The trainer can evaluate completed feature tables, but selection currently uses
held-out labels before cross-validation, parcel/year grouping does not hold out
entire fields across years, and no fitted inference model is saved. Restored
source does not establish completed extraction or trustworthy model scores.

See [batch_pipeline/README.md](../batch_pipeline/README.md) for command paths and
[SCRIPT_GUIDE.md](SCRIPT_GUIDE.md#d-batch-crop-classification-inside-the-yield-project)
for the detailed process of each script.

## E. Full-field monthly pixel preprocessing (script still missing)

Present inputs:

- `data/raw/pixel_crop_features_aci_s2_monthly_csv_quarter_sections_fullfield.csv`
- `reference_inputs/Crop Yield Data - Crop Data 2021-2023-2.csv`
- `reference_inputs/pixel_level_all_crop_training_features_2021_2023.csv`

The expected `scripts/build_fullfield_pixel_training_table.py` has not been
supplied. Its historical specification is to normalize parcel/crop/year keys,
replace `-9999`, derive NDVI summaries, join observed yields, and build a
`row_id` grouped modeling table. This describes planned/recovered behavior to
implement or obtain, not a command available in this checkout.

Use a new generated output under `data/processed/` when this step is restored.
Keep the supplied reference files unchanged. The monthly export's `ndvi_m04`
style schema is different from the prepared table's `NDVI_max` style schema.

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
Their import, unit normalization and field-boundary matching are future work in
[NEXT_STEPS.md](NEXT_STEPS.md).

`pyproject.toml` installs the package and optional ML dependencies;
`requirements.txt` adds geospatial/network tools. Dependencies cannot replace
missing project source. [HANDOFF_NOTES.md](HANDOFF_NOTES.md) lists current assets
and moved paths; [SCRIPT_GUIDE.md](SCRIPT_GUIDE.md) explains implementation details.
