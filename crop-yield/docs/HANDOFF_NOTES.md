# Transfer inventory and layout changes

Updated 2026-10-22 after the projects were reorganized: `batch_pipeline/` moved
to `crop-classification/`, and the full-field builder was recovered from the old
repository. This inventory distinguishes present files from still-missing components.

## Present source

- `src/yield_prediction/`: restored `alberta_ats.py`, `yield_index.py`, CLI,
  data module, historical/pixel modeling, NDVI redistribution and harvest import.
- `gee/sentinel2_quarter_sections.js`: supplied seasonal GEE exporter.
- `scripts/`: basic launcher, harvest importer, full-field builder and three
  pixel modeling/redistribution wrappers.
- `config/harvest_review.json`: saved harvest crop confirmations and exclusions.
- `pyproject.toml`, `requirements.txt`, `tests/` and current documentation.

## Present local data

| Location | Purpose |
| --- | --- |
| `data/raw/yield_df.csv` | Tracked historical example |
| `data/raw/alberta_quarter_sections.geojson` | Supplied province-wide ATS geometry |
| `data/raw/pixel_crop_features_aci_s2_monthly_csv_quarter_sections_fullfield.csv` | Supplied monthly full-field pixel export |
| `reference_inputs/Crop Yield Data - Crop Data 2021-2023-2.csv` | Observed-yield reference |
| `reference_inputs/pixel_level_all_crop_training_features_2021_2023.csv` | Prepared pixel modeling/lookup table |
| `../harvest-data/*.xls` | Separate original 2020–2025 reports shared by both projects |

Large geometry, raw exports and reference inputs remain local/ignored. A fresh
Git checkout does not contain them: transfer them explicitly to these paths.
Existing model/benchmark outputs were preserved. Historical benchmark CSVs were
produced by the earlier replacement benchmark, not the restored random-split CLI.

## Paths moved during this organization

Paths below are relative to `crop-yield/`. Files were renamed/moved without
rewriting their contents; geometry was not duplicated.

| Previous location | Current location |
| --- | --- |
| `scripts/sentinel2_quarter_sections.js` | `gee/sentinel2_quarter_sections.js` |
| `quarter_sections.geojson` | `data/raw/alberta_quarter_sections.geojson` |
| `data/raw/Crop Yield Data - Crop Data 2021-2023-2.csv` | `reference_inputs/Crop Yield Data - Crop Data 2021-2023-2.csv` |
| `data/raw/pixel_level_all_crop_training_features_2021_2023.csv` | `reference_inputs/pixel_level_all_crop_training_features_2021_2023.csv` |
| `batch_pipeline/` (2026-10-22) | `../crop-classification/batch_pipeline/` |

The monthly full-field export stays in `data/raw/`; the pipeline guide now uses
that path consistently. The recovered data module's outdated `archive/yield_df.csv`
default was corrected to `data/raw/yield_df.csv`. Use explicit `--input` for other
prepared tables.

Files copied from the old-repository zip on 2026-10-22 carried pre-organization
defaults. The three pixel wrappers' input defaults were re-pointed to
`reference_inputs/`, and the full-field builder's defaults to `data/raw/` and
`reference_inputs/`. Check any further files copied from that zip against the
table above before running them. Batch geometry defaults now resolve inside
`crop-classification/`; see [its batch README](../../crop-classification/batch_pipeline/README.md).

## Still missing

No documented source file is missing. Credentials, uploaded GEE assets and a
future seasonal export for G are separate runtime requirements, not missing
Python modules.

## Verification and scientific scope

Local verification includes the restored summary, eight tests, the three pixel
entry points using supplied data with outputs directed to a temporary directory,
and module help/syntax checks. Live downloads and Earth Engine exports are not
part of a directory reorganization. Input hashes are checked across moves.

The 2026-10-22 reorganization check ran all 26 tests, confirmed every script's
default input path exists, and ran the full-field builder on the supplied inputs
(87,176 pixel rows, eight field-years). The moved batch files are byte-identical
to their previous copies.

See [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) for workflow order and
[SCRIPT_GUIDE.md](SCRIPT_GUIDE.md) for implementation behavior.
