# Transfer inventory and layout changes

Updated 2026-10-01 after the additional batch helpers were supplied.
This inventory distinguishes present files from still-missing components.

## Present source

- `src/yield_prediction/`: restored `alberta_ats.py`, `yield_index.py`, CLI,
  data module, historical/pixel modeling, and NDVI redistribution.
- `gee/sentinel2_quarter_sections.js`: supplied seasonal GEE exporter.
- `scripts/`: basic launcher and three pixel modeling/redistribution wrappers.
- `batch_pipeline/`: config and all six scripts: preparation, batch utilities,
  CDSE utilities, scene-manifest builder, runner and binary classifier.
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

The monthly full-field export stays in `data/raw/`; the pipeline guide now uses
that path consistently. The recovered data module's outdated `archive/yield_df.csv`
default was corrected to `data/raw/yield_df.csv`. Pixel wrapper defaults and batch
geometry references were updated during organization. The later batch-file
transfer restored the old geometry references in preparation and config: supply
`--geojson data/raw/alberta_quarter_sections.geojson` to preparation and set config
`geojson.path` to `../data/raw/alberta_quarter_sections.geojson`. The documentation
update records that mismatch without changing the supplied Python/config files.
Use explicit `--input` for other prepared tables.

## Still missing

- `scripts/build_fullfield_pixel_training_table.py`

The preferred direct-CDSE extraction branch is still unimplemented. No placeholder
files were added for these gaps. Scene discovery is now implemented, but the
manifest builder does not download/process imagery and the runner does not consume
its output. Credentials, uploaded GEE assets and a future
seasonal export are separate runtime requirements, not missing Python modules.

## Verification and scientific scope

Local verification includes the restored summary, eight tests, the three pixel
entry points using supplied data with outputs directed to a temporary directory,
and module help/syntax checks. Live downloads and Earth Engine exports are not
part of a directory reorganization. Input hashes are checked across moves.

The 2026-10-01 documentation update checked coverage of all six batch scripts,
their syntax and local Markdown links. Preparation and trainer `--help` passed.
Manifest-builder and runner help checks failed because `requests` is absent from
the current `.venv`; the files themselves are present. No live scene discovery
or batch feature extraction was validated by those checks.

See [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) for workflow order and
[SCRIPT_GUIDE.md](SCRIPT_GUIDE.md) for implementation behavior.
