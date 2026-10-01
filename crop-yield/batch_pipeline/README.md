# Crop classification batch pipeline

Updated 2026-10-01 from the six Python files now present in this folder.
This is workflow D in [the yield pipeline guide](../docs/PIPELINE_GUIDE.md).
It aims to classify **crop types** using AAFC labels and Sentinel-2 optical
features. It does not predict harvest yield or import the harvest Excel reports.

All previously missing batch helpers are present. Preparation and scene discovery
have implementations; the combined runner's preferred direct Copernicus Data
Space (CDSE) extraction branch still raises `NotImplementedError`. A manifest
contains scene metadata, not downloaded imagery or model-ready predictors.

## Files and model roles

| File | What it does | What it contributes to the model |
| --- | --- | --- |
| [prepare_crop_classification_batches.py](prepare_crop_classification_batches.py) | Counts filtered GeoJSON parcels and writes 40 deterministic batch ranges and a tracker | Divides feature-extraction work |
| [quarter_section_batch_utils.py](quarter_section_batch_utils.py) | Reads config/batches, selects polygons, provides IDs, writes GeoJSON and updates tracker | Shared parcel keys and processing state |
| [copernicus_data_space_utils.py](copernicus_data_space_utils.py) | Builds weekly date windows/bounds, searches STAC, looks up OData products and constructs URLs | Candidate Sentinel-2 scene metadata |
| [build_copernicus_scene_manifest.py](build_copernicus_scene_manifest.py) | Searches weekly candidates for one batch and saves a ranked manifest | A discovery record for a future extractor |
| [run_crop_classification_batch.py](run_crop_classification_batch.py) | Samples AAFC labels; includes a legacy Statistics API optical extractor and index summaries | Intended quarter-section/year training table |
| [train_one_vs_rest_crop_models.py](train_one_vs_rest_crop_models.py) | Evaluates balanced logistic classifiers, one crop versus all others | Cross-validation probabilities and classification metrics |
| [crop_classification_pipeline_config.json](crop_classification_pipeline_config.json) | Describes sources, crops, season, batching and output paths | Intended workflow settings; some are not enforced |

See [SCRIPT_GUIDE.md](../docs/SCRIPT_GUIDE.md#d-batch-crop-classification-inside-the-yield-project)
for each file's detailed process, controls, outputs and limitations.

## Intended flow and current gap

1. Prepare `quarter_section_batches.csv` and `quarter_section_tracker.csv`.
2. Discover candidate scenes → `scene_manifests/<batch_id>_cdse_scene_manifest.csv`.
3. Download/mask imagery and extract weekly optical predictors, join AAFC labels
   → `batch_features/<batch_id>_features.csv`. **The direct-CDSE implementation
   connecting discovery to this table is unfinished.** The runner does not read
   the manifest.
4. Evaluate binary classifiers → `model_outputs/binary_crop_model_metrics.csv`
   and `binary_crop_predictions.csv`.

These are generated outputs, not files that must have been transferred with the
source. A `manifest_ready` status means discovery finished; `features_ready`
requires actual extraction. Training currently saves no fitted model bundle.

## Paths and prerequisites

Commands below run from `/Users/meeko/Work/crop-prediction/crop-yield` using its
existing `.venv`. The scripts are in `batch_pipeline/`; the old
`upstream/crop_classification_batch_pipeline/` command paths no longer apply.

The actual geometry is `data/raw/alberta_quarter_sections.geojson`. The newly
copied preparation default and JSON configuration refer to the former
`quarter_sections.geojson` location. Before discovery/extraction, change the
config's `geojson.path` to:

```json
"path": "../data/raw/alberta_quarter_sections.geojson"
```

Configured geometry/output paths resolve relative to `batch_pipeline/`, even
with a config file saved elsewhere. Preparation has its own `--geojson` default,
so also supply the explicit path in its command below. Keep the geometry order
and road-allowance filtering unchanged once batches are planned.

Dependencies vary by stage: preparation uses the standard library; batch helpers
and manifest building need `pandas`, and CDSE helpers need `requests`; training
also needs `numpy` and `scikit-learn`. Local AAFC raster sampling needs `rasterio`;
GEE label sampling needs Earth Engine authentication and project access.

The 2026-10-01 local check found `requests` missing from the existing `.venv`.
Manifest building and the combined runner therefore fail even at `--help` until
that dependency is installed. `requests` is listed in `requirements.txt`, but
the package's `ml` extra does not include it. Preparation and trainer help checks
passed; script syntax and documentation links were checked separately.

The CDSE client factory requires `CDSE_USERNAME` and `CDSE_PASSWORD` environment
variables. Its search/product-lookup calls currently send no authorization header,
and manifest building does not call its token method. The code contains endpoint
settings; service access was not validated by this documentation update.

## Commands

Prepare the plan and tracker:

```bash
.venv/bin/python batch_pipeline/prepare_crop_classification_batches.py \
  --geojson data/raw/alberta_quarter_sections.geojson
```

Preparation overwrites the existing plan/tracker. Preserve progress before
regenerating them. The streaming parser has a buffer-growth limitation described
in the script guide; these commands are not evidence of a successful full-size run.

After correcting `geojson.path`, prepare a scene manifest for one batch:

```bash
.venv/bin/python batch_pipeline/build_copernicus_scene_manifest.py \
  --batch-id batch_01 --years 2021 2022 2023 --limit-per-week 8
```

This command discovers scenes and writes metadata. It does not produce feature
CSVs. Searches use only the first returned page; cloud ranking applies to those
returned candidates.

Inspect the combined runner's available options:

```bash
.venv/bin/python batch_pipeline/run_crop_classification_batch.py --help
```

Its default label route needs a populated `label_source.local_raster_pattern`.
The explicit `--label-source gee_aafc_only` route uses GEE for labels and a
hard-coded project. Neither choice implements the preferred CDSE optical route.
The alternative `--sentinel-source sentinel_hub_process_api` actually calls the
Statistics API. It needs a `statistics_endpoint`, compatible OAuth settings,
`SENTINEL_HUB_CLIENT_ID` and `SENTINEL_HUB_CLIENT_SECRET`; the supplied config
lacks the Statistics endpoint.

Once valid feature CSVs exist, the current trainer can be invoked with:

```bash
.venv/bin/python batch_pipeline/train_one_vs_rest_crop_models.py \
  --batch-ids batch_01 --cv-splits 5 --top-features-per-crop 32
```

Omit `--batch-ids` to load all matching feature CSVs. Training uses every observed
crop label, rather than only configured `crops_of_interest`. It selects features
using the full dataset before cross-validation, so scores can be optimistic.
Fix selection within folds and choose suitable field/year holdouts before relying
on results. The current grouping is parcel/year, and the same parcel in another
year can appear in a different fold.

## Next steps for this workflow

- Correct the copied geometry config and preparation default, preserving batch
  identity if a plan already exists.
- Check/fix the streaming parser and prevent concurrent tracker updates from
  overwriting one another before large batch runs.
- Verify scene discovery on a small batch; add pagination, defined empty output
  columns and failure/progress handling.
- Implement manifest-to-feature extraction: imagery access, CRS handling,
  cloud/invalid-pixel masking, parcel statistics and weekly summaries with valid
  observation counts. Join AAFC labels on stable parcel/year keys.
- Move feature selection inside cross-validation, enforce a predictor allowlist,
  and evaluate explicit new-field and future-year holdouts.
- Save a fitted preprocessing/model bundle and feature schema for inference.

For harvest-yield work, follow [the yield next steps](../docs/NEXT_STEPS.md)
instead; this classification route is a separate workflow.
