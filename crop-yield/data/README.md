# Yield data locations

- `raw/yield_df.csv`: bundled historical example, tracked in Git.
- `raw/alberta_quarter_sections.geojson`: supplied ATS geometry, moved from the
  yield project root; remains local and unchanged.
- `raw/pixel_crop_features_aci_s2_monthly_csv_quarter_sections_fullfield.csv`:
  supplied monthly full-field satellite export; remains local and unchanged.
- `processed/`: generated model results and relative-index outputs.
- `processed/harvest/`: generated operation/candidate CSVs, exceptions CSV and JSON summary
  from `scripts/import_harvest_data.py`; staging observations awaiting review and
  field/satellite matching, not a model-ready training table.

Saved crop confirmations and temporary exclusions live separately in
`../config/harvest_review.json`, so regenerating staging files preserves them.

Observed-yield and prepared pixel/lookup reference CSVs live in
`../reference_inputs/`, as described by the pipeline guide. Original harvest
workbooks live at repository-level `harvest-data/`.

Existing `processed/benchmark/metrics.csv` and `predictions.csv` came from the
previous chronological benchmark implementation. The restored historical
benchmark prints a different multi-model/random-split evaluation and does not
regenerate those CSVs. Keep that provenance distinction when comparing results.
