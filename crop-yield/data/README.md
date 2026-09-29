# Yield data locations

- `raw/yield_df.csv`: bundled historical example, tracked in Git.
- `raw/alberta_quarter_sections.geojson`: supplied ATS geometry, moved from the
  yield project root; remains local and unchanged.
- `raw/pixel_crop_features_aci_s2_monthly_csv_quarter_sections_fullfield.csv`:
  supplied monthly full-field satellite export; remains local and unchanged.
- `processed/`: generated model results and relative-index outputs.

Observed-yield and prepared pixel/lookup reference CSVs live in
`../reference_inputs/`, as described by the pipeline guide. Original harvest
workbooks live at repository-level `harvest-data/`.

Existing `processed/benchmark/metrics.csv` and `predictions.csv` came from the
previous chronological benchmark implementation. The restored historical
benchmark prints a different multi-model/random-split evaluation and does not
regenerate those CSVs. Keep that provenance distinction when comparing results.
