# Crop yield: next steps with the harvest records

Updated 2026-10-01 from the current source and [harvest-data inspection](../../harvest-data/README.md). Unchecked steps below are proposed work, not implemented features.

## Objective and current position

Build a model that predicts a farm field's yield from information available by a defined date. Start at field/crop/year resolution because that is the scale of the supplied observations. The 206 harvest operations cover 2020–2025, with mixed crops and units; they are not 206 verified independent training examples yet.

The additional transfer restored the ATS downloader, Earth Engine exporter, yield-potential index and both pixel-training functions, plus geometry, a monthly raw export and two reference tables. Pixel wrappers now use the supplied prepared table. The historical benchmark uses a different dataset/schema and its restored implementation uses random splits. The full-field preprocessor was recovered on 2026-10-22; it rebuilds the pixel table from the monthly export but adds no field-years. The batch crop-classification workflow moved to `crop-classification/batch_pipeline/` on 2026-10-22; see [its README](../../crop-classification/batch_pipeline/README.md) for that separate workflow and its next steps. `scripts/import_harvest_data.py` now stages the harvest Excel observations and review flags; they are not yet joined to training features. See [the pipeline guide](PIPELINE_GUIDE.md) for the current A–G workflow map.

## 1. Import and audit the six workbooks

- [x] Implement a read-only `.xls` importer that carries field headings across page breaks, joins wrapped text/units and ignores repeated report headers and timestamps.
- [x] Produce `data/processed/harvest/operations.csv` with source coordinates and explicit units, plus `exceptions.csv` and `summary.json`.
- [x] Reconcile the operation counts: 25, 32, 36, 36, 36 and 41 for 2020–2025 respectively.
- [x] Check positive harvested areas, dates against operation year, rate/total units, and area × rate versus total, allowing source rounding.
- [ ] Preserve repeated operations. Review the mixed-crop headings at 2022 `Sheet1!A62:M64` and 2023 `Sheet1!A118:M120` rather than dropping them as duplicates.

**Completion evidence:** every output record has source coordinates; 206 operations are accounted for as included or flagged; no unparsed unit is silently treated as a number.

## 2. Review crop identities and target units

- [ ] Maintain a crop/variety mapping table, keeping original text.
- [x] Confirm the 2025 `DK 902 TF 2025 (Round up Ready)` label as canola: user confirmation on 2026-10-06, saved in `config/harvest_review.json`.
- [ ] Choose initial crops based on the number of usable independent fields and years after geometry matching. Oats, barley, wheat and confirmed canola/pea records are candidates, not an automatic inclusion list.
- [ ] Use one consistent yield unit per crop for the first benchmark. Keep the 34 `lb/ac` operations separate from the 172 `bu/ac` operations unless a reviewed conversion is applied.
- [ ] Verify whether reported rates are measured, calculated, estimated, or adjusted for moisture. Arithmetic agreement with total does not resolve that question.

**Completion evidence:** each training target has a reviewed crop, unit and measurement meaning; unsupported or ambiguous records remain traceable outside the pilot.

## 3. Match records to actual harvested boundaries

The user temporarily excluded the six name-only operations and four mixed-crop
operations on 2026-10-06. Their operation IDs and reasons are saved in
`config/harvest_review.json`; all remain traceable in `operations.csv`. The 196
remaining processing candidates still need verified harvested footprints.

On 2026-10-22 `scripts/match_harvest_boundaries.py` matched all 196 candidates to
41 ATS quarter sections (half sections become two quarters). 39 operations share
a quarter-year with another crop and 1 reports more acres than its quarter; the
other 156 are eligible for a boundary pilot. 76 operations cover under half of
their quarter, so quarter-wide satellite averages would mix in other land.

- [ ] Build a stable farm-field registry with geometry and aliases across years.
- [x] Parse quarter- and half-section descriptions and look up candidate geometries from the supplied `data/raw/alberta_quarter_sections.geojson` (the classification project also retains its own source geometry).
- [ ] Request mapped boundaries for name-only fields and subfields. Reported acres are now compared with the candidate polygon area (`area_fraction` and flags in `boundary_matches.csv`); unused land and changing harvested area still need review.
- [ ] Resolve cases where one ATS parcel contains multiple fields/crops. They are flagged `shared_multiple_crops` and excluded from the pilot; same-crop sharing is flagged `shared_same_crop` and needs aggregation. Do not duplicate one yield onto several quarter sections as if they were independent observations.
- [x] Produce proposed `data/processed/harvest/field_boundaries.geojson` and a geometry-match report (`boundary_matches.csv`, `boundary_match_summary.json`).
- [ ] Restrict features to the reported crop within each quarter (for example with the AAFC crop inventory) where coverage is low.

**Completion evidence:** every pilot target refers to a reviewed harvested footprint; unmatched or ambiguous cases are explicitly excluded. Do not use the Lethbridge study-area sample as the harvest geometry.

## 4. Build a small satellite feature pilot

On 2026-10-22 `scripts/export_harvest_features.py` exported 163 pilot-eligible
quarter-section years (2020–2025) to `data/processed/harvest/satellite/`. Every
month has a cloud-free composite and Sentinel-1 values. The crop-inventory mask
agrees with the farm records for canola (median 81% of the quarter), wheat (90%)
and peas (74%), none under 10%. It mostly disagrees for oats and rye (median 0%;
8 of 15 and 6 of 10 under 10%), and 7 of 25 barley quarter-years are under 10%.

- [ ] Choose a small set of unambiguous fields and years, then define a prediction cutoff: whole-season retrospective estimate or a specific in-season forecast.
- [x] Adapt the classification polygon extractor to these field boundaries: `harvest_features.py` reuses its bands, indices, months and cloud mask, keyed by `pid` and year, and adds crop-inventory masking.
- [x] Export field-level optical/vegetation summaries for the correct seasons, with per-month valid-pixel shares.
- [x] Check the precipitation source: CHIRPS ends at 50°N, so no precipitation is exported for these ~55°N fields. Weather still needs another source (for example ERA5-Land).
- [x] Preserve `pid`, `year`, operation IDs, valid-observation shares and extraction configuration (`harvest_features_manifest.json`).
- [ ] Decide when to use crop-masked (`crop_*`) versus whole-quarter (`quarter_*`) features, using `aci_crop_fraction`; investigate the oats/rye disagreements.

**Completion evidence:** a small joined table contains satellite predictors and independently supplied harvest targets for the same land and year, with no silent row multiplication. Then extend to all usable records.

## 5. Implement a field-yield benchmark

- [ ] Add a separate field-data loader and training entry point. Preserve the existing historical benchmark as its own example.
- [ ] Define an explicit predictor allowlist. Exclude observed yield, harvest total, rate-derived columns, source row IDs, and data collected after the prediction cutoff. Use observed crop only when it will also be known at prediction time.
- [ ] Compare a crop-specific training-mean baseline with a simple regularized regression and a modest tree-based model. With this small dataset, prioritize reliable evaluation over model complexity.
- [ ] Fit imputers, scalers and feature selection inside training folds only.
- [ ] Use 2020–2023 for initial development, 2024 for validation, and 2025 as a proposed final chronological holdout, subject to adequate per-crop coverage after cleaning. Tune using development data rather than the final holdout.
- [ ] Add a separate evaluation holding out entire fields across all years. A later-year test on familiar fields and a new-field test answer different questions.
- [ ] Report MAE/RMSE in each crop's target unit, R² where meaningful, bias, and independent field/year counts. Do not pool incompatible units into one error score.

**Completion evidence:** reproducible splits and predictions outperform—or transparently fail to outperform—the baseline on truly held-out observations. A strong result from the unrelated historical CSV is not evidence for this farm dataset.

## 6. Save an inference-ready model and document scope

- [ ] Save the fitted preprocessing/model bundle, ordered feature schema, crop/unit mapping, training years, feature cutoff and software versions.
- [ ] Add inference for new field features without requiring the actual harvest yield.
- [ ] Check feature/schema mismatch behavior and reload-to-prediction consistency.
- [ ] Record exclusions and limits: data from one farm/report source do not establish province-wide accuracy.

**Completion evidence:** a saved model can predict a held-out field/season from valid inputs alone, with its target unit and applicability visible.

## 7. Consider pixels after the field benchmark works

The existing `pixel_yield.py` redistributes a **known** field yield according to NDVI. That can illustrate within-field variability, but its pixel values are estimates, not independently observed labels.

- [ ] If using redistribution, supply a prepared pixel table with the required NDVI summaries and a consistent `yield_bu_ac` within each group. Do not feed pound-based observations into that column.
- [ ] Retain pixel coordinates/IDs in any new exports; the classification pixel exporter currently omits geometry.
- [ ] Keep every pixel from a field together during field-generalization evaluation and evaluate predictions at the measurement scale.
- [ ] Obtain spatial harvest-monitor measurements if the goal is to validate genuine pixel-yield accuracy.

Copying a field average onto thousands of pixels does not create thousands of independent yield observations. Do not train and test on NDVI-generated pseudo-targets and present the result as measured pixel-yield performance.

## Immediate recommended sequence

Import and review harvest records → resolve boundaries and crop/unit mappings → export a small field-level feature pilot → evaluate a field-yield baseline → expand.

See [SCRIPT_GUIDE.md](SCRIPT_GUIDE.md) for what the existing files actually implement and where restoration is needed.
