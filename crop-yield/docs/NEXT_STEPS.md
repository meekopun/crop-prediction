# Crop yield: next steps with the harvest records

Prepared 2026-09-29 from the current source and [harvest-data inspection](../../harvest-data/README.md). Unchecked steps below are proposed work, not implemented features.

## Objective and current position

Build a model that predicts a farm field's yield from information available by a defined date. Start at field/crop/year resolution because that is the scale of the supplied observations. The 206 harvest operations cover 2020–2025, with mixed crops and units; they are not 206 verified independent training examples yet.

The additional transfer restored the ATS downloader, Earth Engine exporter, yield-potential index and both pixel-training functions, plus geometry, a monthly raw export and two reference tables. Pixel wrappers now use the supplied prepared table. The historical benchmark uses a different dataset/schema and its restored implementation uses random splits. The full-field preprocessor and three batch helpers remain missing; direct CDSE extraction remains unimplemented. No existing script imports the harvest Excel reports. See [the pipeline guide](PIPELINE_GUIDE.md) for the current A–G workflow map.

## 1. Import and audit the six workbooks

- [ ] Implement a read-only `.xls` importer that carries field headings across page breaks, joins wrapped text/units and ignores repeated report headers and timestamps.
- [ ] Produce a proposed `data/processed/harvest/operations.csv` with the data contract in the harvest assessment, plus an exceptions report.
- [ ] Reconcile the operation counts: 25, 32, 36, 36, 36 and 41 for 2020–2025 respectively.
- [ ] Check positive harvested areas, dates against operation year, rate/total units, and area × rate versus total, allowing source rounding.
- [ ] Preserve repeated operations. Review the mixed-crop headings at 2022 `Sheet1!A62:M64` and 2023 `Sheet1!A118:M120` rather than dropping them as duplicates.

**Completion evidence:** every output record has source coordinates; 206 operations are accounted for as included or flagged; no unparsed unit is silently treated as a number.

## 2. Review crop identities and target units

- [ ] Maintain a crop/variety mapping table, keeping original text.
- [ ] Confirm ambiguous variety-only entries, particularly the 2025 `DK 902 TF 2025 (Round up Ready)` label.
- [ ] Choose initial crops based on the number of usable independent fields and years after geometry matching. Oats, barley, wheat and confirmed canola/pea records are candidates, not an automatic inclusion list.
- [ ] Use one consistent yield unit per crop for the first benchmark. Keep the 34 `lb/ac` operations separate from the 172 `bu/ac` operations unless a reviewed conversion is applied.
- [ ] Verify whether reported rates are measured, calculated, estimated, or adjusted for moisture. Arithmetic agreement with total does not resolve that question.

**Completion evidence:** each training target has a reviewed crop, unit and measurement meaning; unsupported or ambiguous records remain traceable outside the pilot.

## 3. Match records to actual harvested boundaries

- [ ] Build a stable farm-field registry with geometry and aliases across years.
- [ ] Parse quarter- and half-section descriptions and look up candidate geometries from the supplied `data/raw/alberta_quarter_sections.geojson` (the classification project also retains its own source geometry).
- [ ] Request mapped boundaries for name-only fields and subfields. Compare reported acres with the candidate polygon area, accounting for unused land and changing harvested area.
- [ ] Resolve cases where one ATS parcel contains multiple fields/crops. Do not duplicate one yield onto several quarter sections as if they were independent observations.
- [ ] Produce proposed `data/processed/harvest/field_boundaries.geojson` and a geometry-match report.

**Completion evidence:** every pilot target refers to a reviewed harvested footprint; unmatched or ambiguous cases are explicitly excluded. Do not use the Lethbridge study-area sample as the harvest geometry.

## 4. Build a small satellite feature pilot

- [ ] Choose a small set of unambiguous fields and years, then define a prediction cutoff: whole-season retrospective estimate or a specific in-season forecast.
- [ ] Choose the restored `gee/sentinel2_quarter_sections.js` route or adapt the classification polygon extractor to these field boundaries. The classification extractor requires `properties.pid`, creates monthly features, and assumes quarter-section identifiers; the restored GEE exporter has its own asset ID/year configuration. Keep a separate field-ID mapping if using it temporarily.
- [ ] Export field-level optical/vegetation summaries for the correct seasons. Check cloud coverage, missing months and data availability before adding radar/weather features.
- [ ] Check the geographic coverage of the precipitation source hard-coded in the classification extractors before reusing it for these fields. A requested export does not establish valid local rainfall coverage.
- [ ] Preserve `field_id`, `year`, feature dates, feature units, valid-observation counts and extraction configuration. Join by reviewed field identity and year, with uniqueness checks.

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
