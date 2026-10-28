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

On 2026-10-22 `scripts/build_field_yield_table.py` built 144 field-years and
`scripts/benchmark_field_yield.py` scored six crops with at least 10 of them.
Holding out whole sections, every model/feature combination beat the baseline for
canola (skill +0.42 to +0.50) and wheat (+0.35 to +0.52). Holding out whole
seasons, canola stayed positive (+0.08 to +0.42) but the average across crops was
near zero, and peas never beat the baseline. Season-level prediction is the weak
point: models help in unusual years and can hurt in typical ones.

- [x] Add a separate field-data loader and training entry point (`field_yield.py`, `field_benchmark.py`). The historical benchmark is unchanged.
- [x] Define an explicit predictor allowlist: only `field_<band>_<mon>` satellite columns (`FEATURE_SETS`). Whole-season features suit a retrospective estimate; an in-season forecast needs a month cutoff.
- [x] Compare a crop-specific training-mean baseline with ridge (penalty chosen inside each training fold) and a shallow gradient-boosting model.
- [x] Fit imputers, scalers and penalty selection inside training folds only.
- [x] Score a final chronological holdout once. On 2026-10-22 `scripts/holdout_field_yield.py` chose one configuration per crop by leave-one-year-out on 2020–2024 (baseline included), refitted it, and scored 2025. Wheat improved (skill +0.69, 3 field-years); peas and oats kept the baseline; fescue (−0.09), barley (−0.83) and canola (−6.08) were worse. Canola over-predicted every 2025 field (bias +11.9 bu/ac) while 2025 yields sat near the long-run mean. With five training seasons, season-level yield is not learnable from satellite features alone. 2025 has now been used; further model development should use 2020–2024 only, or wait for 2026 data.
- [x] Add a separate evaluation holding out entire fields across all years (`section` split, grouped by ATS section).
- [x] Report MAE/RMSE in each crop's target unit, R² where meaningful, bias, and independent field/year counts.
- [x] Reduce selection optimism: the holdout chooses one configuration per crop from development years only.
- [x] Separate season level from field level (2026-10-22, 2020–2024 only, leave one season out; `scripts/evaluate_season_field_model.py`). Findings:
  - ERA5-Land weather (`scripts/export_harvest_weather.py`: May–Aug rain, May–Aug GDD5, July mean high) does not predict the season mean from four training seasons: its error is 45% of the crop mean versus 38% for the training mean, and it gets the 2021 heat year backwards.
  - Season means of peak-season satellite indices (July NDVI/NDRE/EVI) do better (34%).
  - Within-season deviations from those three features help barley clearly (given the true season mean, skill 0.21 → 0.51) and wheat slightly; the planned 49-index set overfits and hurts every crop.
  - Best achievable combination (satellite season + peak deviation): barley +0.45, canola +0.16, wheat +0.11, fescue +0.11, oats −0.04, peas −0.44. These feature sets were chosen after inspecting 2020–2024 diagnostics, so the scores are optimistic; 2026 data is the next clean test.
  - Knowing the true season mean would give +0.40 to +0.56 for every crop except peas, so the season level is the main remaining lever.
- [x] Supply the season mean from published regional yields (2026-10-23). `scripts/prepare_regional_yields.py` extracts Statistics Canada table 32-10-0002-01 (downloaded 2026-10-23, 4,108,329 bytes, SHA-256 `948da55f…42dd62`; released 2026-02-12). Small Area Data Region 70 is Census Agricultural Region 7 (divisions 17–19, including Grande Prairie County No. 1). `--season-features regional` scales each crop's published yield by its training farm-to-region ratio. On 2020–2024, leaving one season out, season-model skill was barley +0.31, wheat +0.17, oats +0.13, canola +0.08, peas −0.76 (fescue has no series). Season error was 33.0% of the crop mean versus 31.2% for satellite season means and 34.2% for the training mean (24 crop-seasons without fescue). It beats the satellite option only for wheat.
  - Why it falls short of the "known season" ceiling: the farm departs from the region in exactly the extreme years. In 2021 the region's canola averaged 30.4 bu/ac against the farm's 14.8; in 2023 the farm out-yielded the region by 10–105% across crops. Farm-to-region ratios vary from 0.49 to 2.05 by year, and correlations between farm and regional season means over 2020–2024 are 0.55–0.72 (peas 0.21). The ceiling used the farm's own other fields, which share local weather; a region-wide average does not.
- [x] Learn the season level from many region-seasons (2026-10-23). `scripts/export_regional_features.py` averaged July NDVI/NDRE/EVI over each crop's inventory pixels in all eight Alberta regions, 2019–2025 (56 region-years, ≥419 pixels each), paired with StatCan yields (277 region-crop-years). `--season-features regional_satellite` fits one ridge per crop on every region-year except the held-out season and applies it to the farm's own July anomaly. Design fixed before running.
  - On the regions themselves (leave one year out, ~55 region-years per crop) skill was barley +0.46, wheat +0.35, canola +0.31, oats +0.24, peas +0.21.
  - On the farm, 2020–2024, season-model skill: barley +0.47, canola +0.36, oats +0.34, wheat +0.29, peas −0.36; with the peak field deviation: barley +0.65, wheat +0.36, canola +0.35, oats +0.12. Season error fell to 24.8% of the crop mean (farm-only satellite 31.2%, regional yields 33.0%, training mean 34.2%). It predicted the 2021 heat year correctly (barley 34.4 vs 34.9, wheat 29.0 vs 28.7) and 2023 as high (canola 31.9 vs 43.6, under).
  - Why it works: the satellite-to-yield slope is estimated from ~48 region-years per crop instead of four farm seasons, and the farm's own July anomaly carries its local conditions.
  - Peas stay worse than the baseline; fescue has no regional series.
- [x] Lock `regional_satellite` + `peak` with `--min-improvement 0.1` for the 2026 holdout (2026-10-28, before any 2026 record existed). `scripts/fit_season_field_bundle.py` chose per crop by leaving one season out over 2020–2025, with regional models trained on 2019–2025 region-years, and saved `data/processed/harvest/models/season_field_2026/` (`model.joblib` SHA-256 `30bd2a9909e28f9a27c764d91c51281803e155c3454c7f0d11b9439400608d5e`). Model kept for barley (development skill +0.21), canola (+0.27), oats (+0.12) and wheat (+0.43); training mean for peas (−0.46) and fescue (no regional series). Rye and the other small crops have no model.
  - A one-off 2025 look, using a bundle fitted on 2020–2024, is not clean (2025 had been seen). It went badly: pooled bu/ac skill −0.69; wheat +0.66, barley −1.52, canola −2.44, oats −0.92, every crop biased low. July 2025 indices were low across Alberta while yields were near normal. The regional model under-predicted 2025 in all eight regions (by 7–27% on average per crop), so its ~56 region-years behave like 6–7 independent seasons. Including 2025 in selection cut barley's development skill from +0.63 to +0.21.
- [ ] Score the locked bundle on 2026 once. Do not refit it after seeing 2026 yields:

  ```bash
  python -m pip install -e '.[ml,harvest]'      # xlrd reads the workbook
  # put 2026-harvest-data.xls in ../harvest-data/, then review new exclusions in config/harvest_review.json
  python scripts/import_harvest_data.py
  python scripts/match_harvest_boundaries.py
  python scripts/export_harvest_features.py --years 2020 2021 2022 2023 2024 2025 2026  # list every year: the combined CSV holds only listed years
  python scripts/build_field_yield_table.py
  python scripts/predict_season_field.py --score  # writes score_2026.{csv,md} beside the bundle; refuses a second run
  ```
- [ ] Optionally add June/August indices or more years (Landsat before 2019) to the regional set; get municipality-level yields or nearby farms for a closer season source.
- [x] Prefer fewer features: three peak-season indices beat 49 indices for field deviations on every crop.
- [x] Require a chosen model to beat the baseline by a margin on development years (`--min-improvement`). Backtests that treat 2023 and 2024 as the final year, each using only earlier years: with no margin, mean skill over 12 crop-tests was −0.76 (peas 2024 −8.70); with a 10% margin −0.06, and pooled bu/ac error fell to 88.9% of the baseline's (98.5% with no margin, 93.1% with 20%). Use `--min-improvement 0.1` for the 2026 holdout. Models won in 2023 (canola +0.48, barley +0.41, wheat +0.22) and lost in 2024, matching the season-level pattern above.
- [ ] Try pooling crops on relative yield and an in-season month cutoff.

**Completion evidence:** reproducible splits and predictions outperform—or transparently fail to outperform—the baseline on truly held-out observations. A strong result from the unrelated historical CSV is not evidence for this farm dataset.

## 6. Save an inference-ready model and document scope

- [x] Save the fitted preprocessing/model bundle, ordered feature schema, crop/unit mapping, training years, feature cutoff and software versions (2026-10-28, `season_field_bundle.py`: `model.joblib` plus `manifest.json` with input and model SHA-256s).
- [x] Add inference for new field features without requiring the actual harvest yield (`scripts/predict_season_field.py`). It needs all of a crop's fields for a season together. Features for a season before harvest would need a field/crop list that does not come from the harvest workbook.
- [x] Check feature/schema mismatch behavior and reload-to-prediction consistency (`tests/test_season_field_bundle.py`). The bundle reproduces the evaluation's held-out 2025 predictions exactly.
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
