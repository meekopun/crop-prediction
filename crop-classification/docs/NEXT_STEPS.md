# Crop classification: next steps

Prepared 2026-09-29 from current source and [harvest records](../../harvest-data/README.md). These steps are a development plan; no new exports or training runs were performed for this guide.

## Objective and current position

Predict crop identity from satellite features. The project already contains polygon and pixel workflows, a training table, saved models and prediction outputs. Existing defaults target Lethbridge; the harvest reports identify Beaverlodge farm fields and different legal parcels.

Use the harvest crop names as additional reference labels after field/crop/year matching. Yield values are primarily for `crop-yield` and must not be passed to the classifier as predictors.

## 1. Establish reproducible baseline evidence

- [ ] Record the input dataset, year, field/pixel counts, class counts, feature families, model settings, split strategy and model output path for each existing run.
- [ ] Evaluate using `--split-strategy grouped`; the training script defaults to random splits. Review `pid` formatting and the default six-character group prefix before treating it as a spatial grouping.
- [ ] Report precision, recall, F1 and support per crop, macro-F1 and a confusion matrix, alongside overall accuracy.
- [ ] Preserve old models before rerunning: training writes to fixed model-type directories and has no output-directory argument.

The saved LightGBM multiclass report lists 90% accuracy on 60 held-out rows, but the metrics JSON alone does not record the split strategy or dataset provenance. The model bundle carries some settings. Do not treat that score as a validated Beaverlodge result or compare runs on different datasets as if only the algorithm changed.

**Completion evidence:** a repeatable baseline with known geography, years, class balance and no group overlap between training and evaluation.

## 2. Prepare farm-record labels

- [ ] Reuse a shared harvest importer and reviewed field registry from the [yield plan](../../crop-yield/docs/NEXT_STEPS.md), rather than creating inconsistent copies.
- [ ] Map clearly named barley, canola, oats, wheat and field peas to existing classes. Preserve varieties as metadata and confirm variety-only entries.
- [ ] Explicitly exclude unsupported crops from the first pilot or add new classes only with sufficient independent examples. The records also contain fescue, ryegrass, rye, faba beans, sunflower and triticale.
- [ ] Resolve mixed-crop parcels and partial-field footprints before assigning labels to pixels or polygon means.
- [ ] Maintain `label_source` (farm report versus AAFC), source references and mapping status separately from predictors.

The current AAFC mapping uses code 122 (pasture/forages) as an **alfalfa proxy**. It does not establish species-level alfalfa identity and must not justify relabeling recorded fescue as alfalfa. No harvest evidence for corn or alfalfa was established by this inspection.

**Completion evidence:** a reviewed field/crop/year label table with spatially defensible labels and explicit unsupported classes.

## 3. Prepare the correct geography and observations

- [ ] Select the actual harvest parcels from the Alberta source, then refine to harvested field/subfield boundaries. Preserve a stable field identity across years.
- [ ] Adapt or replace `00_extract_study_area.py` for this selection. Changing `STUDY_BBOX` alone is insufficient: it also hard-codes a `-113.`/`-112.` longitude text filter, checks only the first coordinate, and assumes one feature per source line.
- [ ] Use distinct input/output names for each geography and year. Preserve previous outputs and checkpoint files; old checkpoints must not be reused for a changed export configuration.
- [ ] Export satellite observations for the same years as the labels. Check availability of each requested AAFC year if using AAFC labels; do not substitute another year's labels silently.
- [ ] Verify precipitation coverage for the target fields before reusing the hard-coded CHIRPS source. Start with a feature set whose observations are valid for the study area.

**Completion evidence:** a small farm-field export with matching year/geometry, valid observations and documented missingness before scaling up.

## 4. Make joins and feature selection explicit

- [ ] Add year-aware joins and validate key cardinality. `04_join_labels_features.py` currently joins only on `pid`; feeding multiple years into it can create wrong matches or multiply rows.
- [ ] Add a predictor allowlist or explicit metadata exclusions before including harvest provenance and geometry fields. The trainer currently treats every remaining column as a feature; its exclusion list does not know the proposed harvest schema.
- [ ] Keep crop text, variety text, harvest totals, observed yields, AAFC label codes and label provenance out of predictor columns.
- [ ] Decide how to handle missing satellite measurements. The polygon join drops rows with any missing value, and the trainer rejects missing/non-numeric features. Log exclusions by crop and year.
- [ ] Retain pixel IDs/coordinates in future exports and inference outputs if mapping is a goal. Current exports use `geometries=False`; current prediction output retains only `pid` plus labels/probabilities.

**Completion evidence:** one correctly labeled row per intended observation, with an audited predictor schema and traceable spatial identity.

## 5. Evaluate transfer to the harvest fields

- [ ] Reserve a reviewed set of harvest fields as an external farm-label test before using those labels for training or tuning.
- [ ] Compare the existing model on that set with a model trained or adapted using separate farm fields. Keep evaluation data out of both training and feature selection.
- [ ] For multiple years, implement explicit year holdouts as well as entire-field holdouts across years. Current grouped splitting is based on `pid` prefixes, not a chronological holdout.
- [ ] Treat thousands of pixels from one field as correlated observations. Report independent fields as well as pixel counts, and include field-level summaries where appropriate.
- [ ] Evaluate supported crop classes separately from unsupported crops. One-vs-rest prediction currently always selects the highest scoring known crop; it has no unknown-class rejection rule.

**Completion evidence:** per-crop performance on farm records from fields/years not used to fit or choose the model, including poor results and insufficient-support classes.

## 6. Compare feature choices fairly

- [ ] Use identical observations and folds to compare optical features against optical + radar and, only where valid, precipitation.
- [ ] Use `07_compare_feature_sets.py` for the existing crop-wise comparison; verify the requested families actually exist in the input. Requesting `s1` does not create missing radar observations.
- [ ] Compare whole-season versus earlier cutoff months if early identification is the goal. All temporal summaries must respect that cutoff.
- [ ] Review wheat/barley confusion and rare-class support before expanding model complexity.

**Completion evidence:** a comparison report that states which input signals improved crop-specific performance under the same evaluation conditions.

## 7. Package useful predictions

- [ ] Save model, feature schema, class definitions, label sources, training years, geography, split settings and evaluation results together.
- [ ] Add probability calibration or an uncertainty/unknown policy using validation data if predictions will guide decisions. One-vs-rest probabilities are separate binary scores and need not sum to one.
- [ ] Join predictions back to field/pixel geometry and year for maps. Keep observed crops separate from predicted crops.
- [ ] If classification becomes an upstream yield input, evaluate the yield model using out-of-sample predicted crop labels as well as known crops, so classification errors are represented.

**Completion evidence:** a reproducible field/pixel prediction table or map with stable IDs, year, crop scores and documented applicability.

## Immediate recommended sequence

Audit baseline → normalize harvest crop labels and field boundaries → export matching farm observations → evaluate on held-out farm fields → improve features and packaging.

See [SCRIPT_GUIDE.md](SCRIPT_GUIDE.md) for the detailed behavior of every Python file in `scripts/` and `preprocessing/`.
