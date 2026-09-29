# Supplied reference inputs

These original CSVs were moved from `data/raw/` to the pipeline guide's
`reference_inputs/` directory without changing their contents. They stay local
and are ignored by Git. Generated replacements/results belong in `data/processed/`.

| File | Role |
| --- | --- |
| `Crop Yield Data - Crop Data 2021-2023-2.csv` | 15 observed-yield reference rows with parcel, crop, year, yield/unit columns and coordinates; input to the still-missing full-field preprocessor |
| `pixel_level_all_crop_training_features_2021_2023.csv` | Supplied prepared pixel features/lookup: 749 rows, eight `row_id` groups, four crop labels; default input to the three pixel wrappers |

The raw monthly full-field export remains in `../data/raw/`. Prepared targets
include `yield_bu_ac`; fields such as `yield_original` and `yield_original_unit`
retain source context. Availability does not establish that every source-unit
conversion or spatial match has been independently validated.

The six original 2020–2025 harvest Excel workbooks are separate, at repository
root `harvest-data/`; they have not been merged into these reference CSVs.
