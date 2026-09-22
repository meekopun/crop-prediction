# Raw source data

Place the province-wide source at `quarter_sections.geojson` in this directory.
It is large, stays local, and is ignored by Git.

From `crop-classification/`, run `python scripts/00_extract_study_area.py` to
stream the study-area subset into `data/study_area_quarter_sections.geojson`.
Repeat the command after a checkpoint until it reports completion.
