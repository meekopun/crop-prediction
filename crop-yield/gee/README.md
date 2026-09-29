# Seasonal Earth Engine export (workflow C)

`sentinel2_quarter_sections.js` is the supplied Google Earth Engine Code Editor
script. It is kept here, separate from local Python launchers in `scripts/`.
Moving it did not change its code or configured asset/year placeholders.

1. Upload the intended subset of `../data/raw/alberta_quarter_sections.geojson`
   as a GEE table asset. A small pilot is preferable to a full-province export.
2. Open the JavaScript in the Earth Engine Code Editor.
3. Set `QUARTER_SECTIONS_ASSET` and `QUARTER_ID_FIELD` to real asset values.
   The supplied ID default is uppercase `PID`; inspect the uploaded properties
   rather than assuming that case matches.
4. Review `START_YEAR`, `END_YEAR`, `COVERAGE_YEARS`, coverage thresholds, and
   `LIMIT_FEATURES`. Supplied export years are 2018–2023 and coverage years
   2021–2023; change them deliberately for a different study.
5. Execute a pilot, inspect feature counts and missing values, then start the
   Drive export. Download its CSV into `../data/raw/`.

The output contains polygon/year seasonal optical statistics, AAFC crop
fractions, SMAP, ERA5-Land weather and static terrain/soil context. It is not the
monthly full-field pixel export supplied separately. `yield_prediction.yield_index`
consumes this seasonal column naming convention. Inspect band scaling and cloud
mask behavior before scientific use; organization and JavaScript syntax checks
are not validation of Earth Engine results.

See [the pipeline guide](../docs/PIPELINE_GUIDE.md) for downstream workflows.
