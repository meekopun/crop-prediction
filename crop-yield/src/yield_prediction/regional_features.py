"""Export crop-masked July satellite means for each Alberta small-area region and year.

Pairs with Statistics Canada regional yields (``regional_yields.py``) to give the
season-level model many region-seasons to learn from instead of one farm's few.
For each region and year, July Sentinel-2 SR composites (same cloud mask and
indices as ``harvest_features.py``) are averaged over the pixels the AAFC Annual
Crop Inventory labels as each crop. The reduction runs at 250 m so a whole region
fits in one interactive request; 250 m and 500 m means agreed to about 0.005 NDVI
in a 2021 Peace-region test.

Regions are unions of census divisions (FAO GAUL 2015 level 2); Sentinel-2 SR is
sparse over Alberta before 2019, so years start there.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .regional_yields import ALBERTA_REGION_DIVISIONS, PACKAGE_ROOT

DEFAULT_REGIONAL_FEATURES_PATH = PACKAGE_ROOT / "data/processed/regional/alberta_regional_features.csv"
DEFAULT_REGION_YEARS = tuple(range(2019, 2026))
# Inventory class -> harvest crop name; undifferentiated wheat (140) counts as spring.
CROP_CLASSES = {133: "barley", 136: "oats", 140: "wheat", 146: "wheat", 153: "canola", 162: "field_peas"}
REGIONAL_FEATURES = ["field_NDVI_jul", "field_NDRE_jul", "field_EVI_jul"]
COLUMNS = ["region", "year", "crop", "pixels"] + REGIONAL_FEATURES
SCALE_M = 250
CLOUD_SCL_CLASSES = [3, 8, 9, 10]
MAX_CLOUD_PERCENT = 60


def _division_name(number: int) -> str:
    # GAUL pads one-digit division numbers with an extra space.
    return f"Division No. {number:>2}"


def extract_region_year(region: str, year: int, *, scale: int = SCALE_M) -> list[dict]:
    """Return one row per crop with its July crop-masked means for one region-year."""
    import ee

    divisions = [_division_name(number) for number in ALBERTA_REGION_DIVISIONS[region]]
    geometry = (ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq("ADM1_NAME", "Alberta"))
                .filter(ee.Filter.inList("ADM2_NAME", divisions)).geometry())
    aci = (ee.ImageCollection("AAFC/ACI").filter(ee.Filter.calendarRange(year, year, "year"))
           .first().select("landcover"))
    codes = list(CROP_CLASSES)
    targets = [146 if code == 140 else code for code in codes]
    classes = aci.remap(codes, targets, 0).rename("landcover").selfMask()

    def mask_s2(image):
        clear = image.select("SCL").remap(CLOUD_SCL_CLASSES, [0] * len(CLOUD_SCL_CLASSES), 1).eq(1)
        return image.updateMask(clear).divide(10000).toFloat()

    start = ee.Date.fromYMD(year, 7, 1)
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(geometry)
          .filterDate(start, start.advance(1, "month"))
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_PERCENT))
          .map(mask_s2).select(["B2", "B4", "B5", "B8"]).median())
    peak = ee.Image.cat([
        s2.normalizedDifference(["B8", "B4"]).rename("NDVI"),
        s2.normalizedDifference(["B8", "B5"]).rename("NDRE"),
        s2.expression("2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)",
                      {"nir": s2.select("B8"), "red": s2.select("B4"), "blue": s2.select("B2")}).rename("EVI"),
    ])
    # A copy of NDVI feeds the pixel count; the class band is the group key.
    stack = peak.addBands(peak.select("NDVI").rename("valid")).addBands(classes)
    reducer = (ee.Reducer.mean().repeat(3).combine(ee.Reducer.count(), sharedInputs=False)
               .group(groupField=4, groupName="landcover"))
    groups = stack.reduceRegion(reducer, geometry, scale=scale, maxPixels=1e10, tileScale=4).getInfo()["groups"]
    names = {146: "wheat", **{code: crop for code, crop in CROP_CLASSES.items() if code != 140}}
    return [{"region": region, "year": year, "crop": names[int(group["landcover"])], "pixels": group["count"],
             **dict(zip(REGIONAL_FEATURES, group["mean"]))}
            for group in groups if int(group["landcover"]) in names]


def export_regional_features(output_path: Path | str = DEFAULT_REGIONAL_FEATURES_PATH, *,
                             years: tuple[int, ...] = DEFAULT_REGION_YEARS,
                             regions: tuple[str, ...] = tuple(ALBERTA_REGION_DIVISIONS),
                             project: str | None = None, overwrite: bool = False,
                             scale: int = SCALE_M, progress=print) -> list[dict]:
    """Export every region-year, appending as it goes so an interrupted run resumes."""
    import ee

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if destination.exists() and not overwrite:
        with destination.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    done = {(row["region"], int(row["year"])) for row in rows}
    ee.Initialize(project=project)
    if overwrite or not destination.exists():
        with destination.open("w", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=COLUMNS).writeheader()
        rows, done = [], set()
    for region in regions:
        for year in years:
            if (region, year) in done:
                continue
            new_rows = extract_region_year(region, year, scale=scale)
            with destination.open("a", encoding="utf-8", newline="") as handle:
                csv.DictWriter(handle, fieldnames=COLUMNS).writerows(new_rows)
            rows.extend(new_rows)
            progress(f"{region} {year}: {len(new_rows)} crops")
    return rows
