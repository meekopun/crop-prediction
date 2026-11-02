"""Export crop-masked monthly satellite means for each Alberta small-area region and year.

Pairs with Statistics Canada regional yields (``regional_yields.py``) to give the
season-level model many region-seasons to learn from instead of one farm's few.
For each region and year, a monthly Sentinel-2 SR composite (same cloud mask and
indices as ``harvest_features.py``) is averaged over the pixels the AAFC Annual
Crop Inventory labels as each crop. The reduction runs at 250 m so a whole region
fits in one interactive request; 250 m and 500 m means agreed to about 0.005 NDVI
in a 2021 Peace-region test.

Each month has its own file so finished months are kept; July, the original
export, keeps the unsuffixed name. Regions are unions of census divisions (FAO
GAUL 2015 level 2); Sentinel-2 SR is sparse over Alberta before 2019, so years
start there.

The same reduction over a fixed rectangle around the farm (``LOCAL_AREA_BBOX``,
30 m) gives a local season reading that does not depend on which fields the
farm grew a crop on, or on whether the crop map recognises them.

``sensor="landsat"`` uses Landsat 8/9 Collection 2 surface reflectance instead,
from 2013, so the regional model can learn from more seasons. Landsat has no
red-edge band, so its features are NDVI and EVI only; files add ``_landsat``.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .regional_yields import ALBERTA_REGION_DIVISIONS, PACKAGE_ROOT

DEFAULT_REGIONAL_FEATURES_PATH = PACKAGE_ROOT / "data/processed/regional/alberta_regional_features.csv"
DEFAULT_LOCAL_FEATURES_PATH = PACKAGE_ROOT / "data/processed/harvest/satellite/local_area_features.csv"
DEFAULT_REGION_YEARS = tuple(range(2019, 2026))
# The Beaverlodge quarters' extent (2020-2025) plus about 10 km; fixed so every year covers the same land.
LOCAL_AREA_BBOX = (-119.64, 54.98, -119.04, 55.36)
LOCAL_SCALE_M = 30
LOCAL_MONTHS = ("jul", "aug")
# Inventory class -> harvest crop name; undifferentiated wheat (140) counts as spring.
CROP_CLASSES = {133: "barley", 136: "oats", 140: "wheat", 146: "wheat", 153: "canola", 162: "field_peas"}
MONTHS = {"jun": 6, "jul": 7, "aug": 8}
INDICES = ["NDVI", "NDRE", "EVI"]
SENSOR_INDICES = {"s2": INDICES, "landsat": ["NDVI", "EVI"]}
LANDSAT_REGION_YEARS = tuple(range(2013, 2026))
# Landsat 8/9 Collection 2 Level 2: (collection, blue, red, NIR) surface reflectance bands.
LANDSAT_COLLECTIONS = [("LANDSAT/LC08/C02/T1_L2", "SR_B2", "SR_B4", "SR_B5"),
                       ("LANDSAT/LC09/C02/T1_L2", "SR_B2", "SR_B4", "SR_B5")]
# QA_PIXEL bits 1-5: dilated cloud, cirrus, cloud, cloud shadow, snow.
LANDSAT_MASK_BITS = 0b111110


def month_features(month: str, sensor: str = "s2") -> list[str]:
    """Feature columns for one month, named like the farm table's ``field_<index>_<month>``."""
    return [f"field_{index}_{month}" for index in SENSOR_INDICES[sensor]]


def regional_features_path(month: str, sensor: str = "s2") -> Path:
    suffix = ("_landsat" if sensor == "landsat" else "") + ("" if month == "jul" and sensor == "s2" else f"_{month}")
    return DEFAULT_REGIONAL_FEATURES_PATH.with_name(f"{DEFAULT_REGIONAL_FEATURES_PATH.stem}{suffix}.csv")


REGIONAL_FEATURES = month_features("jul")
SCALE_M = 250
CLOUD_SCL_CLASSES = [3, 8, 9, 10]
MAX_CLOUD_PERCENT = 60


def _division_name(number: int) -> str:
    # GAUL pads one-digit division numbers with an extra space.
    return f"Division No. {number:>2}"


def extract_region_year(region: str, year: int, *, month: str = "jul", scale: int = SCALE_M,
                        sensor: str = "s2") -> list[dict]:
    """Return one row per crop with its crop-masked means for one region-year and month."""
    import ee

    divisions = [_division_name(number) for number in ALBERTA_REGION_DIVISIONS[region]]
    geometry = (ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq("ADM1_NAME", "Alberta"))
                .filter(ee.Filter.inList("ADM2_NAME", divisions)).geometry())
    return [{"region": region, "year": year, **row}
            for row in _crop_masked_means(geometry, year, month, scale, sensor)]


def _s2_composite(geometry, start, end):
    """Monthly median of cloud-masked Sentinel-2 SR as blue, red, red-edge and NIR reflectance."""
    import ee

    def mask_s2(image):
        clear = image.select("SCL").remap(CLOUD_SCL_CLASSES, [0] * len(CLOUD_SCL_CLASSES), 1).eq(1)
        return image.updateMask(clear).divide(10000).toFloat()

    return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(geometry).filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_PERCENT))
            .map(mask_s2).select(["B2", "B4", "B5", "B8"], ["blue", "red", "rededge", "nir"]).median())


def _landsat_composite(geometry, start, end):
    """Monthly median of cloud-masked Landsat 8/9 SR as blue, red and NIR reflectance."""
    import ee

    def prepare(blue, red, nir):
        def apply(image):
            clear = (image.select("QA_PIXEL").bitwiseAnd(LANDSAT_MASK_BITS).eq(0)
                     .And(image.select("QA_RADSAT").eq(0)))
            return (image.select([blue, red, nir], ["blue", "red", "nir"]).multiply(0.0000275).add(-0.2)
                    .updateMask(clear).toFloat())
        return apply

    # A fully masked image keeps the bands when a month has no usable scene.
    images = ee.ImageCollection([ee.Image.constant([0, 0, 0]).rename(["blue", "red", "nir"])
                                 .updateMask(0).toFloat()])
    for name, blue, red, nir in LANDSAT_COLLECTIONS:
        images = images.merge(ee.ImageCollection(name).filterBounds(geometry).filterDate(start, end)
                              .filter(ee.Filter.lt("CLOUD_COVER", MAX_CLOUD_PERCENT))
                              .map(prepare(blue, red, nir)))
    return images.median()


def _crop_masked_means(geometry, year: int, month: str, scale: int, sensor: str = "s2") -> list[dict]:
    """One row per crop inside ``geometry``: pixel count and the month's crop-masked index means."""
    import ee

    features = month_features(month, sensor)
    aci = (ee.ImageCollection("AAFC/ACI").filter(ee.Filter.calendarRange(year, year, "year"))
           .first().select("landcover"))
    codes = list(CROP_CLASSES)
    targets = [146 if code == 140 else code for code in codes]
    classes = aci.remap(codes, targets, 0).rename("landcover").selfMask()

    start = ee.Date.fromYMD(year, MONTHS[month], 1)
    end = start.advance(1, "month")
    image = _s2_composite(geometry, start, end) if sensor == "s2" else _landsat_composite(geometry, start, end)
    indices = {
        "NDVI": lambda: image.normalizedDifference(["nir", "red"]),
        "NDRE": lambda: image.normalizedDifference(["nir", "rededge"]),
        "EVI": lambda: image.expression("2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)",
                                        {"nir": image.select("nir"), "red": image.select("red"),
                                         "blue": image.select("blue")}),
    }
    index_names = SENSOR_INDICES[sensor]
    peak = ee.Image.cat([indices[name]().rename(name) for name in index_names])
    # A copy of NDVI feeds the pixel count; the class band is the group key.
    stack = peak.addBands(peak.select("NDVI").rename("valid")).addBands(classes)
    reducer = (ee.Reducer.mean().repeat(len(index_names)).combine(ee.Reducer.count(), sharedInputs=False)
               .group(groupField=len(index_names) + 1, groupName="landcover"))
    groups = stack.reduceRegion(reducer, geometry, scale=scale, maxPixels=1e10, tileScale=4).getInfo()["groups"]
    names = {146: "wheat", **{code: crop for code, crop in CROP_CLASSES.items() if code != 140}}
    return [{"crop": names[int(group["landcover"])], "pixels": group["count"], **dict(zip(features, group["mean"]))}
            for group in groups if int(group["landcover"]) in names]


def extract_local_year(year: int, *, months: tuple[str, ...] = LOCAL_MONTHS,
                       bbox: tuple[float, ...] = LOCAL_AREA_BBOX, scale: int = LOCAL_SCALE_M) -> list[dict]:
    """One row per crop with crop-masked means for every month over the fixed area around the farm."""
    import ee

    geometry = ee.Geometry.Rectangle(list(bbox))
    rows: dict[str, dict] = {}
    for month in months:
        for row in _crop_masked_means(geometry, year, month, scale):
            entry = rows.setdefault(row["crop"], {"year": year, "crop": row["crop"]})
            entry[f"pixels_{month}"] = row["pixels"]
            entry.update({key: value for key, value in row.items() if key.startswith("field_")})
    return list(rows.values())


def export_local_features(output_path: Path | str = DEFAULT_LOCAL_FEATURES_PATH, *,
                          years: tuple[int, ...] = DEFAULT_REGION_YEARS, months: tuple[str, ...] = LOCAL_MONTHS,
                          project: str | None = None, overwrite: bool = False,
                          scale: int = LOCAL_SCALE_M, progress=print) -> list[dict]:
    """Export the local area year by year, appending as it goes so an interrupted run resumes."""
    import ee

    columns = (["year", "crop"] + [f"pixels_{month}" for month in months]
               + [column for month in months for column in month_features(month)])
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if destination.exists() and not overwrite:
        with destination.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != columns:
                raise ValueError(f"{destination} has other columns; pass overwrite to start again.")
            rows = list(reader)
    ee.Initialize(project=project)
    if overwrite or not destination.exists():
        with destination.open("w", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=columns).writeheader()
        rows = []
    done = {int(row["year"]) for row in rows}
    for year in years:
        if year in done:
            continue
        new_rows = extract_local_year(year, months=months, scale=scale)
        with destination.open("a", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=columns, restval="").writerows(new_rows)
        rows.extend(new_rows)
        progress(f"local area {year}: {len(new_rows)} crops")
    return rows


def export_regional_features(output_path: Path | str | None = None, *, month: str = "jul",
                             years: tuple[int, ...] | None = None,
                             regions: tuple[str, ...] = tuple(ALBERTA_REGION_DIVISIONS),
                             project: str | None = None, overwrite: bool = False,
                             scale: int = SCALE_M, sensor: str = "s2", progress=print) -> list[dict]:
    """Export every region-year for one month, appending as it goes so an interrupted run resumes.

    ``years`` defaults to 2019-2025 for Sentinel-2 and 2013-2025 for Landsat.
    """
    import ee

    if years is None:
        years = LANDSAT_REGION_YEARS if sensor == "landsat" else DEFAULT_REGION_YEARS
    columns = ["region", "year", "crop", "pixels"] + month_features(month, sensor)
    destination = Path(output_path) if output_path is not None else regional_features_path(month, sensor)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if destination.exists() and not overwrite:
        with destination.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    done = {(row["region"], int(row["year"])) for row in rows}
    ee.Initialize(project=project)
    if overwrite or not destination.exists():
        with destination.open("w", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=columns).writeheader()
        rows, done = [], set()
    for region in regions:
        for year in years:
            if (region, year) in done:
                continue
            new_rows = extract_region_year(region, year, month=month, scale=scale, sensor=sensor)
            with destination.open("a", encoding="utf-8", newline="") as handle:
                csv.DictWriter(handle, fieldnames=columns).writerows(new_rows)
            rows.extend(new_rows)
            progress(f"{region} {year} {month} {sensor}: {len(new_rows)} crops")
    return rows
