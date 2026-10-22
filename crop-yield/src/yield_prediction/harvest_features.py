"""Export Sentinel-2/Sentinel-1 features for matched harvest quarter sections.

Each pilot-eligible ``(pid, year)`` from ``boundary_matches.csv`` gets monthly
(Apr-Oct) composite means twice: over the whole quarter (``quarter_*``) and over
only the pixels the AAFC Annual Crop Inventory labels as the reported crop
(``crop_*``). Many harvested fields cover part of their quarter, so the crop-masked
columns describe the harvested crop more closely. Band, index and month names
mirror crop-classification's ``scripts/02_gee_sentinel_features.py``.

Earth Engine is imported lazily; only the export step needs it and a login.
"""

from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path

from .harvest import DEFAULT_OUTPUT_DIR

DEFAULT_MATCHES_PATH = DEFAULT_OUTPUT_DIR / "boundary_matches.csv"
DEFAULT_BOUNDARIES_PATH = DEFAULT_OUTPUT_DIR / "field_boundaries.geojson"
DEFAULT_FEATURE_DIR = DEFAULT_OUTPUT_DIR / "satellite"
DEFAULT_YEARS = tuple(range(2020, 2026))

# AAFC Annual Crop Inventory classes for each normalized harvest crop. Forage
# seed crops fall under Pasture / Forages.
CROP_ACI_CODES = {
    "barley": (133,), "canola": (153,), "faba_beans": (168,), "fescue": (122,),
    "field_peas": (162,), "oats": (136,), "rye": (137,), "ryegrass": (122,),
    "sunflower": (157,), "tall_fescue": (122,), "triticale": (139,),
    "wheat": (140, 145, 146),
}
MONTHS = {4: "apr", 5: "may", 6: "jun", 7: "jul", 8: "aug", 9: "sep", 10: "oct"}
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
S2_INDICES = ["NDVI", "NDYI", "GNDVI", "NDRE", "NDWI", "EVI", "SWIR_RATIO"]
S1_BANDS = ["VV", "VH", "VV_minus_VH"]
FEATURE_BANDS = S2_BANDS + S2_INDICES + S1_BANDS
CLOUD_SCL_CLASSES = [3, 8, 9, 10]  # cloud shadow, medium/high cloud, cirrus
MAX_CLOUD_PERCENT = 60
# Painted into an image so one reduction can mask each quarter to its own crop.
CROP_INDEX = {crop: index for index, crop in enumerate(sorted(CROP_ACI_CODES), start=1)}
ID_COLUMNS = ["pid", "year", "crop", "aci_codes", "operation_ids"]


@dataclass(frozen=True)
class FeatureUnit:
    pid: str
    year: int
    crop: str
    operation_ids: tuple[str, ...]
    geometry: dict

    @property
    def aci_codes(self) -> tuple[int, ...]:
        return CROP_ACI_CODES.get(self.crop, ())


def value_columns() -> list[str]:
    columns = ["aci_crop_fraction"]
    for month in MONTHS.values():
        columns.append(f"s2_valid_{month}")
        columns.extend(f"quarter_{band}_{month}" for band in FEATURE_BANDS)
        columns.extend(f"crop_{band}_{month}" for band in FEATURE_BANDS)
    return columns


def feature_columns() -> list[str]:
    return ID_COLUMNS + value_columns()


def load_feature_units(matches_path: Path | str = DEFAULT_MATCHES_PATH,
                       boundaries_path: Path | str = DEFAULT_BOUNDARIES_PATH) -> list[FeatureUnit]:
    """Return one unit per pilot-eligible quarter section and year.

    Eligible rows never share a quarter-year with another crop, so each unit has
    exactly one crop; a half section contributes a unit for each of its quarters.
    """
    with Path(boundaries_path).open(encoding="utf-8") as handle:
        geometries = {feature["properties"]["pid"]: feature["geometry"]
                      for feature in json.load(handle)["features"]}
    crops: dict[tuple[str, int], set[str]] = defaultdict(set)
    operations: dict[tuple[str, int], list[str]] = defaultdict(list)
    with Path(matches_path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["boundary_pilot_eligible"] != "True":
                continue
            for pid in row["pids"].split("|"):
                key = (pid, int(row["year"]))
                crops[key].add(row["crop"])
                operations[key].append(row["operation_id"])
    units = []
    for (pid, year), names in sorted(crops.items()):
        if len(names) != 1:
            raise ValueError(f"{pid} {year} has several eligible crops: {sorted(names)}")
        if pid not in geometries:
            raise ValueError(f"{pid} is eligible but absent from {boundaries_path}")
        units.append(FeatureUnit(pid, year, names.pop(), tuple(operations[(pid, year)]), geometries[pid]))
    return units


def _feature_image(ee, collection, year: int):
    """Stack monthly quarter-wide and crop-masked bands plus coverage bands."""
    aoi = collection.geometry().bounds()
    painted = ee.Image(0).paint(collection, "crop_index")
    aci = (ee.ImageCollection("AAFC/ACI").filter(ee.Filter.calendarRange(year, year, "year"))
           .first().select("landcover"))
    match = ee.Image(0)
    for crop, index in CROP_INDEX.items():
        codes = list(CROP_ACI_CODES[crop])
        match = match.Or(painted.eq(index).And(aci.remap(codes, [1] * len(codes), 0)))

    def mask_s2(image):
        clear = image.select("SCL").remap(CLOUD_SCL_CLASSES, [0] * len(CLOUD_SCL_CLASSES), 1).eq(1)
        return image.updateMask(clear).divide(10000).toFloat()

    def masked_placeholder(bands):
        # Keeps band names when a month has no scenes; every pixel stays masked.
        return ee.Image.constant([0] * len(bands)).rename(bands).float().updateMask(ee.Image(0))

    image = match.rename("aci_crop_fraction").toFloat()
    for month, name in MONTHS.items():
        start = ee.Date.fromYMD(year, month, 1)
        end = start.advance(1, "month")
        s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(aoi).filterDate(start, end)
              .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_PERCENT)).map(mask_s2)
              .select(S2_BANDS).merge(ee.ImageCollection([masked_placeholder(S2_BANDS)])).median())
        indices = [
            s2.normalizedDifference(["B8", "B4"]).rename("NDVI"),
            s2.normalizedDifference(["B3", "B2"]).rename("NDYI"),
            s2.normalizedDifference(["B8", "B3"]).rename("GNDVI"),
            s2.normalizedDifference(["B8", "B5"]).rename("NDRE"),
            s2.normalizedDifference(["B8", "B11"]).rename("NDWI"),
            s2.expression("2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)",
                          {"nir": s2.select("B8"), "red": s2.select("B4"),
                           "blue": s2.select("B2")}).rename("EVI"),
            s2.select("B11").divide(s2.select("B12")).rename("SWIR_RATIO"),
        ]
        s1 = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(aoi).filterDate(start, end)
              .filter(ee.Filter.eq("instrumentMode", "IW"))
              .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
              .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
              .select(["VV", "VH"]).map(lambda image: image.toFloat())
              .merge(ee.ImageCollection([masked_placeholder(["VV", "VH"])])).median())
        s1 = s1.addBands(s1.select("VV").subtract(s1.select("VH")).rename("VV_minus_VH"))
        monthly = s2.addBands(indices).addBands(s1).select(FEATURE_BANDS).toFloat()
        valid = ee.Image(1).updateMask(s2.select("B4").mask()).unmask(0).rename(f"s2_valid_{name}")
        image = image.addBands([
            valid.toFloat(),
            monthly.rename([f"quarter_{band}_{name}" for band in FEATURE_BANDS]),
            monthly.updateMask(match).rename([f"crop_{band}_{name}" for band in FEATURE_BANDS]),
        ])
    return image


def extract_year(units: list[FeatureUnit], year: int, *, scale: int = 10) -> list[dict]:
    """Reduce one year's feature stack over its units and return one row per unit."""
    import ee

    year_units = [unit for unit in units if unit.year == year]
    if not year_units:
        return []
    if ee.ImageCollection("AAFC/ACI").filter(ee.Filter.calendarRange(year, year, "year")).size().getInfo() == 0:
        raise RuntimeError(f"No AAFC crop inventory image for {year}; crop masks cannot be built.")
    collection = ee.FeatureCollection([
        ee.Feature(unit.geometry, {"pid": unit.pid, "crop_index": CROP_INDEX.get(unit.crop, 0)})
        for unit in year_units
    ])
    image = _feature_image(ee, collection, year)
    reduced = image.reduceRegions(collection=collection, reducer=ee.Reducer.mean(), scale=scale)
    values = {feature["properties"]["pid"]: feature["properties"]
              for feature in reduced.getInfo()["features"]}
    rows = []
    for unit in year_units:
        row = {column: values[unit.pid].get(column) for column in value_columns()}
        row.update({"pid": unit.pid, "year": unit.year, "crop": unit.crop,
                    "aci_codes": "|".join(str(code) for code in unit.aci_codes),
                    "operation_ids": "|".join(unit.operation_ids)})
        rows.append(row)
    return rows


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=feature_columns(), restval="")
        writer.writeheader()
        writer.writerows(rows)


def export_harvest_features(matches_path: Path | str = DEFAULT_MATCHES_PATH,
                            boundaries_path: Path | str = DEFAULT_BOUNDARIES_PATH,
                            output_dir: Path | str = DEFAULT_FEATURE_DIR,
                            *, years: tuple[int, ...] = DEFAULT_YEARS, scale: int = 10,
                            project: str | None = None, overwrite: bool = False,
                            progress=print) -> dict:
    """Export each year to its own CSV (skipping finished years), then combine them."""
    import ee

    units = load_feature_units(matches_path, boundaries_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    ee.Initialize(project=project)
    written = {}
    for year in years:
        path = destination / f"harvest_features_{year}.csv"
        if path.exists() and not overwrite:
            progress(f"{year}: kept existing {path.name}")
        else:
            rows = extract_year(units, year, scale=scale)
            _write_rows(path, rows)
            progress(f"{year}: wrote {len(rows)} quarter sections to {path.name}")
        written[year] = path
    combined = []
    for year, path in written.items():
        with path.open(encoding="utf-8", newline="") as handle:
            combined.extend(csv.DictReader(handle))
    _write_rows(destination / "harvest_satellite_features.csv", combined)
    manifest = {
        "schema_version": 1,
        "created_on": date.today().isoformat(),
        "matches_path": str(Path(matches_path).resolve()),
        "boundaries_path": str(Path(boundaries_path).resolve()),
        "years": list(years),
        "unit_count": len(combined),
        "units_by_year": {str(year): sum(1 for unit in units if unit.year == year) for year in years},
        "scale_m": scale,
        "months": list(MONTHS.values()),
        "sentinel2": {"collection": "COPERNICUS/S2_SR_HARMONIZED", "max_cloudy_pixel_percent": MAX_CLOUD_PERCENT,
                      "masked_scl_classes": CLOUD_SCL_CLASSES, "composite": "monthly median",
                      "reflectance_scale": "SR / 10000"},
        "sentinel1": {"collection": "COPERNICUS/S1_GRD", "mode": "IW", "polarisations": ["VV", "VH"],
                      "composite": "monthly median", "units": "dB"},
        "crop_mask": {"collection": "AAFC/ACI", "band": "landcover",
                      "codes": {crop: list(codes) for crop, codes in CROP_ACI_CODES.items()}},
        "columns": {"quarter_*": "mean over the whole quarter section",
                    "crop_*": "mean over pixels the crop inventory labels as the reported crop",
                    "s2_valid_<month>": "share of quarter pixels with a cloud-free Sentinel-2 composite",
                    "aci_crop_fraction": "share of quarter pixels labelled as the reported crop"},
    }
    (destination / "harvest_features_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                               encoding="utf-8")
    return manifest
