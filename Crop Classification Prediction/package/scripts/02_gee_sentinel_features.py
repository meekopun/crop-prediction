"""
Step 2: Pull Sentinel-2, Sentinel-1, and precipitation phenology/spectral
features per quarter section from Google Earth Engine.

REQUIRES ONE-TIME SETUP (cannot be done on your behalf -- it needs your own
Google account and a browser):
    1. pip install earthengine-api
    2. Register a Google Cloud project for Earth Engine at
       https://console.cloud.google.com/earth-engine (free for research/
       education/nonprofit use).
    3. Run `earthengine authenticate` once in a terminal (or, in a notebook,
       `ee.Authenticate()`), and sign in with the account tied to that project.

What this script does:
    - Loads the sampled quarter sections from `01_select_training_sample.py`.
    - For each growing-season month (Apr-Oct), builds a cloud-masked
      Sentinel-2 surface-reflectance median composite over the study area.
    - Also builds a Sentinel-1 GRD C-band median composite for the same month
      using dual-polarization VV/VH backscatter.
    - Also sums CHIRPS daily precipitation to a monthly total for the same
      month.
    - Computes spectral indices per composite, including NDYI (yellowness --
      the "is it yellow in July?" canola signal), NDVI, GNDVI, EVI, NDRE, and
      an SWIR moisture/senescence ratio, plus a VV-VH radar separation term.
    - Runs reduceRegions to get the mean of every band/index per polygon per
      month, producing one row per quarter section with a wide set of
      phenological columns (e.g. NDYI_jul, NDVI_aug, ...).
    - Exports the result as a CSV, either directly (small samples) or via a
      Google Drive export task (recommended -- avoids interactive timeouts).

Usage:
    python 02_gee_sentinel_features.py --year 2024 --project YOUR_GCP_PROJECT_ID
"""
import argparse
import time

import ee
import pandas as pd

from common import (
    GROWING_SEASON_MONTHS,
    MONTH_NAMES,
    PRECIP_BANDS,
    S1_BANDS,
    S2_BANDS,
    load_geojson,
    quarter_section_id,
)

DATA_DIR = "data"

# Cloud/quality masking + scaling for the harmonized S2 SR collection.
# QA60 stopped carrying real cloud polygons in 2022; from 2024-02-28 onward
# it's rebuilt from MSK_CLASSI. SCL gives an independent, always-available
# per-pixel scene classification, so we mask on both for robustness.
CLOUD_SCL_CLASSES = [3, 8, 9, 10]  # cloud shadow, med/high prob cloud, cirrus


def mask_s2(image):
    scl = image.select("SCL")
    scl_mask = scl.remap(CLOUD_SCL_CLASSES, [0] * len(CLOUD_SCL_CLASSES), 1).eq(1)
    return image.updateMask(scl_mask).divide(10000).copyProperties(
        image, image.propertyNames()
    )


def add_indices(image):
    b2, b3, b4 = image.select("B2"), image.select("B3"), image.select("B4")
    b5, b8, b11, b12 = (image.select("B5"), image.select("B8"),
                        image.select("B11"), image.select("B12"))

    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndyi = image.normalizedDifference(["B3", "B2"]).rename("NDYI")  # yellowness (canola)
    gndvi = image.normalizedDifference(["B8", "B3"]).rename("GNDVI")
    ndre = image.normalizedDifference(["B8", "B5"]).rename("NDRE")
    ndwi = image.normalizedDifference(["B8", "B11"]).rename("NDWI")
    evi = image.expression(
        "2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)",
        {"nir": b8, "red": b4, "blue": b2},
    ).rename("EVI")
    swir_ratio = b11.divide(b12).rename("SWIR_RATIO")

    return image.addBands([ndvi, ndyi, gndvi, ndre, ndwi, evi, swir_ratio])


def add_s1_indices(image):
    vv = image.select("VV")
    vh = image.select("VH")
    vv_minus_vh = vv.subtract(vh).rename("VV_minus_VH")
    return image.addBands([vv_minus_vh])


def monthly_s2_composite(aoi, year, month):
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
        .map(mask_s2)
    )
    composite = col.select(S2_BANDS + ["SCL"]).median()
    composite = add_indices(composite)
    return composite.set("month", month)


def monthly_s1_composite(aoi, year, month):
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")
    col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("resolution_meters", 10))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )
    composite = col.select(S1_BANDS).median()
    composite = add_s1_indices(composite)
    return composite.set("month", month)


def monthly_precip_composite(aoi, year, month):
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")
    total = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterBounds(aoi)
        .filterDate(start, end)
        .select("precipitation")
        .sum()
        .rename(PRECIP_BANDS)
    )
    return total.set("month", month)


def monthly_composite(aoi, year, month):
    s2 = monthly_s2_composite(aoi, year, month)
    s1 = monthly_s1_composite(aoi, year, month)
    precip = monthly_precip_composite(aoi, year, month)
    return s2.addBands(s1).addBands(precip).set("month", month)


def build_feature_collection(geojson_path):
    fc_dict = load_geojson(geojson_path)
    # Keep only the id we need as a property so the exported table stays small.
    features = []
    for feat in fc_dict["features"]:
        props = {"pid": quarter_section_id(feat)}
        features.append(ee.Feature(feat["geometry"], props))
    return ee.FeatureCollection(features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=f"{DATA_DIR}/training_sample.geojson")
    parser.add_argument("--year", type=int, default=2024,
                         help="Growing season year (match your AAFC label year)")
    parser.add_argument("--project",
                         help="Override the default saved with earthengine set_project")
    parser.add_argument("--scale", type=int, default=10, help="Reduction scale in meters")
    parser.add_argument("--export", choices=["drive", "local"], default="drive",
                         help="'drive' starts a Drive export task (recommended); "
                              "'local' pulls results directly via getInfo (only "
                              "safe for small samples, <~500 polygons)")
    parser.add_argument("--out-name", default="sentinel2_features")
    args = parser.parse_args()

    ee.Initialize(project=args.project)

    fc = build_feature_collection(args.input)
    aoi = fc.geometry().bounds()

    reduced = fc
    for month in GROWING_SEASON_MONTHS:
        composite = monthly_composite(aoi, args.year, month)
        band_names = composite.bandNames().getInfo()
        suffix = f"_{MONTH_NAMES[month]}"
        renamed = composite.select(band_names, [b + suffix for b in band_names])
        reduced = renamed.reduceRegions(
            collection=reduced, reducer=ee.Reducer.mean(), scale=args.scale
        )
        print(f"Queued reduceRegions for {MONTH_NAMES[month]} {args.year}")

    if args.export == "drive":
        task = ee.batch.Export.table.toDrive(
            collection=reduced,
            description=args.out_name,
            fileFormat="CSV",
        )
        task.start()
        print(f"Started Drive export task '{args.out_name}'. Check progress with:")
        print("  ee.batch.Task.list() in a Python shell, or the Earth Engine "
              "'Tasks' tab at https://code.earthengine.google.com/tasks")
        while task.active():
            time.sleep(15)
            print("  ...still running")
        status = task.status()
        print("Export finished with status:", status["state"])
        if status["state"] == "COMPLETED":
            print(f"CSV saved to Google Drive as '{args.out_name}.csv'. "
                  f"Download it into {DATA_DIR}/ before running step 04.")
    else:
        info = reduced.getInfo()
        rows = [f["properties"] for f in info["features"]]
        df = pd.DataFrame(rows)
        out_path = f"{DATA_DIR}/{args.out_name}.csv"
        df.to_csv(out_path, index=False)
        print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
