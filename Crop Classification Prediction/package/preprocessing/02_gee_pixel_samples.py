"""
Step 2 (pixel mode): Export per-pixel Sentinel-2, Sentinel-1, and
precipitation features, optionally joined to AAFC pixel labels, from Google
Earth Engine.

This is the pixel-level alternative to `02_gee_sentinel_features.py`. Instead
of averaging over each polygon, it samples one training row per 30 m pixel
inside the input polygons. When `--include-labels` is enabled, it also pulls
the AAFC Annual Crop Inventory label for the same pixel directly from Earth
Engine, so steps 03-04 are not needed for this training path.

For larger runs, use Drive export with polygon batching so Earth Engine does
not need to materialize the entire sample table in one task.

Usage:
    python 02_gee_pixel_samples.py --year 2024 --project YOUR_GCP_PROJECT_ID
"""
import argparse
import json
import os
import time
from pathlib import Path

import ee
import pandas as pd

from common import (
    AAFC_CODE_TO_CROP,
    DATA_DIR,
    GROWING_SEASON_MONTHS,
    MONTH_NAMES,
    PRECIP_BANDS,
    S1_BANDS,
    STATE_DIR,
    S2_BANDS,
    load_geojson,
    load_tracker,
    quarter_section_id,
    save_tracker,
)
CLOUD_SCL_CLASSES = [3, 8, 9, 10]


def mask_s2(image):
    scl = image.select("SCL")
    scl_mask = scl.remap(CLOUD_SCL_CLASSES, [0] * len(CLOUD_SCL_CLASSES), 1).eq(1)
    return image.updateMask(scl_mask).divide(10000).copyProperties(
        image, image.propertyNames()
    )


def add_indices(image):
    b2, b3, b4 = image.select("B2"), image.select("B3"), image.select("B4")
    b5, b8, b11, b12 = (
        image.select("B5"),
        image.select("B8"),
        image.select("B11"),
        image.select("B12"),
    )

    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndyi = image.normalizedDifference(["B3", "B2"]).rename("NDYI")
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
    return add_indices(col.select(S2_BANDS + ["SCL"]).median()).set("month", month)


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
    return add_s1_indices(col.select(S1_BANDS).median()).set("month", month)


def monthly_precip_composite(aoi, year, month):
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")
    return (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterBounds(aoi)
        .filterDate(start, end)
        .select("precipitation")
        .sum()
        .rename(PRECIP_BANDS)
        .set("month", month)
    )


def monthly_composite(aoi, year, month):
    s2 = monthly_s2_composite(aoi, year, month)
    s1 = monthly_s1_composite(aoi, year, month)
    precip = monthly_precip_composite(aoi, year, month)
    return s2.addBands(s1).addBands(precip).set("month", month)


def build_feature_collection(geojson_path):
    fc_dict = load_geojson(geojson_path)
    features = []
    for feat in fc_dict["features"]:
        props = {"pid": quarter_section_id(feat)}
        features.append(ee.Feature(feat["geometry"], props))
    return ee.FeatureCollection(features)


def feature_collection_batches(fc, batch_size):
    total = fc.size().getInfo()
    if batch_size <= 0 or total <= batch_size:
        return [(0, fc, total)]

    batches = []
    for offset in range(0, total, batch_size):
        size = min(batch_size, total - offset)
        batch_fc = ee.FeatureCollection(fc.toList(size, offset))
        batches.append((offset // batch_size, batch_fc, size))
    return batches


def default_checkpoint_path(out_name):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return str(STATE_DIR / f"{out_name}_checkpoint.json")


def load_checkpoint(path):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {"completed_batches": [], "failed_batches": {}, "batches": {}}
    with open(path) as f:
        return json.load(f)


def save_checkpoint(path, state):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def update_batch_state(state, batch_idx, **updates):
    batch_key = str(batch_idx)
    batch_state = state.setdefault("batches", {}).setdefault(batch_key, {})
    batch_state.update(updates)


def update_global_tracker(args, checkpoint_path, batch_name=None, batch_status=None):
    tracker = load_tracker()
    export_key = args.out_name
    export_state = tracker["pixel_exports"].setdefault(
        export_key,
        {
            "input": args.input,
            "year": args.year,
            "label_year": args.label_year or args.year if args.include_labels else None,
            "scale": args.scale,
            "polygon_batch_size": args.polygon_batch_size,
            "include_labels": args.include_labels,
            "keep_other": args.keep_other,
            "checkpoint": checkpoint_path,
            "batches": {},
        },
    )
    if batch_name is not None:
        export_state["batches"][batch_name] = batch_status
    save_tracker(tracker)


def build_stacked_image(aoi, year):
    stacked = None
    for month in GROWING_SEASON_MONTHS:
        composite = monthly_composite(aoi, year, month)
        band_names = composite.bandNames().getInfo()
        suffix = f"_{MONTH_NAMES[month]}"
        renamed = composite.select(band_names, [b + suffix for b in band_names])
        stacked = renamed if stacked is None else stacked.addBands(renamed)
        print(f"Prepared {MONTH_NAMES[month]} {year} composite")
    return stacked


def add_aafc_labels(sample_fc, year, keep_other):
    mapping = ee.Dictionary({str(k): v for k, v in AAFC_CODE_TO_CROP.items()})

    labeled = sample_fc.map(
        lambda feat: feat.set(
            "crop_label",
            ee.String(
                mapping.get(ee.Number(feat.get("aafc_code")).format("%.0f"), "other")
            ),
        )
    )
    if keep_other:
        return labeled
    return labeled.filter(ee.Filter.neq("crop_label", "other"))


def export_or_download(fc, out_name, export_mode):
    if export_mode == "drive":
        task = ee.batch.Export.table.toDrive(
            collection=fc,
            description=out_name,
            fileFormat="CSV",
        )
        task.start()
        print(f"Started Drive export task '{out_name}'.")
        while task.active():
            time.sleep(15)
            print("  ...still running")
        status = task.status()
        print("Export finished with status:", status["state"])
        if status["state"] == "COMPLETED":
            print(f"CSV saved to Google Drive as '{out_name}.csv'.")
        return status

    info = fc.getInfo()
    rows = [f["properties"] for f in info["features"]]
    out_path = DATA_DIR / f"{out_name}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {len(rows)} rows to {out_path}")
    return {"state": "COMPLETED", "out_path": str(out_path), "row_count": len(rows)}


def sample_pixels_for_fc(sample_image, fc, scale, tile_scale):
    return sample_image.sampleRegions(
        collection=fc,
        properties=["pid"],
        scale=scale,
        tileScale=tile_scale,
        geometries=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DATA_DIR / "training_sample.geojson"))
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--project", required=True)
    parser.add_argument(
        "--label-year",
        type=int,
        default=None,
        help="AAFC label year. Defaults to the Sentinel feature year.",
    )
    parser.add_argument("--scale", type=int, default=30, help="Pixel sampling scale in meters")
    parser.add_argument(
        "--export", choices=["drive", "local"], default="drive",
        help="'drive' is recommended for large pixel tables"
    )
    parser.add_argument("--tile-scale", type=int, default=4)
    parser.add_argument(
        "--polygon-batch-size",
        type=int,
        default=250,
        help="Number of input polygons per export batch when using Drive export",
    )
    parser.add_argument("--out-name", default="sentinel2_pixel_samples")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="JSON checkpoint path for resumable Drive batch exports "
             "(default: preprocessing/state/<out-name>_checkpoint.json)",
    )
    parser.add_argument(
        "--include-labels",
        action="store_true",
        help="Attach AAFC pixel labels in Earth Engine so steps 03-04 are unnecessary.",
    )
    parser.add_argument(
        "--keep-other",
        action="store_true",
        help="Keep non-target AAFC classes instead of filtering them out.",
    )
    args = parser.parse_args()
    checkpoint_path = args.checkpoint or default_checkpoint_path(args.out_name)
    update_global_tracker(args, checkpoint_path)

    ee.Initialize(project=args.project)

    fc = build_feature_collection(args.input)
    aoi = fc.geometry().bounds()
    stacked = build_stacked_image(aoi, args.year)

    sample_image = stacked
    if args.include_labels:
        label_year = args.label_year or args.year
        aafc = (
            ee.ImageCollection("AAFC/ACI")
            .filterDate(f"{label_year}-01-01", f"{label_year + 1}-01-01")
            .first()
            .select("landcover")
            .rename("aafc_code")
        )
        sample_image = sample_image.addBands(aafc)

    if args.export == "local":
        sample_fc = sample_pixels_for_fc(sample_image, fc, args.scale, args.tile_scale)
        print(f"Prepared pixel samples at {args.scale} m resolution")
        if args.include_labels:
            label_year = args.label_year or args.year
            sample_fc = add_aafc_labels(sample_fc, label_year, args.keep_other)
            print(f"Attached AAFC labels for {label_year}")
        export_or_download(sample_fc, args.out_name, args.export)
        return

    batches = feature_collection_batches(fc, args.polygon_batch_size)
    state = load_checkpoint(checkpoint_path)
    state["config"] = {
        "input": args.input,
        "year": args.year,
        "label_year": args.label_year or args.year if args.include_labels else None,
        "scale": args.scale,
        "polygon_batch_size": args.polygon_batch_size,
        "include_labels": args.include_labels,
        "keep_other": args.keep_other,
        "out_name": args.out_name,
    }
    state["total_batches"] = len(batches)
    save_checkpoint(checkpoint_path, state)
    print(
        f"Prepared {len(batches)} export batch(es) at {args.scale} m resolution "
        f"with up to {args.polygon_batch_size} polygons each"
    )
    print(f"Checkpoint file: {checkpoint_path}")
    for batch_idx, batch_fc, batch_len in batches:
        if batch_idx in state.get("completed_batches", []):
            print(f"Skipping completed batch {batch_idx + 1}/{len(batches)}")
            continue

        sample_fc = sample_pixels_for_fc(sample_image, batch_fc, args.scale, args.tile_scale)
        if args.include_labels:
            label_year = args.label_year or args.year
            sample_fc = add_aafc_labels(sample_fc, label_year, args.keep_other)
        batch_name = f"{args.out_name}_part{batch_idx:03d}"
        print(f"Starting batch {batch_idx + 1}/{len(batches)}: {batch_len} polygons -> {batch_name}")
        update_batch_state(
            state,
            batch_idx,
            batch_name=batch_name,
            polygon_count=batch_len,
            status="RUNNING",
        )
        save_checkpoint(checkpoint_path, state)
        status = export_or_download(sample_fc, batch_name, "drive")
        batch_status = status["state"]
        update_global_tracker(args, checkpoint_path, batch_name=batch_name, batch_status=batch_status)
        update_batch_state(state, batch_idx, status=batch_status)
        if batch_status == "COMPLETED":
            if batch_idx not in state["completed_batches"]:
                state["completed_batches"].append(batch_idx)
            state["failed_batches"].pop(str(batch_idx), None)
        else:
            state.setdefault("failed_batches", {})[str(batch_idx)] = batch_status
        save_checkpoint(checkpoint_path, state)
        if batch_status != "COMPLETED":
            print(
                f"Batch {batch_idx + 1} failed with status {batch_status}. "
                f"Fix the issue and rerun; completed batches will be skipped."
            )
            return

    print("All batches completed.")


if __name__ == "__main__":
    main()
