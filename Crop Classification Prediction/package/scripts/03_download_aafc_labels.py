"""
Step 3: Download AAFC Annual Crop Inventory labels for the study area
directly from AAFC (not via Earth Engine), and assign a majority crop label
to each quarter section.

Data source: AAFC's public ImageServer REST API (Open Government Licence -
Canada). No account or API key needed -- it's a public government endpoint:
    https://agriculture.canada.ca/imagery-images/rest/services/annual_crop_inventory/{year}/ImageServer

NOTE ON NETWORK ACCESS: this script makes outbound HTTPS requests to
agriculture.canada.ca. If you're running it inside a network-restricted
sandbox (e.g. a locked-down CI box or an egress-allowlisted VM), that domain
may need to be added to the allowlist first. It works from a normal
developer machine with unrestricted internet.

What this script does:
    - Calls AAFC's ImageServer `computeStatisticsHistograms` endpoint for
      each quarter-section polygon.
    - Uses the returned per-class histogram to take the majority (mode)
      land-cover code as that quarter section's label.
    - Records the dominant-class purity within each polygon so you can filter
      out mixed quarter sections before training.
    - Maps AAFC codes to the target crops (alfalfa via AAFC forage code 122,
      plus corn, canola, wheat, barley, oats, peas); anything else is
      labeled "other" and, by default, excluded from the training table.

Usage:
    python 03_download_aafc_labels.py --year 2024
"""
import argparse
import csv
import json
import math
from collections import Counter

import requests

from common import AAFC_CODE_TO_CROP, load_geojson, quarter_section_id

DATA_DIR = "data"
IMAGESERVER_HISTOGRAM_URL = (
    "https://agriculture.canada.ca/imagery-images/rest/services/"
    "annual_crop_inventory/{year}/ImageServer/computeStatisticsHistograms"
)


def geojson_to_esri_polygon(geometry):
    geom_type = geometry["type"]
    coords = geometry["coordinates"]
    if geom_type == "Polygon":
        rings = coords
    elif geom_type == "MultiPolygon":
        rings = [ring for polygon in coords for ring in polygon]
    else:
        raise ValueError(f"Unsupported geometry type: {geom_type}")
    return {
        "rings": [project_ring_to_web_mercator(ring) for ring in rings],
        "spatialReference": {"wkid": 3857},
    }


def lonlat_to_web_mercator(lon, lat):
    # Clamp latitude to the valid Web Mercator range.
    lat = max(min(lat, 85.05112878), -85.05112878)
    x = lon * 20037508.34 / 180.0
    y = math.log(math.tan((90.0 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
    y = y * 20037508.34 / 180.0
    return [x, y]


def project_ring_to_web_mercator(ring):
    return [lonlat_to_web_mercator(lon, lat) for lon, lat in ring]


def summarize_histogram(hist):
    counts = hist["counts"]
    if not counts:
        return None

    bin_width = (hist["max"] - hist["min"]) / hist["size"]
    best_code = None
    best_count = -1
    total_count = 0
    for idx, count in enumerate(counts):
        if count <= 0:
            continue

        # AAFC class rasters are thematic integers; histogram bins are
        # centered on each class code (for example, -0.5..250.5 for 0..250).
        code = int(round(hist["min"] + (idx + 0.5) * bin_width))
        if code == 0:
            continue
        total_count += int(count)
        if count > best_count:
            best_code = code
            best_count = count

    if best_code is None or total_count == 0:
        return None

    return {
        "majority_code": best_code,
        "majority_fraction": best_count / total_count,
        "pixel_count": total_count,
    }


def fetch_histogram(session, url, geometry, pixel_size, timeout):
    params = {
        "geometry": json.dumps(geojson_to_esri_polygon(geometry)),
        "geometryType": "esriGeometryPolygon",
        "pixelSize": json.dumps(
            {"x": pixel_size, "y": pixel_size, "spatialReference": {"wkid": 3857}}
        ),
        "f": "json",
    }
    resp = session.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    if "error" in payload:
        raise RuntimeError(f"AAFC ImageServer error: {payload['error']}")
    histograms = payload.get("histograms", [])
    return histograms[0] if histograms else None


def majority_labels(year, fc, pixel_size=30, timeout=120):
    url = IMAGESERVER_HISTOGRAM_URL.format(year=year)
    session = requests.Session()
    labels = []
    total = len(fc["features"])

    for idx, feat in enumerate(fc["features"], start=1):
        hist = fetch_histogram(session, url, feat["geometry"], pixel_size, timeout)
        summary = summarize_histogram(hist) if hist else None
        if summary is None:
            labels.append((quarter_section_id(feat), None, "no_data", None, 0))
        else:
            crop = AAFC_CODE_TO_CROP.get(summary["majority_code"], "other")
            labels.append(
                (
                    quarter_section_id(feat),
                    summary["majority_code"],
                    crop,
                    summary["majority_fraction"],
                    summary["pixel_count"],
                )
            )

        if idx == 1 or idx % 100 == 0 or idx == total:
            print(f"Processed {idx}/{total} polygons")

    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=f"{DATA_DIR}/training_sample.geojson")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--labels-out", default=f"{DATA_DIR}/aafc_labels.csv")
    parser.add_argument("--pixel-size", type=int, default=30,
                        help="Pixel size in meters for ImageServer histogram queries")
    parser.add_argument("--keep-other", action="store_true",
                        help="Keep quarter sections whose majority class isn't "
                             "one of the configured target crops (labeled 'other')")
    parser.add_argument(
        "--min-purity",
        type=float,
        default=0.0,
        help="Minimum dominant-class fraction required to keep a label "
             "(for example 0.7 keeps only polygons that are at least 70%% one crop)",
    )
    args = parser.parse_args()

    fc = load_geojson(args.input)
    print("Requesting per-polygon AAFC histograms from")
    print(f"  {IMAGESERVER_HISTOGRAM_URL.format(year=args.year)}")

    labels = majority_labels(args.year, fc, pixel_size=args.pixel_size)

    n_kept = 0
    n_filtered_purity = 0
    with open(args.labels_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pid", "aafc_code", "crop_label", "label_purity", "aafc_pixel_count"])
        for pid, code, crop, purity, pixel_count in labels:
            if crop == "other" and not args.keep_other:
                continue
            if crop == "no_data":
                continue
            if purity is None or purity < args.min_purity:
                n_filtered_purity += 1
                continue
            writer.writerow([pid, code, crop, purity, pixel_count])
            n_kept += 1

    print(f"Wrote {n_kept} labeled quarter sections to {args.labels_out}")
    if args.min_purity > 0:
        print(f"Filtered out {n_filtered_purity} low-purity labels below {args.min_purity:.2f}")
    print("Label counts:")
    print(Counter(c for _, _, c, _, _ in labels))


if __name__ == "__main__":
    main()
