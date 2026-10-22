#!/usr/bin/env python3
"""Run one crop classification batch: quarter-section subset, labels, Sentinel-2 stats, and feature engineering."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from quarter_section_batch_utils import (
    ROOT,
    collect_batch_features,
    feature_identifier,
    get_batch,
    iso_date,
    load_config,
    output_paths,
    tracker_note,
    update_tracker,
    write_geojson,
)


AAFC_CROP_CODES = {
    133: "Barley",
    136: "Oats",
    140: "Wheat",
    145: "Winter Wheat",
    146: "Spring Wheat",
    147: "Corn for Grain",
    153: "Canola and Rapeseed",
    154: "Flaxseed",
    158: "Soybeans",
    162: "Peas",
    174: "Lentils",
}

S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL", "dataMask"]
DERIVED_WEEKLY_FEATURES = [
    "ndvi",
    "ndre1",
    "ndre2",
    "gndvi",
    "ndmi",
    "evi",
    "yellow_blue_ratio",
    "vari",
    "ngrdi",
    "flowering_contrast",
    "red_green_ratio",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "crop_classification_pipeline_config.json",
    )
    parser.add_argument("--batch-id", required=True)
    parser.add_argument(
        "--label-source",
        choices=["preferred", "local_aafc_raster", "gee_aafc_only"],
        default="preferred",
    )
    parser.add_argument(
        "--sentinel-source",
        choices=["preferred", "copernicus_data_space", "sentinel_hub_process_api"],
        default="preferred",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=None,
        help="Optional override for configured years.",
    )
    return parser


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def polygon_centroid(geometry: dict[str, Any]) -> tuple[float, float]:
    coords = geometry["coordinates"]
    pts: list[tuple[float, float]] = []
    if geometry["type"] == "Polygon":
        rings = coords
        for ring in rings[:1]:
            pts.extend((float(lon), float(lat)) for lon, lat in ring[:-1])
    elif geometry["type"] == "MultiPolygon":
        for polygon in coords[:1]:
            for ring in polygon[:1]:
                pts.extend((float(lon), float(lat)) for lon, lat in ring[:-1])
    else:
        raise ValueError(f"Unsupported geometry type: {geometry['type']}")
    lon = sum(point[0] for point in pts) / len(pts)
    lat = sum(point[1] for point in pts) / len(pts)
    return lon, lat


def bbox_from_geometry(geometry: dict[str, Any]) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    coords = geometry["coordinates"]
    if geometry["type"] == "Polygon":
        polygons = [coords]
    elif geometry["type"] == "MultiPolygon":
        polygons = coords
    else:
        raise ValueError(f"Unsupported geometry type: {geometry['type']}")
    for polygon in polygons:
        for ring in polygon:
            for lon, lat in ring:
                xs.append(float(lon))
                ys.append(float(lat))
    return min(xs), min(ys), max(xs), max(ys)


def label_source_name(config: dict[str, Any], requested: str) -> str:
    if requested != "preferred":
        return requested
    return str(config["label_source"]["preferred"])


def sentinel_source_name(config: dict[str, Any], requested: str) -> str:
    if requested != "preferred":
        return requested
    return str(config["sentinel2_source"]["preferred_api"])


def assign_labels_from_local_raster(
    features: list[dict[str, Any]],
    *,
    raster_path: Path,
) -> pd.DataFrame:
    try:
        import rasterio
        from rasterio.mask import mask
        from shapely.geometry import shape
    except ImportError as exc:
        raise RuntimeError("Local AAFC raster labeling requires rasterio and shapely.") from exc

    rows: list[dict[str, Any]] = []
    with rasterio.open(raster_path) as dataset:
        nodata = dataset.nodata
        for feature_index, feature in enumerate(features):
            geom = feature["geometry"]
            out, _transform = mask(dataset, [geom], crop=True, filled=False)
            values = out[0].compressed()
            if nodata is not None:
                values = values[values != nodata]
            values = values[values > 0]
            if len(values) == 0:
                label_code = None
            else:
                unique, counts = np.unique(values.astype(int), return_counts=True)
                label_code = int(unique[np.argmax(counts)])
            rows.append(
                {
                    "quarter_section": feature_identifier(feature, feature_index),
                    "label_code": label_code,
                    "crop_label": AAFC_CROP_CODES.get(label_code),
                }
            )
    return pd.DataFrame(rows)


def assign_labels_from_gee(
    features: list[dict[str, Any]],
    *,
    years: list[int],
) -> pd.DataFrame:
    import ee

    ee.Initialize(project="satellite-analysis-489120")
    rows: list[dict[str, Any]] = []
    for feature_index, feature in enumerate(features):
        feature_id = feature_identifier(feature, feature_index)
        geom = ee.Geometry(feature["geometry"])
        for year in years:
            image = (
                ee.ImageCollection("AAFC/ACI")
                .filterDate(f"{year}-01-01", f"{year}-12-31")
                .first()
                .select("landcover")
            )
            result = image.reduceRegion(
                reducer=ee.Reducer.mode(),
                geometry=geom,
                scale=10,
                maxPixels=10_000_000,
            ).getInfo()
            label_code = result.get("landcover") if result else None
            label_code = int(label_code) if label_code is not None else None
            rows.append(
                {
                    "quarter_section": feature_id,
                    "year": year,
                    "label_code": label_code,
                    "crop_label": AAFC_CROP_CODES.get(label_code),
                }
            )
    return pd.DataFrame(rows)


class SentinelHubStatsClient:
    def __init__(self, *, client_id: str, client_secret: str, token_url: str, statistics_endpoint: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        self.statistics_endpoint = statistics_endpoint
        self._token: str | None = None

    def token(self) -> str:
        if self._token is None:
            response = requests.post(
                self.token_url,
                headers={"content-type": "application/x-www-form-urlencoded"},
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
                timeout=60,
            )
            response.raise_for_status()
            self._token = response.json()["access_token"]
        return self._token

    def band_evalscript(self) -> str:
        return """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12","SCL","dataMask"] }],
    output: [
      { id: "spectral", bands: 10, sampleType: "FLOAT32" },
      { id: "scl", bands: 1, sampleType: "UINT8" },
      { id: "dataMask", bands: 1, sampleType: "UINT8" }
    ]
  };
}
function evaluatePixel(samples) {
  return {
    spectral: [samples.B02, samples.B03, samples.B04, samples.B05, samples.B06, samples.B07, samples.B08, samples.B8A, samples.B11, samples.B12],
    scl: [samples.SCL],
    dataMask: [samples.dataMask]
  };
}
""".strip()

    def fetch_weekly_stats(
        self,
        *,
        geometry: dict[str, Any],
        start_date: str,
        end_date: str,
        resolution_m: int,
        max_cloud_coverage: int,
        aggregation_interval: str,
    ) -> dict[str, Any]:
        minx, miny, maxx, maxy = bbox_from_geometry(geometry)
        payload = {
            "input": {
                "bounds": {
                    "bbox": [minx, miny, maxx, maxy],
                    "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
                    "geometry": geometry,
                },
                "data": [
                    {
                        "type": "sentinel-2-l2a",
                        "dataFilter": {"maxCloudCoverage": max_cloud_coverage},
                        "processing": {"mosaickingOrder": "leastCC"},
                    }
                ],
            },
            "aggregation": {
                "timeRange": {"from": f"{start_date}T00:00:00Z", "to": f"{end_date}T00:00:00Z"},
                "aggregationInterval": {"of": aggregation_interval},
                "resx": resolution_m,
                "resy": resolution_m,
                "evalscript": self.band_evalscript(),
            },
            "calculations": {
                "spectral": {
                    "statistics": {
                        "default": {"percentiles": {"k": [10, 50, 90]}}
                    }
                },
                "scl": {
                    "statistics": {
                        "default": {"histograms": {"default": {"nBins": 12, "lowEdge": 0, "highEdge": 12}}}
                    }
                },
            },
        }
        response = requests.post(
            self.statistics_endpoint,
            headers={"Authorization": f"Bearer {self.token()}"},
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        return response.json()


def week_prefix(index: int) -> str:
    return f"w{index:02d}"


def safe_div(numerator: float, denominator: float) -> float:
    eps = 1e-6
    return float(numerator) / float(denominator + eps)


def flatten_interval_stats(interval_payload: dict[str, Any], week_index: int) -> dict[str, float]:
    outputs = interval_payload["outputs"]
    spectral_bands = outputs["spectral"]["bands"]
    band_means = {}
    order = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    for band_name, stats in zip(order, spectral_bands.values(), strict=False):
        band_means[band_name] = float(stats["stats"]["mean"])

    b2 = band_means["B02"]
    b3 = band_means["B03"]
    b4 = band_means["B04"]
    b5 = band_means["B05"]
    b6 = band_means["B06"]
    b8 = band_means["B08"]
    b11 = band_means["B11"]

    derived = {
        "ndvi": safe_div(b8 - b4, b8 + b4),
        "ndre1": safe_div(b8 - b5, b8 + b5),
        "ndre2": safe_div(b8 - b6, b8 + b6),
        "gndvi": safe_div(b8 - b3, b8 + b3),
        "ndmi": safe_div(b8 - b11, b8 + b11),
        "evi": 2.5 * safe_div(b8 - b4, b8 + 6 * b4 - 7.5 * b2 + 1),
        "yellow_blue_ratio": safe_div(b3 + b4, 2 * b2),
        "vari": safe_div(b3 - b4, b3 + b4 - b2),
        "ngrdi": safe_div(b3 - b4, b3 + b4),
        "flowering_contrast": safe_div(b3 + b4, b8 + b2),
        "red_green_ratio": safe_div(b4, b3),
    }

    flattened: dict[str, float] = {}
    prefix = week_prefix(week_index)
    for band_name, mean_value in band_means.items():
        flattened[f"{band_name.lower()}_{prefix}"] = mean_value
    for feature_name, feature_value in derived.items():
        flattened[f"{feature_name}_{prefix}"] = feature_value
    return flattened


def summarize_weekly_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for feature_name in DERIVED_WEEKLY_FEATURES:
        weekly_columns = [col for col in result.columns if col.startswith(f"{feature_name}_w")]
        if not weekly_columns:
            continue
        weekly_columns = sorted(weekly_columns)
        values = result[weekly_columns].to_numpy(dtype=float)
        week_numbers = np.arange(1, len(weekly_columns) + 1, dtype=float)
        result[f"{feature_name}_mean"] = np.nanmean(values, axis=1)
        result[f"{feature_name}_max"] = np.nanmax(values, axis=1)
        result[f"{feature_name}_min"] = np.nanmin(values, axis=1)
        result[f"{feature_name}_amplitude"] = result[f"{feature_name}_max"] - result[f"{feature_name}_min"]
        peak_idx = np.where(np.isnan(values).all(axis=1), np.nan, np.nanargmax(values, axis=1) + 1)
        result[f"{feature_name}_peak_week"] = peak_idx
        early = values[:, : max(1, len(weekly_columns) // 3)]
        mid = values[:, len(weekly_columns) // 3 : 2 * len(weekly_columns) // 3]
        late = values[:, 2 * len(weekly_columns) // 3 :]
        result[f"{feature_name}_early_mean"] = np.nanmean(early, axis=1)
        result[f"{feature_name}_mid_mean"] = np.nanmean(mid, axis=1)
        result[f"{feature_name}_late_mean"] = np.nanmean(late, axis=1)
        result[f"{feature_name}_greenup_delta"] = result[f"{feature_name}_mid_mean"] - result[f"{feature_name}_early_mean"]
        result[f"{feature_name}_senescence_delta"] = result[f"{feature_name}_late_mean"] - result[f"{feature_name}_mid_mean"]
    return result


def sentinel_hub_client_from_env(config: dict[str, Any]) -> SentinelHubStatsClient:
    client_id = os.environ.get("SENTINEL_HUB_CLIENT_ID")
    client_secret = os.environ.get("SENTINEL_HUB_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError("Set SENTINEL_HUB_CLIENT_ID and SENTINEL_HUB_CLIENT_SECRET.")
    source_cfg = config["sentinel2_source"]
    return SentinelHubStatsClient(
        client_id=client_id,
        client_secret=client_secret,
        token_url=source_cfg["oauth_token_url"],
        statistics_endpoint=source_cfg["statistics_endpoint"],
    )


def run() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    paths = output_paths(config)
    for directory_key in ("batch_geojson_dir", "batch_feature_dir", "model_output_dir"):
        paths[directory_key].mkdir(parents=True, exist_ok=True)

    batch = get_batch(paths["batches_csv"], args.batch_id)
    years = args.years or [2021, 2022, 2023]
    geojson_cfg = config["geojson"]
    exclude_road_allowances = bool(geojson_cfg["exclude_road_allowances"])
    tracker_path = paths["tracker_csv"]

    update_tracker(
        tracker_path,
        batch.batch_id,
        {
            "status": "running",
            "started_at": now_utc(),
        },
    )

    try:
        geojson_path = (ROOT / geojson_cfg["path"]).resolve()
        features = collect_batch_features(
            geojson_path,
            feature_start=batch.feature_start,
            feature_end=batch.feature_end,
            exclude_road_allowances=exclude_road_allowances,
        )
        batch_geojson_path = paths["batch_geojson_dir"] / f"{batch.batch_id}.geojson"
        write_geojson(batch_geojson_path, features)
        tracker_note(tracker_path, batch.batch_id, f"Saved batch GeoJSON to {batch_geojson_path.name}")

        selected_label_source = label_source_name(config, args.label_source)
        if selected_label_source == "local_aafc_raster":
            raster_pattern = str(config["label_source"].get("local_raster_pattern", "")).strip()
            if not raster_pattern:
                raise RuntimeError(
                    "crop_classification_pipeline_config.json "
                    "`label_source.local_raster_pattern` is empty."
                )
            label_frames = []
            for year in years:
                raster_path = Path(raster_pattern.format(year=year))
                label_frame = assign_labels_from_local_raster(features, raster_path=raster_path)
                label_frame["year"] = year
                label_frames.append(label_frame)
            labels = pd.concat(label_frames, ignore_index=True)
        elif selected_label_source == "gee_aafc_only":
            labels = assign_labels_from_gee(features, years=years)
        else:
            raise ValueError(f"Unsupported label source: {selected_label_source}")

        update_tracker(tracker_path, batch.batch_id, {"labels_status": "completed"})

        selected_s2_source = sentinel_source_name(config, args.sentinel_source)
        if selected_s2_source == "copernicus_data_space":
            raise NotImplementedError(
                "The combined batch runner has not been ported to direct Copernicus Data Space extraction yet. "
                "Use `build_copernicus_scene_manifest.py` for CDSE scene "
                "discovery and keep this runner only for the older Sentinel "
                "Hub path."
            )
        if selected_s2_source != "sentinel_hub_process_api":
            raise ValueError(f"Unsupported Sentinel-2 source: {selected_s2_source}")
        if "statistics_endpoint" not in config["sentinel2_source"]:
            raise RuntimeError(
                "Current config does not define Sentinel Hub statistics settings. "
                "Use Copernicus Data Space or restore the old Sentinel Hub config values."
            )
        client = sentinel_hub_client_from_env(config)
        season_cfg = config["season"]

        feature_rows: list[dict[str, Any]] = []
        for feature_index, feature in enumerate(features):
            feature_id = feature_identifier(feature, feature_index)
            props = feature.get("properties", {}) or {}
            geometry = feature["geometry"]
            lon, lat = polygon_centroid(geometry)
            for year in years:
                start_date = iso_date(year, season_cfg["start_month"], season_cfg["start_day"])
                end_date = iso_date(year, season_cfg["end_month"], season_cfg["end_day"])
                stats = client.fetch_weekly_stats(
                    geometry=geometry,
                    start_date=start_date,
                    end_date=end_date,
                    resolution_m=int(season_cfg["resolution_m"]),
                    max_cloud_coverage=int(season_cfg["max_cloud_coverage"]),
                    aggregation_interval=str(season_cfg["aggregation_interval"]),
                )
                row = {
                    "quarter_section": feature_id,
                    "year": year,
                    "lat": lat,
                    "lon": lon,
                    "descriptor": props.get("descriptor"),
                    "meridian": props.get("m"),
                    "range": props.get("rge"),
                    "township": props.get("twp"),
                    "section": props.get("sec"),
                    "quarter": props.get("qs"),
                }
                for week_idx, interval in enumerate(stats["data"], start=1):
                    row.update(flatten_interval_stats(interval, week_idx))
                feature_rows.append(row)

        feature_frame = pd.DataFrame(feature_rows)
        feature_frame = feature_frame.merge(labels, on=["quarter_section", "year"], how="left")
        feature_frame = summarize_weekly_features(feature_frame)

        feature_csv = paths["batch_feature_dir"] / f"{batch.batch_id}_features.csv"
        feature_frame.to_csv(feature_csv, index=False)
        update_tracker(
            tracker_path,
            batch.batch_id,
            {
                "sentinel2_status": "completed",
                "status": "features_ready",
                "completed_at": now_utc(),
            },
        )
        tracker_note(tracker_path, batch.batch_id, f"Wrote engineered features to {feature_csv.name}")
        return 0
    except Exception as exc:
        update_tracker(tracker_path, batch.batch_id, {"status": "failed"})
        tracker_note(tracker_path, batch.batch_id, f"Failure: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(run())
