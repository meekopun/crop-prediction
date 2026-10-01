#!/usr/bin/env python3
"""Build a Copernicus Data Space Sentinel-2 weekly scene manifest for one batch."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from copernicus_data_space_utils import (
    bbox_union,
    cdse_client_from_env,
    iso_date_range,
    weekly_ranges,
)
from quarter_section_batch_utils import (
    ROOT,
    collect_batch_features,
    get_batch,
    load_config,
    output_paths,
    tracker_note,
    update_tracker,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "crop_classification_pipeline_config.json",
    )
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--years", nargs="+", type=int, default=None)
    parser.add_argument("--limit-per-week", type=int, default=8)
    return parser


def run() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    paths = output_paths(config)
    paths["scene_manifest_dir"].mkdir(parents=True, exist_ok=True)

    batch = get_batch(paths["batches_csv"], args.batch_id)
    geojson_cfg = config["geojson"]
    features = collect_batch_features(
        (ROOT / geojson_cfg["path"]).resolve(),
        feature_start=batch.feature_start,
        feature_end=batch.feature_end,
        exclude_road_allowances=bool(geojson_cfg["exclude_road_allowances"]),
    )
    batch_bbox = bbox_union(features)

    years = args.years or [2021, 2022, 2023]
    season_cfg = config["season"]
    client = cdse_client_from_env(config)

    rows: list[dict[str, object]] = []
    update_tracker(paths["tracker_csv"], batch.batch_id, {"sentinel2_status": "running"})
    for year in years:
        start_date, end_date = iso_date_range(
            year,
            int(season_cfg["start_month"]),
            int(season_cfg["start_day"]),
            int(season_cfg["end_month"]),
            int(season_cfg["end_day"]),
        )
        for week_index, week_start, week_end in weekly_ranges(start_date, end_date):
            scenes = client.search_sentinel2_l2a(
                bbox=batch_bbox,
                start_iso=week_start.isoformat(),
                end_iso=week_end.isoformat(),
                max_cloud_coverage=int(season_cfg["max_cloud_coverage"]),
                limit=args.limit_per_week,
            )
            scenes = sorted(
                scenes,
                key=lambda item: (item.cloud_cover if item.cloud_cover is not None else 999.0, item.datetime),
            )
            for rank, scene in enumerate(scenes, start=1):
                product_uuid = client.resolve_product_uuid(scene.product_name)
                rows.append(
                    {
                        "batch_id": batch.batch_id,
                        "year": year,
                        "week_index": week_index,
                        "week_start": week_start.isoformat(),
                        "week_end": week_end.isoformat(),
                        "rank": rank,
                        "scene_id": scene.scene_id,
                        "product_name": scene.product_name,
                        "product_uuid": product_uuid,
                        "download_url": client.product_download_url(product_uuid) if product_uuid else "",
                        "datetime": scene.datetime,
                        "eo_cloud_cover": scene.cloud_cover,
                        "collection": scene.collection,
                    }
                )

    manifest = pd.DataFrame(rows)
    manifest_path = paths["scene_manifest_dir"] / f"{batch.batch_id}_cdse_scene_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    update_tracker(paths["tracker_csv"], batch.batch_id, {"sentinel2_status": "manifest_ready"})
    tracker_note(paths["tracker_csv"], batch.batch_id, f"Built CDSE scene manifest {manifest_path.name}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
