#!/usr/bin/env python3
"""Prepare crop-only quarter-section batch and tracker files for the crop classification pipeline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_GEOJSON = ROOT.parent / "data" / "raw" / "alberta_quarter_sections.geojson"
DEFAULT_CONFIG = ROOT / "crop_classification_pipeline_config.json"
DEFAULT_BATCHES = ROOT / "quarter_section_batches.csv"
DEFAULT_TRACKER = ROOT / "quarter_section_tracker.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geojson", type=Path, default=DEFAULT_GEOJSON)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--batches-out", type=Path, default=DEFAULT_BATCHES)
    parser.add_argument("--tracker-out", type=Path, default=DEFAULT_TRACKER)
    return parser


def iter_geojson_features(path: Path):
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        in_features = False
        eof = False
        while True:
            if not eof and len(buffer) < 65536:
                chunk = handle.read(65536)
                if chunk:
                    buffer += chunk
                else:
                    eof = True

            if not in_features:
                idx = buffer.find('"features"')
                if idx == -1:
                    if eof:
                        raise ValueError("GeoJSON does not contain a `features` array.")
                    continue
                array_idx = buffer.find("[", idx)
                if array_idx == -1:
                    if eof:
                        raise ValueError("GeoJSON `features` array is not well formed.")
                    continue
                buffer = buffer[array_idx + 1 :]
                in_features = True

            buffer = buffer.lstrip()
            if not buffer:
                if eof:
                    break
                continue
            if buffer[0] == "]":
                break
            if buffer[0] == ",":
                buffer = buffer[1:]
                continue

            try:
                feature, end_idx = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof:
                    raise
                continue

            yield feature
            buffer = buffer[end_idx:]


def include_feature(properties: dict[str, object], *, exclude_road_allowances: bool) -> bool:
    if not exclude_road_allowances:
        return True
    return properties.get("ra") in (None, "", "null")


def load_config(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_ranges(total_count: int, batch_count: int) -> list[tuple[int, int, int, int]]:
    ranges: list[tuple[int, int, int, int]] = []
    for idx in range(batch_count):
        start = (total_count * idx) // batch_count
        end = (total_count * (idx + 1)) // batch_count
        ranges.append((idx + 1, start, end, end - start))
    return ranges


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    batching = config["batching"]
    geojson_cfg = config["geojson"]
    batch_count = int(batching["batch_count"])
    assignees = list(batching["assignees"])
    exclude_road_allowances = bool(geojson_cfg["exclude_road_allowances"])

    total_count = 0
    for feature in iter_geojson_features(args.geojson):
        if include_feature(feature.get("properties", {}) or {}, exclude_road_allowances=exclude_road_allowances):
            total_count += 1

    ranges = build_ranges(total_count, batch_count)

    args.batches_out.parent.mkdir(parents=True, exist_ok=True)
    with args.batches_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "batch_id",
                "assignee",
                "feature_start",
                "feature_end",
                "feature_count",
                "exclude_road_allowances",
            ],
        )
        writer.writeheader()
        for batch_id, start, end, count in ranges:
            writer.writerow(
                {
                    "batch_id": f"batch_{batch_id:02d}",
                    "assignee": assignees[(batch_id - 1) % len(assignees)],
                    "feature_start": start,
                    "feature_end": end,
                    "feature_count": count,
                    "exclude_road_allowances": str(exclude_road_allowances).lower(),
                }
            )

    with args.tracker_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "batch_id",
                "assignee",
                "feature_start",
                "feature_end",
                "feature_count",
                "quarter_sections_total",
                "quarter_sections_completed",
                "quarter_sections_successful",
                "quarter_sections_failed",
                "status",
                "labels_status",
                "sentinel2_status",
                "model_status",
                "started_at",
                "completed_at",
                "notes",
            ],
        )
        writer.writeheader()
        for batch_id, start, end, count in ranges:
            writer.writerow(
                {
                    "batch_id": f"batch_{batch_id:02d}",
                    "assignee": assignees[(batch_id - 1) % len(assignees)],
                    "feature_start": start,
                    "feature_end": end,
                    "feature_count": count,
                    "quarter_sections_total": count,
                    "quarter_sections_completed": 0,
                    "quarter_sections_successful": 0,
                    "quarter_sections_failed": 0,
                    "status": "pending",
                    "labels_status": "pending",
                    "sentinel2_status": "pending",
                    "model_status": "pending",
                    "started_at": "",
                    "completed_at": "",
                    "notes": "",
                }
            )

    print(f"Quarter sections kept: {total_count}")
    print(f"Wrote batches to {args.batches_out}")
    print(f"Wrote tracker to {args.tracker_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
