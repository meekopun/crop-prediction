from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class BatchSpec:
    batch_id: str
    assignee: str
    feature_start: int
    feature_end: int
    feature_count: int


def load_config(config_path: Path | str) -> dict[str, Any]:
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def resolve_path(base_dir: Path, relative_or_absolute: str) -> Path:
    path = Path(relative_or_absolute)
    return path if path.is_absolute() else (base_dir / path).resolve()


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


def include_feature(properties: dict[str, Any], *, exclude_road_allowances: bool) -> bool:
    if not exclude_road_allowances:
        return True
    return properties.get("ra") in (None, "", "null")


def iter_filtered_geojson_features(geojson_path: Path, *, exclude_road_allowances: bool):
    for feature in iter_geojson_features(geojson_path):
        props = feature.get("properties", {}) or {}
        if include_feature(props, exclude_road_allowances=exclude_road_allowances):
            yield feature


def feature_identifier(feature: dict[str, Any], fallback_index: int) -> str:
    props = feature.get("properties", {}) or {}
    value = props.get("pid") or props.get("PID") or props.get("id") or props.get("ID")
    return str(value) if value is not None else f"feature_{fallback_index:07d}"


def load_batches(path: Path | str) -> list[BatchSpec]:
    frame = pd.read_csv(path)
    return [
        BatchSpec(
            batch_id=str(row.batch_id),
            assignee=str(row.assignee),
            feature_start=int(row.feature_start),
            feature_end=int(row.feature_end),
            feature_count=int(row.feature_count),
        )
        for row in frame.itertuples(index=False)
    ]


def get_batch(path: Path | str, batch_id: str) -> BatchSpec:
    for batch in load_batches(path):
        if batch.batch_id == batch_id:
            return batch
    raise ValueError(f"Unknown batch_id: {batch_id}")


def collect_batch_features(
    geojson_path: Path,
    *,
    feature_start: int,
    feature_end: int,
    exclude_road_allowances: bool,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    kept_index = 0
    for feature in iter_filtered_geojson_features(
        geojson_path,
        exclude_road_allowances=exclude_road_allowances,
    ):
        if kept_index < feature_start:
            kept_index += 1
            continue
        if kept_index >= feature_end:
            break
        selected.append(feature)
        kept_index += 1
    return selected


def write_geojson(path: Path, features: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    collection = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(collection), encoding="utf-8")


def load_tracker(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("")


def update_tracker(
    tracker_path: Path | str,
    batch_id: str,
    updates: dict[str, Any],
) -> None:
    tracker_path = Path(tracker_path)
    frame = load_tracker(tracker_path)
    mask = frame["batch_id"] == batch_id
    if not mask.any():
        raise ValueError(f"Batch id not found in tracker: {batch_id}")
    for key, value in updates.items():
        if key not in frame.columns:
            frame[key] = ""
        frame.loc[mask, key] = "" if value is None else str(value)
    frame.to_csv(tracker_path, index=False)


def append_note(existing: str, new_note: str) -> str:
    existing = (existing or "").strip()
    new_note = new_note.strip()
    if not existing:
        return new_note
    if not new_note:
        return existing
    return f"{existing} | {new_note}"


def tracker_note(tracker_path: Path | str, batch_id: str, message: str) -> None:
    frame = load_tracker(tracker_path)
    mask = frame["batch_id"] == batch_id
    if not mask.any():
        raise ValueError(f"Batch id not found in tracker: {batch_id}")
    current = frame.loc[mask, "notes"].iloc[0]
    frame.loc[mask, "notes"] = append_note(current, message)
    Path(tracker_path).write_text(frame.to_csv(index=False), encoding="utf-8")


def output_paths(config: dict[str, Any]) -> dict[str, Path]:
    path_cfg = config["paths"]
    return {
        "batches_csv": ROOT / path_cfg["batches_csv"],
        "tracker_csv": ROOT / path_cfg["tracker_csv"],
        "batch_geojson_dir": ROOT / path_cfg["batch_geojson_dir"],
        "scene_manifest_dir": ROOT / path_cfg["scene_manifest_dir"],
        "batch_feature_dir": ROOT / path_cfg["batch_feature_dir"],
        "model_output_dir": ROOT / path_cfg["model_output_dir"],
    }


def iso_date(year: int, month: int, day: int) -> str:
    return f"{year:04d}-{month:02d}-{day:02d}"
