"""
Step 0: Extract just the study-area quarter sections from the full
province-wide quarter_sections.geojson.

Why this exists: quarter_sections.geojson covers all of Alberta and is
~1.2GB / ~1.9M features -- far too large to hand to Earth Engine or a local
classifier directly. This script streams through it line-by-line (it's
formatted with one Feature per line) so it never loads the whole file into
memory, and keeps only the polygons inside STUDY_BBOX (see common.py --
currently set to Lethbridge County, chosen for its mix of irrigated and
dryland farming, which supports corn, canola, wheat, barley, oats, and peas
all in one place).

This step is checkpointed: it saves its progress (byte offset into the
source file) every TIME_BUDGET seconds and can be safely re-run to resume
where it left off (useful in environments where long-running processes get
cut off). Once it prints "ALL DONE", the study-area file is complete.

Usage:
    python 00_extract_study_area.py
    # if interrupted, just run it again -- it resumes automatically
"""
import json
import os
import time

from common import STUDY_BBOX

SRC = "raw_data/quarter_sections.geojson"
OUT = "data/study_area_quarter_sections.geojson"
LOG = "data/.extract_study_area.log"
STATE = "data/.extract_study_area.state"

TIME_BUDGET = 30  # seconds per invocation before checkpointing


def first_coord(geom):
    coords = geom["coordinates"]
    if geom["type"] == "MultiPolygon":
        return coords[0][0][0]
    return coords[0][0]


def in_bbox(lon, lat):
    return (STUDY_BBOX["lon_min"] <= lon <= STUDY_BBOX["lon_max"]
            and STUDY_BBOX["lat_min"] <= lat <= STUDY_BBOX["lat_max"])


def load_state():
    if os.path.exists(STATE) and os.path.getsize(STATE) > 0:
        with open(STATE) as f:
            return json.load(f)
    return {"offset": 0, "n_lines": 0, "n_candidates": 0, "n_matched": 0, "done": False}


def save_state(state):
    with open(STATE, "w") as f:
        json.dump(state, f)


def main():
    os.makedirs("data", exist_ok=True)
    state = load_state()
    if state["done"]:
        print("ALL DONE (already complete) -- see", OUT)
        return

    start_time = time.time()
    fresh_start = state["offset"] == 0
    out_mode = "w" if fresh_start else "a"

    with open(SRC, "r") as fin, open(OUT, out_mode) as fout, open(LOG, "a") as flog:
        fin.seek(state["offset"])
        if out_mode == "w":
            fout.write(
                '{ "type": "FeatureCollection", "name": "study_area_quarter_sections", '
                '"crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:EPSG::4269" } }, '
                '"features": [\n'
            )
        need_comma = state["n_matched"] > 0

        while True:
            if time.time() - start_time > TIME_BUDGET:
                state["offset"] = fin.tell()
                save_state(state)
                flog.write(f"CHECKPOINT at line {state['n_lines']}, "
                           f"{state['n_matched']} matched\n")
                print(f"Checkpointed at {state['n_lines']} lines, "
                      f"{state['n_matched']} matched so far. Re-run to continue.")
                return

            line = fin.readline()
            if not line:
                fout.write("\n] }\n")
                state["done"] = True
                save_state(state)
                flog.write(f"ALL DONE. {state['n_lines']} lines, {state['n_matched']} matched\n")
                print(f"ALL DONE. {state['n_matched']} quarter sections written to {OUT}")
                return

            state["n_lines"] += 1
            stripped = line.strip()
            if not stripped.startswith('{ "type": "Feature"'):
                continue
            if "-113." not in stripped and "-112." not in stripped:
                continue
            state["n_candidates"] += 1

            payload = stripped[:-1] if stripped.endswith(",") else stripped
            try:
                feat = json.loads(payload)
                lon, lat = first_coord(feat["geometry"])
            except Exception:
                continue

            if in_bbox(lon, lat):
                state["n_matched"] += 1
                if need_comma:
                    fout.write(",\n")
                fout.write(json.dumps(feat))
                need_comma = True


if __name__ == "__main__":
    main()
