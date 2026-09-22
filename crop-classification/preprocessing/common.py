"""
Shared constants and helpers for the standalone pixel preprocessing folder.
"""
import json
from pathlib import Path

AAFC_CODE_TO_CROP = {
    122: "alfalfa",
    147: "corn",
    153: "canola",
    140: "wheat",
    145: "wheat",
    146: "wheat",
    133: "barley",
    136: "oats",
    162: "peas",
}

S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
S1_BANDS = ["VV", "VH"]
PRECIP_BANDS = ["PRECIP"]
GROWING_SEASON_MONTHS = [4, 5, 6, 7, 8, 9, 10]
MONTH_NAMES = {
    4: "apr",
    5: "may",
    6: "jun",
    7: "jul",
    8: "aug",
    9: "sep",
    10: "oct",
}

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PREPROCESSING_DIR = ROOT / "preprocessing"
STATE_DIR = PREPROCESSING_DIR / "state"
TRACKER_PATH = STATE_DIR / "preprocessing_progress.json"


def load_geojson(path):
    with open(path, "r") as f:
        return json.load(f)


def save_geojson(path, feature_collection):
    with open(path, "w") as f:
        json.dump(feature_collection, f)


def quarter_section_id(feature):
    return feature["properties"]["pid"]


def load_tracker(path=TRACKER_PATH):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        return {
            "quarter_section_sets": {},
            "pixel_exports": {},
            "merged_outputs": {},
        }
    with open(path) as f:
        return json.load(f)


def save_tracker(state, path=TRACKER_PATH):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
