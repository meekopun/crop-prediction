"""
Shared constants and helpers for the crop classification pipeline.
"""
import json

# ---------------------------------------------------------------------------
# Study area: Lethbridge County, southern Alberta.
# Chosen for crop diversity: irrigated + dryland farming supports corn,
# canola, spring wheat/durum, barley, oats, peas, and forage production in
# the same region.
# ---------------------------------------------------------------------------
STUDY_BBOX = {
    "lon_min": -113.35,
    "lon_max": -112.50,
    "lat_min": 49.55,
    "lat_max": 50.05,
}

# ---------------------------------------------------------------------------
# AAFC Annual Crop Inventory landcover codes -> target crop classes.
# Source: https://developers.google.com/earth-engine/datasets/catalog/AAFC_ACI
# Wheat variants (general/winter/spring) are pooled into one "wheat" class.
# AAFC does not expose a dedicated alfalfa class in the public ACI table, so
# code 122 ("Pasture and Forages") is used as the closest forage proxy.
# ---------------------------------------------------------------------------
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

TARGET_CROPS = sorted(set(AAFC_CODE_TO_CROP.values()))

# Sentinel-2 surface reflectance bands used for feature extraction.
# Excludes B1 (aerosol), B9 (water vapor), B10 (cirrus, absent in SR product).
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
S1_BANDS = ["VV", "VH"]
PRECIP_BANDS = ["PRECIP"]

# Growing-season months to sample phenology across (Alberta prairie crops).
GROWING_SEASON_MONTHS = [4, 5, 6, 7, 8, 9, 10]  # Apr - Oct

MONTH_NAMES = {
    4: "apr", 5: "may", 6: "jun", 7: "jul",
    8: "aug", 9: "sep", 10: "oct",
}


def load_geojson(path):
    with open(path, "r") as f:
        return json.load(f)


def save_geojson(path, feature_collection):
    with open(path, "w") as f:
        json.dump(feature_collection, f)


def quarter_section_id(feature):
    """Stable unique identifier for a quarter-section feature (its ATS pid)."""
    return feature["properties"]["pid"]
