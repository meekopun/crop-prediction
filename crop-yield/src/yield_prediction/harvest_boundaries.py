"""Match staged harvest operations to candidate ATS quarter-section boundaries.

Legal land descriptions such as ``SE 30-71-9 W6M`` (or ``N1/2 15-72-10 W6M`` for
two quarters) become ATS ``pid`` values, which are looked up in the province-wide
quarter-section GeoJSON by streaming it line by line. A matched quarter section is
a candidate boundary only: several farm fields can share one quarter, and the
harvested acres can cover part of it. No yields are converted or aggregated here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

from .harvest import DEFAULT_OUTPUT_DIR

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OPERATIONS_PATH = DEFAULT_OUTPUT_DIR / "operations.csv"
DEFAULT_GEOMETRY_PATH = PACKAGE_ROOT / "data/raw/alberta_quarter_sections.geojson"

# ATS pid = meridian (1) + range (2) + township (3) + section (2) + quarter (1)
# + road-allowance suffix (2); "00" is the quarter itself, 01-03 are road strips.
QUARTER_CODES = {"SE": "5", "SW": "6", "NW": "7", "NE": "8"}
HALF_QUARTERS = {"N1/2": ("NE", "NW"), "S1/2": ("SE", "SW")}
LEGAL_PATTERN = re.compile(
    r"(NE|NW|SE|SW|N1/2|S1/2)\s*(\d{1,2})\s*[-.]\s*(\d{1,3})\s*-\s*(\d{1,2})\s*W\s*(\d)\s*M",
    re.IGNORECASE,
)
FEATURE_LINE = re.compile(r'^\s*\{\s*"type":\s*"Feature"\s*,')
SQUARE_METRES_PER_ACRE = 4046.8564224
EARTH_RADIUS_M = 6_371_008.8
# Reported harvested area above this multiple of the matched parcel area is flagged.
AREA_TOLERANCE = 1.05
# A share of the parcel below this means the quarter likely also holds other land use.
LOW_COVERAGE = 0.5
STATUS_ORDER = ("single_operation", "shared_same_crop", "shared_multiple_crops")
REQUIRED_COLUMNS = ("operation_id", "source_file", "source_row", "year", "crop", "crop_raw",
                    "field_name_raw", "legal_description_raw", "area_ac", "yield_value",
                    "yield_unit", "processing_status")
MATCH_COLUMNS = ["operation_id", "source_file", "source_row", "year", "crop", "crop_raw",
                 "field_name_raw", "legal_description_raw", "legal_description", "pids",
                 "match_status", "parcel_acres", "area_ac", "area_fraction", "yield_value",
                 "yield_unit", "quarter_year_status", "quarter_year_operation_count",
                 "quarter_year_crops", "quarter_year_area_ac", "boundary_pilot_eligible", "flags"]


@dataclass(frozen=True)
class LegalDescription:
    kind: str
    section: int
    township: int
    range_: int
    meridian: int

    @property
    def quarters(self) -> tuple[str, ...]:
        return HALF_QUARTERS.get(self.kind, (self.kind,))

    @property
    def pids(self) -> tuple[str, ...]:
        return tuple(ats_pid(self.meridian, self.range_, self.township, self.section, quarter)
                     for quarter in self.quarters)

    @property
    def label(self) -> str:
        return f"{self.kind} {self.section}-{self.township}-{self.range_} W{self.meridian}M"


def ats_pid(meridian: int, range_: int, township: int, section: int, quarter: str) -> str:
    return f"{meridian}{range_:02d}{township:03d}{section:02d}{QUARTER_CODES[quarter]}00"


def parse_legal_description(text: str) -> LegalDescription | None:
    """Parse a quarter or half-section description; return None if unsupported."""
    match = LEGAL_PATTERN.fullmatch(text.strip())
    if not match:
        return None
    kind = match[1].upper()
    section, township, range_, meridian = (int(value) for value in match.groups()[1:])
    if not (1 <= section <= 36 and 1 <= township <= 126 and 1 <= range_ <= 34 and 4 <= meridian <= 6):
        return None
    return LegalDescription(kind, section, township, range_, meridian)


def find_quarter_sections(path: Path | str, pids: set[str]) -> tuple[dict[str, dict], str | None]:
    """Stream a one-feature-per-line GeoJSON (as GDAL writes it) for wanted pids.

    Only lines containing a wanted pid are decoded, so the 1.2 GB provincial file
    is never loaded whole. Returns the features by pid and the declared CRS name.
    """
    found: dict[str, dict] = {}
    crs_name = None
    feature_lines = 0
    marker = '"pid": "'
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not FEATURE_LINE.match(line):
                if crs_name is None and '"crs"' in line:
                    crs_match = re.search(r'"name":\s*"([^"]+)"', line)
                    crs_name = crs_match[1] if crs_match else None
                continue
            feature_lines += 1
            start = line.find(marker)
            if start < 0:
                continue
            start += len(marker)
            pid = line[start:line.find('"', start)]
            if pid in pids and pid not in found:
                feature = json.loads(line.rstrip().rstrip(","))
                if feature["properties"].get("ra") in (None, ""):
                    found[pid] = feature
    if not feature_lines:
        raise ValueError(f"{path}: expected one GeoJSON feature per line, as written by GDAL")
    return found, crs_name


def _ring_area_m2(ring: list) -> float:
    # Local equirectangular projection; accurate to well under 1% at parcel scale.
    lat0 = math.radians(sum(point[1] for point in ring) / len(ring))
    kx = EARTH_RADIUS_M * math.cos(lat0) * math.pi / 180
    ky = EARTH_RADIUS_M * math.pi / 180
    points = [(point[0] * kx, point[1] * ky) for point in ring]
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2)
                   in zip(points, points[1:] + points[:1]))) / 2


def polygon_acres(geometry: dict) -> float:
    """Approximate area of a lon/lat Polygon or MultiPolygon in acres."""
    polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
    square_metres = sum(_ring_area_m2(rings[0]) - sum(_ring_area_m2(hole) for hole in rings[1:])
                        for rings in polygons)
    return square_metres / SQUARE_METRES_PER_ACRE


def _number(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _write_csv(path: Path, rows: list[dict], column_names: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=column_names)
        writer.writeheader()
        writer.writerows(rows)


def match_harvest_boundaries(operations_path: Path | str = DEFAULT_OPERATIONS_PATH,
                             geometry_path: Path | str = DEFAULT_GEOMETRY_PATH,
                             output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> dict:
    """Match pending operations to quarter sections; write boundaries, matches and summary.

    Every operation with a legal description, including temporarily excluded ones,
    counts when deciding whether a quarter section is shared within a year.
    """
    with Path(operations_path).open(encoding="utf-8", newline="") as handle:
        operations = list(csv.DictReader(handle))
    missing = [column for column in REQUIRED_COLUMNS if operations and column not in operations[0]]
    if missing:
        raise ValueError(f"{operations_path} is missing columns: {', '.join(missing)}")
    legal = {op["operation_id"]: parse_legal_description(op["legal_description_raw"]) for op in operations}
    candidates = [op for op in operations if op["processing_status"] == "pending_boundary_review"]
    wanted = {pid for op in candidates if legal[op["operation_id"]] for pid in legal[op["operation_id"]].pids}
    features, crs_name = find_quarter_sections(geometry_path, wanted)
    parcel_acres = {pid: polygon_acres(feature["geometry"]) for pid, feature in features.items()}

    usage: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for op in operations:
        if legal[op["operation_id"]]:
            for pid in legal[op["operation_id"]].pids:
                usage[(pid, op["year"])].append(op)

    rows = []
    for op in candidates:
        description = legal[op["operation_id"]]
        pids = description.pids if description else ()
        matched = [pid for pid in pids if pid in features]
        flags = []
        if not description:
            match_status = "unparsed"
            flags.append("unparsed_legal_description")
        else:
            match_status = "matched" if len(matched) == len(pids) else "partial" if matched else "unmatched"
            if match_status != "matched":
                flags.append("parcel_not_found")
        sharing = {other["operation_id"]: other for pid in pids for other in usage[(pid, op["year"])]}
        crops = sorted({other["crop"] or other["crop_raw"].casefold() for other in sharing.values()})
        status = ("single_operation" if len(sharing) <= 1
                  else "shared_same_crop" if len(crops) == 1 else "shared_multiple_crops")
        if status != "single_operation":
            flags.append(status)
        area = _number(op["area_ac"])
        acres = sum(parcel_acres[pid] for pid in matched) if match_status == "matched" else None
        fraction = area / acres if area is not None and acres else None
        shared_area = sum(_number(other["area_ac"]) or 0.0 for other in sharing.values())
        if fraction is not None and fraction > AREA_TOLERANCE:
            flags.append("area_exceeds_parcel")
        elif fraction is not None and fraction < LOW_COVERAGE:
            flags.append("low_parcel_coverage")
        if acres and len(sharing) > 1 and shared_area > acres * AREA_TOLERANCE:
            flags.append("quarter_year_area_exceeds_parcel")
        eligible = (match_status == "matched" and status != "shared_multiple_crops"
                    and "area_exceeds_parcel" not in flags)
        rows.append({
            "operation_id": op["operation_id"], "source_file": op["source_file"],
            "source_row": op["source_row"], "year": op["year"], "crop": op["crop"],
            "crop_raw": op["crop_raw"], "field_name_raw": op["field_name_raw"],
            "legal_description_raw": op["legal_description_raw"],
            "legal_description": description.label if description else "",
            "pids": "|".join(pids), "match_status": match_status,
            "parcel_acres": f"{acres:.1f}" if acres else "", "area_ac": op["area_ac"],
            "area_fraction": f"{fraction:.3f}" if fraction is not None else "",
            "yield_value": op["yield_value"], "yield_unit": op["yield_unit"],
            "quarter_year_status": status, "quarter_year_operation_count": len(sharing),
            "quarter_year_crops": "|".join(crops), "quarter_year_area_ac": f"{shared_area:.1f}",
            "boundary_pilot_eligible": eligible, "flags": "|".join(flags),
        })

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _write_csv(destination / "boundary_matches.csv", rows, MATCH_COLUMNS)
    boundary_features = []
    for pid in sorted(features):
        properties = features[pid]["properties"]
        boundary_features.append({
            "type": "Feature",
            "properties": {"pid": pid, "descriptor": properties.get("descriptor"),
                           "m": properties.get("m"), "rge": properties.get("rge"),
                           "twp": properties.get("twp"), "sec": properties.get("sec"),
                           "qs": properties.get("qs"), "parcel_acres": round(parcel_acres[pid], 1),
                           "boundary_source": "ats_quarter_section"},
            "geometry": features[pid]["geometry"],
        })
    collection = {"type": "FeatureCollection", "name": "harvest_field_boundaries"}
    if crs_name:
        collection["crs"] = {"type": "name", "properties": {"name": crs_name}}
    collection["features"] = boundary_features
    (destination / "field_boundaries.geojson").write_text(json.dumps(collection) + "\n", encoding="utf-8")

    eligible_rows = [row for row in rows if row["boundary_pilot_eligible"]]
    summary = {
        "schema_version": 1,
        "operations_path": str(Path(operations_path).resolve()),
        "geometry_path": str(Path(geometry_path).resolve()),
        "geometry_crs": crs_name,
        "candidate_count": len(candidates),
        "candidates_by_match_status": dict(sorted(Counter(row["match_status"] for row in rows).items())),
        "quarter_sections_requested": len(wanted),
        "quarter_sections_found": len(features),
        "quarter_sections_missing": sorted(wanted - set(features)),
        "candidates_by_quarter_year_status": dict(sorted(
            Counter(row["quarter_year_status"] for row in rows).items(),
            key=lambda item: STATUS_ORDER.index(item[0]))),
        "flags": dict(sorted(Counter(flag for row in rows for flag in row["flags"].split("|") if flag).items())),
        "boundary_pilot_eligible_count": len(eligible_rows),
        "eligible_operations_by_crop_unit": dict(sorted(Counter(
            f"{row['crop']} ({row['yield_unit']})" for row in eligible_rows).items())),
        "eligible_parcel_years_by_crop_unit": dict(sorted(Counter(
            f"{crop} ({unit})" for crop, unit, _, _ in
            {(row["crop"], row["yield_unit"], row["pids"], row["year"]) for row in eligible_rows}).items())),
        "notes": [
            "Boundaries are ATS quarter sections (ra null), not surveyed farm-field outlines.",
            "Quarter sharing counts every operation with a legal description, including exclusions.",
            "Same-crop operations sharing a parcel-year are eligible but must be aggregated "
            "(total production / total area) before training.",
            "Parcel acres are approximate (local equirectangular projection of lon/lat coordinates).",
            "No unit conversion or yield aggregation performed.",
        ],
    }
    (destination / "boundary_match_summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                            encoding="utf-8")
    return summary
