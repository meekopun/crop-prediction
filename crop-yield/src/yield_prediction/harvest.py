"""Read the farm's printed harvest reports into traceable operation records.

This imports observations only. It does not convert yield units, resolve field
geometry, aggregate operations, or approve records for model training.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from datetime import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import re
from typing import Iterable, Sequence


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PACKAGE_ROOT.parent / "harvest-data"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "data/processed/harvest"
DEFAULT_REVIEW_PATH = PACKAGE_ROOT / "config/harvest_review.json"

# These are source column positions, not a general-purpose Excel schema.
COLUMNS = {"year": 1, "harvest_date": 2, "crop_raw": 4,
           "item_raw": 6, "area_raw": 9, "yield_raw": 11, "total_raw": 12}
HEADER = {0: "Type", 1: "Year", 2: "Date", 4: "Crop", 6: "Item",
          9: "Area", 11: "Rate", 12: "Total"}
LEGAL_DESCRIPTION = re.compile(
    r"\b(?:N[EW]|S[EW]|[NS]1/2)\s*\d{1,2}\s*[-.]\s*\d{1,3}"
    r"\s*-\s*\d{1,2}\s*W\s*\d\s*M\b", re.IGNORECASE
)
CROP_NAMES = (
    (r"tall\s+fescue", "tall_fescue"),
    (r"(?:perennial\s+)?rye\s*grass", "ryegrass"),
    (r"faba\s*beans?", "faba_beans"),
    (r"field\s+peas?", "field_peas"),
    (r"sunflowers?", "sunflower"),
    (r"barley", "barley"), (r"canola", "canola"),
    (r"fescue", "fescue"), (r"oats", "oats"),
    (r"wheat", "wheat"), (r"triticale", "triticale"), (r"rye", "rye"),
)
CROP_PATTERN = re.compile("|".join(rf"(?P<{crop}>\b{pattern}\b)"
                                   for pattern, crop in CROP_NAMES), re.IGNORECASE)


@dataclass
class HarvestOperation:
    source_file: str
    source_sheet: str
    source_row: int
    source_row_end: int
    operation_id: str
    field_heading_row: int | None
    field_heading_raw: str
    field_name_raw: str
    legal_description_raw: str
    year_raw: str
    harvest_date_raw: str
    year: int | None
    harvest_date: str | None
    crop_raw: str
    item_raw: str
    crop: str | None
    variety_raw: str
    crop_mapping_status: str
    area_raw: str
    yield_raw: str
    total_raw: str
    area_ac: float | None
    yield_value: float | None
    yield_unit: str | None
    total_value: float | None
    total_unit: str | None
    field_id: str = ""
    geometry_id: str = ""
    geometry_status: str = "unmatched"
    included_in_pilot: bool = False
    qa_flags: tuple[str, ...] = ()
    processing_status: str = "pending_boundary_review"
    exclusion_reason: str = ""
    crop_mapping_note: str = ""


@dataclass(frozen=True)
class HarvestIssue:
    source_file: str
    source_sheet: str
    source_row: int
    operation_id: str
    severity: str
    code: str
    column: str
    raw_value: str
    message: str


@dataclass
class _PendingOperation:
    start: int
    end: int
    field: str
    field_row: int | None
    parts: dict[str, list[str]]


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def parse_measurement(value: object, continuation: object = None) -> tuple[float, str | None]:
    """Parse a number and optional unit; retain unsupported units for validation.

    Missing/invalid numbers raise ValueError. Missing units remain None; they
    are never inferred from neighboring operations or converted to bushels.
    """
    text = " ".join(part for part in (_text(value), _text(continuation)) if part)
    match = re.fullmatch(
        r"([+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+))\s*([^\d]*?)",
        text,
    )
    if not match:
        raise ValueError(f"Invalid measurement: {text!r}")
    number = float(match[1].replace(",", ""))
    if not math.isfinite(number):
        raise ValueError(f"Non-finite measurement: {text!r}")
    unit = re.sub(r"\s+", "", match[2].lower()) or None
    return number, unit


def _row_kind(row: Sequence[object]) -> str:
    first = _text(row[0])
    if first.casefold() == "harvesting":
        return "operation"
    if first.casefold() == "type":
        return "header"
    if first == "Operation History Summary Report" or re.fullmatch(
        r"Page\s+\d+\s+of\s+\d+", first, re.IGNORECASE
    ):
        return "report"
    if not any(_text(value) for value in row):
        return "blank"
    if not first and _text(row[3]) and not any(
        _text(value) for i, value in enumerate(row) if i != 3
    ):
        return "report"  # Repeated farm name in column D.
    if first and not any(_text(value) for value in row[1:]):
        return "field"
    if not first and all(not _text(value) for i, value in enumerate(row)
                         if i not in COLUMNS.values()):
        return "continuation"
    return "unexpected"


def _normalize_crop(text: str) -> tuple[str | None, str, str]:
    matches = list(CROP_PATTERN.finditer(text))
    crop_names = {match.lastgroup for match in matches}
    if len(crop_names) == 1:
        # An explicit name may follow a variety (e.g. "Dekalb ... canola").
        # Only remove a leading crop name; retain embedded variety text intact.
        first = matches[0]
        variety = text[first.end():].strip() if first.start() == 0 else text
        return first.lastgroup, variety, "recognized_name"
    return None, text, "needs_review"


def _parse_date(text: str) -> str:
    for date_format in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, date_format).date().isoformat()
        except ValueError:
            pass
    raise ValueError(f"Invalid harvest date: {text!r}")


def validate_operation(record: HarvestOperation) -> list[HarvestIssue]:
    """Check measurement meaning without repairing or discarding source data."""
    issues: list[HarvestIssue] = []

    def issue(code: str, column: str, raw: object, message: str,
              severity: str = "error") -> None:
        issues.append(HarvestIssue(record.source_file, record.source_sheet,
                                   record.source_row, record.operation_id,
                                   severity, code, column, _text(raw), message))

    if not record.field_heading_raw:
        issue("missing_field_heading", "A", "", "No field heading precedes this operation.")
    elif not record.legal_description_raw:
        issue("missing_legal_description", "A", record.field_heading_raw,
              "Name-only field needs a reviewed boundary.", "warning")
    if record.year is None:
        issue("invalid_year", "B", record.year_raw, "Operation year is missing or invalid.")
    if record.harvest_date is None:
        issue("invalid_harvest_date", "C", record.harvest_date_raw, "Harvest date is missing or invalid.")
    elif record.year is not None and int(record.harvest_date[:4]) != record.year:
        issue("date_year_mismatch", "C", record.harvest_date,
              "Harvest date does not match the operation year.")
    if not record.crop_raw:
        issue("missing_crop", "E", "", "Crop label is missing.")
    elif record.crop is None:
        issue("unmapped_crop", "E", record.crop_raw,
              "Crop cannot be identified from an explicit crop name; review this label.", "warning")
    for name, value, raw, column, positive in (
        ("area", record.area_ac, record.area_raw, "J", True),
        ("yield", record.yield_value, record.yield_raw, "L", False),
        ("total", record.total_value, record.total_raw, "M", False),
    ):
        if value is None:
            issue(f"invalid_{name}_measurement", column, raw,
                  f"{name.title()} has no valid number (area must be in ac).")
        elif not math.isfinite(value) or value < 0 or (positive and value == 0):
            issue(f"invalid_{name}_value", column, raw,
                  f"{name.title()} must be {'positive' if positive else 'nonnegative'} and finite.")
    for name, unit, allowed, raw, column in (
        ("yield", record.yield_unit, {"bu/ac", "lb/ac"}, record.yield_raw, "L"),
        ("total", record.total_unit, {"bu", "lb"}, record.total_raw, "M"),
    ):
        if unit not in allowed:
            issue(f"unsupported_{name}_unit", column, raw,
                  f"Explicit unit must be one of {sorted(allowed)}; got {unit!r}.")
    expected_unit = {"bu/ac": "bu", "lb/ac": "lb"}.get(record.yield_unit)
    if expected_unit and record.total_unit in {"bu", "lb"}:
        if record.total_unit != expected_unit:
            issue("rate_total_unit_mismatch", "L/M", f"{record.yield_unit}, {record.total_unit}",
                  "Yield rate and total use different units.")
        elif all(value is not None and math.isfinite(value) and value >= 0
                 for value in (record.area_ac, record.yield_value, record.total_value)):
            expected_total = record.area_ac * record.yield_value
            tolerance = max(1.0, 0.001 * abs(record.total_value))
            if abs(expected_total - record.total_value) > tolerance:
                issue("area_rate_total_mismatch", "J/L/M", record.total_raw,
                      f"Area × rate = {expected_total:g}, reported total = {record.total_value:g}; "
                      f"tolerance = {tolerance:g}.")
    return issues


def parse_rows(rows: Iterable[Sequence[object]], *, source_file: str,
               source_sheet: str = "Sheet1") -> tuple[list[HarvestOperation], list[HarvestIssue]]:
    """Parse worksheet rows. Row numbers in results are one-based Excel rows."""
    records: list[HarvestOperation] = []
    issues: list[HarvestIssue] = []
    field_heading = ""
    field_heading_row: int | None = None
    pending: _PendingOperation | None = None

    def finish() -> None:
        nonlocal pending
        if pending is None:
            return
        raw = {name: " ".join(parts) for name, parts in pending.parts.items()}
        op_id = hashlib.sha256(
            f"{source_file}\0{source_sheet}\0{pending.start}".encode()
        ).hexdigest()[:20]
        try:
            year_number = float(raw["year"])
            year = int(year_number) if year_number.is_integer() and 1900 <= year_number <= 2100 else None
        except ValueError:
            year = None
        try:
            harvest_date = _parse_date(raw["harvest_date"])
        except ValueError:
            harvest_date = None
        measurements = {}
        for name in ("area", "yield", "total"):
            try:
                measurements[name] = parse_measurement(raw[f"{name}_raw"])
            except ValueError:
                measurements[name] = (None, None)
        heading = pending.field
        legal = LEGAL_DESCRIPTION.search(heading)
        crop, variety, mapping_status = _normalize_crop(raw["crop_raw"])
        record = HarvestOperation(
            source_file=source_file, source_sheet=source_sheet,
            source_row=pending.start, source_row_end=pending.end, operation_id=op_id,
            field_heading_row=pending.field_row, field_heading_raw=heading,
            field_name_raw=heading[:legal.start()].strip() if legal else heading.strip(),
            legal_description_raw=legal[0] if legal else "",
            year_raw=raw["year"], harvest_date_raw=raw["harvest_date"],
            year=year, harvest_date=harvest_date, crop_raw=raw["crop_raw"], item_raw=raw["item_raw"],
            crop=crop, variety_raw=variety, crop_mapping_status=mapping_status,
            area_raw=raw["area_raw"], yield_raw=raw["yield_raw"], total_raw=raw["total_raw"],
            area_ac=measurements["area"][0] if measurements["area"][1] == "ac" else None,
            yield_value=measurements["yield"][0], yield_unit=measurements["yield"][1],
            total_value=measurements["total"][0], total_unit=measurements["total"][1],
        )
        record_issues = validate_operation(record)
        file_year = re.search(r"\b(?:19|20)\d{2}\b", source_file)
        if file_year and year is not None and int(file_year[0]) != year:
            record_issues.append(HarvestIssue(source_file, source_sheet, record.source_row,
                op_id, "error", "file_year_mismatch", "B", raw["year"],
                "Operation year differs from the year in the source filename."))
        record.qa_flags = tuple(problem.code for problem in record_issues)
        records.append(record)
        issues.extend(record_issues)
        pending = None

    for row_number, values in enumerate(rows, start=1):
        row = list(values)
        if len(row) < 13:
            row.extend([""] * (13 - len(row)))
        kind = _row_kind(row)
        if kind == "operation":
            finish()
            pending = _PendingOperation(row_number, row_number, field_heading, field_heading_row,
                {name: [_text(row[column])] for name, column in COLUMNS.items()})
        elif kind == "field":
            finish()
            field_heading, field_heading_row = str(row[0]), row_number
        elif kind == "continuation" and pending is not None:
            for name, column in COLUMNS.items():
                if _text(row[column]):
                    pending.parts[name].append(_text(row[column]))
            pending.end = row_number
        elif kind == "header":
            finish()
            if any(_text(row[column]).casefold() != label.casefold()
                   for column, label in HEADER.items()):
                raise ValueError(f"{source_file}:{source_sheet}!A{row_number}: "
                                 "report columns differ from the expected harvest layout")
        elif kind == "report":
            # Preserve field identity across pages, but never append footer dates.
            pass
        elif kind != "blank":
            finish()
            issues.append(HarvestIssue(source_file, source_sheet, row_number, "", "error",
                "unexpected_row", "", json.dumps(row, default=str),
                "Unrecognized report row; field context cleared to avoid a false match."))
            field_heading, field_heading_row = "", None
    finish()
    return records, issues


def parse_workbook(path: Path | str) -> tuple[list[HarvestOperation], list[HarvestIssue]]:
    """Read Sheet1 from an .xls file without modifying the source workbook."""
    try:
        import xlrd
    except ImportError as exc:
        raise RuntimeError("Harvest import requires xlrd. Install with `pip install -e '.[harvest]'`.") from exc
    source = Path(path)
    diagnostics = io.StringIO()
    try:
        book = xlrd.open_workbook(str(source), logfile=diagnostics)
    except xlrd.XLRDError as exc:
        raise ValueError(f"Cannot read {source}: {exc}") from exc
    try:
        if "Sheet1" not in book.sheet_names():
            raise ValueError(f"{source}: expected worksheet 'Sheet1'")
        sheet = book.sheet_by_name("Sheet1")
        if sheet.ncols < 13:
            raise ValueError(f"{source}: expected at least 13 report columns")

        def rows():
            for row_number in range(sheet.nrows):
                row = sheet.row_values(row_number)
                if sheet.cell_type(row_number, 2) == xlrd.XL_CELL_DATE:
                    try:
                        row[2] = xlrd.xldate_as_datetime(row[2], book.datemode).date().isoformat()
                    except (ValueError, OverflowError):
                        row[2] = f"Invalid Excel date: {row[2]}"
                yield row

        records, issues = parse_rows(rows(), source_file=source.name)
        for message in diagnostics.getvalue().splitlines():
            if message.strip():
                issues.append(HarvestIssue(source.name, "Sheet1", 0, "", "warning",
                    "workbook_warning", "", "", message.strip()))
        if not records:
            raise ValueError(f"{source}: no Harvesting operations found; check the report layout")
        return records, issues
    finally:
        book.release_resources()


def _flag_shared_fields(records: list[HarvestOperation]) -> list[HarvestIssue]:
    groups: dict[tuple[str, int | None], list[HarvestOperation]] = defaultdict(list)
    for record in records:
        if record.field_heading_raw:
            key = (" ".join(record.field_heading_raw.lower().split()), record.year)
            groups[key].append(record)
    issues = []
    for group in groups.values():
        if len(group) < 2:
            continue
        crops = {record.crop or record.crop_raw.casefold() for record in group}
        code = "multiple_crops_same_field_year" if len(crops) > 1 else "multiple_operations_same_field_year"
        for record in group:
            record.qa_flags += (code,)
            issues.append(HarvestIssue(record.source_file, record.source_sheet,
                record.source_row, record.operation_id, "warning", code, "A/E",
                record.field_heading_raw,
                "Multiple harvest operations share this field heading/year; "
                "review subareas and repeated passes before aggregating."))
    return issues


def _write_csv(path: Path, rows: Iterable[dict], column_names: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=column_names)
        writer.writeheader()
        writer.writerows(rows)


def apply_harvest_review(records: list[HarvestOperation], issues: list[HarvestIssue],
                         review_path: Path | str) -> list[HarvestIssue]:
    """Apply saved crop confirmations and temporary, operation-specific exclusions.

    Original crop labels and all records remain available. Exclusions never
    approve any remaining record's geometry or turn it into a training example.
    """
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    if not isinstance(review, dict) or review.get("schema_version") != 1:
        raise ValueError("Harvest review must have schema_version=1.")
    aliases = review.get("crop_aliases", {})
    exclusions = review.get("excluded_operations", {})
    if not isinstance(aliases, dict) or not isinstance(exclusions, dict):
        raise ValueError("crop_aliases and excluded_operations must be objects.")
    allowed_crops = {crop for _, crop in CROP_NAMES}
    if any(not isinstance(label, str) or not label.strip() or not isinstance(crop, str) or crop not in allowed_crops
           for label, crop in aliases.items()):
        raise ValueError("Each crop alias needs a nonempty label and a supported normalized crop.")
    if any(not isinstance(reason, str) or not reason.strip() for reason in exclusions.values()):
        raise ValueError("Each excluded operation must have a nonempty reason.")
    missing_ids = set(exclusions) - {record.operation_id for record in records}
    if missing_ids:
        raise ValueError("Review exclusions reference operations absent from this import: "
                         + ", ".join(sorted(missing_ids))
                         + ". Check source filenames/rows or use --no-review for unrelated reports.")
    normalized_aliases = {" ".join(label.casefold().split()): crop for label, crop in aliases.items()}
    if len(normalized_aliases) != len(aliases):
        raise ValueError("Crop aliases contain duplicate labels after normalization.")
    confirmed_ids = set()
    for record in records:
        label = " ".join(record.crop_raw.casefold().split())
        if label in normalized_aliases:
            record.crop = normalized_aliases[label]
            record.crop_mapping_status = "confirmed_by_review"
            record.crop_mapping_note = (
                f"{review.get('reviewed_by', 'reviewer')}, {review.get('reviewed_on', 'undated')}: "
                f"{record.crop_raw} confirmed as {record.crop}."
            )
            record.qa_flags = tuple(flag for flag in record.qa_flags if flag != "unmapped_crop")
            confirmed_ids.add(record.operation_id)
        if record.operation_id in exclusions:
            record.processing_status = "excluded"
            record.exclusion_reason = exclusions[record.operation_id]
    return [issue for issue in issues
            if not (issue.code == "unmapped_crop" and issue.operation_id in confirmed_ids)]


def import_harvest_directory(input_dir: Path | str = DEFAULT_INPUT_DIR,
                             output_dir: Path | str = DEFAULT_OUTPUT_DIR,
                             *, review_file: Path | str | None = None) -> dict:
    """Read all .xls reports, then write operations, exceptions and summary.

    A workbook/layout failure aborts before writing outputs. Row-level issues
    are retained in the outputs, including operations with invalid measurements.
    """
    source_dir, destination = Path(input_dir), Path(output_dir)
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Harvest input directory does not exist: {source_dir}")
    paths = sorted(path for path in source_dir.iterdir()
                   if path.is_file() and path.suffix.lower() == ".xls" and not path.name.startswith("~$"))
    if not paths:
        raise FileNotFoundError(f"No .xls harvest reports found in {source_dir}")
    records: list[HarvestOperation] = []
    issues: list[HarvestIssue] = []
    for path in paths:
        file_records, file_issues = parse_workbook(path)
        records.extend(file_records)
        issues.extend(file_issues)
    if review_file is not None:
        issues = apply_harvest_review(records, issues, review_file)
    issues.extend(_flag_shared_fields(records))
    error_ids = {issue.operation_id for issue in issues if issue.severity == "error" and issue.operation_id}
    for record in records:
        unresolved = {"missing_legal_description", "multiple_crops_same_field_year"}
        if record.processing_status != "excluded" and (
            record.operation_id in error_ids or record.crop is None or unresolved.intersection(record.qa_flags)
        ):
            record.processing_status = "needs_review"
    candidates = [record for record in records if record.processing_status == "pending_boundary_review"]
    summary = {
        "schema_version": 1,
        "input_dir": str(source_dir.resolve()),
        "workbook_count": len(paths),
        "operation_count": len(records),
        "operations_by_workbook": dict(sorted(Counter(record.source_file for record in records).items())),
        "operations_by_year": dict(sorted(Counter(str(record.year) for record in records).items())),
        "operations_by_yield_unit": dict(sorted(Counter(record.yield_unit or "missing" for record in records).items())),
        "operations_by_crop": dict(sorted(Counter(record.crop or "unmapped" for record in records).items())),
        "operations_with_errors": len(error_ids),
        "operations_with_flags": sum(bool(record.qa_flags) for record in records),
        "issue_count": len(issues),
        "issues_by_code": dict(sorted(Counter(issue.code for issue in issues).items())),
        "issues_by_severity": dict(sorted(Counter(issue.severity for issue in issues).items())),
        "geometry_unmatched_count": len(records),
        "included_in_pilot_count": 0,
        "review_file": str(Path(review_file).resolve()) if review_file is not None else None,
        "confirmed_crop_mapping_count": sum(record.crop_mapping_status == "confirmed_by_review" for record in records),
        "excluded_operation_count": sum(record.processing_status == "excluded" for record in records),
        "processing_candidate_count": len(candidates),
        "operations_by_processing_status": dict(sorted(Counter(record.processing_status for record in records).items())),
        "notes": ["No unit conversion, aggregation or geometry matching performed.",
                  "Recognized crop names still require review; all records start excluded from the pilot.",
                  "processing_candidates.csv omits temporary exclusions and unresolved crop/measurement errors; "
                  "its records still require verified boundaries and matching features.",
                  "source_row and field_heading_row use one-based Excel row numbers; "
                  "source_row=0 on an issue denotes a workbook-level diagnostic."],
    }
    destination.mkdir(parents=True, exist_ok=True)
    operation_rows = []
    for record in records:
        row = asdict(record)
        row["qa_flags"] = "|".join(record.qa_flags)
        operation_rows.append(row)
    _write_csv(destination / "operations.csv", operation_rows, [field.name for field in fields(HarvestOperation)])
    _write_csv(destination / "processing_candidates.csv",
               (row for row in operation_rows if row["processing_status"] == "pending_boundary_review"),
               [field.name for field in fields(HarvestOperation)])
    _write_csv(destination / "exceptions.csv", (asdict(issue) for issue in issues),
               [field.name for field in fields(HarvestIssue)])
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
