from collections import Counter
import csv
from datetime import date as datetime_date
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from yield_prediction.harvest import (
    DEFAULT_INPUT_DIR,
    DEFAULT_REVIEW_PATH,
    apply_harvest_review,
    import_harvest_directory,
    parse_measurement,
    parse_rows,
    parse_workbook,
)


def row(**cells):
    values = [""] * 13
    for column, value in cells.items():
        values[ord(column) - ord("A")] = value
    return values


def header():
    return row(A="Type", B="Year", C="Date", E="Crop", G="Item",
               J="Area", L="Rate", M="Total")


def operation(**overrides):
    cells = dict(A="Harvesting", B="2025", C="9/1/2025", E="Oats (Waldern)",
                 G="Oats (Waldern)", J="10 ac", L="50 bu/ac", M="500 bu")
    cells.update(overrides)
    return row(**cells)


class HarvestParserTests(unittest.TestCase):
    def test_measurements_keep_units_and_reject_invalid_numbers(self):
        self.assertEqual(parse_measurement("21,840.0000", "bu"), (21840.0, "bu"))
        self.assertEqual(parse_measurement("244.7368 LB / AC"), (244.7368, "lb/ac"))
        self.assertEqual(parse_measurement("0 bu/ac"), (0.0, "bu/ac"))
        self.assertEqual(parse_measurement("50"), (50.0, None))
        self.assertEqual(parse_measurement("50 kg/ac"), (50.0, "kg/ac"))
        for value in ("", "NaN bu", "inf bu", "5,00 bu", "50 bu 100 lb"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_measurement(value)

    def test_page_breaks_preserve_heading_and_wrapped_text(self):
        rows = [row(A="Field 3  SE 30-71-9 W6M"), header(), row(),
                row(A="Page 1 of 4", M="8/18/2026 3:12 PM"),
                row(D="Beaverlodge Seed Farm"), row(A="Operation History Summary Report"),
                operation(E="Barley (Alberta", G="Barley (Alberta Select)",
                          L="50", M="500"),
                row(E="Select)", L="bu/ac", M="bu"),
                row(A="Page 2 of 4", M="8/18/2026 3:12 PM")]
        records, issues = parse_rows(rows, source_file="2025-harvest-data.xls")
        self.assertEqual(issues, [])
        record = records[0]
        self.assertEqual((record.source_row, record.source_row_end, record.field_heading_row), (7, 8, 1))
        self.assertEqual(record.crop_raw, "Barley (Alberta Select)")
        self.assertEqual(record.crop, "barley")
        self.assertEqual(record.variety_raw, "(Alberta Select)")
        self.assertEqual(record.harvest_date, "2025-09-01")
        self.assertEqual(record.field_name_raw, "Field 3")
        self.assertEqual(record.legal_description_raw, "SE 30-71-9 W6M")
        self.assertEqual(record.yield_value, 50)
        self.assertFalse(record.included_in_pilot)
        self.assertEqual(record.geometry_status, "unmatched")

    def test_wrapped_operation_can_continue_across_footer(self):
        rows = [row(A="Field NE3-72-10 W6M"), header(), operation(L="50"),
                row(A="Page 1 of 2", M="8/18/2026 3:12 PM"),
                row(D="Beaverlodge Seed Farm"), row(A="Operation History Summary Report"),
                row(L="bu/ac")]
        records, issues = parse_rows(rows, source_file="2025.xls")
        self.assertEqual(issues, [])
        self.assertEqual(records[0].yield_raw, "50 bu/ac")

    def test_separate_operations_and_unmatched_variety_are_preserved(self):
        rows = [row(A="Ron Jones"), header(), operation(E="DK 902 TF 2025"),
                row(E="(Round up Ready)"), operation(E="Fescue (Brynn)", L="50 lb/ac", M="500 lb")]
        records, issues = parse_rows(rows, source_file="2025.xls")
        self.assertEqual(len(records), 2)
        self.assertNotEqual(records[0].operation_id, records[1].operation_id)
        self.assertIsNone(records[0].crop)
        self.assertIn("unmapped_crop", records[0].qa_flags)
        self.assertEqual(records[1].crop, "fescue")
        self.assertEqual(records[1].yield_unit, "lb/ac")
        self.assertEqual(records[1].yield_value, 50)
        repeated, _ = parse_rows(rows, source_file="2025.xls")
        self.assertEqual([r.operation_id for r in records], [r.operation_id for r in repeated])

    def test_explicit_embedded_crop_names_are_recognized(self):
        for label, expected in [("Dekalb 96 SC canola (truflex)", "canola"),
                                ("Rye Grass (Perennial)", "ryegrass"),
                                ("Tall Fescue", "tall_fescue")]:
            with self.subTest(label=label):
                records, _ = parse_rows([row(A="Field SW9-72-10 W6M"), operation(E=label)],
                                        source_file="2025.xls")
                self.assertEqual(records[0].crop, expected)

    def test_validation_exposes_errors_without_dropping_operation(self):
        rows = [operation(B="2024", C="9/1/2025", J="0 ac", L="50 bu/ac", M="500 lb")]
        records, issues = parse_rows(rows, source_file="2025.xls")
        self.assertEqual(len(records), 1)
        codes = {issue.code for issue in issues}
        self.assertTrue({"missing_field_heading", "date_year_mismatch", "file_year_mismatch",
                         "invalid_area_value", "rate_total_unit_mismatch"}.issubset(codes))

    def test_missing_units_and_bad_dates_are_never_inferred(self):
        records, issues = parse_rows([row(A="Field SW9-72-10 W6M"),
            operation(B="bad year", C="not a date", L="50", M="NaN bu")], source_file="report.xls")
        self.assertIsNone(records[0].yield_unit)
        by_code = {issue.code: issue for issue in issues}
        self.assertEqual(by_code["invalid_year"].raw_value, "bad year")
        self.assertEqual(by_code["invalid_harvest_date"].raw_value, "not a date")
        self.assertIn("unsupported_yield_unit", by_code)
        self.assertIn("invalid_total_measurement", by_code)

    def test_arithmetic_tolerance_and_zero_yield(self):
        for total, has_error in [("500.4 bu", False), ("502 bu", True), ("0 bu", False)]:
            rate = "0 bu/ac" if total == "0 bu" else "50 bu/ac"
            records, issues = parse_rows([row(A="Field SW9-72-10 W6M"),
                operation(L=rate, M=total)], source_file="2025.xls")
            self.assertEqual(any(issue.code == "area_rate_total_mismatch" for issue in issues), has_error)

    def test_layout_mismatch_aborts_and_unknown_rows_clear_field_context(self):
        changed_header = header()
        changed_header[11] = "Cost"
        with self.assertRaisesRegex(ValueError, "columns differ"):
            parse_rows([changed_header, operation()], source_file="report.xls")
        records, issues = parse_rows([row(A="Field SW9-72-10 W6M"),
            row(A="Other operation", J="10 ac"), operation()], source_file="2025.xls")
        self.assertEqual(records[0].field_heading_raw, "")
        self.assertIn("unexpected_row", {issue.code for issue in issues})

    def test_export_keeps_multiple_crops_and_flags_aggregation(self):
        rows = [row(A="Field SW9-72-10 W6M"), operation(),
                operation(E="Fescue (Brynn)", L="50 lb/ac", M="500 lb")]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input"
            source.mkdir()
            (source / "2025.xls").touch()
            destination = Path(directory) / "output"
            with patch("yield_prediction.harvest.parse_workbook",
                       side_effect=lambda path: parse_rows(rows, source_file=path.name)):
                summary = import_harvest_directory(source, destination)
            self.assertEqual(summary["operation_count"], 2)
            self.assertEqual(summary["issues_by_code"]["multiple_crops_same_field_year"], 2)
            with (destination / "operations.csv").open() as handle:
                exported = list(csv.DictReader(handle))
            self.assertEqual(len(exported), 2)
            self.assertEqual(exported[1]["yield_unit"], "lb/ac")
            self.assertIn("multiple_crops_same_field_year", exported[0]["qa_flags"])
            self.assertEqual(json.loads((destination / "summary.json").read_text()), summary)

    def test_no_input_or_workbook_failure_creates_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            destination = source / "output"
            with self.assertRaises(FileNotFoundError):
                import_harvest_directory(source, destination)
            (source / "bad.xls").touch()
            with patch("yield_prediction.harvest.parse_workbook", side_effect=ValueError("bad layout")):
                with self.assertRaisesRegex(ValueError, "bad layout"):
                    import_harvest_directory(source, destination)
            self.assertFalse(destination.exists())

    def test_review_confirms_canola_and_excludes_only_named_operations(self):
        raw_rows = [row(A="Field SW9-72-10 W6M"),
                    operation(E="DK 902 TF 2025 (Round up Ready)"),
                    operation(E="DK 902 TF 2026 (Round up Ready)"),
                    operation()]
        records, issues = parse_rows(raw_rows, source_file="2025.xls")
        review = {"schema_version": 1, "reviewed_by": "user", "reviewed_on": "2026-10-06",
                  "crop_aliases": {"DK 902 TF 2025 (Round up Ready)": "canola"},
                  "excluded_operations": {records[2].operation_id: "Awaiting crop-area boundaries."}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.json"
            path.write_text(json.dumps(review))
            remaining = apply_harvest_review(records, issues, path)
        self.assertEqual(records[0].crop, "canola")
        self.assertEqual(records[0].crop_raw, "DK 902 TF 2025 (Round up Ready)")
        self.assertEqual(records[0].crop_mapping_status, "confirmed_by_review")
        self.assertNotIn("unmapped_crop", records[0].qa_flags)
        self.assertFalse([issue for issue in remaining if issue.operation_id == records[0].operation_id])
        self.assertIsNone(records[1].crop)
        self.assertIn("unmapped_crop", records[1].qa_flags)
        self.assertEqual(records[2].processing_status, "excluded")
        self.assertEqual(records[0].geometry_status, "unmatched")
        self.assertFalse(any(record.included_in_pilot for record in records))

    def test_stale_review_exclusions_abort_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input"
            source.mkdir()
            (source / "2025.xls").touch()
            path = Path(directory) / "review.json"
            path.write_text(json.dumps({"schema_version": 1,
                "excluded_operations": {"missing_id": "awaiting boundary"}}))
            output = Path(directory) / "output"
            with patch("yield_prediction.harvest.parse_workbook",
                       return_value=parse_rows([operation()], source_file="2025.xls")):
                with self.assertRaisesRegex(ValueError, "absent from this import"):
                    import_harvest_directory(source, output, review_file=path)
            self.assertFalse(output.exists())


@unittest.skipUnless(importlib.util.find_spec("xlrd"), "requires optional xlrd dependency")
class HarvestWorkbookTests(unittest.TestCase):
    def test_excel_date_cells_use_workbook_date_system(self):
        import xlrd
        from types import SimpleNamespace
        from unittest.mock import Mock

        # Check the 1904 date system rather than assuming the 1900 default.
        date_serial = (datetime_date(2025, 9, 1) - datetime_date(1904, 1, 1)).days
        raw_rows = [row(A="Field SW9-72-10 W6M"), operation(C=date_serial)]
        sheet = SimpleNamespace(nrows=2, ncols=13, row_values=lambda i: raw_rows[i].copy(),
            cell_type=lambda i, j: xlrd.XL_CELL_DATE if i == 1 and j == 2 else xlrd.XL_CELL_TEXT)
        book = SimpleNamespace(datemode=1, sheet_names=lambda: ["Sheet1"],
                               sheet_by_name=lambda name: sheet, release_resources=Mock())
        with patch("xlrd.open_workbook", return_value=book):
            records, issues = parse_workbook("2025.xls")
        self.assertEqual(issues, [])
        self.assertEqual(records[0].harvest_date, "2025-09-01")
        book.release_resources.assert_called_once()

    @unittest.skipUnless(DEFAULT_INPUT_DIR.is_dir(), "local farm reports unavailable")
    def test_local_workbooks_reconcile_to_source_counts_and_units(self):
        expected = {2020: (25, 17, 8), 2021: (32, 28, 4), 2022: (36, 27, 9),
                    2023: (36, 28, 8), 2024: (36, 34, 2), 2025: (41, 38, 3)}
        paths = sorted(DEFAULT_INPUT_DIR.glob("*.xls"))
        if len(paths) != 6:
            self.skipTest("requires the six local 2020–2025 reports")
        for path in paths:
            year = int(path.name[:4])
            records, issues = parse_workbook(path)
            with self.subTest(year=year):
                units = Counter(record.yield_unit for record in records)
                self.assertEqual((len(records), units["bu/ac"], units["lb/ac"]), expected[year])
                self.assertFalse([issue for issue in issues if issue.severity == "error"])
                self.assertTrue(all(record.field_heading_row is not None for record in records))
                if year == 2025:
                    target = next(record for record in records if record.source_row == 47)
                    self.assertEqual(target.field_heading_row, 40)
                    self.assertEqual(target.field_name_raw, "3")

    @unittest.skipUnless(DEFAULT_INPUT_DIR.is_dir(), "local farm reports unavailable")
    def test_cli_imports_all_reports_from_another_directory(self):
        root = Path(__file__).resolve().parents[1]
        if len(list(DEFAULT_INPUT_DIR.glob("*.xls"))) != 6:
            self.skipTest("requires the six local harvest reports")
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "output"
            result = subprocess.run([sys.executable, str(root / "scripts/import_harvest_data.py"),
                "--output-dir", str(destination)], cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Parsed 206 operations", result.stdout)
            summary = json.loads((destination / "summary.json").read_text())
            self.assertEqual(summary["operations_by_yield_unit"], {"bu/ac": 172, "lb/ac": 34})
            self.assertEqual(summary["confirmed_crop_mapping_count"], 8)
            self.assertEqual(summary["excluded_operation_count"], 10)
            self.assertEqual(summary["processing_candidate_count"], 196)
            self.assertNotIn("unmapped_crop", summary["issues_by_code"])
            with (destination / "operations.csv").open() as handle:
                all_rows = list(csv.DictReader(handle))
            with (destination / "processing_candidates.csv").open() as handle:
                candidates = list(csv.DictReader(handle))
            self.assertEqual(len(all_rows), 206)
            self.assertEqual(len(candidates), 196)
            review = json.loads(DEFAULT_REVIEW_PATH.read_text())
            candidate_ids = {record["operation_id"] for record in candidates}
            self.assertFalse(candidate_ids.intersection(review["excluded_operations"]))
            self.assertTrue(all(record["included_in_pilot"] == "False" for record in candidates))


if __name__ == "__main__":
    unittest.main()
