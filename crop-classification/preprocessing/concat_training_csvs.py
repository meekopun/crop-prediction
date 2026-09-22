"""
Concatenate one or more CSV files into a single training dataset.

Useful for appending newly exported pixel-level samples to an existing
training CSV. Headers must match exactly.
"""
import argparse
import csv
import glob
from pathlib import Path

from common import load_tracker, save_tracker

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def row_key(row, key_cols):
    if not key_cols:
        return None
    return tuple(row[col] for col in key_cols)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pattern",
        required=True,
        help="Glob pattern for input CSVs, quoted to avoid shell expansion issues",
    )
    parser.add_argument(
        "--output",
        default=str(DATA_DIR / "sentinel2_pixel_samples_all.csv"),
        help="Merged output CSV path",
    )
    parser.add_argument(
        "--dedupe-cols",
        default="",
        help="Optional comma-separated columns to deduplicate on, e.g. 'system:index'",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched pattern: {args.pattern}")

    dedupe_cols = [c.strip() for c in args.dedupe_cols.split(",") if c.strip()]
    seen = set()
    header = None
    rows_written = 0
    duplicates_skipped = 0

    output_path = Path(args.output)
    with output_path.open("w", newline="") as fout:
        writer = None
        for path in files:
            with open(path, newline="") as fin:
                reader = csv.DictReader(fin)
                if header is None:
                    header = reader.fieldnames
                    writer = csv.DictWriter(fout, fieldnames=header)
                    writer.writeheader()
                elif reader.fieldnames != header:
                    raise SystemExit(f"Header mismatch in {path}")

                for row in reader:
                    key = row_key(row, dedupe_cols)
                    if key is not None:
                        if key in seen:
                            duplicates_skipped += 1
                            continue
                        seen.add(key)
                    writer.writerow(row)
                    rows_written += 1

    print(f"Merged {len(files)} files into {output_path}")
    print(f"Wrote {rows_written} data rows")
    if dedupe_cols:
        print(f"Skipped {duplicates_skipped} duplicate rows using columns: {dedupe_cols}")

    tracker = load_tracker()
    tracker["merged_outputs"][str(output_path)] = {
        "source_pattern": args.pattern,
        "source_files": files,
        "row_count": rows_written,
        "duplicates_skipped": duplicates_skipped,
        "dedupe_cols": dedupe_cols,
        "type": "concat_training_csvs",
    }
    save_tracker(tracker)


if __name__ == "__main__":
    main()
