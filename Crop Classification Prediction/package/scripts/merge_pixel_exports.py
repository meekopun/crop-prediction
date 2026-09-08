"""
Merge batched pixel export CSVs into a single training table.

Usage:
    python scripts/merge_pixel_exports.py
    python scripts/merge_pixel_exports.py --pattern "data/my_pixels_part*.csv" --output data/my_pixels.csv
"""
import argparse
import csv
import glob
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pattern",
        default="data/sentinel2_pixel_samples_part*.csv",
        help="Glob pattern for batched pixel export CSVs",
    )
    parser.add_argument(
        "--output",
        default="data/sentinel2_pixel_samples.csv",
        help="Merged output CSV path",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched pattern: {args.pattern}")

    output_path = Path(args.output)
    header = None
    row_count = 0

    with output_path.open("w", newline="") as fout:
        writer = None
        for path in files:
            with open(path, newline="") as fin:
                reader = csv.reader(fin)
                this_header = next(reader)
                if header is None:
                    header = this_header
                    writer = csv.writer(fout)
                    writer.writerow(header)
                elif this_header != header:
                    raise SystemExit(f"Header mismatch in {path}")

                for row in reader:
                    writer.writerow(row)
                    row_count += 1

    print(f"Merged {len(files)} files into {output_path}")
    print(f"Wrote {row_count} data rows")


if __name__ == "__main__":
    main()
