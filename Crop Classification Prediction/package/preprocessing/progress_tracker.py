"""
Summarize preprocessing progress for quarter-section sampling and pixel export
batches.

Usage:
    python preprocessing/progress_tracker.py
"""
import argparse
import json
from pathlib import Path

from common import DATA_DIR, TRACKER_PATH, load_tracker


def summarize_checkpoint(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {"completed": 0, "failed": 0, "total": 0}
    with open(path) as f:
        state = json.load(f)
    return {
        "completed": len(state.get("completed_batches", [])),
        "failed": len(state.get("failed_batches", {})),
        "total": state.get("total_batches", 0),
    }


def count_rows(path):
    with open(path, "r") as f:
        return max(sum(1 for _ in f) - 1, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the raw tracker JSON instead of a summary",
    )
    args = parser.parse_args()

    tracker = load_tracker()
    if args.json:
        print(json.dumps(tracker, indent=2, sort_keys=True))
        return

    print(f"Tracker file: {TRACKER_PATH}")
    print("\nQuarter-section sets:")
    if not tracker["quarter_section_sets"]:
        print("  none recorded")
    for path, info in sorted(tracker["quarter_section_sets"].items()):
        print(f"  {path}: {info['count']} sections")
        if info.get("exclude"):
            print(f"    excludes: {info['exclude']}")

    print("\nPixel exports:")
    if not tracker["pixel_exports"]:
        print("  none recorded")
    for name, info in sorted(tracker["pixel_exports"].items()):
        ckpt = summarize_checkpoint(info["checkpoint"])
        print(
            f"  {name}: {ckpt['completed']}/{ckpt['total']} batches completed, "
            f"{ckpt['failed']} failed"
        )
        print(f"    checkpoint: {info['checkpoint']}")
        downloaded = sorted(DATA_DIR.glob(f"{name}_part*.csv"))
        if downloaded:
            print(f"    downloaded CSV parts: {len(downloaded)}")

    print("\nMerged CSVs:")
    if not tracker["merged_outputs"]:
        print("  none found")
    for path, info in sorted(tracker["merged_outputs"].items()):
        try:
            rows = count_rows(path)
            print(f"  {path}: {rows} data rows")
            print(f"    created by: {info.get('type')}")
        except OSError:
            print(f"  {path}: unreadable")


if __name__ == "__main__":
    main()
