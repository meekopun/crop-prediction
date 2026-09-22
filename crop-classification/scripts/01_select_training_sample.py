"""
Step 1: Down-sample the county-level quarter-section subset to a manageable
training/testing set.

Why this step exists: the full province-wide quarter_sections.geojson is
~1.2GB / ~1.9M features, far too large to push through Google Earth Engine
or a local classifier in one go. `00_extract_study_area.py` already narrowed
that down to one county (Lethbridge County, ~9,760 quarter sections). This
script narrows it further to a random sample sized for a first training run
-- large enough for a decision tree to learn real splits, small enough that
Earth Engine calls and local processing stay fast and free-tier-friendly.

Usage:
    python 01_select_training_sample.py --n 1500 --seed 42
    python 01_select_training_sample.py --n 1000 --exclude data/training_sample.geojson --output data/training_sample_extra.geojson
"""
import argparse
import random

from common import load_geojson, quarter_section_id, save_geojson

DATA_DIR = "data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=f"{DATA_DIR}/study_area_quarter_sections.geojson")
    parser.add_argument("--output", default=f"{DATA_DIR}/training_sample.geojson")
    parser.add_argument(
        "--exclude",
        default=None,
        help="Optional GeoJSON of already-sampled quarter sections to exclude",
    )
    parser.add_argument("--n", type=int, default=1500,
                         help="Number of quarter sections to sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fc = load_geojson(args.input)
    features = fc["features"]
    print(f"Loaded {len(features)} quarter sections from {args.input}")

    if args.exclude:
        exclude_fc = load_geojson(args.exclude)
        exclude_pids = {quarter_section_id(feat) for feat in exclude_fc["features"]}
        features = [feat for feat in features if quarter_section_id(feat) not in exclude_pids]
        print(f"Excluded {len(exclude_pids)} existing quarter sections from {args.exclude}")
        print(f"{len(features)} quarter sections remain eligible for sampling")

    n = min(args.n, len(features))
    rng = random.Random(args.seed)
    sample = rng.sample(features, n)

    out_fc = {
        "type": "FeatureCollection",
        "name": "training_sample",
        "crs": fc.get("crs"),
        "features": sample,
    }
    save_geojson(args.output, out_fc)
    print(f"Wrote {n} sampled quarter sections to {args.output}")


if __name__ == "__main__":
    main()
