"""
Step 4: Join Sentinel-2 spectral/phenology features (step 2) with AAFC crop
labels (step 3) into one training table, keyed on quarter-section pid.

Usage:
    python 04_join_labels_features.py
"""
import argparse

import pandas as pd

DATA_DIR = "data"
DROP_COLS = {"system:index", ".geo"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=f"{DATA_DIR}/sentinel2_features.csv")
    parser.add_argument("--labels", default=f"{DATA_DIR}/aafc_labels.csv")
    parser.add_argument("--output", default=f"{DATA_DIR}/training_table.csv")
    args = parser.parse_args()

    features = pd.read_csv(args.features)
    labels = pd.read_csv(args.labels)

    drop_cols = [c for c in DROP_COLS if c in features.columns]
    if drop_cols:
        features = features.drop(columns=drop_cols)
        print(f"Dropped non-feature export columns: {drop_cols}")

    if "pid" not in features.columns:
        raise SystemExit(
            "Expected a 'pid' column in the features CSV. If you exported via "
            "Google Drive, make sure the FeatureCollection still carries the "
            "'pid' property through reduceRegions (it does by default in "
            "02_gee_sentinel_features.py)."
        )

    merged = features.merge(labels, on="pid", how="inner")
    dropped = len(features) - len(merged)
    if dropped:
        print(f"Dropped {dropped} quarter sections with no matching label "
              f"(cloud, non-target crop, or outside the AAFC clip).")

    merged = merged.dropna(axis=0, how="any")
    merged.to_csv(args.output, index=False)
    print(f"Wrote {len(merged)} labeled rows to {args.output}")
    print("Class distribution:")
    print(merged["crop_label"].value_counts())


if __name__ == "__main__":
    main()
