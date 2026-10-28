#!/usr/bin/env python3
"""Fit and save the locked season + field yield model for a held-out year.

The configuration is fixed in `yield_prediction.season_field_bundle`: season level
`regional_satellite`, field deviation `peak`, 10% margin over the training mean.
Only years before `--test-year` are used. Run `scripts/build_field_yield_table.py`,
`scripts/prepare_regional_yields.py --all-alberta` and
`scripts/export_regional_features.py` first.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Also support a source checkout before installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yield_prediction.field_yield import DEFAULT_TABLE_PATH
from yield_prediction.regional_features import DEFAULT_REGIONAL_FEATURES_PATH
from yield_prediction.regional_yields import DEFAULT_ALBERTA_YIELDS_PATH
from yield_prediction.season_field_bundle import (
    MODEL_VARIANT, TEST_YEAR, default_bundle_dir, fit_season_field_bundle, save_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_TABLE_PATH, help="field_yield_table.csv.")
    parser.add_argument("--test-year", type=int, default=TEST_YEAR,
                        help="Held-out year; the model trains on every earlier year.")
    parser.add_argument("--output-dir", type=Path,
                        help="Bundle directory (default: data/processed/harvest/models/season_field_<test-year>).")
    parser.add_argument("--regional-yields", type=Path, default=DEFAULT_ALBERTA_YIELDS_PATH,
                        help="All-Alberta regional yields from prepare_regional_yields.py --all-alberta.")
    parser.add_argument("--regional-features", type=Path, default=DEFAULT_REGIONAL_FEATURES_PATH,
                        help="Regional crop-masked July features from export_regional_features.py.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing bundle.")
    return parser


def build_summary(manifest: dict) -> str:
    config = manifest["config"]
    lines = [f"# Season + Field Model Bundle for {manifest['test_year']}", "",
             f"Fitted {manifest['created']} on {min(manifest['training_years'])}-{max(manifest['training_years'])} "
             f"farm seasons and {len(manifest['region_training_years'])} regional years "
             f"({min(manifest['region_training_years'])}-{max(manifest['region_training_years'])}). "
             f"Season level `{config['season_features']}`, field deviation `{config['deviation_features']}`. "
             f"A crop uses the model only if its leave-one-season-out error beats the training mean's by at "
             f"least {config['min_improvement']:.0%}.", "",
             f"Feature cutoff: {manifest['feature_cutoff']}", "",
             f"`model.joblib` SHA-256: `{manifest['model_sha256']}`", "",
             "| Crop | Dev field-years | Seasons | Dev MAE baseline | Dev MAE model | Dev skill | Chosen | "
             "Season model | Deviation model |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for crop, entry in manifest["crops"].items():
        chosen = "model" if entry["variant"] == MODEL_VARIANT else "training mean"
        lines.append(f"| {crop} ({entry['yield_unit']}) | {entry['dev_field_years']} | {entry['dev_seasons']} | "
                     f"{entry['dev_mae_baseline']:.1f} | {entry['dev_mae_model']:.1f} | {entry['dev_skill']:+.2f} | "
                     f"{chosen} | {'yes' if entry['has_season_model'] else 'no'} | "
                     f"{'yes' if entry['has_deviation_model'] else 'no'} |")
    if manifest["later_rows_at_fit"]:
        lines.extend(["", f"The input table already held rows from {manifest['test_year']} or later "
                          f"({manifest['later_rows_at_fit']}); they were not used."])
    return "\n".join(lines)


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir or default_bundle_dir(args.test_year)
    if (output_dir / "model.joblib").exists() and not args.overwrite:
        print(f"{output_dir / 'model.joblib'} already exists; pass --overwrite to replace it.", file=sys.stderr)
        return 1
    try:
        bundle = fit_season_field_bundle(args.input, test_year=args.test_year,
                                         regional_yields_path=args.regional_yields,
                                         regional_features_path=args.regional_features)
        manifest = save_bundle(bundle, output_dir, overwrite=args.overwrite)
    except (OSError, KeyError, ValueError) as exc:
        print(f"Fitting the season + field bundle failed: {exc}", file=sys.stderr)
        return 1
    summary = build_summary(manifest)
    (output_dir / "summary.md").write_text(summary + "\n", encoding="utf-8")
    print(summary)
    print(f"Wrote model.joblib, manifest.json and summary.md to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
