#!/usr/bin/env python3
"""Evaluate the two-part yield model (season level + satellite field deviation).

Holds out one season at a time over development years only (2020-2024 by default).
Run `scripts/export_harvest_weather.py` and `scripts/build_field_yield_table.py`
first, and `scripts/prepare_regional_yields.py` for `--season-features regional`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Also support a source checkout before installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yield_prediction.field_yield import DEFAULT_TABLE_PATH
from yield_prediction.regional_yields import DEFAULT_REGIONAL_PATH
from yield_prediction.season_field_model import (
    DEFAULT_OUTPUT_DIR, DEV_YEARS, DEVIATION_FEATURE_SETS, SEASON_FEATURE_SETS, VARIANTS,
    evaluate_season_field_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_TABLE_PATH, help="field_yield_table.csv with weather.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory for metrics, season table, predictions and the summary.")
    parser.add_argument("--years", type=int, nargs="+", default=list(DEV_YEARS),
                        help="Development years; one season is held out at a time.")
    parser.add_argument("--min-rows", type=int, default=10, help="Minimum development field-years per crop.")
    parser.add_argument("--deviation-features", choices=sorted(DEVIATION_FEATURE_SETS), default="indices",
                        help="Satellite features for the field-deviation model.")
    parser.add_argument("--season-features", choices=sorted(SEASON_FEATURE_SETS), default="weather",
                        help="Season level: ERA5-Land weather, season-mean peak satellite indices, "
                             "or published regional yields.")
    parser.add_argument("--regional", type=Path, default=DEFAULT_REGIONAL_PATH,
                        help="regional_yields.csv for --season-features regional.")
    return parser


def build_summary(metrics: pd.DataFrame, seasons: pd.DataFrame, years: list[int],
                  deviation_features: str, season_features: str) -> str:
    lines = [f"# Season + Field Yield Model ({min(years)}-{max(years)}, leave one season out)", "",
             f"Field deviation: `{deviation_features}` features "
             f"({len(DEVIATION_FEATURE_SETS[deviation_features])} columns, centred on the crop-season mean). "
             + (f"Season level: published regional yield x the crop's training farm-to-region ratio."
              if season_features == "regional" else
              f"Season level: `{season_features}` features "
              f"({', '.join(f'`{column}`' for column in SEASON_FEATURE_SETS[season_features])})."), "",
             "Mean absolute error per crop in its unit; `skill` is the share of the baseline's error",
             "removed. `known_season` variants use the crop's other fields' actual season mean, a",
             "stand-in for published regional yields: a ceiling, not an achievable forecast here.", "",
             "| Crop | Field-years | " + " | ".join(VARIANTS) + " |",
             "| --- | --- | " + " | ".join("---" for _ in VARIANTS) + " |"]
    for (crop, unit), frame in metrics.groupby(["crop", "yield_unit"], sort=True):
        frame = frame.set_index("variant")
        cells = [f"{frame.loc[variant, 'mae']:.1f} ({frame.loc[variant, 'skill']:+.2f})" for variant in VARIANTS]
        lines.append(f"| {crop} ({unit}) | {int(frame['field_years'].iloc[0])} | " + " | ".join(cells) + " |")
    modelled_error = np.mean(np.abs(seasons["modelled_mean"] - seasons["actual_season_mean"])
                             / seasons["baseline_mean"])
    baseline_error = np.mean(np.abs(seasons["baseline_mean"] - seasons["actual_season_mean"])
                             / seasons["baseline_mean"])
    lines.extend(["", "## Season level", "",
                  f"Across {len(seasons)} crop-seasons the season model's mean error is "
                  f"{modelled_error:.1%} of the crop mean, versus {baseline_error:.1%} for the training mean.", "",
                  "| Crop | Year | Fields | Actual | Training mean | Modelled | Alpha or ratio |",
                  "| --- | --- | --- | --- | --- | --- | --- |"])
    for row in seasons.sort_values(["crop", "year"]).to_dict("records"):
        ratio = row.get("farm_to_region_ratio")
        setting = (f"{row['season_alpha']:g}" if not pd.isna(row["season_alpha"])
                   else "no published yield" if ratio is None or pd.isna(ratio) else f"x{ratio:.2f}")
        lines.append(f"| {row['crop']} | {row['year']} | {row['fields']} | {row['actual_season_mean']:.1f} | "
                     f"{row['baseline_mean']:.1f} | {row['modelled_mean']:.1f} | {setting} |")
    return "\n".join(lines)


def main() -> int:
    args = build_parser().parse_args()
    try:
        artifact = evaluate_season_field_model(args.input, years=tuple(args.years), min_rows=args.min_rows,
                                               deviation_features=args.deviation_features,
                                               season_features=args.season_features,
                                               regional_path=args.regional)
    except (OSError, KeyError, ValueError) as exc:
        print(f"Season + field evaluation failed: {exc}", file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"season_field_{args.season_features}_{args.deviation_features}"
    artifact.metrics.to_csv(args.output_dir / f"{prefix}_metrics.csv", index=False)
    artifact.seasons.to_csv(args.output_dir / f"{prefix}_seasons.csv", index=False)
    artifact.predictions.to_csv(args.output_dir / f"{prefix}_predictions.csv", index=False)
    summary = build_summary(artifact.metrics, artifact.seasons, args.years,
                            args.deviation_features, args.season_features)
    (args.output_dir / f"{prefix}_summary.md").write_text(summary + "\n", encoding="utf-8")
    print(summary)
    print(f"Wrote {prefix}_* outputs to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
