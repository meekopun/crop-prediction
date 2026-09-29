from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

from .data import DEFAULT_DATASET_PATH, clean_records, load_records, summarize_records
from .modeling import benchmark_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Yield prediction utilities")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the yield_df.csv dataset",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("summary", help="Show dataset summary")
    subparsers.add_parser("benchmark", help="Run model benchmark")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "summary":
        summary = summarize_records(clean_records(load_records(args.data)))
        for key, value in asdict(summary).items():
            print(f"{key}: {value}")
        return 0

    if args.command == "benchmark":
        try:
            results = benchmark_models(args.data)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        for result in results:
            print(
                f"{result.model}: "
                f"test_r2={result.test_r2:.4f}, "
                f"mse={result.mse:.2f}, "
                f"mae={result.mae:.2f}, "
                f"mape={result.mape:.4f}, "
                f"cv_mean_r2={result.cv_mean_r2:.4f}"
            )
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
