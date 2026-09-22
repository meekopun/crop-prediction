"""Commands for the bundled historical crop yield dataset."""

import argparse
from pathlib import Path
import sys

from .data import load_data, summarize


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/yield_df.csv"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("summary", help="Validate and summarize the input CSV")
    benchmark = commands.add_parser("benchmark", help="Train and evaluate on held-out later years")
    benchmark.add_argument("--output-dir", type=Path, default=Path("data/processed/benchmark"))
    args = parser.parse_args(argv)
    try:
        if args.command == "summary":
            for key, value in summarize(load_data(args.data)).items():
                print(f"{key}: {value}")
        else:
            from .modeling import run_benchmark

            print("Training historical yield benchmark...", flush=True)
            result = run_benchmark(args.data, args.output_dir)
            for key, value in result.items():
                if key != "metrics":
                    print(f"{key}: {value}")
            for metric in result["metrics"]:
                print(f"{metric['model']}: MAE={metric['mae_hg_ha']:.2f} hg/ha, "
                      f"RMSE={metric['rmse_hg_ha']:.2f} hg/ha, R²={metric['r2']:.4f}")
            print(f"Results: {args.output_dir.resolve()}")
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
