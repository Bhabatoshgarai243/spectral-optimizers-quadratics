"""Run all manuscript experiments in a dedicated output directory."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


EXPERIMENTS = ("P1_2.py", "P3.py", "P4.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce all figures and tables from the manuscript."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for generated figures and tables (default: results).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository_root = Path(__file__).resolve().parent
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["PYTHONUNBUFFERED"] = "1"

    for experiment in EXPERIMENTS:
        script = repository_root / experiment
        print(f"Running {experiment} ...", flush=True)
        subprocess.run(
            [sys.executable, str(script)],
            cwd=output_dir,
            env=environment,
            check=True,
        )

    print(f"Completed. Outputs are in {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
