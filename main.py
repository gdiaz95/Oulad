#!/usr/bin/env python3
"""Fit NPGC on pre-processed students data and generate synthetic samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from source.npgc import NPGC

DEFAULT_INPUT = Path("data/pre_processed/students.csv")
DEFAULT_OUTPUT = Path("data/post_processed/students.csv")


def resolve_input_path(input_path: str | None) -> Path:
    """Resolve input path from CLI value or the default pre-processed dataset."""
    path = Path(input_path) if input_path else DEFAULT_INPUT
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}. Use --input to provide a valid path."
        )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit NPGC to a pre-processed students CSV and create a synthetic students CSV."
        )
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10_000,
        help="Number of synthetic samples to generate.",
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        default=None,
        help=(
            "Optional path to the input CSV (default: data/pre_processed/students.csv)."
        ),
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        default=str(DEFAULT_OUTPUT),
        help="Path to save synthetic CSV (default: data/post_processed/students.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.n_samples <= 0:
        raise ValueError("n_samples must be a positive integer.")

    input_path = resolve_input_path(args.input_path)
    output_path = Path(args.output_path)

    students_df = pd.read_csv(input_path)

    synthesizer = NPGC()
    synthesizer.fit(students_df)
    synthetic_df = synthesizer.sample(args.n_samples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_df.to_csv(output_path, index=False)

    print(
        f"Fitted NPGC on {input_path} ({len(students_df)} rows); generated "
        f"{len(synthetic_df)} rows to {output_path}."
    )


if __name__ == "__main__":
    main()
