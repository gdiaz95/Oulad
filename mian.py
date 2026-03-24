#!/usr/bin/env python3
"""Fit NPGC on pre-processed students data and generate synthetic samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from source.npgc import NPGC

DEFAULT_INPUT_CANDIDATES = [
    Path("data/pre_processed/studentcs.csv"),
    Path("data/pre_processed/students.csv"),
]
DEFAULT_OUTPUT = Path("data/post_procesed/students.csv")


def resolve_input_path(input_path: str | None) -> Path:
    """Resolve input path from CLI value or from default candidates."""
    if input_path:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return path

    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.exists():
            return candidate

    candidates = ", ".join(str(path) for path in DEFAULT_INPUT_CANDIDATES)
    raise FileNotFoundError(
        "No input dataset found. Checked: "
        f"{candidates}. Use --input to provide a path."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit NPGC to a pre-processed students CSV and create a synthetic students CSV."
        )
    )
    parser.add_argument(
        "n_samples",
        type=int,
        help="Number of synthetic samples to generate.",
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        default=None,
        help=(
            "Optional path to the input CSV. If omitted, the script checks "
            "data/pre_processed/studentcs.csv then data/pre_processed/students.csv."
        ),
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        default=str(DEFAULT_OUTPUT),
        help="Path to save synthetic CSV (default: data/post_procesed/students.csv).",
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
