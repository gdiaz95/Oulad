#!/usr/bin/env python3
"""Build the OULAD students dataset by joining student tables.

The OULAD student tables are keyed by the enrollment-level identifiers
``code_module``, ``code_presentation``, and ``id_student``. Joining only on
``id_student`` can create a many-to-many expansion for students who appear in
multiple module presentations, so this script joins on the full shared key,
drops ``id_student`` from the resulting dataset, and writes the combined
records to ``data/pre_processed/students.csv`` by default.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_STUDENT_INFO = Path("data/raw/studentInfo.csv")
DEFAULT_STUDENT_REGISTRATION = Path("data/raw/studentRegistration.csv")
DEFAULT_OUTPUT = Path("data/pre_processed/students.csv")
JOIN_COLUMNS = ["code_module", "code_presentation", "id_student"]


def build_students_dataset(
    student_info_path: Path = DEFAULT_STUDENT_INFO,
    student_registration_path: Path = DEFAULT_STUDENT_REGISTRATION,
    output_path: Path = DEFAULT_OUTPUT,
) -> pd.DataFrame:
    """Join the raw OULAD student tables and write the combined dataset."""
    missing_inputs = [
        str(path)
        for path in (student_info_path, student_registration_path)
        if not path.exists()
    ]
    if missing_inputs:
        missing = ", ".join(missing_inputs)
        raise FileNotFoundError(
            "Missing required raw input file(s): "
            f"{missing}. Place the OULAD CSV files in data/raw/ and rerun the script."
        )

    student_info = pd.read_csv(student_info_path)
    student_registration = pd.read_csv(student_registration_path)

    for name, dataframe in {
        "studentInfo": student_info,
        "studentRegistration": student_registration,
    }.items():
        missing_columns = [column for column in JOIN_COLUMNS if column not in dataframe.columns]
        if missing_columns:
            raise KeyError(f"{name} is missing required join column(s): {missing_columns}")

    students = student_info.merge(
        student_registration,
        on=JOIN_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    students = students.drop(columns=["id_student"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    students.to_csv(output_path, index=False)
    return students


if __name__ == "__main__":
    dataset = build_students_dataset()
    print(
        f"Saved {len(dataset):,} rows and {len(dataset.columns):,} columns to "
        f"{DEFAULT_OUTPUT}"
    )
