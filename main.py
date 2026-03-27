#!/usr/bin/env python3
"""Fit NPGC, generate synthetic data, and run evaluation metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sdv.metadata import SingleTableMetadata
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from source.metrics import (
    evaluate_and_save_reports,
    get_metrics,
    run_bivariate_distribution_tests,
    run_conclusion_consistency_for_all_pairs,
    run_tstr_evaluation,
    run_univariate_hypothesis_tests,
    summarize_hypothesis_test_runs,
)
from source.npgc import NPGC

DEFAULT_REAL = Path("data/pre_processed/students.csv")
DEFAULT_SYNTH = Path("data/post_processed/students.csv")
DEFAULT_REPORTS_DIR = Path("reports")
DEFAULT_TARGET = "final_result"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit NPGC on students data, synthesize rows, and evaluate metrics."
    )
    parser.add_argument("--real", default=str(DEFAULT_REAL), help="Path to real CSV.")
    parser.add_argument(
        "--synth",
        default=str(DEFAULT_SYNTH),
        help="Path where synthetic CSV will be saved/read.",
    )
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Target column for TSTR.")
    parser.add_argument(
        "--reports-dir",
        default=str(DEFAULT_REPORTS_DIR),
        help="Directory where report JSON should be written.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Holdout size for train/holdout split on real data.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=20_000,
        help="Synthetic sample count (default: number of rows in real data).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for univariate hypothesis tests.",
    )
    parser.add_argument(
        "--critical-pair",
        action="append",
        default=[],
        help=(
            "Business-critical pair to track in bivariate tests, format 'column_a:column_b'. "
            "Repeat the flag to pass multiple pairs."
        ),
    )
    parser.add_argument(
        "--conclusion-predictor",
        default="gender",
        help="Binary predictor column for conclusion consistency test.",
    )
    parser.add_argument(
        "--conclusion-outcome",
        default="final_result",
        help="Outcome column for conclusion consistency test.",
    )
    parser.add_argument(
        "--conclusion-positive-outcome",
        action="append",
        default=["Pass", "Distinction"],
        help=(
            "Outcome label treated as positive class for conclusion consistency test. "
            "Repeat flag for multiple labels."
        ),
    )
    parser.add_argument(
        "--hypothesis-runs",
        type=int,
        default=50,
        help="Number of synthetic draws used to summarize hypothesis-testing metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    real_path = Path(args.real)
    synth_path = Path(args.synth)
    reports_dir = Path(args.reports_dir)

    if not real_path.exists():
        raise FileNotFoundError(f"Real data not found: {real_path}")

    real_data = pd.read_csv(real_path)
    print("Data loaded")
    if args.target not in real_data.columns:
        raise KeyError(f"Target column '{args.target}' not present in real data")

    n_samples = args.n_samples if args.n_samples is not None else len(real_data)
    if n_samples <= 0:
        raise ValueError("n-samples must be a positive integer.")

    synthesizer = NPGC()
    synthesizer.fit(real_data)
    print("Learned_from_data")
    synthetic_data = synthesizer.sample(n_samples, seed=args.seed)

    synth_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_data.to_csv(synth_path, index=False)
    print("Sampled and saved")

    if args.target not in synthetic_data.columns:
        raise KeyError(f"Target column '{args.target}' not present in synthetic data")

    real_train, real_holdout = train_test_split(
        real_data,
        test_size=args.test_size,
        random_state=args.seed,
    )

    print("Running metrics")
    metrics_qa = get_metrics(real_train, synthetic_data, real_holdout)

    model = RandomForestClassifier(n_estimators=200, random_state=args.seed)
    score_real, score_synthetic, performance_drop = run_tstr_evaluation(
        real_data=real_data,
        synthetic_data=synthetic_data,
        target_column=args.target,
        model=model,
        metric_func=accuracy_score,
        test_size=args.test_size,
        random_state=args.seed,
    )

    tstr_results = {
        "target_column": args.target,
        "metric": "accuracy",
        "real_model_score": score_real,
        "synthetic_model_score": score_synthetic,
        "performance_drop": performance_drop,
    }
    if args.hypothesis_runs <= 0:
        raise ValueError("--hypothesis-runs must be a positive integer.")

    parsed_critical_pairs = []
    for raw_pair in args.critical_pair:
        if ":" not in raw_pair:
            raise ValueError(
                f"Invalid --critical-pair value '{raw_pair}'. Expected format 'column_a:column_b'."
            )
        left, right = [value.strip() for value in raw_pair.split(":", maxsplit=1)]
        if not left or not right:
            raise ValueError(
                f"Invalid --critical-pair value '{raw_pair}'. Both column names are required."
            )
        parsed_critical_pairs.append((left, right))

    if not parsed_critical_pairs and {"age_band", "final_result"}.issubset(real_data.columns):
        parsed_critical_pairs.append(("age_band", "final_result"))

    hypothesis_runs = []
    for run_idx in range(args.hypothesis_runs):
        run_synthetic = synthetic_data if run_idx == 0 else synthesizer.sample(
            n_samples,
            seed=args.seed + run_idx,
        )
        univariate_hypothesis_tests = run_univariate_hypothesis_tests(
            real_data=real_data,
            synthetic_data=run_synthetic,
            alpha=args.alpha,
        )
        bivariate_distribution_tests = run_bivariate_distribution_tests(
            real_data=real_data,
            synthetic_data=run_synthetic,
            alpha=args.alpha,
            critical_pairs=parsed_critical_pairs,
        )
        conclusion_consistency_test = run_conclusion_consistency_for_all_pairs(
            real_data=real_data,
            synthetic_data=run_synthetic,
            alpha=args.alpha,
            positive_outcomes_by_column={
                args.conclusion_outcome: args.conclusion_positive_outcome
            },
        )
        hypothesis_runs.append(
            {
                "univariate_hypothesis_tests": univariate_hypothesis_tests,
                "bivariate_distribution_tests": bivariate_distribution_tests,
                "conclusion_consistency_test": conclusion_consistency_test,
            }
        )

    hypothesis_testing_summary = summarize_hypothesis_test_runs(hypothesis_runs)

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=real_data)

    dataset_name = real_path.stem
    report_path = reports_dir / f"{dataset_name}.json"
    evaluate_and_save_reports(
        original_real_data=real_data,
        synthetic_data=synthetic_data,
        metadata=metadata,
        report_path=report_path,
        metrics_qa=metrics_qa,
        tstr_results=tstr_results,
        univariate_hypothesis_tests=univariate_hypothesis_tests,
        bivariate_distribution_tests=bivariate_distribution_tests,
        conclusion_consistency_test=conclusion_consistency_test,
        hypothesis_testing_summary=hypothesis_testing_summary,
    )

    print(f"Generated synthetic data at: {synth_path}")
    print(f"Saved metrics report to: {report_path}")


if __name__ == "__main__":
    main()
