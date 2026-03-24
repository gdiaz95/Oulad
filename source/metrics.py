#!/usr/bin/env python3
"""Metric utilities for synthetic tabular data evaluation."""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from mostlyai import qa
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


def get_metrics(data_train: pd.DataFrame, data_gen: pd.DataFrame, data_holdout: pd.DataFrame):
    """Compute Mostly AI QA metrics using train/synthetic/holdout tables."""
    _, metrics = qa.report(
        syn_tgt_data=data_gen,
        trn_tgt_data=data_train,
        hol_tgt_data=data_holdout,
    )

    report_file = Path("model-report.html")
    if report_file.exists():
        report_file.unlink()

    return metrics


def run_tstr_evaluation(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    target_column: str,
    model,
    metric_func,
    test_size: float = 0.3,
    random_state: int = 42,
):
    """Run Train-on-Synthetic, Test-on-Real evaluation."""
    try:
        real_train, real_test = train_test_split(
            real_data,
            test_size=test_size,
            random_state=random_state,
            stratify=real_data[target_column],
        )
    except ValueError:
        warnings.warn(
            f"Could not stratify on '{target_column}'. Proceeding without stratification.",
            stacklevel=2,
        )
        real_train, real_test = train_test_split(
            real_data,
            test_size=test_size,
            random_state=random_state,
        )

    X_real_train = real_train.drop(columns=[target_column])
    y_real_train = real_train[target_column]
    X_real_test = real_test.drop(columns=[target_column])
    y_real_test = real_test[target_column]

    X_synthetic_train = synthetic_data.drop(columns=[target_column])
    y_synthetic_train = synthetic_data[target_column]

    categorical_features = X_real_train.select_dtypes(include=["object", "category"]).columns
    numeric_features = X_real_train.select_dtypes(include=np.number).columns

    feature_preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical_features,
            ),
        ],
        remainder="passthrough",
    )

    target_encoder = LabelEncoder()

    X_real_train = feature_preprocessor.fit_transform(X_real_train)
    y_real_train = target_encoder.fit_transform(y_real_train)

    X_real_test = feature_preprocessor.transform(X_real_test)
    y_real_test = target_encoder.transform(y_real_test)

    X_synthetic_train = feature_preprocessor.transform(X_synthetic_train)
    y_synthetic_train = target_encoder.transform(y_synthetic_train)

    model_real = clone(model)
    model_real.fit(X_real_train, y_real_train)

    model_synthetic = clone(model)
    model_synthetic.fit(X_synthetic_train, y_synthetic_train)

    preds_real = model_real.predict(X_real_test)
    score_real = metric_func(y_real_test, preds_real)

    preds_synthetic = model_synthetic.predict(X_real_test)
    score_synthetic = metric_func(y_real_test, preds_synthetic)

    if score_real == 0:
        performance_gap_pct = float("inf") if score_synthetic != 0 else 0.0
    else:
        performance_gap_pct = ((score_real - score_synthetic) / score_real) * 100

    return score_real, score_synthetic, performance_gap_pct


def evaluate_and_save_reports(
    original_real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    metadata,
    report_path: Path,
    metrics_qa,
    tstr_results: dict,
):
    """Generate SDV + Mostly AI reports and save JSON report."""
    diagnostic_report = run_diagnostic(original_real_data, synthetic_data, metadata)
    quality_report = evaluate_quality(original_real_data, synthetic_data, metadata)

    combined_report_data = {
        "diagnostic_report": {
            "properties": diagnostic_report.get_properties().to_dict("records")
        },
        "quality_report": {
            "overall_score": quality_report.get_score(),
            "properties": quality_report.get_properties().to_dict("records"),
        },
        "metrics_qa": {
            "overall_accuracy": metrics_qa.accuracy.overall,
            "univariate_accuracy": metrics_qa.accuracy.univariate,
            "bivariate_accuracy": metrics_qa.accuracy.bivariate,
            "discriminator_auc": metrics_qa.similarity.discriminator_auc_training_synthetic,
            "identical_matches": metrics_qa.distances.ims_training,
            "dcr_training": metrics_qa.distances.dcr_training,
            "dcr_holdout": metrics_qa.distances.dcr_holdout,
            "dcr_share": metrics_qa.distances.dcr_share,
        },
        "tstr_evaluation": tstr_results,
    }

    os.makedirs(report_path.parent, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(combined_report_data, f, indent=4)

    return combined_report_data
