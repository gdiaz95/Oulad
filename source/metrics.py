#!/usr/bin/env python3
"""Metric utilities for synthetic tabular data evaluation."""

from __future__ import annotations

import json
import os
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from mostlyai import qa
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic
from scipy import stats
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


def run_univariate_hypothesis_tests(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    alpha: float = 0.05,
) -> dict:
    """Run per-column statistical hypothesis tests between real and synthetic data."""
    shared_columns = [c for c in real_data.columns if c in synthetic_data.columns]
    per_column_results = []

    for column in shared_columns:
        real_col = real_data[column]
        synthetic_col = synthetic_data[column]

        if pd.api.types.is_numeric_dtype(real_col):
            real_values = real_col.dropna().to_numpy()
            synthetic_values = synthetic_col.dropna().to_numpy()

            if len(real_values) < 2 or len(synthetic_values) < 2:
                per_column_results.append(
                    {
                        "column": column,
                        "column_type": "numeric",
                        "test_name": "ks_2samp",
                        "p_value": None,
                        "statistic": None,
                        "reject_null": None,
                        "conclusion": "insufficient_data",
                    }
                )
                continue

            ks_result = stats.ks_2samp(real_values, synthetic_values)
            reject = bool(ks_result.pvalue < alpha)
            per_column_results.append(
                {
                    "column": column,
                    "column_type": "numeric",
                    "test_name": "ks_2samp",
                    "p_value": float(ks_result.pvalue),
                    "statistic": float(ks_result.statistic),
                    "reject_null": reject,
                    "conclusion": "different_distribution" if reject else "no_significant_difference",
                }
            )
        else:
            real_counts = real_col.astype("string").fillna("<NA>").value_counts()
            synthetic_counts = synthetic_col.astype("string").fillna("<NA>").value_counts()
            category_index = real_counts.index.union(synthetic_counts.index)
            contingency = np.vstack(
                [
                    real_counts.reindex(category_index, fill_value=0).to_numpy(),
                    synthetic_counts.reindex(category_index, fill_value=0).to_numpy(),
                ]
            )

            if contingency.shape[1] < 2:
                per_column_results.append(
                    {
                        "column": column,
                        "column_type": "categorical",
                        "test_name": "chi2_contingency",
                        "p_value": 1.0,
                        "statistic": 0.0,
                        "reject_null": False,
                        "conclusion": "no_significant_difference",
                    }
                )
                continue

            chi2_stat, p_value, _, _ = stats.chi2_contingency(contingency, correction=False)
            reject = bool(p_value < alpha)
            per_column_results.append(
                {
                    "column": column,
                    "column_type": "categorical",
                    "test_name": "chi2_contingency",
                    "p_value": float(p_value),
                    "statistic": float(chi2_stat),
                    "reject_null": reject,
                    "conclusion": "different_distribution" if reject else "no_significant_difference",
                }
            )

    valid_tests = [item for item in per_column_results if item["reject_null"] is not None]
    rejected_count = sum(item["reject_null"] for item in valid_tests)

    return {
        "alpha": alpha,
        "n_columns_compared": len(shared_columns),
        "n_valid_tests": len(valid_tests),
        "n_rejected": int(rejected_count),
        "rejection_rate": (rejected_count / len(valid_tests)) if valid_tests else None,
        "columns": per_column_results,
    }


def _safe_float(value):
    """Convert to float and handle NaN-like values as None."""
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_categorical(series: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(series)


def _cramers_v_from_table(table: pd.DataFrame) -> float | None:
    if table.empty or table.shape[0] < 2 or table.shape[1] < 2:
        return None
    chi2, _, _, _ = stats.chi2_contingency(table.to_numpy(), correction=False)
    total = table.to_numpy().sum()
    if total == 0:
        return None
    phi2 = chi2 / total
    denom = min(table.shape[0] - 1, table.shape[1] - 1)
    if denom <= 0:
        return None
    return float(np.sqrt(phi2 / denom))


def _bh_adjust(p_values: list[float | None]) -> list[float | None]:
    indexed = [(idx, pv) for idx, pv in enumerate(p_values) if pv is not None]
    if not indexed:
        return [None] * len(p_values)

    sorted_pairs = sorted(indexed, key=lambda x: x[1])
    m = len(sorted_pairs)
    adjusted = [None] * len(p_values)
    running_min = 1.0

    for rank in range(m, 0, -1):
        idx, pv = sorted_pairs[rank - 1]
        scaled = (pv * m) / rank
        running_min = min(running_min, scaled)
        adjusted[idx] = float(min(1.0, running_min))

    return adjusted


def run_bivariate_distribution_tests(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    alpha: float = 0.05,
    tolerances: dict | None = None,
    critical_pairs: list[tuple[str, str]] | None = None,
) -> dict:
    """Evaluate bivariate fidelity across cat-cat, num-cat, and num-num pairs."""
    tolerances = tolerances or {
        "cat_cat_tvd_max": 0.10,
        "cat_cat_cramers_v_gap_max": 0.10,
        "num_cat_wasserstein_norm_max": 0.30,
        "num_num_corr_gap_max": 0.10,
        "num_num_density_tvd_max": 0.15,
    }
    critical_pairs = critical_pairs or []

    shared_columns = [c for c in real_data.columns if c in synthetic_data.columns]
    pair_results = []

    for left, right in combinations(shared_columns, 2):
        real_left = real_data[left]
        real_right = real_data[right]
        synth_left = synthetic_data[left]
        synth_right = synthetic_data[right]

        left_cat = _is_categorical(real_left)
        right_cat = _is_categorical(real_right)

        result = {
            "pair": [left, right],
            "pair_key": f"{left}__{right}",
            "p_value": None,
            "p_value_fdr_bh": None,
            "reject_null_raw": None,
            "reject_null_fdr": None,
            "within_tolerance": None,
        }

        if left_cat and right_cat:
            result["pair_type"] = "cat_cat"
            real_table = pd.crosstab(
                real_left.astype("string").fillna("<NA>"),
                real_right.astype("string").fillna("<NA>"),
                dropna=False,
            )
            synth_table = pd.crosstab(
                synth_left.astype("string").fillna("<NA>"),
                synth_right.astype("string").fillna("<NA>"),
                dropna=False,
            )

            idx = real_table.index.union(synth_table.index)
            cols = real_table.columns.union(synth_table.columns)
            real_table = real_table.reindex(index=idx, columns=cols, fill_value=0)
            synth_table = synth_table.reindex(index=idx, columns=cols, fill_value=0)

            real_total = real_table.to_numpy().sum()
            synth_total = synth_table.to_numpy().sum()
            real_prob = real_table.to_numpy() / real_total if real_total else np.zeros_like(real_table)
            synth_prob = synth_table.to_numpy() / synth_total if synth_total else np.zeros_like(synth_table)

            tvd = 0.5 * np.abs(real_prob - synth_prob).sum()
            cramers_real = _cramers_v_from_table(real_table)
            cramers_synth = _cramers_v_from_table(synth_table)
            cramers_gap = (
                None
                if (cramers_real is None or cramers_synth is None)
                else abs(cramers_real - cramers_synth)
            )

            expected = real_prob.ravel() * synth_total
            observed = synth_table.to_numpy().ravel()
            valid_mask = expected > 0
            if valid_mask.any():
                chi_result = stats.chisquare(f_obs=observed[valid_mask], f_exp=expected[valid_mask])
                p_value = float(chi_result.pvalue)
                statistic = float(chi_result.statistic)
            else:
                p_value = None
                statistic = None

            within_tolerance = (tvd <= tolerances["cat_cat_tvd_max"]) and (
                cramers_gap is None
                or cramers_gap <= tolerances["cat_cat_cramers_v_gap_max"]
            )

            result.update(
                {
                    "test_name": "joint_table_chisquare",
                    "statistic": statistic,
                    "tvd_joint": float(tvd),
                    "cramers_v_real": _safe_float(cramers_real),
                    "cramers_v_synth": _safe_float(cramers_synth),
                    "cramers_v_gap": _safe_float(cramers_gap),
                    "p_value": p_value,
                    "within_tolerance": bool(within_tolerance),
                    "divergence_score": float(max(tvd, cramers_gap or 0.0)),
                    "interpretation": (
                        "association_strength_preserved"
                        if within_tolerance
                        else "association_shifted_or_weakened"
                    ),
                }
            )
        elif left_cat ^ right_cat:
            result["pair_type"] = "num_cat"
            numeric_col = right if left_cat else left
            categorical_col = left if left_cat else right

            real_numeric = real_data[numeric_col]
            synth_numeric = synthetic_data[numeric_col]
            real_categorical = real_data[categorical_col].astype("string").fillna("<NA>")
            synth_categorical = synthetic_data[categorical_col].astype("string").fillna("<NA>")
            categories = real_categorical.dropna().unique().tolist()
            categories = sorted(set(categories) | set(synth_categorical.dropna().unique().tolist()))

            group_summaries = []
            wasserstein_norm_values = []
            group_p_values = []
            real_groups = []
            synth_groups = []

            real_std = real_numeric.dropna().std()
            scale = float(real_std) if real_std and real_std > 0 else 1.0

            for cat in categories:
                real_vals = real_numeric[real_categorical == cat].dropna().to_numpy()
                synth_vals = synth_numeric[synth_categorical == cat].dropna().to_numpy()

                if len(real_vals) > 0:
                    real_groups.append(real_vals)
                if len(synth_vals) > 0:
                    synth_groups.append(synth_vals)

                if len(real_vals) > 0 and len(synth_vals) > 0:
                    wasserstein = float(stats.wasserstein_distance(real_vals, synth_vals))
                    ks_res = stats.ks_2samp(real_vals, synth_vals)
                    group_p = float(ks_res.pvalue)
                    group_stat = float(ks_res.statistic)
                    wasserstein_norm = wasserstein / scale
                    wasserstein_norm_values.append(wasserstein_norm)
                    group_p_values.append(group_p)
                else:
                    wasserstein = None
                    wasserstein_norm = None
                    group_p = None
                    group_stat = None

                group_summaries.append(
                    {
                        "category": cat,
                        "real_count": int(len(real_vals)),
                        "synth_count": int(len(synth_vals)),
                        "real_mean": _safe_float(np.mean(real_vals) if len(real_vals) else None),
                        "synth_mean": _safe_float(np.mean(synth_vals) if len(synth_vals) else None),
                        "real_median": _safe_float(np.median(real_vals) if len(real_vals) else None),
                        "synth_median": _safe_float(np.median(synth_vals) if len(synth_vals) else None),
                        "real_std": _safe_float(np.std(real_vals, ddof=1) if len(real_vals) > 1 else None),
                        "synth_std": _safe_float(
                            np.std(synth_vals, ddof=1) if len(synth_vals) > 1 else None
                        ),
                        "real_iqr": _safe_float(
                            np.subtract(*np.percentile(real_vals, [75, 25]))
                            if len(real_vals) > 1
                            else None
                        ),
                        "synth_iqr": _safe_float(
                            np.subtract(*np.percentile(synth_vals, [75, 25]))
                            if len(synth_vals) > 1
                            else None
                        ),
                        "wasserstein": _safe_float(wasserstein),
                        "wasserstein_normalized": _safe_float(wasserstein_norm),
                        "ks_statistic": _safe_float(group_stat),
                        "ks_p_value": _safe_float(group_p),
                    }
                )

            real_anova_p = None
            synth_anova_p = None
            real_kruskal_p = None
            synth_kruskal_p = None
            if len(real_groups) >= 2 and all(len(g) > 1 for g in real_groups):
                real_anova_p = float(stats.f_oneway(*real_groups).pvalue)
                real_kruskal_p = float(stats.kruskal(*real_groups).pvalue)
            if len(synth_groups) >= 2 and all(len(g) > 1 for g in synth_groups):
                synth_anova_p = float(stats.f_oneway(*synth_groups).pvalue)
                synth_kruskal_p = float(stats.kruskal(*synth_groups).pvalue)

            pair_p_value = min(group_p_values) if group_p_values else None
            mean_wass_norm = (
                float(np.mean(wasserstein_norm_values)) if wasserstein_norm_values else None
            )
            within_tolerance = (
                mean_wass_norm is not None
                and mean_wass_norm <= tolerances["num_cat_wasserstein_norm_max"]
            )

            result.update(
                {
                    "test_name": "per_category_ks_plus_wasserstein",
                    "numeric_column": numeric_col,
                    "categorical_column": categorical_col,
                    "group_summaries": group_summaries,
                    "mean_wasserstein_normalized": _safe_float(mean_wass_norm),
                    "anova_p_real": _safe_float(real_anova_p),
                    "anova_p_synth": _safe_float(synth_anova_p),
                    "kruskal_p_real": _safe_float(real_kruskal_p),
                    "kruskal_p_synth": _safe_float(synth_kruskal_p),
                    "p_value": _safe_float(pair_p_value),
                    "within_tolerance": bool(within_tolerance),
                    "divergence_score": float(mean_wass_norm or 0.0),
                    "interpretation": (
                        f"{numeric_col}_by_{categorical_col}_preserved"
                        if within_tolerance
                        else f"{numeric_col}_by_{categorical_col}_shifted"
                    ),
                }
            )
        else:
            result["pair_type"] = "num_num"

            real_pair = real_data[[left, right]].dropna()
            synth_pair = synthetic_data[[left, right]].dropna()
            if len(real_pair) >= 3 and len(synth_pair) >= 3:
                pearson_real = float(real_pair[left].corr(real_pair[right], method="pearson"))
                pearson_synth = float(synth_pair[left].corr(synth_pair[right], method="pearson"))
                spearman_real = float(real_pair[left].corr(real_pair[right], method="spearman"))
                spearman_synth = float(synth_pair[left].corr(synth_pair[right], method="spearman"))
            else:
                pearson_real = pearson_synth = spearman_real = spearman_synth = None

            if len(real_pair) >= 10 and len(synth_pair) >= 10:
                x_edges = np.unique(np.quantile(real_pair[left], np.linspace(0, 1, 11)))
                y_edges = np.unique(np.quantile(real_pair[right], np.linspace(0, 1, 11)))
                if len(x_edges) > 2 and len(y_edges) > 2:
                    real_hist, _, _ = np.histogram2d(
                        real_pair[left], real_pair[right], bins=[x_edges, y_edges]
                    )
                    synth_hist, _, _ = np.histogram2d(
                        synth_pair[left], synth_pair[right], bins=[x_edges, y_edges]
                    )
                    real_prob = real_hist / real_hist.sum() if real_hist.sum() else real_hist
                    synth_prob = synth_hist / synth_hist.sum() if synth_hist.sum() else synth_hist
                    density_tvd = float(0.5 * np.abs(real_prob - synth_prob).sum())
                else:
                    density_tvd = None
            else:
                density_tvd = None

            pearson_gap = (
                None if (pearson_real is None or pearson_synth is None) else abs(pearson_real - pearson_synth)
            )
            spearman_gap = (
                None
                if (spearman_real is None or spearman_synth is None)
                else abs(spearman_real - spearman_synth)
            )
            corr_gap = max([v for v in [pearson_gap, spearman_gap] if v is not None], default=None)
            within_tolerance = (
                corr_gap is not None
                and corr_gap <= tolerances["num_num_corr_gap_max"]
                and (density_tvd is None or density_tvd <= tolerances["num_num_density_tvd_max"])
            )

            result.update(
                {
                    "test_name": "correlation_gap_plus_density_tvd",
                    "pearson_real": _safe_float(pearson_real),
                    "pearson_synth": _safe_float(pearson_synth),
                    "pearson_gap": _safe_float(pearson_gap),
                    "spearman_real": _safe_float(spearman_real),
                    "spearman_synth": _safe_float(spearman_synth),
                    "spearman_gap": _safe_float(spearman_gap),
                    "density_tvd": _safe_float(density_tvd),
                    "within_tolerance": bool(within_tolerance),
                    "divergence_score": float(max(corr_gap or 0.0, density_tvd or 0.0)),
                    "interpretation": (
                        "linear_and_rank_dependencies_retained"
                        if within_tolerance
                        else "dependency_flattened_or_exaggerated"
                    ),
                }
            )

        pair_results.append(result)

    raw_p_values = [item["p_value"] for item in pair_results]
    p_values_fdr = _bh_adjust(raw_p_values)

    for item, adjusted_p in zip(pair_results, p_values_fdr):
        item["p_value_fdr_bh"] = adjusted_p
        if item["p_value"] is not None:
            item["reject_null_raw"] = bool(item["p_value"] < alpha)
            item["reject_null_fdr"] = bool(adjusted_p is not None and adjusted_p < alpha)

    within_count = sum(bool(item["within_tolerance"]) for item in pair_results)
    pair_type_breakdown = {
        "cat_cat": sum(item["pair_type"] == "cat_cat" for item in pair_results),
        "num_cat": sum(item["pair_type"] == "num_cat" for item in pair_results),
        "num_num": sum(item["pair_type"] == "num_num" for item in pair_results),
    }

    sorted_worst = sorted(
        pair_results,
        key=lambda x: x.get("divergence_score", 0.0),
        reverse=True,
    )
    top_10_worst_pairs = [
        {
            "pair": item["pair"],
            "pair_type": item["pair_type"],
            "divergence_score": item.get("divergence_score"),
            "interpretation": item.get("interpretation"),
        }
        for item in sorted_worst[:10]
    ]

    critical_lookup = {}
    for pair in critical_pairs:
        forward = f"{pair[0]}__{pair[1]}"
        reverse = f"{pair[1]}__{pair[0]}"
        matched = next(
            (item for item in pair_results if item["pair_key"] in {forward, reverse}),
            None,
        )
        critical_lookup[f"{pair[0]}__{pair[1]}"] = (
            {
                "status": "within_tolerance" if matched and matched["within_tolerance"] else "outside_tolerance",
                "pair_type": matched["pair_type"] if matched else None,
                "divergence_score": matched.get("divergence_score") if matched else None,
            }
            if matched
            else {"status": "pair_not_found", "pair_type": None, "divergence_score": None}
        )

    return {
        "alpha": alpha,
        "multiple_testing_correction": "benjamini-hochberg_fdr",
        "tolerances": tolerances,
        "n_pairs_evaluated": len(pair_results),
        "pair_type_breakdown": pair_type_breakdown,
        "pairs_within_tolerance": within_count,
        "percent_pairs_within_tolerance": (
            (within_count / len(pair_results)) * 100 if pair_results else None
        ),
        "top_10_worst_pairs": top_10_worst_pairs,
        "critical_pairs_status": critical_lookup,
        "pairs": pair_results,
    }


def evaluate_and_save_reports(
    original_real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    metadata,
    report_path: Path,
    metrics_qa,
    tstr_results: dict,
    univariate_hypothesis_tests: dict,
    bivariate_distribution_tests: dict,
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
        "univariate_hypothesis_tests": univariate_hypothesis_tests,
        "bivariate_distribution_tests": bivariate_distribution_tests,
    }

    os.makedirs(report_path.parent, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(combined_report_data, f, indent=4)

    return combined_report_data
