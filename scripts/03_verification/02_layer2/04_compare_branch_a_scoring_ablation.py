"""
04_compare_branch_a_scoring_ablation.py

Compare two Branch A Layer 2 scoring variants:

1. Signal-only scoring
   Uses entity, operator--value, quantitative completeness, temporal,
   history, context, and computability support. It excludes Layer 1
   deterministic signals.

2. Operational scoring
   Uses the complete Branch A Layer 2 score, including Layer 1 support.
   This is the version used for Layer 3 routing.

This script does not modify extraction outputs, Layer 1 results, or the
existing Layer 2 scores. It only produces comparison and ablation tables.

Inputs:
    outputs/verification/layer2/branch_a/
        layer2_branch_a_leaf_risk_scores.csv

    outputs/evaluation/pre_verification/
        semantic_manual_pre_verification_A_B_summary/
            reviewed_semantic_clause_labels_A_B.csv

Outputs:
    outputs/verification/layer2/branch_a/
        ablation_signal_only_vs_operational/

Run from the repository root:
python scripts/03_verification/02_layer2/04_compare_branch_a_scoring_ablation.py
"""

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


REVIEW_PRIORITY_THRESHOLD = 0.60
HIGH_SUPPORT_REFERENCE_THRESHOLD = 0.85

ROUTING_LABEL_ORDER = [
    "high_support_not_auto_accept",
    "intermediate_support",
    "review_priority",
]

ROUTING_SEVERITY = {
    "high_support_not_auto_accept": 0,
    "intermediate_support": 1,
    "review_priority": 2,
}

SIGNAL_ONLY_COMPONENTS = [
    "entity_support",
    "operator_value_support",
    "quantitative_completeness_support",
    "temporal_support",
    "history_support",
    "context_support",
    "computability_support",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def normalize_text(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()


def norm_col_name(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        s = normalize_text(x)
        if not s:
            return default
        value = float(s)
        if math.isnan(value):
            return default
        return value
    except Exception:
        return default


def assign_risk_label(leaf_support: float) -> str:
    if leaf_support < REVIEW_PRIORITY_THRESHOLD:
        return "review_priority"
    if leaf_support >= HIGH_SUPPORT_REFERENCE_THRESHOLD:
        return "high_support_not_auto_accept"
    return "intermediate_support"


def normalize_manual_label(x: Any) -> Optional[str]:
    s = normalize_text(x).lower()
    if not s or s in {"nan", "none", "null", "na", "n/a", "missing", "unlabeled", "not_labeled"}:
        return None

    exact_map = {
        "correct": "correct",
        "c": "correct",
        "ok": "correct",
        "partial": "partial",
        "partially_correct": "partial",
        "partially correct": "partial",
        "p": "partial",
        "incorrect": "incorrect",
        "wrong": "incorrect",
        "error": "incorrect",
        "i": "incorrect",
        "unsupported": "incorrect",
    }
    if s in exact_map:
        return exact_map[s]

    # Important: check incorrect before correct.
    if "incorrect" in s or "wrong" in s or "unsupported" in s:
        return "incorrect"
    if "partial" in s:
        return "partial"
    if "correct" in s:
        return "correct"
    return None


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def crosstab_with_order(df: pd.DataFrame, row_col: str, col_col: str) -> pd.DataFrame:
    tab = pd.crosstab(df[row_col], df[col_col], margins=True)

    # Keep risk columns ordered when applicable.
    cols = [c for c in ROUTING_LABEL_ORDER if c in tab.columns]
    other_cols = [c for c in tab.columns if c not in cols and c != "All"]
    if "All" in tab.columns:
        tab = tab[cols + other_cols + ["All"]]
    else:
        tab = tab[cols + other_cols]

    return tab.reset_index()


def auroc_rank(y_true: List[int], scores: List[float]) -> Optional[float]:
    n = len(y_true)
    if n == 0:
        return None
    n_pos = sum(y_true)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    pairs = sorted(zip(scores, y_true), key=lambda x: x[0])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1

    sum_pos_ranks = sum(rank for rank, (_, y) in zip(ranks, pairs) if y == 1)
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y_true: List[int], scores: List[float]) -> Optional[float]:
    n_pos = sum(y_true)
    if n_pos == 0:
        return None
    pairs = sorted(zip(scores, y_true), key=lambda x: x[0], reverse=True)
    tp = 0
    precisions = []
    for i, (_, y) in enumerate(pairs, start=1):
        if y == 1:
            tp += 1
            precisions.append(tp / i)
    return float(sum(precisions) / n_pos) if precisions else None


def try_metrics(y_true: List[int], scores: List[float]) -> Tuple[Optional[float], Optional[float]]:
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        if len(set(y_true)) < 2:
            return None, None
        return float(roc_auc_score(y_true, scores)), float(average_precision_score(y_true, scores))
    except Exception:
        return auroc_rank(y_true, scores), average_precision(y_true, scores)

def binary_metrics(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
    }

# ---------------------------------------------------------------------
# Manual labels
# ---------------------------------------------------------------------

def ensure_manual_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    norm_to_original = {norm_col_name(c): c for c in df.columns}

    if "criterion_id" not in df.columns:
        for cand in ["criterion_id", "leaf_id", "leaf_uid", "criterion_uid"]:
            key = norm_col_name(cand)
            if key in norm_to_original:
                df["criterion_id"] = df[norm_to_original[key]].astype(str)
                break

    if "criterion_id" not in df.columns:
        item_col = norm_to_original.get("item_uid") or norm_to_original.get("item_id")
        clause_col = norm_to_original.get("clause_id") or norm_to_original.get("clause")
        if item_col and clause_col:
            df["criterion_id"] = df[item_col].astype(str) + "_" + df[clause_col].astype(str)

    return df

def load_manual_a_labels(
    root: Path,
) -> Tuple[Optional[pd.DataFrame], Optional[Path], Optional[str]]:
    manual_path = (
        root
        / "outputs"
        / "evaluation"
        / "pre_verification"
        / "semantic_manual_pre_verification_A_B_summary"
        / "reviewed_semantic_clause_labels_A_B.csv"
    )

    if not manual_path.exists():
        return None, None, None

    df = pd.read_csv(manual_path)
    if df.empty:
        return None, manual_path, None

    # Keep the remainder of this function unchanged.

    preferred = [
        "manual_A_leaf_label",
        "manual_a_leaf_label",
        "manual_A_label",
        "manual_a_label",
        "A_label",
        "A_leaf_label",
        "manual_label_A",
        "label_A",
    ]
    norm_to_original = {norm_col_name(c): c for c in df.columns}

    label_col = "manual_A_leaf_label"
    for c in preferred:
        key = norm_col_name(c)
        if key in norm_to_original:
            candidate = norm_to_original[key]
            labels = df[candidate].map(normalize_manual_label)
            if labels.notna().sum() > 0:
                label_col = candidate
                break

    if label_col not in df.columns:
        return None, manual_path, None

    col_norm = norm_col_name(label_col)
    forbidden = ["manual_b", "b_leaf", "branch_b", "manual_c", "c_strict", "c_relaxed", "branch_c"]
    if any(x in col_norm for x in forbidden):
        raise RuntimeError(f"Wrong manual label column detected for Branch A: {label_col}")

    df = ensure_manual_keys(df)
    if "criterion_id" not in df.columns:
        return None, manual_path, label_col

    manual = pd.DataFrame({
        "criterion_id": df["criterion_id"].astype(str),
        "manual_label_raw": df[label_col],
        "manual_label": df[label_col].map(normalize_manual_label),
    })
    manual = manual[manual["manual_label"].notna()].copy()
    manual = manual.drop_duplicates(subset=["criterion_id"], keep="first")

    return manual, manual_path, label_col

def validation_metrics_for_variant(df, risk_col):
    out = {}

    out["high_support_as_correct"] = binary_metrics(
        y_true=(df["manual_label"] == "correct").astype(int),
        y_pred=(df[risk_col] == "high_support_not_auto_accept").astype(int),
    )

    out["review_priority_as_partial_or_incorrect"] = binary_metrics(
        y_true=df["manual_label"].isin(["partial", "incorrect"]).astype(int),
        y_pred=(df[risk_col] == "review_priority").astype(int),
    )

    out["review_priority_as_incorrect_only"] = binary_metrics(
        y_true=(df["manual_label"] == "incorrect").astype(int),
        y_pred=(df[risk_col] == "review_priority").astype(int),
    )

    return out

# ---------------------------------------------------------------------
# Core ablation
# ---------------------------------------------------------------------

def compute_signal_only_support(df: pd.DataFrame) -> pd.Series:
    available = [c for c in SIGNAL_ONLY_COMPONENTS if c in df.columns]
    missing = [c for c in SIGNAL_ONLY_COMPONENTS if c not in df.columns]
    if missing:
        print("WARNING: Missing signal-only support components:", missing)
    if not available:
        raise ValueError("No signal-only support components found in score file.")

    temp = df[available].apply(pd.to_numeric, errors="coerce")
    return temp.min(axis=1)


def add_ablation_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "criterion_id" not in df.columns:
        raise ValueError("Score CSV must contain criterion_id.")

    df["leaf_support_signal_only"] = compute_signal_only_support(df)
    df["risk_score_signal_only"] = 1.0 - df["leaf_support_signal_only"]
    df["risk_label_signal_only"] = df["leaf_support_signal_only"].map(assign_risk_label)

    if "leaf_support" in df.columns:
        df["leaf_support_operational"] = pd.to_numeric(df["leaf_support"], errors="coerce")
    else:
        layer1 = pd.to_numeric(df.get("layer1_support", 1.0), errors="coerce").fillna(1.0)
        df["leaf_support_operational"] = pd.concat([df["leaf_support_signal_only"], layer1], axis=1).min(axis=1)

    df["risk_score_operational"] = 1.0 - df["leaf_support_operational"]

    if "risk_label" in df.columns:
        df["risk_label_operational"] = df["risk_label"].astype(str)
    else:
        df["risk_label_operational"] = df["leaf_support_operational"].map(assign_risk_label)

    if "layer1_issue_count" in df.columns:
        layer1_count = pd.to_numeric(df["layer1_issue_count"], errors="coerce").fillna(0)
        df["has_layer1_issue"] = layer1_count > 0
    else:
        layer1_bool_cols = [
            c for c in ["has_layer1_inventory_issue", "has_layer1_policy_issue", "has_layer1d_issue"]
            if c in df.columns
        ]
        if layer1_bool_cols:
            temp = df[layer1_bool_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
            df["has_layer1_issue"] = temp.max(axis=1) > 0
        else:
            df["has_layer1_issue"] = False

    df["signal_only_severity"] = df["risk_label_signal_only"].map(ROUTING_SEVERITY)
    df["operational_severity"] = df["risk_label_operational"].map(ROUTING_SEVERITY)
    df["operational_minus_signal_only_severity"] = df["operational_severity"] - df["signal_only_severity"]
    df["made_riskier_by_layer1_component"] = df["operational_minus_signal_only_severity"] > 0

    return df


def summarize_overlap(df: pd.DataFrame) -> Dict[str, Any]:
    l1 = df["has_layer1_issue"].astype(bool)
    sig_review = df["risk_label_signal_only"].eq("review_priority")
    op_review = df["risk_label_operational"].eq("review_priority")

    sig_prioritized = df["risk_label_signal_only"].isin(["intermediate_support", "review_priority"])
    op_prioritized = df["risk_label_operational"].isin(["intermediate_support", "review_priority"])

    return {
        "signal_only_review_priority": int(sig_review.sum()),
        "operational_review_priority": int(op_review.sum()),
        "layer1_and_signal_only_review_priority": int((l1 & sig_review).sum()),
        "layer1_and_operational_review_priority": int((l1 & op_review).sum()),
        "signal_only_review_priority_without_layer1": int((~l1 & sig_review).sum()),
        "operational_review_priority_without_layer1": int((~l1 & op_review).sum()),
        "layer1_flagged_not_signal_only_review_priority": int((l1 & ~sig_review).sum()),
        "layer1_flagged_not_operational_review_priority": int((l1 & ~op_review).sum()),
        "layer1_and_signal_only_intermediate_or_review": int((l1 & sig_prioritized).sum()),
        "layer1_and_operational_intermediate_or_review": int((l1 & op_prioritized).sum()),
    }


# ---------------------------------------------------------------------
# Manual validation summaries
# ---------------------------------------------------------------------

def manual_error_by_variant_and_risk(labeled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    variants = [
        ("signal_only", "risk_label_signal_only", "risk_score_signal_only", "leaf_support_signal_only"),
        ("operational", "risk_label_operational", "risk_score_operational", "leaf_support_operational"),
    ]

    for variant, label_col, risk_col, support_col in variants:
        for risk in ROUTING_LABEL_ORDER:
            sub = labeled[labeled[label_col] == risk]
            n = len(sub)
            rows.append({
                "variant": variant,
                "risk_label": risk,
                "n": int(n),
                "mean_risk_score": None if n == 0 else float(sub[risk_col].mean()),
                "mean_leaf_support": None if n == 0 else float(sub[support_col].mean()),
                "correct_rate": None if n == 0 else float((sub["manual_label"] == "correct").mean()),
                "partial_rate": None if n == 0 else float((sub["manual_label"] == "partial").mean()),
                "incorrect_rate": None if n == 0 else float((sub["manual_label"] == "incorrect").mean()),
                "error_rate_partial_or_incorrect": None if n == 0 else float(sub["manual_error_partial_or_incorrect"].mean()),
                "error_rate_incorrect_only": None if n == 0 else float(sub["manual_error_incorrect_only"].mean()),
            })

    return pd.DataFrame(rows)


def review_priority_detection_by_variant(labeled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    variants = [
        ("signal_only", "risk_label_signal_only", "risk_score_signal_only"),
        ("operational", "risk_label_operational", "risk_score_operational"),
    ]
    outcomes = [
        ("partial_or_incorrect", "manual_error_partial_or_incorrect"),
        ("incorrect_only", "manual_error_incorrect_only"),
    ]

    for variant, label_col, score_col in variants:
        review_priority = labeled[label_col].eq("review_priority")

        for outcome_name, outcome_col in outcomes:
            y = labeled[outcome_col].astype(int)
            n_error = int(y.sum())
            n_review_priority = int(review_priority.sum())
            true_errors_in_review_priority = int((review_priority & y.eq(1)).sum())

            auroc, auprc = try_metrics(
                y.tolist(),
                labeled[score_col].astype(float).tolist(),
            )

            rows.append({
                "variant": variant,
                "outcome": outcome_name,
                "n_labeled": int(len(labeled)),
                "n_errors": n_error,
                "n_review_priority": n_review_priority,
                "true_errors_in_review_priority": true_errors_in_review_priority,
                "review_priority_precision": (
                    None if n_review_priority == 0
                    else true_errors_in_review_priority / n_review_priority
                ),
                "review_priority_recall": (
                    None if n_error == 0
                    else true_errors_in_review_priority / n_error
                ),
                "auroc": auroc,
                "auprc": auprc,
            })

    return pd.DataFrame(rows)


def risk_coverage_by_variant(labeled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    variants = [
        ("signal_only", "risk_label_signal_only"),
        ("operational", "risk_label_operational"),
    ]

    policies = [
        (
            "not_review_high_support_reference_only",
            ["high_support_not_auto_accept"],
        ),
        (
            "not_review_high_and_intermediate_support",
            ["high_support_not_auto_accept", "intermediate_support"],
        ),
        (
            "all_labeled_leaves_reference",
            [
                "high_support_not_auto_accept",
                "intermediate_support",
                "review_priority",
            ],
        ),
    ]

    total = len(labeled)

    for variant, label_col in variants:
        for policy, not_reviewed_labels in policies:
            not_reviewed = labeled[labeled[label_col].isin(not_reviewed_labels)]
            reviewed = labeled[~labeled[label_col].isin(not_reviewed_labels)]

            n_not_reviewed = len(not_reviewed)
            n_reviewed = len(reviewed)

            rows.append({
                "variant": variant,
                "policy": policy,
                "not_reviewed_routing_labels": "|".join(not_reviewed_labels),
                "not_reviewed_n": int(n_not_reviewed),
                "reviewed_n": int(n_reviewed),
                "coverage_not_reviewed_among_labeled": (
                    None if total == 0 else n_not_reviewed / total
                ),
                "partial_or_incorrect_rate_among_not_reviewed": (
                    None if n_not_reviewed == 0
                    else float(not_reviewed["manual_error_partial_or_incorrect"].mean())
                ),
                "incorrect_only_rate_among_not_reviewed": (
                    None if n_not_reviewed == 0
                    else float(not_reviewed["manual_error_incorrect_only"].mean())
                ),
                "correct_rate_among_not_reviewed": (
                    None if n_not_reviewed == 0
                    else float((not_reviewed["manual_label"] == "correct").mean())
                ),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    root = Path(__file__).resolve().parents[3]

    score_path = (
        root
        / "outputs"
        / "verification"
        / "layer2"
        / "branch_a"
        / "layer2_branch_a_leaf_risk_scores.csv"
    )

    out_dir = (
        root
        / "outputs"
        / "verification"
        / "layer2"
        / "branch_a"
        / "ablation_signal_only_vs_operational"
    )

    if not score_path.exists():
        raise FileNotFoundError(f"Score file not found: {score_path}")

    print("\nLayer 2 Branch A ablation: signal-only vs operational")
    print("Score input:", score_path)
    print("Output dir:", out_dir)

    scores = pd.read_csv(score_path)
    df = add_ablation_columns(scores)

    expected_labels = set(ROUTING_LABEL_ORDER)

    for col in ["risk_label_signal_only", "risk_label_operational"]:
        observed = set(df[col].dropna().astype(str))
        old = observed.intersection({"low", "medium", "high"})
        unknown = observed - expected_labels

        if old:
            raise RuntimeError(
                f"Old low/medium/high labels found in {col}: {sorted(old)}. "
                "Rerun 02_score_branch_a_leaf_risk.py and check this script."
            )

        if unknown:
            raise RuntimeError(
                f"Unexpected routing labels found in {col}: {sorted(unknown)}. "
                "Update ROUTING_LABEL_ORDER or check the score file."
            )

    # Write leaf-level ablation file.
    leaf_out = out_dir / "layer2_branch_a_ablation_leaf_level.csv"
    write_csv(leaf_out, df)

    # Crosstabs.
    c1 = crosstab_with_order(df, "has_layer1_issue", "risk_label_signal_only")
    c2 = crosstab_with_order(df, "has_layer1_issue", "risk_label_operational")
    c3 = crosstab_with_order(df, "risk_label_signal_only", "risk_label_operational")

    write_csv(out_dir / "crosstab_layer1_vs_signal_only_risk.csv", c1)
    write_csv(out_dir / "crosstab_layer1_vs_operational_risk.csv", c2)
    write_csv(out_dir / "crosstab_signal_only_vs_operational_risk.csv", c3)

    summary: Dict[str, Any] = {
        "description": "Ablation comparing Layer 2 signal-only risk with operational Layer 2 risk including Layer 1 support.",
        "score_input": str(score_path),
        "leaf_output": str(leaf_out),
        "thresholds": {
            "review_priority_if_leaf_support_below": REVIEW_PRIORITY_THRESHOLD,
            "high_support_reference_if_leaf_support_at_least": HIGH_SUPPORT_REFERENCE_THRESHOLD,
            "intermediate_support_if_leaf_support_between": [
                REVIEW_PRIORITY_THRESHOLD,
                HIGH_SUPPORT_REFERENCE_THRESHOLD,
            ],
        },
        "signal_only_components": [c for c in SIGNAL_ONLY_COMPONENTS if c in df.columns],
        "operational_score_definition": "Current Branch A leaf_support and risk_label, including layer1_support as a support component.",
        "overlap_summary": summarize_overlap(df),
        "risk_counts_signal_only": df["risk_label_signal_only"].value_counts().to_dict(),
        "risk_counts_operational": df["risk_label_operational"].value_counts().to_dict(),
    }

    # Optional manual validation comparison.
    manual, manual_path, manual_label_col = load_manual_a_labels(root)
    if manual is not None and not manual.empty:
        labeled = df.merge(
            manual[["criterion_id", "manual_label_raw", "manual_label"]],
            on="criterion_id",
            how="inner",
            validate="one_to_one",
        )
        labeled["manual_error_partial_or_incorrect"] = labeled["manual_label"].isin(["partial", "incorrect"]).astype(int)
        labeled["manual_error_incorrect_only"] = labeled["manual_label"].eq("incorrect").astype(int)

        err_by_risk = manual_error_by_variant_and_risk(labeled)
        review_det = review_priority_detection_by_variant(labeled)
        coverage = risk_coverage_by_variant(labeled)

        write_csv(out_dir / "manual_error_by_variant_and_risk.csv", err_by_risk)
        write_csv(out_dir / "manual_review_priority_detection_by_variant.csv", review_det)
        write_csv(out_dir / "manual_risk_coverage_by_variant.csv", coverage)
        write_csv(out_dir / "manual_ablation_merged_labeled.csv", labeled)

        summary["manual_validation"] = {
            "manual_input": str(manual_path),
            "manual_label_column": manual_label_col,
            "n_labeled_matched": int(len(labeled)),
            "manual_label_counts": labeled["manual_label"].value_counts().to_dict(),
            "manual_binary_metrics": {
                "signal_only": validation_metrics_for_variant(
                    labeled,
                    "risk_label_signal_only",
                ),
                "operational": validation_metrics_for_variant(
                    labeled,
                    "risk_label_operational",
                ),
            },
            "outputs": {
                "manual_error_by_variant_and_risk": str(out_dir / "manual_error_by_variant_and_risk.csv"),
                "manual_review_priority_detection_by_variant": str(
                    out_dir / "manual_review_priority_detection_by_variant.csv"
                ),
                "manual_risk_coverage_by_variant": str(out_dir / "manual_risk_coverage_by_variant.csv"),
                "manual_ablation_merged_labeled": str(out_dir / "manual_ablation_merged_labeled.csv"),
            },
        }
    else:
        summary["manual_validation"] = {
            "status": "manual labels not found or not usable",
        }

    summary_out = out_dir / "layer2_branch_a_ablation_summary.json"
    write_json(summary_out, summary)

    print("DONE")
    print("Total leaves:", len(df))
    print("Signal-only risk counts:", summary["risk_counts_signal_only"])
    print("Operational risk counts:", summary["risk_counts_operational"])
    print("Overlap summary:", summary["overlap_summary"])

    if "n_labeled_matched" in summary.get("manual_validation", {}):
        print("Manual labels matched:", summary["manual_validation"]["n_labeled_matched"])
        print("Manual label column:", summary["manual_validation"]["manual_label_column"])

    print("Wrote summary:", summary_out)


if __name__ == "__main__":
    main()

# Run from the repository root:
# python scripts/03_verification/02_layer2/04_compare_branch_a_scoring_ablation.py