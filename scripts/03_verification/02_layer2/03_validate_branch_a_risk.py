"""
03_validate_branch_a_risk.py

Validate the Branch A Layer 2 risk score against the manually reviewed
semantic labels.

This script does not modify the logical rule tree and does not change
the previously calculated support or risk scores.

Main question:
    Are leaves assigned to review_priority enriched for partial or
    incorrect manual labels?

Inputs:
    outputs/verification/layer2/branch_a/
        layer2_branch_a_leaf_risk_scores.csv

    outputs/evaluation/pre_verification/semantic_manual_pre_verification_A_B_summary/
        reviewed_semantic_clause_labels_A_B.csv

Outputs:
    outputs/verification/layer2/branch_a/
        layer2_branch_a_manual_validation_merged.csv
        layer2_branch_a_risk_manual_crosstab.csv
        layer2_branch_a_error_by_risk_label.csv
        layer2_branch_a_risk_coverage_table.csv
        layer2_branch_a_validation_examples_high_support_error.csv
        layer2_branch_a_validation_examples_review_priority_correct.csv
        layer2_branch_a_threshold_sensitivity_table.csv
        layer2_branch_a_manual_validation_summary.json

Run from the repository root:
python scripts/03_verification/02_layer2/03_validate_branch_a_risk.py
"""

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def normalize_text(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()


def norm_col_name(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def normalize_manual_label(x: Any) -> Optional[str]:
    """
    Normalize manual labels to: correct / partial / incorrect.

    Important:
    Check 'incorrect' before 'correct', because 'incorrect' contains 'correct'.
    """
    s = normalize_text(x).lower()
    if not s or s in {"nan", "none", "null", "na", "n/a", "not_labeled", "unlabeled", "missing"}:
        return None

    # Exact/common forms first.
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

    # Conservative substring fallback.
    if "incorrect" in s or "wrong" in s or "unsupported" in s:
        return "incorrect"
    if "partial" in s:
        return "partial"
    if "correct" in s:
        return "correct"

    return None


def to_int_bool(x: Any) -> int:
    s = normalize_text(x).lower()
    if s in {"1", "true", "yes", "y"}:
        return 1
    if s in {"0", "false", "no", "n", "", "nan", "none", "null"}:
        return 0
    try:
        return int(float(s) != 0)
    except Exception:
        return 0


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, float) and math.isnan(x):
            return None
        s = normalize_text(x)
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------
# Locate and prepare manual labels
# ---------------------------------------------------------------------

def candidate_manual_paths(root: Path) -> List[Path]:
    """
    Use the manually reviewed pre-verification A/B semantic labels
    employed in the thesis evaluation.
    """
    manual_path = (
        root
        / "outputs"
        / "evaluation"
        / "pre_verification"
        / "semantic_manual_pre_verification_A_B_summary"
        / "reviewed_semantic_clause_labels_A_B.csv"
    )

    if not manual_path.exists():
        raise FileNotFoundError(
            f"Reviewed A/B semantic label file not found: {manual_path}"
        )

    return [manual_path]

def detect_manual_label_column(df: pd.DataFrame) -> Optional[str]:
    """
    Detect the Branch A manual semantic label column.

    Works for:
    - wide files with A_label / A_leaf_label / label_A
    - simple files with one manual label column
    """
    if df.empty:
        return None

    # Exact/preferred names first.
    preferred_names = [
        "manual_A_leaf_label",
        "manual_a_leaf_label",
        "manual_A_label",
        "manual_a_label",
        "A_label",
        "A_leaf_label",
        "A_manual_label",
        "A_semantic_label",
        "manual_label_A",
        "semantic_label_A",
        "leaf_label_A",
        "label_A",
        "branch_a_label",
        "branch_a_manual_label",
        "branch_A_label",
        "manual_semantic_label_A",
        "manual_label",
        "semantic_label",
        "leaf_label",
        "label",
    ]

    norm_to_original = {norm_col_name(c): c for c in df.columns}
    for name in preferred_names:
        key = norm_col_name(name)
        if key in norm_to_original:
            c = norm_to_original[key]
            labels = df[c].map(normalize_manual_label)
            if labels.notna().sum() > 0:
                return c

    # Heuristic: candidate columns whose values look like correct/partial/incorrect.
    scored: List[Tuple[float, str]] = []
    for c in df.columns:
        cn = norm_col_name(c)

        # Exclude obvious issue/reason columns.
        if any(bad in cn for bad in ["issue", "reason", "comment", "notes", "text", "evidence", "entity"]):
            continue

        labels = df[c].map(normalize_manual_label)
        n_label = int(labels.notna().sum())
        if n_label == 0:
            continue

        frac = n_label / max(len(df), 1)

        score = frac

        # Prefer Branch A columns.
        if cn.startswith("a_") or cn.endswith("_a") or "branch_a" in cn or "bert" in cn:
            score += 1.0

        # Prefer columns containing label/correctness.
        if "label" in cn or "correct" in cn or "semantic" in cn:
            score += 0.5

        # Penalize B/C columns.
        if cn.startswith("b_") or cn.endswith("_b") or "branch_b" in cn or "llm" in cn:
            score -= 1.0
        if cn.startswith("c_") or cn.endswith("_c") or "branch_c" in cn or "strict" in cn or "relaxed" in cn:
            score -= 1.0

        scored.append((score, c))

    if not scored:
        return None

    scored.sort(reverse=True)
    return scored[0][1]


def ensure_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure criterion_id exists if possible.
    """
    df = df.copy()

    # Normalize common criterion_id variants.
    norm_to_original = {norm_col_name(c): c for c in df.columns}
    for cand in ["criterion_id", "leaf_id", "leaf_uid", "criterion_uid"]:
        key = norm_col_name(cand)
        if key in norm_to_original and "criterion_id" not in df.columns:
            df["criterion_id"] = df[norm_to_original[key]].astype(str)
            break

    # Build from item_uid + clause_id if needed.
    if "criterion_id" not in df.columns:
        item_col = None
        clause_col = None
        for cand in ["item_uid", "item_id"]:
            if norm_col_name(cand) in norm_to_original:
                item_col = norm_to_original[norm_col_name(cand)]
                break
        for cand in ["clause_id", "clause"]:
            if norm_col_name(cand) in norm_to_original:
                clause_col = norm_to_original[norm_col_name(cand)]
                break

        if item_col and clause_col:
            df["criterion_id"] = df[item_col].astype(str) + "_" + df[clause_col].astype(str)

    # Also keep item_uid/clause_id if variants exist.
    norm_to_original = {norm_col_name(c): c for c in df.columns}
    if "item_uid" not in df.columns and "item_uid" in norm_to_original:
        df["item_uid"] = df[norm_to_original["item_uid"]]
    if "clause_id" not in df.columns and "clause_id" in norm_to_original:
        df["clause_id"] = df[norm_to_original["clause_id"]]

    return df


def load_manual_labels(root: Path) -> Tuple[pd.DataFrame, Path, str]:
    paths = candidate_manual_paths(root)

    debug_info = []
    for path in paths:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            debug_info.append(f"{path}: could not read ({e})")
            continue

        if df.empty:
            debug_info.append(f"{path}: empty")
            continue

        label_col = "manual_A_leaf_label"

        if label_col not in df.columns:
            debug_info.append(
                f"{path}: expected column {label_col}, but columns are {list(df.columns)[:30]}"
            )
            continue

        df = ensure_key_columns(df)
        if "criterion_id" not in df.columns:
            debug_info.append(f"{path}: label col {label_col}, but no criterion_id or item_uid+clause_id key")
            continue

        manual = pd.DataFrame({
            "criterion_id": df["criterion_id"].astype(str),
            "manual_label_raw": df[label_col],
            "manual_label": df[label_col].map(normalize_manual_label),
        })

        # Keep useful context columns if present.
        for c in ["item_uid", "clause_id", "entity_text", "evidence_text", "clause_text"]:
            if c in df.columns and c not in manual.columns:
                manual[c + "_manual"] = df[c]

        manual = manual[manual["manual_label"].notna()].copy()

        if manual.empty:
            debug_info.append(f"{path}: label col {label_col}, but no normalized labels")
            continue

        # If duplicate labels exist for the same criterion_id, keep the first non-empty one.
        manual = manual.drop_duplicates(subset=["criterion_id"], keep="first")

        return manual, path, label_col

    message = (
        "Could not find a usable manual clause-label CSV.\n\n"
        "Tried:\n"
        + "\n".join(str(p) for p in paths)
        + "\n\nDebug:\n"
        + "\n".join(debug_info)
        + "\n\nExpected a file with criterion_id OR item_uid+clause_id and a Branch A label column "
        "containing values like correct / partial / incorrect."
    )
    raise FileNotFoundError(message)


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def auc_roc_binary(y_true: List[int], scores: List[float]) -> Optional[float]:
    """
    Rank-based AUROC fallback. Returns None if only one class is present.
    """
    n = len(y_true)
    if n == 0:
        return None

    n_pos = sum(y_true)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    # Average ranks for ties, ranks start at 1.
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
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def average_precision_binary(y_true: List[int], scores: List[float]) -> Optional[float]:
    """
    Average precision / AUPRC fallback.
    """
    n_pos = sum(y_true)
    if n_pos == 0:
        return None

    pairs = sorted(zip(scores, y_true), key=lambda x: x[0], reverse=True)

    tp = 0
    precisions_at_pos = []
    for i, (_, y) in enumerate(pairs, start=1):
        if y == 1:
            tp += 1
            precisions_at_pos.append(tp / i)

    if not precisions_at_pos:
        return None

    return float(sum(precisions_at_pos) / n_pos)


def try_sklearn_metrics(y_true: List[int], scores: List[float]) -> Tuple[Optional[float], Optional[float]]:
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        if len(set(y_true)) < 2:
            return None, None
        return float(roc_auc_score(y_true, scores)), float(average_precision_score(y_true, scores))
    except Exception:
        return auc_roc_binary(y_true, scores), average_precision_binary(y_true, scores)

ROUTING_LABEL_ORDER = [
    "high_support_not_auto_accept",
    "intermediate_support",
    "review_priority",
]

# ---------------------------------------------------------------------
# Validation tables
# ---------------------------------------------------------------------

def make_risk_manual_crosstab(df: pd.DataFrame) -> pd.DataFrame:
    label_order = ["correct", "partial", "incorrect"]

    tab = pd.crosstab(df["risk_label"], df["manual_label"])

    for r in ROUTING_LABEL_ORDER:
        if r not in tab.index:
            tab.loc[r] = 0

    for c in label_order:
        if c not in tab.columns:
            tab[c] = 0

    tab = tab.loc[ROUTING_LABEL_ORDER, label_order]
    tab["total"] = tab.sum(axis=1)
    tab["error_count"] = tab["partial"] + tab["incorrect"]
    tab["error_rate"] = tab["error_count"] / tab["total"].replace(0, pd.NA)

    for c in label_order:
        tab[c + "_pct"] = tab[c] / tab["total"].replace(0, pd.NA)

    return tab.reset_index()


def make_error_by_risk_label(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for risk in ROUTING_LABEL_ORDER:
        sub = df[df["risk_label"] == risk]
        n = len(sub)

        if n == 0:
            rows.append({
                "risk_label": risk,
                "n": 0,
                "mean_risk_score": None,
                "mean_leaf_support": None,
                "correct_rate": None,
                "partial_rate": None,
                "incorrect_rate": None,
                "error_rate_partial_or_incorrect": None,
            })
            continue

        rows.append({
            "risk_label": risk,
            "n": n,
            "mean_risk_score": sub["risk_score"].astype(float).mean(),
            "mean_leaf_support": sub["leaf_support"].astype(float).mean(),
            "correct_rate": (sub["manual_label"] == "correct").mean(),
            "partial_rate": (sub["manual_label"] == "partial").mean(),
            "incorrect_rate": (sub["manual_label"] == "incorrect").mean(),
            "error_rate_partial_or_incorrect": (sub["manual_error"] == 1).mean(),
        })

    return pd.DataFrame(rows)


def make_risk_coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selective review table:
    what error rate remains if selected support groups are not prioritized for review?

    This is not automatic acceptance.
    """
    groups = [
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

    total = len(df)
    rows = []

    for name, labels in groups:
        not_reviewed = df[df["risk_label"].isin(labels)]
        reviewed = df[~df["risk_label"].isin(labels)]

        n_not_reviewed = len(not_reviewed)
        n_reviewed = len(reviewed)

        rows.append({
            "policy": name,
            "not_reviewed_routing_labels": "|".join(labels),
            "not_reviewed_n": n_not_reviewed,
            "reviewed_n": n_reviewed,
            "coverage_not_reviewed_among_labeled": n_not_reviewed / total if total else None,
            "error_rate_among_not_reviewed": not_reviewed["manual_error"].mean() if n_not_reviewed else None,
            "correct_rate_among_not_reviewed": (not_reviewed["manual_label"] == "correct").mean() if n_not_reviewed else None,
            "partial_rate_among_not_reviewed": (not_reviewed["manual_label"] == "partial").mean() if n_not_reviewed else None,
            "incorrect_rate_among_not_reviewed": (not_reviewed["manual_label"] == "incorrect").mean() if n_not_reviewed else None,
        })

    return pd.DataFrame(rows)

def make_threshold_sensitivity_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate different leaf_support cutoffs.

    For each cutoff t:
      accepted = leaf_support >= t
      rejected/reviewed = leaf_support < t

    This does not recalibrate the model. It shows the coverage-error tradeoff
    on the manually labeled subset.
    """
    rows = []
    total = len(df)
    total_errors = int(df["manual_error"].sum())

    thresholds = [round(x / 100, 2) for x in range(50, 96, 5)]

    for t in thresholds:
        accepted = df[df["leaf_support"].astype(float) >= t]
        rejected = df[df["leaf_support"].astype(float) < t]

        n_acc = len(accepted)
        n_rej = len(rejected)

        acc_errors = int(accepted["manual_error"].sum()) if n_acc else 0
        rej_errors = int(rejected["manual_error"].sum()) if n_rej else 0

        rows.append({
            "leaf_support_threshold": t,
            "accepted_n": n_acc,
            "rejected_n": n_rej,
            "coverage_among_labeled": n_acc / total if total else None,

            "accepted_error_n": acc_errors,
            "accepted_error_rate": acc_errors / n_acc if n_acc else None,

            "rejected_error_n": rej_errors,
            "rejected_error_rate": rej_errors / n_rej if n_rej else None,

            "error_recall_among_rejected": rej_errors / total_errors if total_errors else None,
            "correct_rejected_n": int((rejected["manual_label"] == "correct").sum()) if n_rej else 0,
        })

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    ROOT = Path(__file__).resolve().parents[3]

    score_path = (
        ROOT
        / "outputs"
        / "verification"
        / "layer2"
        / "branch_a"
        / "layer2_branch_a_leaf_risk_scores.csv"
    )

    out_dir = ROOT / "outputs" / "verification" / "layer2" / "branch_a"

    merged_out = out_dir / "layer2_branch_a_manual_validation_merged.csv"
    crosstab_out = out_dir / "layer2_branch_a_risk_manual_crosstab.csv"
    error_by_risk_out = out_dir / "layer2_branch_a_error_by_risk_label.csv"
    coverage_out = out_dir / "layer2_branch_a_risk_coverage_table.csv"
    high_support_error_out = out_dir / "layer2_branch_a_validation_examples_high_support_error.csv"
    review_priority_correct_out = out_dir / "layer2_branch_a_validation_examples_review_priority_correct.csv"
    summary_out = out_dir / "layer2_branch_a_manual_validation_summary.json"
    threshold_sensitivity_out = out_dir / "layer2_branch_a_threshold_sensitivity_table.csv"

    if not score_path.exists():
        raise FileNotFoundError(f"Layer 2 score file not found: {score_path}")

    print("\nLayer 2 Branch A validation against manual labels")
    print("Score input:", score_path)

    scores = pd.read_csv(score_path)
    if "criterion_id" not in scores.columns:
        raise ValueError("Score CSV must contain criterion_id.")

    # Make sure numeric columns are numeric.
    for c in ["leaf_support", "risk_score"]:
        if c in scores.columns:
            scores[c] = pd.to_numeric(scores[c], errors="coerce")

    manual, manual_path, manual_label_col = load_manual_labels(ROOT)

    label_col_norm = norm_col_name(manual_label_col)

    forbidden_label_patterns = [
        "manual_b",
        "b_leaf",
        "branch_b",
        "manual_c",
        "c_strict",
        "c_relaxed",
        "branch_c",
    ]

    if any(pattern in label_col_norm for pattern in forbidden_label_patterns):
        raise RuntimeError(
            f"Wrong manual label column detected for Branch A validation: {manual_label_col}. "
            "This script must use manual_A_leaf_label or another Branch A label column."
        )

    print("Manual input:", manual_path)
    print("Manual label column detected:", manual_label_col)
    print("Manual labeled rows:", len(manual))

    merged = scores.merge(
        manual[["criterion_id", "manual_label_raw", "manual_label"]],
        on="criterion_id",
        how="left",
        validate="one_to_one",
    )

    labeled = merged[merged["manual_label"].notna()].copy()

    if labeled.empty:
        raise RuntimeError(
            "No scored leaves matched manual labels. Check criterion_id / item_uid / clause_id alignment."
        )

    labeled["manual_error"] = labeled["manual_label"].isin(["partial", "incorrect"]).astype(int)

    observed_routing_labels = set(labeled["risk_label"].dropna().astype(str))
    expected_routing_labels = set(ROUTING_LABEL_ORDER)

    unknown_routing_labels = observed_routing_labels - expected_routing_labels
    if unknown_routing_labels:
        raise RuntimeError(
            "Unexpected risk_label values found in score file: "
            f"{sorted(unknown_routing_labels)}. "
            "Rerun 02_score_branch_a_leaf_risk.py or update ROUTING_LABEL_ORDER."
        )

    old_labels = observed_routing_labels.intersection({"low", "medium", "high"})
    if old_labels:
        raise RuntimeError(
            "Old low/medium/high labels detected in score file. "
            "Rerun 02_score_branch_a_leaf_risk.py after the routing-label update."
        )

    # Ensure risk_score exists.
    if "risk_score" not in labeled.columns or labeled["risk_score"].isna().all():
        if "leaf_support" not in labeled.columns:
            raise ValueError("Need risk_score or leaf_support in score file.")
        labeled["risk_score"] = 1.0 - labeled["leaf_support"].astype(float)

    crosstab = make_risk_manual_crosstab(labeled)
    error_by_risk = make_error_by_risk_label(labeled)
    coverage = make_risk_coverage_table(labeled)
    threshold_sensitivity = make_threshold_sensitivity_table(labeled)

    y_true = labeled["manual_error"].astype(int).tolist()
    risk_scores = labeled["risk_score"].astype(float).tolist()
    auroc, auprc = try_sklearn_metrics(y_true, risk_scores)

    # Useful example exports.
    context_cols = [
        "criterion_id",
        "risk_label",
        "leaf_support",
        "risk_score",
        "manual_label",
        "risk_reasons",
        "risk_bottleneck_components",
        "entity_text",
        "entity_type",
        "operator",
        "value_json",
        "unit",
        "evidence_text",
        "best_anchor_text",
        "best_anchor_score",
        "layer1_issue_sources",
    ]
    context_cols = [c for c in context_cols if c in labeled.columns]

    high_support_error = (
        labeled[
            (labeled["risk_label"] == "high_support_not_auto_accept")
            & (labeled["manual_error"] == 1)
        ]
        .sort_values(["risk_score"], ascending=True)
        [context_cols]
        .head(30)
    )

    review_priority_correct = (
        labeled[
            (labeled["risk_label"] == "review_priority")
            & (labeled["manual_label"] == "correct")
        ]
        .sort_values(["risk_score"], ascending=False)
        [context_cols]
        .head(30)
    )

    write_csv(merged_out, labeled)
    write_csv(crosstab_out, crosstab)
    write_csv(error_by_risk_out, error_by_risk)
    write_csv(coverage_out, coverage)
    write_csv(threshold_sensitivity_out, threshold_sensitivity)
    write_csv(high_support_error_out, high_support_error)
    write_csv(review_priority_correct_out, review_priority_correct)

    risk_counts = labeled["risk_label"].value_counts().to_dict()
    manual_counts = labeled["manual_label"].value_counts().to_dict()

    # Enrichment check: high-risk error rate should be greater than low-risk error rate.
    err_table = error_by_risk.set_index("risk_label")
    def get_error_rate(label: str):
        if label not in err_table.index:
            return None
        value = err_table.loc[label, "error_rate_partial_or_incorrect"]
        if pd.isna(value):
            return None
        return float(value)


    high_support_err = get_error_rate("high_support_not_auto_accept")
    intermediate_err = get_error_rate("intermediate_support")
    review_priority_err = get_error_rate("review_priority")

    review_priority_has_higher_error_than_high_support = (
        None
        if high_support_err is None or review_priority_err is None
        else bool(review_priority_err > high_support_err)
    )

    summary = {
        "description": "Validation of Layer 2 Branch A risk score against manual semantic clause labels.",
        "score_input": str(score_path),
        "manual_input": str(manual_path),
        "manual_label_column_detected": manual_label_col,
        "n_scores_total": int(len(scores)),
        "n_manual_labeled_unique": int(len(manual)),
        "n_scored_with_manual_label": int(len(labeled)),
        "manual_label_counts": {str(k): int(v) for k, v in manual_counts.items()},
        "risk_label_counts_among_labeled": {str(k): int(v) for k, v in risk_counts.items()},
        "manual_error_definition": "partial_or_incorrect",
        "manual_error_rate_overall": float(labeled["manual_error"].mean()),
        "auroc_detecting_partial_or_incorrect": auroc,
        "auprc_detecting_partial_or_incorrect": auprc,
        "high_support_error_rate": high_support_err,
        "intermediate_support_error_rate": intermediate_err,
        "review_priority_error_rate": review_priority_err,
        "review_priority_has_higher_error_than_high_support": review_priority_has_higher_error_than_high_support,
        "outputs": {
            "merged": str(merged_out),
            "risk_manual_crosstab": str(crosstab_out),
            "error_by_risk_label": str(error_by_risk_out),
            "risk_coverage_table": str(coverage_out),
            "threshold_sensitivity_table": str(threshold_sensitivity_out),
            "high_support_error_examples": str(high_support_error_out),
            "review_priority_correct_examples": str(review_priority_correct_out),
        },
    }

    write_json(summary_out, summary)

    print("DONE")
    print("Scored leaves total:", len(scores))
    print("Manual labeled leaves matched:", len(labeled))
    print("Manual label counts:", manual_counts)
    print("Risk label counts among labeled:", risk_counts)
    print("Overall manual error rate:", round(float(labeled["manual_error"].mean()), 4))
    print("AUROC detecting partial/incorrect:", None if auroc is None else round(float(auroc), 4))
    print("AUPRC detecting partial/incorrect:", None if auprc is None else round(float(auprc), 4))
    print("Error rate by risk label:")
    print(error_by_risk.to_string(index=False))
    print("Risk-coverage table:")
    print(coverage.to_string(index=False))

    print("Threshold sensitivity table:")
    print(threshold_sensitivity.to_string(index=False))

    print("Wrote summary:", summary_out)


if __name__ == "__main__":
    main()

# Run from the repository root: 
# # python scripts/03_verification/02_layer2/03_validate_branch_a_risk.py