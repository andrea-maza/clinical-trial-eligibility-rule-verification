"""
07_validate_branch_b_grounding.py

Validate the Branch B Layer 2 grounding and routing screen against the
manually reviewed Branch B semantic labels.

The manual labels measure semantic correctness:
    correct / partial / incorrect

Therefore, semantic grounding support is evaluated against the manual
labels, while execution support is reported separately as a measure of
computability and rule usability.

Inputs:
    outputs/verification/layer2/branch_b/
        layer2_branch_b_grounding_screen_leaf_level.csv

    outputs/evaluation/pre_verification/
        semantic_manual_pre_verification_A_B_summary/
            reviewed_semantic_clause_labels_A_B.csv

Outputs:
    outputs/verification/layer2/branch_b/validation_against_manual/

Run from the repository root:
python scripts/03_verification/02_layer2/07_validate_branch_b_grounding.py
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]

GROUNDING_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "verification"
    / "layer2"
    / "branch_b"
    / "layer2_branch_b_grounding_screen_leaf_level.csv"
)

MANUAL_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "pre_verification"
    / "semantic_manual_pre_verification_A_B_summary"
    / "reviewed_semantic_clause_labels_A_B.csv"
)

OUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "verification"
    / "layer2"
    / "branch_b"
    / "validation_against_manual"
)

OUT_MATCHED_CSV = OUT_DIR / "layer2_branch_b_grounding_manual_matched_rows.csv"
OUT_SEMANTIC_BY_RISK_CSV = OUT_DIR / "semantic_error_by_risk_label.csv"
OUT_SEMANTIC_BY_ROUTING_CSV = OUT_DIR / "semantic_error_by_routing_decision.csv"
OUT_EXECUTION_BY_MANUAL_CSV = OUT_DIR / "execution_risk_vs_manual_label.csv"
OUT_SUMMARY_JSON = OUT_DIR / "layer2_branch_b_grounding_manual_validation_summary.json"
OUT_THRESHOLD_SENSITIVITY_CSV = OUT_DIR / "semantic_grounding_threshold_sensitivity.csv"

MANUAL_LABEL_COL_CANDIDATES = [
    "manual_B_leaf_label",
    "manual_b_leaf_label",
    "manual_B_label",
    "manual_label_B",
]

JOIN_COL_CANDIDATES = [
    "criterion_id",
    "leaf_id",
    "clause_id",
    "pass1_clause_id",
    "id",
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def serialize_cell(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, bool):
        return "1" if x else "0"
    if isinstance(x, (dict, list)):
        return json.dumps(x, ensure_ascii=False)
    return str(x)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    priority = [
        "criterion_id",
        "document_id",
        "manual_B_leaf_label",
        "manual_label_normalized",
        "manual_is_error_partial_or_incorrect",
        "manual_is_incorrect",
        "semantic_grounding_risk_label",
        "semantic_grounding_support",
        "semantic_grounding_reasons",
        "execution_risk_label",
        "execution_support",
        "execution_reasons",
        "final_routing_decision",
        "llm_verifier_candidate",
        "computability_review_candidate",
        "entity_text",
        "operator",
        "value",
        "unit",
        "evidence_text",
    ]

    cols: List[str] = []
    seen = set()
    for c in priority:
        if any(c in r for r in rows):
            cols.append(c)
            seen.add(c)
    for r in rows:
        for c in r:
            if c not in seen:
                cols.append(c)
                seen.add(c)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: serialize_cell(r.get(c, "")) for c in cols})


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def clean(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def norm_label(x: Any) -> str:
    return clean(x).lower().replace(" ", "_")


def to_float(x: Any, default: float = 0.0) -> float:
    try:
        s = clean(x)
        if not s:
            return default
        return float(s)
    except Exception:
        return default


def detect_col(rows: List[Dict[str, str]], candidates: List[str]) -> Optional[str]:
    if not rows:
        return None
    cols = set(rows[0].keys())
    for c in candidates:
        if c in cols:
            return c
    lower_to_actual = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower_to_actual:
            return lower_to_actual[c.lower()]
    return None


def normalize_manual_label(label: Any) -> str:
    x = norm_label(label)
    if x in {"correct", "ok", "true", "valid"}:
        return "correct"
    if x in {"partial", "partially_correct", "partly_correct"}:
        return "partial"
    if x in {"incorrect", "wrong", "false", "error"}:
        return "incorrect"
    return x


def safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def binary_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    y_true = 1 means manual error.
    y_pred = 1 means flagged for semantic review.
    """
    tp = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 1)
    tn = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 0)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 0)

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    npv = safe_div(tn, tn + fn)
    accuracy = safe_div(tp + tn, tp + fp + tn + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    return {
        "n": len(y_true),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision_positive_predictive_value": round(precision, 6),
        "recall_sensitivity": round(recall, 6),
        "specificity": round(specificity, 6),
        "negative_predictive_value": round(npv, 6),
        "accuracy": round(accuracy, 6),
        "f1": round(f1, 6),
    }


def grouped_rates(rows: List[Dict[str, Any]], group_col: str) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(clean(r.get(group_col, "")), []).append(r)

    out: List[Dict[str, Any]] = []
    for group, rs in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        n = len(rs)
        correct = sum(1 for r in rs if r["manual_label_normalized"] == "correct")
        partial = sum(1 for r in rs if r["manual_label_normalized"] == "partial")
        incorrect = sum(1 for r in rs if r["manual_label_normalized"] == "incorrect")
        error = partial + incorrect

        sem = [to_float(r.get("semantic_grounding_support"), math.nan) for r in rs]
        sem = [x for x in sem if not math.isnan(x)]
        exe = [to_float(r.get("execution_support"), math.nan) for r in rs]
        exe = [x for x in exe if not math.isnan(x)]

        out.append({
            group_col: group,
            "n": n,
            "correct_n": correct,
            "partial_n": partial,
            "incorrect_n": incorrect,
            "manual_error_n_partial_or_incorrect": error,
            "correct_rate": round(safe_div(correct, n), 6),
            "partial_rate": round(safe_div(partial, n), 6),
            "incorrect_rate": round(safe_div(incorrect, n), 6),
            "manual_error_rate_partial_or_incorrect": round(safe_div(error, n), 6),
            "manual_incorrect_only_rate": round(safe_div(incorrect, n), 6),
            "mean_semantic_grounding_support": round(sum(sem) / len(sem), 6) if sem else "",
            "mean_execution_support": round(sum(exe) / len(exe), 6) if exe else "",
        })
    return out

def threshold_sensitivity_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Evaluate semantic_grounding_support thresholds against manual_B labels.

    For each threshold t:
      prioritized = semantic_grounding_support < t
      not_prioritized = semantic_grounding_support >= t

    This is not calibration. It is a coverage-error tradeoff table.
    """
    out = []
    total = len(rows)
    total_errors = sum(int(r["manual_is_error_partial_or_incorrect"]) for r in rows)

    thresholds = [round(x / 100, 2) for x in range(40, 96, 5)]

    for t in thresholds:
        prioritized = [
            r for r in rows
            if to_float(r.get("semantic_grounding_support")) < t
        ]
        not_prioritized = [
            r for r in rows
            if to_float(r.get("semantic_grounding_support")) >= t
        ]

        n_p = len(prioritized)
        n_np = len(not_prioritized)

        err_p = sum(int(r["manual_is_error_partial_or_incorrect"]) for r in prioritized)
        err_np = sum(int(r["manual_is_error_partial_or_incorrect"]) for r in not_prioritized)

        incorrect_p = sum(int(r["manual_is_incorrect"]) for r in prioritized)
        incorrect_np = sum(int(r["manual_is_incorrect"]) for r in not_prioritized)

        out.append({
            "semantic_support_threshold": t,
            "prioritized_if_support_below_threshold_n": n_p,
            "not_prioritized_n": n_np,
            "coverage_prioritized_among_labeled": round(safe_div(n_p, total), 6),

            "partial_or_incorrect_in_prioritized_n": err_p,
            "partial_or_incorrect_rate_in_prioritized": round(safe_div(err_p, n_p), 6),
            "partial_or_incorrect_rate_in_not_prioritized": round(safe_div(err_np, n_np), 6),
            "partial_or_incorrect_recall_in_prioritized": round(safe_div(err_p, total_errors), 6),

            "incorrect_only_in_prioritized_n": incorrect_p,
            "incorrect_only_rate_in_prioritized": round(safe_div(incorrect_p, n_p), 6),
            "incorrect_only_rate_in_not_prioritized": round(safe_div(incorrect_np, n_np), 6),
        })

    return out

def confusion_table(rows: List[Dict[str, Any]], row_col: str, col_col: str) -> List[Dict[str, Any]]:
    row_vals = sorted(set(clean(r.get(row_col, "")) for r in rows))
    col_vals = sorted(set(clean(r.get(col_col, "")) for r in rows))
    out = []
    for rv in row_vals:
        subset = [r for r in rows if clean(r.get(row_col, "")) == rv]
        line = {row_col: rv, "All": len(subset)}
        for cv in col_vals:
            line[cv] = sum(1 for r in subset if clean(r.get(col_col, "")) == cv)
        out.append(line)
    return out


def get_join_key(row: Dict[str, str], join_col: str) -> str:
    """
    Return join key.

    Grounding CSV already has criterion_id.
    Manual CSV may not have criterion_id, so build it from item_uid + clause_id.
    """
    if join_col in row and clean(row.get(join_col, "")):
        return clean(row.get(join_col, ""))

    item_uid = clean(row.get("item_uid", ""))
    clause_id = clean(row.get("clause_id", ""))

    if item_uid and clause_id:
        return f"{item_uid}_{clause_id}"

    return ""


def make_index(rows: List[Dict[str, str]], join_col: str) -> Dict[str, Dict[str, str]]:
    index = {}

    for r in rows:
        key = get_join_key(r, join_col)
        if key:
            index[key] = r

    return index


def merge_grounding_with_manual(
    grounding_rows: List[Dict[str, str]],
    manual_rows: List[Dict[str, str]],
    manual_label_col: str,
    join_col: str,
) -> List[Dict[str, Any]]:
    manual_index = make_index(manual_rows, join_col)
    matched: List[Dict[str, Any]] = []

    for gr in grounding_rows:
        key = clean(gr.get(join_col, ""))
        if not key:
            continue
        mr = manual_index.get(key)
        if mr is None:
            continue

        manual_label = normalize_manual_label(mr.get(manual_label_col, ""))
        if manual_label not in {"correct", "partial", "incorrect"}:
            continue

        row = dict(gr)
        for k, v in mr.items():
            if k not in row:
                row[k] = v
            else:
                row[f"manual_file__{k}"] = v

        row["manual_B_leaf_label"] = mr.get(manual_label_col, "")
        row["manual_label_normalized"] = manual_label
        row["manual_is_error_partial_or_incorrect"] = 1 if manual_label in {"partial", "incorrect"} else 0
        row["manual_is_incorrect"] = 1 if manual_label == "incorrect" else 0
        matched.append(row)

    return matched


def semantic_review_intermediate_or_priority(row: Dict[str, Any]) -> int:
    return 1 if norm_label(row.get("semantic_grounding_risk_label", "")) in {
        "reference_intermediate_support",
        "reference_review_priority",
    } else 0


def semantic_review_priority_only(row: Dict[str, Any]) -> int:
    return 1 if norm_label(row.get("semantic_grounding_risk_label", "")) == "reference_review_priority" else 0


def routing_semantic_review(row: Dict[str, Any]) -> int:
    d = norm_label(row.get("final_routing_decision", ""))
    return 1 if d in {"llm_verifier", "optional_llm_verifier_or_review"} else 0


def main() -> None:
    print("\nBranch B Layer 2 grounding screen validation against manual labels")
    print(f"Grounding input: {GROUNDING_CSV}")
    print(f"Manual input: {MANUAL_CSV}")
    print(f"Output dir: {OUT_DIR}")

    grounding_rows = read_csv(GROUNDING_CSV)
    manual_rows = read_csv(MANUAL_CSV)

    manual_label_col = "manual_B_leaf_label"

    if manual_label_col not in manual_rows[0]:
        raise RuntimeError(
            f"Expected {manual_label_col} in manual file, but it was not found."
        )

    # Grounding file uses criterion_id.
    # Manual file may not have criterion_id; if not, get_join_key()
    # will build the manual key from item_uid + "_" + clause_id.
    if "criterion_id" not in grounding_rows[0]:
        raise RuntimeError("Grounding CSV does not contain criterion_id.")

    if "criterion_id" not in manual_rows[0]:
        if not ("item_uid" in manual_rows[0] and "clause_id" in manual_rows[0]):
            raise RuntimeError(
                "Manual CSV needs either criterion_id or item_uid + clause_id."
            )

    join_col = "criterion_id"

    matched = merge_grounding_with_manual(
        grounding_rows=grounding_rows,
        manual_rows=manual_rows,
        manual_label_col=manual_label_col,
        join_col=join_col,
    )
    if not matched:
        raise RuntimeError("No matched manual rows found. Check criterion_id matching.")

    expected_labeled_manual_rows = sum(
        1
        for r in manual_rows
        if normalize_manual_label(r.get(manual_label_col, "")) in {"correct", "partial", "incorrect"}
    )

    expected_semantic_labels = {
        "reference_high_support_not_auto_accept",
        "reference_intermediate_support",
        "reference_review_priority",
    }

    observed_semantic_labels = {
        str(r.get("semantic_grounding_risk_label", "")) for r in matched
    }

    old_labels = observed_semantic_labels.intersection({"low", "medium", "high"})
    if old_labels:
        raise RuntimeError(
            f"Old low/medium/high labels found in Branch B validation: {sorted(old_labels)}"
        )

    unknown_labels = observed_semantic_labels - expected_semantic_labels
    if unknown_labels:
        raise RuntimeError(
            f"Unexpected semantic grounding labels found: {sorted(unknown_labels)}"
        )

    if len(matched) != expected_labeled_manual_rows:
        raise RuntimeError(
            f"Matched {len(matched)} labeled Branch B rows, but manual file contains "
            f"{expected_labeled_manual_rows} valid Branch B labels. "
            "Check criterion_id / item_uid / clause_id alignment."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_MATCHED_CSV, matched)

    semantic_by_risk = grouped_rates(matched, "semantic_grounding_risk_label")
    semantic_by_routing = grouped_rates(matched, "final_routing_decision")
    execution_by_manual = grouped_rates(matched, "execution_risk_label")
    threshold_sensitivity = threshold_sensitivity_table(matched)

    write_csv(OUT_SEMANTIC_BY_RISK_CSV, semantic_by_risk)
    write_csv(OUT_SEMANTIC_BY_ROUTING_CSV, semantic_by_routing)
    write_csv(OUT_EXECUTION_BY_MANUAL_CSV, execution_by_manual)
    write_csv(OUT_THRESHOLD_SENSITIVITY_CSV, threshold_sensitivity)

    y_error = [int(r["manual_is_error_partial_or_incorrect"]) for r in matched]
    y_incorrect = [int(r["manual_is_incorrect"]) for r in matched]

    pred_intermediate_or_priority = [
        semantic_review_intermediate_or_priority(r) for r in matched
    ]
    pred_priority_only = [
        semantic_review_priority_only(r) for r in matched
    ]
    pred_routing_semantic = [routing_semantic_review(r) for r in matched]

    label_counts = Counter(r["manual_label_normalized"] for r in matched)

    summary = {
        "description": (
            "Validation of the Branch B Layer 2 grounding and routing screen "
            "against manual_B_leaf_label. Manual labels evaluate semantic "
            "correctness, while execution and computability support are "
            "reported separately."
        ),
        "inputs": {
            "grounding_screen_csv": str(GROUNDING_CSV),
            "manual_csv": str(MANUAL_CSV),
        },
        "outputs": {
            "matched_rows_csv": str(OUT_MATCHED_CSV),
            "semantic_error_by_risk_label_csv": str(OUT_SEMANTIC_BY_RISK_CSV),
            "semantic_error_by_routing_decision_csv": str(OUT_SEMANTIC_BY_ROUTING_CSV),
            "execution_risk_vs_manual_label_csv": str(OUT_EXECUTION_BY_MANUAL_CSV),
            "summary_json": str(OUT_SUMMARY_JSON),
            "semantic_grounding_threshold_sensitivity_csv": str(OUT_THRESHOLD_SENSITIVITY_CSV),
        },
        "detected_columns": {
            "manual_label_col": manual_label_col,
            "join_col": join_col,
        },
        "n_grounding_rows": len(grounding_rows),
        "n_manual_rows": len(manual_rows),
        "n_matched_labeled_rows": len(matched),
        "manual_label_counts": dict(label_counts),
        "manual_error_rate_partial_or_incorrect": round(safe_div(label_counts.get("partial", 0) + label_counts.get("incorrect", 0), len(matched)), 6),
        "manual_incorrect_only_rate": round(safe_div(label_counts.get("incorrect", 0), len(matched)), 6),
        "semantic_risk_counts_among_labeled": dict(Counter(r.get("semantic_grounding_risk_label", "") for r in matched)),
        "execution_risk_counts_among_labeled": dict(Counter(r.get("execution_risk_label", "") for r in matched)),
        "routing_counts_among_labeled": dict(Counter(r.get("final_routing_decision", "") for r in matched)),
        "metrics_detecting_partial_or_incorrect": {
            "semantic_intermediate_or_review_priority_as_positive": binary_metrics(
                y_error, pred_intermediate_or_priority
            ),
            "semantic_review_priority_only_as_positive": binary_metrics(
                y_error, pred_priority_only
            ),
            "routing_llm_or_optional_as_positive": binary_metrics(y_error, pred_routing_semantic),
        },
        "metrics_detecting_incorrect_only": {
            "semantic_intermediate_or_review_priority_as_positive": binary_metrics(
                y_incorrect, pred_intermediate_or_priority
            ),
            "semantic_review_priority_only_as_positive": binary_metrics(
                y_incorrect, pred_priority_only
            ),
            "routing_llm_or_optional_as_positive": binary_metrics(
                y_incorrect, pred_routing_semantic
            ),
        },
        "semantic_error_by_risk_label": semantic_by_risk,
        "semantic_error_by_routing_decision": semantic_by_routing,
        "execution_risk_vs_manual_label": execution_by_manual,
        "confusion_manual_by_semantic_risk": confusion_table(matched, "semantic_grounding_risk_label", "manual_label_normalized"),
        "confusion_manual_by_routing": confusion_table(matched, "final_routing_decision", "manual_label_normalized"),
        "method_note": (
            "A good screen should ideally have high recall for partial/incorrect leaves when using semantic intermediate/review-priority "
            "or semantic review routing as positive. Precision may be low because Branch B has few incorrect labels and the screen is conservative."
        ),
        "semantic_grounding_threshold_sensitivity": threshold_sensitivity,
    }

    write_json(OUT_SUMMARY_JSON, summary)

    print("DONE")
    print(f"Grounding rows: {len(grounding_rows)}")
    print(f"Manual rows: {len(manual_rows)}")
    print(f"Matched labeled rows: {len(matched)}")
    print(f"Manual label column: {manual_label_col}")
    print(f"Join column: {join_col}")
    print(f"Manual label counts: {dict(label_counts)}")
    print("\nThreshold sensitivity:")
    for row in threshold_sensitivity:
        print(row)

    print("\nSemantic risk counts among labeled:")
    print(summary["semantic_risk_counts_among_labeled"])

    print("\nRouting counts among labeled:")
    print(summary["routing_counts_among_labeled"])

    print("\nMetrics detecting partial/incorrect:")
    for name, metrics in summary["metrics_detecting_partial_or_incorrect"].items():
        print(f"  {name}:")
        print(
            "    precision=", metrics["precision_positive_predictive_value"],
            "recall=", metrics["recall_sensitivity"],
            "specificity=", metrics["specificity"],
            "npv=", metrics["negative_predictive_value"],
            "f1=", metrics["f1"],
        )

    print("\nError rate by semantic risk label:")
    for row in semantic_by_risk:
        print(row)

    print("\nError rate by routing decision:")
    for row in semantic_by_routing:
        print(row)

    print(f"\nWrote summary: {OUT_SUMMARY_JSON}")


if __name__ == "__main__":
    main()

# Run from the repository root:
# python scripts/03_verification/02_layer2/07_validate_branch_b_grounding.py