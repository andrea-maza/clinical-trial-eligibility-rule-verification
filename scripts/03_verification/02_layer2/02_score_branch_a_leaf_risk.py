"""
02_score_branch_a_leaf_risk.py

Compute the Layer 2 support and risk scores for Branch A leaves.

This script reads the Branch A support-signal inventory and calculates
a post-hoc support profile for each completed leaf.

The score is not a calibrated probability of correctness. It combines
PubMedBERT anchor support, operator--value support, quantitative
completeness, temporal and history context, condition or exception
context, computability, and deterministic Layer 1 signals.

Outputs:
    outputs/verification/layer2/branch_a/
        layer2_branch_a_leaf_risk_scores.csv
        layer2_branch_a_risk_summary.json

Run from the repository root:
python scripts/03_verification/02_layer2/02_score_branch_a_leaf_risk.py
"""

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple


SCORE_VERSION = "layer2_branch_a_v1_heuristic_support"

REVIEW_PRIORITY_THRESHOLD = 0.60
HIGH_SUPPORT_REFERENCE_THRESHOLD = 0.85

SUPPORT_COMPONENTS = [
    "entity_support",
    "operator_value_support",
    "quantitative_completeness_support",
    "temporal_support",
    "history_support",
    "context_support",
    "computability_support",
    "layer1_support",
]


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------

def read_csv(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    # Keep original inventory columns first, then scoring columns.
    scoring_cols = [
        "score_version",
        "entity_support",
        "operator_value_support",
        "quantitative_completeness_support",
        "temporal_support",
        "history_support",
        "context_support",
        "computability_support",
        "layer1_support",
        "leaf_support",
        "risk_score",
        "risk_label",
        "risk_reasons",
        "risk_bottleneck_components",
    ]

    keys: List[str] = []
    seen = set()

    for k in rows[0].keys():
        if k not in scoring_cols and k not in seen:
            keys.append(k)
            seen.add(k)

    for k in scoring_cols:
        if k in rows[0] and k not in seen:
            keys.append(k)
            seen.add(k)

    for row in rows:
        for k in row.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------

def norm(text: Any) -> str:
    return str(text or "").strip()


def lower(text: Any) -> str:
    return norm(text).lower()


def is_missing(x: Any) -> bool:
    s = lower(x)
    return s in {"", "na", "nan", "none", "null"}


def to_bool(row: Dict[str, Any], col: str, default: bool = False) -> bool:
    if col not in row:
        return default
    v = lower(row.get(col))
    if v in {"1", "1.0", "true", "yes", "y"}:
        return True
    if v in {"0", "0.0", "false", "no", "n", "", "nan", "none", "null"}:
        return False
    return default


def to_int(row: Dict[str, Any], col: str, default: int = 0) -> int:
    if col not in row:
        return default
    try:
        return int(float(str(row.get(col)).strip()))
    except Exception:
        return default


def to_float(row: Dict[str, Any], col: str, default: Optional[float] = None) -> Optional[float]:
    if col not in row:
        return default
    v = row.get(col)
    if is_missing(v):
        return default
    try:
        f = float(str(v).strip())
        if math.isnan(f):
            return default
        return f
    except Exception:
        return default


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def add_reason(reasons: List[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


# ---------------------------------------------------------------------
# Component support scores
# ---------------------------------------------------------------------

def compute_entity_support(row: Dict[str, Any], reasons: List[str]) -> float:
    """Compute support for the extracted entity field.

    Main evidence:
        - BERT anchor presence and score
        - entity/type agreement with BERT anchor
        - token overlap with any BERT anchor
        - normalized source-text grounding
        - generic entity flag
    """
    has_anchor = to_bool(row, "has_bert_anchor")
    entity_in_evidence = to_bool(row, "entity_text_ci_in_evidence")
    entity_in_item = to_bool(row, "entity_text_ci_in_item")
    type_match = to_bool(row, "entity_type_matches_best_anchor")
    token_overlap_any = to_bool(row, "entity_text_token_overlaps_any_anchor")
    generic_entity = to_bool(row, "generic_entity_text")

    if has_anchor:
        score = to_float(row, "best_anchor_score", default=0.75)
        if score is None:
            score = 0.75
        score = clamp01(score)

        if not type_match:
            score = min(score, 0.50)
            add_reason(reasons, "entity_type_not_supported_by_best_anchor")

        if not token_overlap_any:
            score = min(score, 0.50)
            add_reason(reasons, "entity_text_not_aligned_with_any_anchor")

        if not entity_in_evidence:
            if entity_in_item:
                score = min(score, 0.65)
                add_reason(reasons, "entity_not_in_leaf_evidence_but_in_item")
            else:
                score = min(score, 0.45)
                add_reason(reasons, "entity_not_grounded_in_evidence_or_item")

        if generic_entity:
            score = min(score, 0.40)
            add_reason(reasons, "generic_entity_text")

        return clamp01(score)

    # No BERT anchor. The entity may still be correct if literally grounded,
    # but support is weaker because Branch A is intended to be anchor-grounded.
    add_reason(reasons, "no_bert_anchor")

    if entity_in_evidence and not generic_entity:
        return 0.60
    if entity_in_item and not generic_entity:
        add_reason(reasons, "entity_not_in_leaf_evidence_but_in_item")
        return 0.50
    if entity_in_evidence and generic_entity:
        add_reason(reasons, "generic_entity_text")
        return 0.40

    add_reason(reasons, "entity_not_grounded_in_evidence_or_item")
    if generic_entity:
        add_reason(reasons, "generic_entity_text")
    return 0.25


def compute_operator_value_support(row: Dict[str, Any], reasons: List[str]) -> float:
    """Compute support for operator/value consistency.

    Existence operators do not require a value. For comparison/range/list
    operators, the value should be structurally present and preferably found
    in the evidence text.
    """
    category = lower(row.get("operator_category"))
    op_supported = to_bool(row, "operator_value_structurally_supported")
    value_in_evidence = to_bool(row, "value_text_found_in_evidence")
    operator = norm(row.get("operator"))

    if category == "existence":
        return 1.00

    if category in {"comparison", "range", "list"}:
        if op_supported and value_in_evidence:
            return 0.90
        if op_supported and not value_in_evidence:
            add_reason(reasons, "operator_value_supported_but_value_not_found_in_evidence")
            return 0.70

        add_reason(reasons, "operator_value_not_structurally_supported")
        return 0.30

    add_reason(reasons, f"unknown_or_other_operator_category:{operator or category}")
    return 0.50


def compute_temporal_support(row: Dict[str, Any], reasons: List[str]) -> float:
    temporal_marker = to_bool(row, "temporal_marker_in_evidence")
    temporal_present = to_bool(row, "temporal_context_present")
    temporal_missing = to_bool(row, "temporal_marker_missing_context")

    if not temporal_marker:
        return 1.00

    if temporal_marker and temporal_present and not temporal_missing:
        # If the marker is present and the context exists, this is acceptable.
        # It is still slightly below 1 because the temporal extractor is rule-based.
        return 0.90

    add_reason(reasons, "temporal_marker_without_temporal_context")
    return 0.40


def compute_history_support(row: Dict[str, Any], reasons: List[str]) -> float:
    history_marker = to_bool(row, "history_marker_in_evidence")
    history_present = to_bool(row, "history_context_present")
    history_missing = to_bool(row, "history_marker_missing_context")

    if not history_marker:
        return 1.00

    if history_marker and history_present and not history_missing:
        return 0.90

    add_reason(reasons, "history_marker_without_history_context")
    return 0.50


def compute_context_support(row: Dict[str, Any], reasons: List[str]) -> float:
    condition_or_exception_missing = to_bool(row, "condition_or_exception_missing_context")
    computable_with_unhandled_context = to_bool(row, "computable_with_unhandled_condition_or_exception")
    condition_marker = to_bool(row, "condition_marker_in_evidence")
    exception_marker = to_bool(row, "exception_marker_in_evidence")

    if computable_with_unhandled_context:
        add_reason(reasons, "computable_with_unhandled_condition_or_exception")
        return 0.30

    if condition_or_exception_missing:
        add_reason(reasons, "condition_or_exception_marker_without_context")
        return 0.40

    # If the context is present, the leaf is better supported, but condition/
    # exception clauses are still slightly more complex than simple clauses.
    if condition_marker or exception_marker:
        return 0.85

    return 1.00


def compute_computability_support(row: Dict[str, Any], reasons: List[str]) -> float:
    computability = lower(row.get("computability"))
    non_comp_reason = norm(row.get("non_computable_reason"))

    if computability == "computable":
        return 1.00

    if computability == "partial":
        add_reason(reasons, "computability_partial")
        return 0.70

    if computability == "non_computable":
        add_reason(reasons, "computability_non_computable")
        if non_comp_reason:
            return 0.80
        add_reason(reasons, "non_computable_without_reason")
        return 0.60

    add_reason(reasons, "unknown_computability")
    return 0.50


def compute_layer1_support(row: Dict[str, Any], reasons: List[str]) -> float:
    """
    Compute Layer 1 support using the Branch A Layer 1 policy.

    Layer 1 does not modify the logical rule tree.
    It provides deterministic flags and Branch-A-specific policy hints.
    Layer 2 uses these as routing/risk support signals.
    """
    issue_count = to_int(row, "layer1_issue_count", default=0)

    has_inventory_issue = to_bool(row, "has_layer1_inventory_issue")
    has_policy_issue = to_bool(row, "has_layer1_policy_issue")
    has_layer1d_issue = to_bool(row, "has_layer1d_issue")

    policy_action = lower(row.get("layer1_policy_action_hint"))

    support = 1.00

    # Branch A policy has priority because it already interprets
    # Layer 1A/1B/1C/1D for the Branch A extraction mechanism.
    if policy_action == "mandatory_rescue_or_review_candidate":
        support = min(support, 0.45)
        add_reason(reasons, "layer1_policy_mandatory_rescue_or_review")

    elif policy_action == "computability_review_candidate":
        support = min(support, 0.70)
        add_reason(reasons, "layer1_policy_computability_review")

    elif policy_action == "safe_normalization_candidate":
        support = min(support, 0.90)
        add_reason(reasons, "layer1_policy_safe_normalization_candidate")

    elif has_policy_issue and policy_action not in {"", "continue_to_layer2"}:
        support = min(support, 0.80)
        add_reason(reasons, "layer1_policy_issue_present")

    # Layer 1D is a direct Pass1/Pass2 consistency signal, so keep it explicit.
    if has_layer1d_issue:
        support = min(support, 0.55)
        add_reason(reasons, "layer1d_pass1_pass2_logic_flag")

    # Common Layer 1 inventory flags are still useful, but less specific than
    # the Branch A policy.
    if has_inventory_issue:
        support = min(support, 0.65)
        add_reason(reasons, "layer1_inventory_flag")

    # Fallback in case issue codes exist but source flags were not set.
    if issue_count > 0 and support == 1.00 and (
        has_inventory_issue or has_layer1d_issue or
        (has_policy_issue and policy_action not in {"", "continue_to_layer2"})
    ):
        support = 0.80
        add_reason(reasons, "layer1_issue_present")

    return support

def compute_quantitative_completeness_support(
    row: Dict[str, Any],
    reasons: List[str],
) -> float:
    """
    Penalize leaves where the source text contains a quantitative cue
    but Branch A did not represent it as an operator/value structure.
    """
    if to_bool(row, "quantitative_cue_unhandled"):
        add_reason(reasons, "quantitative_cue_not_represented")
        return 0.35

    if to_bool(row, "exists_with_quantitative_cue"):
        add_reason(reasons, "exists_operator_with_quantitative_threshold")
        return 0.35

    if to_bool(row, "value_missing_with_quantitative_cue"):
        add_reason(reasons, "value_missing_despite_quantitative_cue")
        return 0.40

    return 1.00

# ---------------------------------------------------------------------
# Leaf-level scoring
# ---------------------------------------------------------------------

def assign_risk_label(leaf_support: float) -> str:
    if leaf_support < REVIEW_PRIORITY_THRESHOLD:
        return "review_priority"
    if leaf_support >= HIGH_SUPPORT_REFERENCE_THRESHOLD:
        return "high_support_not_auto_accept"
    return "intermediate_support"


def compute_leaf_score(row: Dict[str, Any]) -> Dict[str, Any]:
    reasons: List[str] = []

    entity_support = compute_entity_support(row, reasons)
    operator_value_support = compute_operator_value_support(row, reasons)
    temporal_support = compute_temporal_support(row, reasons)
    history_support = compute_history_support(row, reasons)
    context_support = compute_context_support(row, reasons)
    computability_support = compute_computability_support(row, reasons)
    quantitative_completeness_support = compute_quantitative_completeness_support(row, reasons)
    layer1_support = compute_layer1_support(row, reasons)

    component_values = {
        "entity_support": entity_support,
        "operator_value_support": operator_value_support,
        "quantitative_completeness_support": quantitative_completeness_support,
        "temporal_support": temporal_support,
        "history_support": history_support,
        "context_support": context_support,
        "computability_support": computability_support,
        "layer1_support": layer1_support,
    }

    leaf_support = min(
        entity_support,
        operator_value_support,
        quantitative_completeness_support,
        temporal_support,
        history_support,
        context_support,
        computability_support,
        layer1_support,
    )
    risk_score = 1.0 - leaf_support
    risk_label = assign_risk_label(leaf_support)

    # Identify bottleneck components. These are useful for examples and audit.
    min_value = leaf_support
    bottlenecks = [
        name for name, value in component_values.items()
        if abs(value - min_value) < 1e-12
    ]

    out = dict(row)
    out.update(
        {
            "score_version": SCORE_VERSION,
            "entity_support": round(entity_support, 6),
            "operator_value_support": round(operator_value_support, 6),
            "quantitative_completeness_support": round(quantitative_completeness_support, 6),
            "temporal_support": round(temporal_support, 6),
            "history_support": round(history_support, 6),
            "context_support": round(context_support, 6),
            "computability_support": round(computability_support, 6),
            "layer1_support": round(layer1_support, 6),
            "leaf_support": round(leaf_support, 6),
            "risk_score": round(risk_score, 6),
            "risk_label": risk_label,
            "risk_reasons": "|".join(reasons),
            "risk_bottleneck_components": "|".join(bottlenecks),
        }
    )

    return out


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def numeric_values(rows: List[Dict[str, Any]], col: str) -> List[float]:
    vals: List[float] = []
    for row in rows:
        value = to_float(row, col, default=None)
        if value is not None:
            vals.append(value)
    return vals


def summarize_numeric(rows: List[Dict[str, Any]], col: str) -> Dict[str, Optional[float]]:
    vals = numeric_values(rows, col)
    if not vals:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": round(mean(vals), 6),
        "median": round(median(vals), 6),
        "min": round(min(vals), 6),
        "max": round(max(vals), 6),
    }


def count_by(rows: List[Dict[str, Any]], col: str) -> Dict[str, int]:
    return dict(Counter(norm(row.get(col)) for row in rows))


def reason_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counter: Counter = Counter()
    for row in rows:
        for reason in norm(row.get("risk_reasons")).split("|"):
            if reason:
                counter[reason] += 1
    return dict(counter.most_common())


def bottleneck_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counter: Counter = Counter()
    for row in rows:
        for reason in norm(row.get("risk_bottleneck_components")).split("|"):
            if reason:
                counter[reason] += 1
    return dict(counter.most_common())


def cross_tab(rows: List[Dict[str, Any]], row_col: str, col_col: str) -> Dict[str, Dict[str, int]]:
    table: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        r = norm(row.get(row_col))
        c = norm(row.get(col_col))
        table[r][c] += 1
    return {k: dict(v) for k, v in table.items()}


def make_summary(rows: List[Dict[str, Any]], input_path: Path, out_csv: Path) -> Dict[str, Any]:
    total = len(rows)
    risk_counts = Counter(row.get("risk_label") for row in rows)

    component_stats = {
        component: summarize_numeric(rows, component)
        for component in SUPPORT_COMPONENTS
    }

    return {
        "description": (
            "Layer 2 Branch A support scoring from support-signal inventory. "
            "The labels are routing/prioritization labels, not calibrated correctness classes."
        ),
        "score_version": SCORE_VERSION,
        "input_inventory_csv": str(input_path),
        "output_risk_csv": str(out_csv),

        "thresholds": {
            "review_priority_if_leaf_support_below": REVIEW_PRIORITY_THRESHOLD,
            "high_support_reference_if_leaf_support_at_least": HIGH_SUPPORT_REFERENCE_THRESHOLD,
            "intermediate_support_if_leaf_support_between": [
                REVIEW_PRIORITY_THRESHOLD,
                HIGH_SUPPORT_REFERENCE_THRESHOLD,
            ],
        },

        "threshold_source": {
            "type": "predefined_heuristic_review_prioritization_cutoffs",
            "review_priority_threshold": REVIEW_PRIORITY_THRESHOLD,
            "high_support_reference_threshold": HIGH_SUPPORT_REFERENCE_THRESHOLD,
            "not_calibrated_probability": True,
            "not_learned_from_manual_labels": True,
            "interpretation": (
                "leaf_support below 0.60 is prioritized for review/rescue. "
                "leaf_support at or above 0.85 is only a high-support reference group, "
                "not automatic acceptance."
            ),
            "required_validation": (
                "Validate against manual semantic labels using error enrichment, "
                "AUROC/AUPRC, and threshold sensitivity."
            ),
        },

        "method_note": (
            "The score is a post-hoc support score, not a calibrated probability. "
            "It uses a conservative minimum over support components. "
            "The thresholds are heuristic routing cutoffs and are evaluated against "
            "manual semantic labels in the validation script."
        ),

        "total_leaves": total,
        "risk_label_counts": dict(risk_counts),
        "risk_label_rates": {
            str(label): round(count / total, 6) if total else None
            for label, count in risk_counts.items()
        },

        "leaf_support_summary": summarize_numeric(rows, "leaf_support"),
        "risk_score_summary": summarize_numeric(rows, "risk_score"),
        "component_support_summary": component_stats,
        "risk_reason_counts": reason_counts(rows),
        "risk_bottleneck_component_counts": bottleneck_counts(rows),
        "risk_by_operator_category": cross_tab(rows, "operator_category", "risk_label"),
        "risk_by_entity_type": cross_tab(rows, "entity_type", "risk_label"),
        "risk_by_computability": cross_tab(rows, "computability", "risk_label"),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    ROOT = Path(__file__).resolve().parents[3]

    in_csv = (
        ROOT
        / "outputs"
        / "verification"
        / "layer2"
        / "branch_a"
        / "layer2_branch_a_support_inventory_leaf_level.csv"
    )

    out_dir = (
        ROOT
        / "outputs"
        / "verification"
        / "layer2"
        / "branch_a"
    )

    out_csv = out_dir / "layer2_branch_a_leaf_risk_scores.csv"
    out_json = out_dir / "layer2_branch_a_risk_summary.json"

    if not in_csv.exists():
        raise FileNotFoundError(
            f"Layer 2 Branch A inventory CSV not found: {in_csv}"
        )

    print("Layer 2 Branch A leaf-risk scoring")
    print("Input inventory:", in_csv)
    print("Output CSV:", out_csv)
    print("Output JSON:", out_json)

    inventory_rows = read_csv(in_csv)
    scored_rows = [compute_leaf_score(row) for row in inventory_rows]
    summary = make_summary(scored_rows, in_csv, out_csv)

    write_csv(out_csv, scored_rows)
    write_json(out_json, summary)

    print("DONE")
    print("Leaves scored:", summary["total_leaves"])
    print("Risk label counts:", summary["risk_label_counts"])
    print("Risk label rates:", summary["risk_label_rates"])
    print("Mean leaf support:", summary["leaf_support_summary"].get("mean"))
    print("Median leaf support:", summary["leaf_support_summary"].get("median"))
    print("Top risk reasons:")
    for reason, count in list(summary["risk_reason_counts"].items())[:10]:
        print("  ", reason, count)

if __name__ == "__main__":
    main()

# Run from the repository root:
# python scripts/03_verification/02_layer2/02_score_branch_a_leaf_risk.py