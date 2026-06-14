"""
12_assign_final_decisions.py

Assign the final post-rescue verification decision to every Branch A
and Branch B leaf.

Inputs:
    - post-rescue Branch A and Branch B Layer 1 policies
    - post-rescue Branch A Layer 2 risk scores
    - post-rescue Branch B Layer 2 grounding screen
    - candidate-selection rescue application audit

Outputs:
    outputs/verification/layer3/final_decisions/

The possible final decisions are:
    - accept
    - partial
    - non_computable
    - human_review

This script does not call the LLM, modify rule trees, perform repairs,
or evaluate against manual labels.

Run from the repository root:
python scripts/03_verification/03_layer3/12_assign_final_decisions.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------

POST_DIR = (
    ROOT
    / "outputs"
    / "verification"
    / "layer3"
    / "post_rescue_verification"
)

A_POLICY_CSV = (
    POST_DIR
    / "layer1"
    / "policy_branch_a"
    / "layer1_policy_branch_a_leaf_level.csv"
)

B_POLICY_CSV = (
    POST_DIR
    / "layer1"
    / "policy_branch_b"
    / "layer1_policy_branch_b_leaf_level.csv"
)

A_LAYER2_CSV = (
    POST_DIR
    / "layer2"
    / "branch_a"
    / "layer2_branch_a_leaf_risk_scores.csv"
)

B_LAYER2_CSV = (
    POST_DIR
    / "layer2"
    / "branch_b"
    / "layer2_branch_b_grounding_screen_leaf_level.csv"
)

RESCUE_APPLY_AUDIT_CSV = (
    ROOT
    / "outputs"
    / "verification"
    / "layer3"
    / "applied_candidate_selection_rescue"
    / "layer3_candidate_selection_apply_audit.csv"
)

# ---------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------

OUT_DIR = (
    ROOT
    / "outputs"
    / "verification"
    / "layer3"
    / "final_decisions"
)

OUT_ALL_CSV = (
    OUT_DIR / "layer3_final_decision_leaf_level.csv"
)
OUT_A_CSV = (
    OUT_DIR / "layer3_final_decision_branch_a.csv"
)
OUT_B_CSV = (
    OUT_DIR / "layer3_final_decision_branch_b.csv"
)
OUT_CROSS_CSV = (
    OUT_DIR / "layer3_final_decision_cross_branch_by_criterion.csv"
)
OUT_SUMMARY_JSON = (
    OUT_DIR / "layer3_final_decision_summary.json"
)

FINAL_DECISION_ORDER = {
    "accept": 0,
    "partial": 1,
    "non_computable": 2,
    "human_review": 3,
}


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------

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
        "branch",
        "document_id",
        "criterion_id",
        "item_uid",
        "clause_id",
        "final_decision",
        "final_decision_severity",
        "final_decision_reasons",
        "final_decision_rule",
        "entity_type",
        "entity_text",
        "operator",
        "value_type",
        "value",
        "unit",
        "computability",
        "non_computable_reason",
        "evidence_text",

        # Branch A signals
        "branch_a_risk_label",
        "branch_a_leaf_support",
        "branch_a_risk_score",
        "branch_a_risk_reasons",
        "branch_a_layer1_action_hint",
        "branch_a_layer1_bucket",
        "branch_a_layer1_severity",

        # Branch B signals
        "branch_b_semantic_grounding_risk_label",
        "branch_b_execution_risk_label",
        "branch_b_final_routing_decision",
        "branch_b_semantic_grounding_support",
        "branch_b_execution_support",
        "branch_b_routing_reasons",
        "branch_b_layer1_action_hint",
        "branch_b_layer1_bucket",
        "branch_b_layer1_severity",

        # Rescue audit
        "rescue_candidate_id",
        "rescue_final_decision",
        "rescue_apply_action",
        "rescue_apply_status",
        "rescue_changed_fields",
        "rescue_skip_reason",

        # Cross branch
        "chosen_branch",
        "chosen_branch_decision",
        "branch_a_final_decision",
        "branch_b_final_decision",
        "cross_branch_reason",
    ]

    cols = []
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


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def clean(x: Any) -> str:
    return str(x or "").strip()


def lower(x: Any) -> str:
    return clean(x).lower()


def first_non_empty(*values: Any) -> str:
    for v in values:
        s = clean(v)
        if s:
            return s
    return ""


def to_float(x: Any, default: float | None = None) -> float | None:
    try:
        s = clean(x)
        if not s:
            return default
        return float(s)
    except Exception:
        return default


def index_by_criterion(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    out = {}

    for r in rows:
        cid = clean(r.get("criterion_id"))
        if cid:
            out[cid] = r

    return out


def index_audit(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, str]]:
    out = {}

    for r in rows:
        branch = first_non_empty(r.get("branch_to_update"), r.get("branch"))
        cid = clean(r.get("criterion_id"))

        if branch and cid:
            out[(branch, cid)] = r

    return out


def add_reason(reasons: List[str], reason: str) -> None:
    reason = clean(reason)

    if reason and reason not in reasons:
        reasons.append(reason)


def severity(decision: str) -> int:
    return FINAL_DECISION_ORDER.get(decision, 99)

def is_applied_branch_b_substitution(audit: Dict[str, str] | None) -> bool:
    """
    True when Branch A was actually replaced by a Branch B-derived leaf.

    In that case, the A leaf should not be judged as a normal BERT/rules leaf
    anymore. Its final decision should follow the corresponding Branch B
    post-rescue decision.
    """
    if not audit:
        return False

    apply_status = lower(audit.get("apply_status"))
    selected_source = lower(audit.get("selected_source"))

    return (
        apply_status == "applied"
        and selected_source in {"b_current", "b_dejure_best"}
    )

# ---------------------------------------------------------------------
# Branch-specific final decision logic
# ---------------------------------------------------------------------

def rescue_forces_human_review(audit: Dict[str, str] | None, reasons: List[str]) -> bool:
    if not audit:
        return False

    rescue_decision = lower(audit.get("final_decision"))
    local_validation_status = lower(audit.get("local_validation_status"))
    apply_status = lower(audit.get("apply_status"))
    skip_reason = lower(audit.get("skip_reason"))

    if rescue_decision == "no_change":
        return False

    if skip_reason.startswith("not_applicable_final_decision:no_change"):
        return False

    if rescue_decision == "human_review":
        add_reason(reasons, "candidate_selection_rescue_returned_human_review")
        return True

    if rescue_decision == "no_repair_possible":
        add_reason(reasons, "candidate_selection_rescue_returned_no_repair_possible")
        return True

    if local_validation_status == "fail":
        add_reason(reasons, "candidate_selection_rescue_failed_local_validation")
        return True

    if apply_status == "skipped" and skip_reason:
        add_reason(reasons, f"candidate_selection_rescue_skipped:{skip_reason}")
        return True

    return False


def decide_branch_a(
    layer2: Dict[str, str],
    policy: Dict[str, str] | None,
    audit: Dict[str, str] | None,
    b_final_row: Dict[str, Any] | None = None,
) -> Tuple[str, List[str], str]:
    reasons: List[str] = []

    policy = policy or {}

    risk_label = lower(layer2.get("risk_label"))
    computability = lower(layer2.get("computability"))
    layer1_action = lower(policy.get("layer1_policy_action_hint"))
    layer1_bucket = lower(policy.get("layer1_policy_bucket"))

    if rescue_forces_human_review(audit, reasons):
        return "human_review", reasons, "A_rescue_unresolved"

    # --------------------------------------------------
    # Hybrid rule:
    # If Branch A was actually replaced by a Branch B-derived leaf,
    # do not judge it as a normal BERT/rules leaf anymore.
    # Use the corresponding Branch B post-rescue final decision.
    # --------------------------------------------------
    if is_applied_branch_b_substitution(audit):
        if b_final_row:
            b_decision = clean(b_final_row.get("final_decision"))

            if b_decision in {"accept", "partial", "non_computable", "human_review"}:
                add_reason(reasons, "branch_a_replaced_by_validated_branch_b_leaf")
                add_reason(reasons, f"using_branch_b_post_rescue_decision:{b_decision}")
                return b_decision, reasons, f"A_hybrid_from_B_{b_decision}"

        add_reason(reasons, "branch_a_replaced_by_branch_b_but_missing_branch_b_final_decision")
        return "human_review", reasons, "A_hybrid_from_B_missing_b_decision"

    if layer1_action == "mandatory_rescue_or_review_candidate":
        add_reason(reasons, "branch_a_layer1_mandatory_rescue_or_review_remains")
        return "human_review", reasons, "A_layer1_hard_issue"

    if risk_label in {"high", "review_priority"}:
        add_reason(reasons, "branch_a_layer2_high_risk_after_rescue")
        return "human_review", reasons, "A_layer2_high_risk"

    if computability == "non_computable":
        add_reason(reasons, "branch_a_non_computable")
        return "non_computable", reasons, "A_non_computable"

    if computability == "partial":
        add_reason(reasons, "branch_a_partial_computability")
        return "partial", reasons, "A_partial_computability"

    if layer1_action == "computability_review_candidate":
        add_reason(reasons, "branch_a_layer1_computability_review")
        return "partial", reasons, "A_computability_review"

    if risk_label in {"medium", "intermediate_support"}:
        add_reason(reasons, "branch_a_medium_risk_after_rescue")
        return "partial", reasons, "A_layer2_medium_risk"

    if risk_label in {"low", "high_support_not_auto_accept"} and computability in {"", "computable"}:
        add_reason(reasons, "branch_a_low_risk_no_hard_issue")
        return "accept", reasons, "A_low_risk_accept"

    if layer1_bucket == "safe_normalization_only":
        add_reason(reasons, "branch_a_safe_normalization_only")
        return "accept", reasons, "A_safe_normalization_accept"

    add_reason(reasons, "branch_a_uncertain_default_partial")
    return "partial", reasons, "A_default_partial"


def decide_branch_b(
    layer2: Dict[str, str],
    policy: Dict[str, str] | None,
    audit: Dict[str, str] | None,
) -> Tuple[str, List[str], str]:
    reasons: List[str] = []

    policy = policy or {}

    routing = lower(layer2.get("final_routing_decision"))
    semantic_risk = lower(layer2.get("semantic_grounding_risk_label"))
    execution_risk = lower(layer2.get("execution_risk_label"))
    computability = lower(layer2.get("computability"))
    layer1_action = lower(policy.get("layer1_policy_action_hint"))

    if rescue_forces_human_review(audit, reasons):
        return "human_review", reasons, "B_rescue_unresolved"

    if layer1_action == "mandatory_verifier_candidate":
        add_reason(reasons, "branch_b_layer1_mandatory_verifier_remains")
        return "human_review", reasons, "B_layer1_hard_issue"

    if routing == "llm_verifier":
        add_reason(reasons, "branch_b_final_routing_llm_verifier")
        return "human_review", reasons, "B_llm_verifier"

    if semantic_risk in {"high", "reference_review_priority"}:
        add_reason(reasons, "branch_b_semantic_grounding_high_risk")
        return "human_review", reasons, "B_semantic_high_risk"

    if computability == "non_computable":
        add_reason(reasons, "branch_b_non_computable")
        return "non_computable", reasons, "B_non_computable"

    if routing == "computability_review":
        add_reason(reasons, "branch_b_computability_review")
        return "partial", reasons, "B_computability_review"

    if computability == "partial":
        add_reason(reasons, "branch_b_partial_computability")
        return "partial", reasons, "B_partial_computability"

    if execution_risk in {"medium", "high", "reference_intermediate_support", "reference_review_priority"}:
        add_reason(reasons, f"branch_b_execution_{execution_risk}_risk")
        return "partial", reasons, "B_execution_risk"

    if routing == "optional_llm_verifier_or_review":
        add_reason(reasons, "branch_b_optional_verifier_or_review")
        return "partial", reasons, "B_optional_review_partial"

    if semantic_risk in {"medium", "reference_intermediate_support"}:
        add_reason(reasons, "branch_b_semantic_grounding_medium_risk")
        return "partial", reasons, "B_semantic_medium_risk"

    if routing in {"accept", "keep_without_layer2_action"} and semantic_risk in {"low", "reference_high_support_not_auto_accept"}:
        add_reason(reasons, "branch_b_accept_routing_low_semantic_risk")
        return "accept", reasons, "B_accept"

    add_reason(reasons, "branch_b_uncertain_default_partial")
    return "partial", reasons, "B_default_partial"


# ---------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------

def build_branch_a_row(
    layer2: Dict[str, str],
    policy: Dict[str, str] | None,
    audit: Dict[str, str] | None,
    b_final_row: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    final_decision, reasons, rule = decide_branch_a(layer2, policy, audit, b_final_row)

    policy = policy or {}
    audit = audit or {}

    return {
        "branch": "A",
        "document_id": first_non_empty(layer2.get("document_id"), policy.get("document_id"), audit.get("document_id")),
        "criterion_id": first_non_empty(layer2.get("criterion_id"), policy.get("criterion_id"), audit.get("criterion_id")),
        "item_uid": first_non_empty(layer2.get("item_uid"), policy.get("item_uid")),
        "clause_id": first_non_empty(layer2.get("clause_id"), policy.get("clause_id")),
        "final_decision": final_decision,
        "final_decision_severity": severity(final_decision),
        "final_decision_reasons": ";".join(reasons),
        "final_decision_rule": rule,

        "entity_type": layer2.get("entity_type"),
        "entity_text": layer2.get("entity_text"),
        "operator": layer2.get("operator"),
        "value_type": layer2.get("value_type"),
        "value": layer2.get("value_json") or layer2.get("value"),
        "unit": layer2.get("unit"),
        "computability": layer2.get("computability"),
        "non_computable_reason": layer2.get("non_computable_reason"),
        "evidence_text": layer2.get("evidence_text"),

        "branch_a_risk_label": layer2.get("risk_label"),
        "branch_a_leaf_support": layer2.get("leaf_support"),
        "branch_a_risk_score": layer2.get("risk_score"),
        "branch_a_risk_reasons": layer2.get("risk_reasons"),
        "branch_a_layer1_action_hint": policy.get("layer1_policy_action_hint"),
        "branch_a_layer1_bucket": policy.get("layer1_policy_bucket"),
        "branch_a_layer1_severity": policy.get("layer1_policy_severity"),

        "rescue_candidate_id": first_non_empty(audit.get("plan_id"), audit.get("candidate_id")),
        "rescue_final_decision": audit.get("final_decision"),
        "rescue_apply_action": audit.get("apply_action"),
        "rescue_apply_status": audit.get("apply_status"),
        "rescue_selected_source": audit.get("selected_source"),
        "rescue_local_validation_status": audit.get("local_validation_status"),
        "rescue_changed_fields": audit.get("changed_fields"),
        "rescue_skip_reason": audit.get("skip_reason"),
        "hybrid_b_reference_decision": clean(b_final_row.get("final_decision")) if b_final_row else "",
        "hybrid_b_reference_rule": clean(b_final_row.get("final_decision_rule")) if b_final_row else "",
    }


def build_branch_b_row(
    layer2: Dict[str, str],
    policy: Dict[str, str] | None,
    audit: Dict[str, str] | None,
) -> Dict[str, Any]:
    final_decision, reasons, rule = decide_branch_b(layer2, policy, audit)

    policy = policy or {}
    audit = audit or {}

    return {
        "branch": "B",
        "document_id": first_non_empty(layer2.get("document_id"), policy.get("document_id"), audit.get("document_id")),
        "criterion_id": first_non_empty(layer2.get("criterion_id"), policy.get("criterion_id"), audit.get("criterion_id")),
        "item_uid": first_non_empty(layer2.get("item_uid"), policy.get("item_uid")),
        "clause_id": first_non_empty(layer2.get("clause_id"), policy.get("clause_id")),
        "final_decision": final_decision,
        "final_decision_severity": severity(final_decision),
        "final_decision_reasons": ";".join(reasons),
        "final_decision_rule": rule,

        "entity_type": layer2.get("entity_type"),
        "entity_text": layer2.get("entity_text"),
        "operator": layer2.get("operator"),
        "value_type": layer2.get("value_type"),
        "value": layer2.get("value"),
        "unit": layer2.get("unit"),
        "computability": layer2.get("computability"),
        "non_computable_reason": layer2.get("non_computable_reason"),
        "evidence_text": layer2.get("evidence_text"),

        "branch_b_semantic_grounding_risk_label": layer2.get("semantic_grounding_risk_label"),
        "branch_b_execution_risk_label": layer2.get("execution_risk_label"),
        "branch_b_final_routing_decision": layer2.get("final_routing_decision"),
        "branch_b_semantic_grounding_support": layer2.get("semantic_grounding_support"),
        "branch_b_execution_support": layer2.get("execution_support"),
        "branch_b_routing_reasons": layer2.get("routing_reasons"),
        "branch_b_layer1_action_hint": policy.get("layer1_policy_action_hint"),
        "branch_b_layer1_bucket": policy.get("layer1_policy_bucket"),
        "branch_b_layer1_severity": policy.get("layer1_policy_severity"),

        "rescue_candidate_id": first_non_empty(
            audit.get("plan_id"),
            audit.get("candidate_id"),
        ),
        "rescue_final_decision": audit.get("final_decision"),
        "rescue_apply_action": audit.get("apply_action"),
        "rescue_apply_status": audit.get("apply_status"),
        "rescue_changed_fields": audit.get("changed_fields"),
        "rescue_skip_reason": audit.get("skip_reason"),
    }


def choose_cross_branch_row(a_row: Dict[str, Any] | None, b_row: Dict[str, Any] | None) -> Dict[str, Any]:
    cid = ""
    document_id = ""

    if a_row:
        cid = clean(a_row.get("criterion_id"))
        document_id = clean(a_row.get("document_id"))

    if b_row and not cid:
        cid = clean(b_row.get("criterion_id"))
        document_id = clean(b_row.get("document_id"))

    candidates = []

    if b_row:
        candidates.append(("B", b_row))

    if a_row:
        candidates.append(("A", a_row))

    if not candidates:
        return {
            "criterion_id": cid,
            "document_id": document_id,
            "chosen_branch": "",
            "chosen_branch_decision": "human_review",
            "branch_a_final_decision": "",
            "branch_b_final_decision": "",
            "cross_branch_reason": "no_branch_rows_found",
        }

    # Prefer accepted leaves. Then partial. Then non-computable. Human review last.
    # Within the same decision class, prefer Branch B because it is the semantic
# LLM Pass 2 branch. This is a design choice, not a manual-label rule.
    candidates = sorted(
        candidates,
        key=lambda x: (severity(clean(x[1].get("final_decision"))), 0 if x[0] == "B" else 1),
    )

    chosen_branch, chosen_row = candidates[0]
    chosen_decision = clean(chosen_row.get("final_decision"))

    return {
        "criterion_id": cid,
        "document_id": document_id,
        "chosen_branch": chosen_branch,
        "chosen_branch_decision": chosen_decision,
        "branch_a_final_decision": clean(a_row.get("final_decision")) if a_row else "",
        "branch_b_final_decision": clean(b_row.get("final_decision")) if b_row else "",
        "branch_a_rule": clean(a_row.get("final_decision_rule")) if a_row else "",
        "branch_b_rule": clean(b_row.get("final_decision_rule")) if b_row else "",
        "cross_branch_reason": f"selected_{chosen_branch}_because_{chosen_decision}_has_best_available_status",
    }


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def summarize(rows: List[Dict[str, Any]], cross_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_branch = defaultdict(list)

    for r in rows:
        by_branch[clean(r.get("branch"))].append(r)

    branch_summaries = {}

    for branch, rs in by_branch.items():
        branch_summaries[branch] = {
            "n": len(rs),
            "final_decision_counts": dict(Counter(clean(r.get("final_decision")) for r in rs).most_common()),
            "final_decision_rule_counts": dict(Counter(clean(r.get("final_decision_rule")) for r in rs).most_common()),
            "rescue_final_decision_counts": dict(Counter(clean(r.get("rescue_final_decision")) for r in rs if clean(r.get("rescue_final_decision"))).most_common()),
        }

    return {
        "stage": "12_assign_final_decisions",
        "description": (
            "Assigns final leaf-level verification decisions after deterministic "
            "repair, candidate-selection rescue, and post-rescue verification."
        ),
        "inputs": {
            "branch_a_layer1_policy_csv": str(A_POLICY_CSV),
            "branch_b_layer1_policy_csv": str(B_POLICY_CSV),
            "branch_a_layer2_score_csv": str(A_LAYER2_CSV),
            "branch_b_layer2_screen_csv": str(B_LAYER2_CSV),
            "rescue_apply_audit_csv": str(RESCUE_APPLY_AUDIT_CSV),
        },
        "outputs": {
            "all_leaf_final_decision_csv": str(OUT_ALL_CSV),
            "branch_a_final_decision_csv": str(OUT_A_CSV),
            "branch_b_final_decision_csv": str(OUT_B_CSV),
            "cross_branch_final_decision_csv": str(OUT_CROSS_CSV),
            "summary_json": str(OUT_SUMMARY_JSON),
        },
        "row_counts": {
            "all_leaf_rows": len(rows),
            "branch_a_rows": len(by_branch.get("A", [])),
            "branch_b_rows": len(by_branch.get("B", [])),
            "cross_branch_rows": len(cross_rows),
        },
        "branch_summaries": branch_summaries,
        "cross_branch_summary": {
            "chosen_branch_counts": dict(Counter(clean(r.get("chosen_branch")) for r in cross_rows).most_common()),
            "chosen_branch_decision_counts": dict(Counter(clean(r.get("chosen_branch_decision")) for r in cross_rows).most_common()),
            "branch_a_vs_branch_b_decision_pairs": dict(
                Counter(
                    f"A={clean(r.get('branch_a_final_decision'))}|B={clean(r.get('branch_b_final_decision'))}"
                    for r in cross_rows
                ).most_common()
            ),
        },
        "decision_rules": {
            "accept": [
                "Branch A: low Layer 2 risk with no hard Layer 1 issue and computable status.",
                "Branch B: accept routing with low semantic grounding risk.",
            ],
            "partial": [
                "Medium risk after rescue.",
                "Computability review.",
                "Partial computability.",
                "Optional verifier/review for Branch B.",
            ],
            "non_computable": [
                "Explicit non_computable status after post-rescue verification.",
            ],
            "human_review": [
                "Hard Layer 1 issue remains.",
                "High Layer 2 risk remains.",
                "Mandatory verifier/rescue remains.",
                "Targeted rescue returned human_review or no_repair_possible.",
                "Targeted rescue failed local validation.",
            ],
        },
        "method_notes": [
            "This script does not call the LLM.",
            "This script does not modify rule-tree files.",
            "This script does not evaluate against manual labels.",
            "Manual pre/post evaluation is performed separately.",
            "The cross-branch table is a comparison summary, not a third extraction branch.",
        ],
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLayer 3 final decisions")
    print("Branch A Layer 2:", A_LAYER2_CSV)
    print("Branch B Layer 2:", B_LAYER2_CSV)
    print("Branch A policy:", A_POLICY_CSV)
    print("Branch B policy:", B_POLICY_CSV)
    print("Rescue audit:", RESCUE_APPLY_AUDIT_CSV)

    a_layer2_rows = read_csv(A_LAYER2_CSV)
    b_layer2_rows = read_csv(B_LAYER2_CSV)
    a_policy_rows = read_csv(A_POLICY_CSV)
    b_policy_rows = read_csv(B_POLICY_CSV)
    audit_rows = read_csv(RESCUE_APPLY_AUDIT_CSV)

    a_policy_index = index_by_criterion(a_policy_rows)
    b_policy_index = index_by_criterion(b_policy_rows)
    audit_index = index_audit(audit_rows)

    a_final_rows = []
    b_final_rows = []

    # Build Branch B first because rescued Branch A leaves may inherit
    # the corresponding Branch B post-rescue decision.
    for r in b_layer2_rows:
        cid = clean(r.get("criterion_id"))
        audit = audit_index.get(("B", cid))
        policy = b_policy_index.get(cid)
        b_final_rows.append(build_branch_b_row(r, policy, audit))

    b_final_by_id = {
        clean(r.get("criterion_id")): r
        for r in b_final_rows
        if clean(r.get("criterion_id"))
    }

    for r in a_layer2_rows:
        cid = clean(r.get("criterion_id"))
        audit = audit_index.get(("A", cid))
        policy = a_policy_index.get(cid)
        b_final_row = b_final_by_id.get(cid)
        a_final_rows.append(build_branch_a_row(r, policy, audit, b_final_row))

    all_rows = a_final_rows + b_final_rows

    a_by_id = {clean(r.get("criterion_id")): r for r in a_final_rows if clean(r.get("criterion_id"))}
    b_by_id = {clean(r.get("criterion_id")): r for r in b_final_rows if clean(r.get("criterion_id"))}

    all_ids = sorted(set(a_by_id.keys()) | set(b_by_id.keys()))
    cross_rows = [
        choose_cross_branch_row(a_by_id.get(cid), b_by_id.get(cid))
        for cid in all_ids
    ]

    write_csv(OUT_A_CSV, a_final_rows)
    write_csv(OUT_B_CSV, b_final_rows)
    write_csv(OUT_ALL_CSV, all_rows)
    write_csv(OUT_CROSS_CSV, cross_rows)

    summary = summarize(all_rows, cross_rows)
    write_json(OUT_SUMMARY_JSON, summary)

    print("\nDONE")
    print("All leaf final decision CSV:", OUT_ALL_CSV)
    print("Branch A final decision CSV:", OUT_A_CSV)
    print("Branch B final decision CSV:", OUT_B_CSV)
    print("Cross-branch final decision CSV:", OUT_CROSS_CSV)
    print("Summary JSON:", OUT_SUMMARY_JSON)

    print("\nRow counts:")
    print(summary["row_counts"])

    print("\nBranch summaries:")
    for branch, s in summary["branch_summaries"].items():
        print(f"\n--- Branch {branch} ---")
        print("Final decision counts:", s["final_decision_counts"])
        print("Final rule counts:", s["final_decision_rule_counts"])

    print("\nCross-branch summary:")
    print("Chosen branch counts:", summary["cross_branch_summary"]["chosen_branch_counts"])
    print("Chosen decision counts:", summary["cross_branch_summary"]["chosen_branch_decision_counts"])


if __name__ == "__main__":
    main()

# Run from the repository root:
# python scripts/03_verification/03_layer3/12_assign_final_decisions.py