"""
01_build_decision_inventory.py

Build the Layer 3 decision inventory for Branch A and Branch B.

This script combines Layer 1 policy outputs and Layer 2 support signals
to assign the next action for every extracted leaf.

Possible actions include:
    - safe deterministic repair
    - conservative recovery candidate
    - targeted LLM rescue candidate
    - optional rescue or review
    - computability review
    - keep without Layer 2 action
    - diagnostic-only routing
    - human review or abstention

This script does not modify the logical rule trees, apply repairs, call
the LLM, or change Layer 1 or Layer 2 scores.

Inputs:
    outputs/verification/layer1/
    outputs/verification/layer2/

Outputs:
    outputs/verification/layer3/decision_inventory/
        layer3_decision_inventory_leaf_level.csv
        layer3_decision_summary.json

Run from the repository root:
python scripts/03_verification/03_layer3/01_build_decision_inventory.py
"""


from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------

LAYER1_INVENTORY_CSV = (
    ROOT
    / "outputs"
    / "verification"
    / "layer1"
    / "deterministic_inventory"
    / "deterministic_verification_inventory_leaf_level.csv"
)

BRANCH_A_POLICY_CSV = (
    ROOT
    / "outputs"
    / "verification"
    / "layer1"
    / "policy_branch_a"
    / "layer1_policy_branch_a_leaf_level.csv"
)

BRANCH_A_LAYER2_CSV = (
    ROOT
    / "outputs"
    / "verification"
    / "layer2"
    / "branch_a"
    / "layer2_branch_a_leaf_risk_scores.csv"
)

BRANCH_B_POLICY_CSV = (
    ROOT
    / "outputs"
    / "verification"
    / "layer1"
    / "policy_branch_b"
    / "layer1_policy_branch_b_leaf_level.csv"
)

BRANCH_B_LAYER2_CSV = (
    ROOT
    / "outputs"
    / "verification"
    / "layer2"
    / "branch_b"
    / "layer2_branch_b_grounding_screen_leaf_level.csv"
)

BRANCH_A_VALIDATION_SUMMARY_JSON = (
    ROOT
    / "outputs"
    / "verification"
    / "layer2"
    / "branch_a"
    / "layer2_branch_a_manual_validation_summary.json"
)

# ---------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------

OUT_DIR = (
    ROOT
    / "outputs"
    / "verification"
    / "layer3"
    / "decision_inventory"
)

OUT_CSV = OUT_DIR / "layer3_decision_inventory_leaf_level.csv"
OUT_JSON = OUT_DIR / "layer3_decision_summary.json"


# ---------------------------------------------------------------------
# Basic IO
# ---------------------------------------------------------------------

def read_csv(path: Path, required: bool = True) -> List[Dict[str, str]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required CSV not found: {path}")
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path, required: bool = False) -> Dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required JSON not found: {path}")
        return {}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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

    priority_cols = [
        "branch",
        "document_id",
        "criterion_id",
        "item_uid",
        "clause_id",
        "path",
        "tree_path",
        "entity_type",
        "entity_text",
        "operator",
        "value_type",
        "value",
        "unit",
        "computability",
        "evidence_text",

        # Layer 1 action hints
        "layer1a_action_category",
        "layer1a_action_hint",
        "layer1_policy_action_hint",
        "layer1_policy_bucket",
        "layer1_policy_severity",
        "layer1_policy_reasons",
        "all_layer1_codes",

        # Branch A Layer 2
        "branch_a_leaf_support",
        "branch_a_risk_label",
        "branch_a_risk_reasons",

        # Branch B Layer 2
        "branch_b_semantic_grounding_support",
        "branch_b_semantic_grounding_risk_label",
        "branch_b_semantic_grounding_reasons",
        "branch_b_execution_support",
        "branch_b_execution_risk_label",
        "branch_b_execution_reasons",
        "branch_b_final_routing_decision",
        "branch_b_routing_reasons",

        # Layer 3 decision
        "layer3_action_family",
        "layer3_primary_action",
        "layer3_priority",
        "layer3_changes_ast",
        "layer3_uses_llm",
        "layer3_requires_reverification",
        "layer3_reason",
        "suggested_rescue_task",
        "suggested_fallback",
    ]

    cols: List[str] = []
    seen = set()

    for c in priority_cols:
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


def to_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    s = clean(x)
    if not s:
        return default
    try:
        return float(s)
    except Exception:
        return default


def to_int(x: Any, default: int = 0) -> int:
    s = clean(x)
    if not s:
        return default
    try:
        return int(float(s))
    except Exception:
        return default


def to_bool_int(x: Any) -> int:
    if isinstance(x, bool):
        return 1 if x else 0
    s = lower(x)
    if s in {"1", "1.0", "true", "yes", "y", "t"}:
        return 1
    return 0


def split_codes(x: Any) -> List[str]:
    if x is None:
        return []

    if isinstance(x, list):
        return [clean(v) for v in x if clean(v)]

    s = clean(x)
    if not s:
        return []

    # JSON list support.
    if s.startswith("[") and s.endswith("]"):
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [clean(v) for v in obj if clean(v)]
        except Exception:
            pass

    # Support semicolon and pipe-separated issue lists.
    return [p.strip() for p in re.split(r"[;|]", s) if p.strip()]


def criterion_id_from_row(row: Dict[str, Any]) -> str:
    cid = clean(row.get("criterion_id"))
    if cid:
        return cid

    item_uid = clean(row.get("item_uid"))
    clause_id = clean(row.get("clause_id"))
    if item_uid and clause_id:
        return f"{item_uid}_{clause_id}"

    return ""


def branch_raw_name(branch: str) -> str:
    if branch == "A":
        return "A_raw"
    if branch == "B":
        return "B_raw"
    return branch


def make_key(branch: str, row: Dict[str, Any]) -> Tuple[str, str]:
    return (branch, criterion_id_from_row(row))


def index_by_branch_and_criterion(
    rows: List[Dict[str, str]],
    branch_name: str,
    accepted_branch_values: Optional[List[str]] = None,
) -> Dict[Tuple[str, str], Dict[str, str]]:
    """
    Index rows by (branch_name, criterion_id).

    accepted_branch_values is used to filter files that contain A_raw/B_raw.
    """
    out: Dict[Tuple[str, str], Dict[str, str]] = {}

    accepted = set(accepted_branch_values or [])

    for r in rows:
        if accepted:
            row_branch = clean(r.get("branch"))
            if row_branch and row_branch not in accepted:
                continue

        cid = criterion_id_from_row(r)
        if not cid:
            continue

        out[(branch_name, cid)] = r

    return out


def pick_first(row: Dict[str, Any], keys: List[str], default: str = "") -> str:
    for k in keys:
        v = row.get(k, "")
        if clean(v):
            return clean(v)
    return default


def collect_layer1_codes(policy_row: Dict[str, Any], det_row: Dict[str, Any]) -> List[str]:
    codes: List[str] = []

    for col in [
        "all_layer1_codes",
        "layer1a_issues",
        "layer1b_flags",
        "layer1c_warnings",
        "layer1d_issues",
    ]:
        codes.extend(split_codes(policy_row.get(col, "")))

    for col in [
        "deterministic_issues",
        "layer1c_source_text_warnings",
    ]:
        codes.extend(split_codes(det_row.get(col, "")))

    return sorted(set(codes))


def infer_rescue_task(
    issue_codes: List[str],
    risk_reasons: List[str],
    policy_action: str,
    branch: str,
) -> str:
    """
    Convert Layer 1/2 diagnosis into a targeted rescue task.

    This does not call the LLM. It only tags what the next rescue script
    should ask the LLM to repair.
    """
    joined = " ".join(issue_codes + risk_reasons + [policy_action]).lower()

    if any(x in joined for x in [
        "entity_not_grounded",
        "entity_not_in",
        "entity_text_empty",
        "generic_entity_text",
        "entity_supported_by_source_token_overlap",
        "entity_partially_supported",
    ]):
        return "entity_regrounding"

    if any(x in joined for x in [
        "comparison_without_scalar_value",
        "equality_without_value",
        "pattern_operator_without_scalar_value",
        "between_without_range_value",
        "range_with_both_bounds_missing",
        "range_with_missing_bound",
        "list_operator_without_list_value",
        "operator_value_not_structurally_supported",
        "value_not_grounded",
        "value_missing_with_quantitative_cue",
        "quantitative_cue_not_represented",
        "quantitative_cue_unhandled",
    ]):
        return "value_or_operator_recovery"

    if any(x in joined for x in [
        "temporal_marker_without_temporal_context",
        "temporal_context_missing",
        "temporal_anchor_mismatch",
        "duration_marker_missing",
    ]):
        return "temporal_context_recovery"

    if any(x in joined for x in [
        "history_marker_without_history_context",
        "history_context_invalid",
    ]):
        return "history_context_recovery"

    if any(x in joined for x in [
        "condition_context_present",
        "condition_or_exception",
        "exception_context",
        "exception_clause",
        "computable_with_exception_context",
        "condition_exception",
    ]):
        return "condition_exception_context_recovery"

    if any(x in joined for x in [
        "negation_clause_with_exists_operator",
        "positive_clause_with_not_exists_operator",
        "negative_entity_with_not_exists_operator",
        "allowance_or_polarity",
        "not_exists",
        "negation",
    ]):
        return "polarity_negation_repair"

    if any(x in joined for x in [
        "entity_type_invalid",
        "entity_type_not_supported",
        "entity_type",
    ]):
        return "entity_type_reclassification"

    if branch == "B":
        return "structured_llm_judge_branch_b"

    return "structured_llm_rescue_branch_a"


def base_leaf_fields(
    branch: str,
    layer2_row: Dict[str, Any],
    policy_row: Dict[str, Any],
    det_row: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build shared output fields.
    """
    cid = criterion_id_from_row(layer2_row) or criterion_id_from_row(policy_row) or criterion_id_from_row(det_row)

    return {
        "branch": branch,
        "document_id": pick_first(layer2_row, ["document_id"], pick_first(policy_row, ["document_id"], pick_first(det_row, ["document_id"]))),
        "criterion_id": cid,
        "item_uid": pick_first(layer2_row, ["item_uid"], pick_first(policy_row, ["item_uid"], pick_first(det_row, ["item_uid"]))),
        "clause_id": pick_first(layer2_row, ["clause_id"], pick_first(policy_row, ["clause_id"], pick_first(det_row, ["clause_id"]))),
        "path": pick_first(layer2_row, ["path", "leaf_path"], pick_first(policy_row, ["path"], pick_first(det_row, ["path"]))),
        "tree_path": pick_first(layer2_row, ["tree_path", "leaf_path"], pick_first(policy_row, ["path"], pick_first(det_row, ["path"]))),

        "entity_type": pick_first(layer2_row, ["entity_type"], pick_first(policy_row, ["entity_type"], pick_first(det_row, ["entity_type"]))),
        "entity_text": pick_first(layer2_row, ["entity_text"], pick_first(policy_row, ["entity_text"], pick_first(det_row, ["entity_text"]))),
        "operator": pick_first(layer2_row, ["operator"], pick_first(policy_row, ["operator"], pick_first(det_row, ["operator"]))),
        "value_type": pick_first(layer2_row, ["value_type"], pick_first(policy_row, ["value_type"], pick_first(det_row, ["value_type"]))),
        "value": pick_first(layer2_row, ["value", "value_json", "value_text"], pick_first(policy_row, ["value"], pick_first(det_row, ["value"]))),
        "unit": pick_first(layer2_row, ["unit"], pick_first(policy_row, ["unit"], pick_first(det_row, ["unit"]))),
        "computability": pick_first(layer2_row, ["computability"], pick_first(policy_row, ["computability"], pick_first(det_row, ["computability"]))),
        "evidence_text": pick_first(layer2_row, ["evidence_text"], pick_first(policy_row, ["evidence_text"], pick_first(det_row, ["evidence_text"]))),

        "layer1a_action_category": pick_first(det_row, ["layer1a_action_category", "repair_category"]),
        "layer1a_action_hint": pick_first(det_row, ["layer1a_action_hint", "proposed_repair"]),

        "layer1_policy_action_hint": pick_first(policy_row, ["layer1_policy_action_hint"]),
        "layer1_policy_bucket": pick_first(policy_row, ["layer1_policy_bucket"]),
        "layer1_policy_severity": pick_first(policy_row, ["layer1_policy_severity"]),
        "layer1_policy_reasons": pick_first(policy_row, ["layer1_policy_reasons"]),
        "all_layer1_codes": pick_first(policy_row, ["all_layer1_codes"]),
    }


# ---------------------------------------------------------------------
# Branch-specific Layer 3 decisions
# ---------------------------------------------------------------------

def decide_branch_a(
    layer2_row: Dict[str, Any],
    policy_row: Dict[str, Any],
    det_row: Dict[str, Any],
) -> Dict[str, Any]:
    base = base_leaf_fields("A", layer2_row, policy_row, det_row)

    risk_label = lower(layer2_row.get("risk_label"))
    leaf_support = to_float(layer2_row.get("leaf_support"), default=None)
    risk_reasons = split_codes(layer2_row.get("risk_reasons", ""))
    policy_action = lower(base["layer1_policy_action_hint"])
    action_category = lower(base["layer1a_action_category"])
    issue_codes = collect_layer1_codes(policy_row, det_row)
    base["all_layer1_codes"] = "|".join(issue_codes)

    base.update({
        "branch_a_leaf_support": "" if leaf_support is None else leaf_support,
        "branch_a_risk_label": risk_label,
        "branch_a_risk_reasons": "|".join(risk_reasons),
    })

    # Priority order matters:
    # 1. deterministic structural candidates
    # 2. hard policy / high risk rescue
    # 3. computability review
    # 4. Branch A low/medium are diagnostic only or optional review
    if action_category == "safe_normalization_candidate" or policy_action == "safe_normalization_candidate":
        action = {
            "layer3_action_family": "structural_repair",
            "layer3_primary_action": "safe_structural_repair_candidate",
            "layer3_priority": 20,
            "layer3_changes_ast": 1,
            "layer3_uses_llm": 0,
            "layer3_requires_reverification": 1,
            "layer3_reason": "Layer 1 detected a safe normalization candidate. Apply deterministic repair in Layer 3, then re-run verification.",
            "suggested_rescue_task": "deterministic_safe_normalization",
            "suggested_fallback": "reverify_after_repair",
        }

    elif action_category == "conservative_rewrite_candidate":
        action = {
            "layer3_action_family": "conservative_downgrade",
            "layer3_primary_action": "conservative_downgrade_candidate",
            "layer3_priority": 30,
            "layer3_changes_ast": 1,
            "layer3_uses_llm": 0,
            "layer3_requires_reverification": 1,
            "layer3_reason": "Layer 1 detected a structural inconsistency that should not be silently accepted. Downgrade to partial/non-computable only if clinically safe.",
            "suggested_rescue_task": "deterministic_conservative_downgrade",
            "suggested_fallback": "human_review_if_downgrade_changes_meaning",
        }

    elif policy_action == "mandatory_rescue_or_review_candidate" or risk_label == "review_priority":
        rescue_task = infer_rescue_task(issue_codes, risk_reasons, policy_action, branch="A")
        action = {
            "layer3_action_family": "llm_rescue_or_review",
            "layer3_primary_action": "targeted_llm_rescue_candidate",
            "layer3_priority": 40,
            "layer3_changes_ast": 1,
            "layer3_uses_llm": 1,
            "layer3_requires_reverification": 1,
            "layer3_reason": "Branch A leaf is high risk or has mandatory Layer 1 rescue/review policy. Use targeted diagnosis-aware rescue, not blind re-extraction.",
            "suggested_rescue_task": rescue_task,
            "suggested_fallback": "mark_partial_or_human_review_if_rescue_fails",
        }

    elif policy_action == "computability_review_candidate":
        action = {
            "layer3_action_family": "computability_review",
            "layer3_primary_action": "computability_review_or_partial",
            "layer3_priority": 50,
            "layer3_changes_ast": 0,
            "layer3_uses_llm": 0,
            "layer3_requires_reverification": 0,
            "layer3_reason": "Layer 1 indicates a computability/execution issue. Review computability status rather than treating this as semantic extraction error.",
            "suggested_rescue_task": "computability_review",
            "suggested_fallback": "mark_partial_or_non_computable_with_reason",
        }

    elif risk_label == "intermediate_support":
        rescue_task = infer_rescue_task(issue_codes, risk_reasons, policy_action, branch="A")
        action = {
            "layer3_action_family": "optional_review",
            "layer3_primary_action": "optional_targeted_rescue_or_review",
            "layer3_priority": 60,
            "layer3_changes_ast": 0,
            "layer3_uses_llm": 1,
            "layer3_requires_reverification": 1,
            "layer3_reason": "Branch A medium risk. Use as optional rescue/review candidate depending on budget.",
            "suggested_rescue_task": rescue_task,
            "suggested_fallback": "manual_review_if_not_rescued",
        }

    else:
        # Important: Branch A low risk is NOT automatic accept.
        action = {
            "layer3_action_family": "diagnostic_only",
            "layer3_primary_action": "branch_a_diagnostic_only_not_auto_accept",
            "layer3_priority": 90,
            "layer3_changes_ast": 0,
            "layer3_uses_llm": 0,
            "layer3_requires_reverification": 0,
            "layer3_reason": "Pilot threshold derivation found no safe Branch A automatic accept threshold. Low risk is only relative within Branch A.",
            "suggested_rescue_task": "none",
            "suggested_fallback": "use_branch_b_or_manual_review_for_final_decision",
        }

    base.update(action)
    return base


def decide_branch_b(
    layer2_row: Dict[str, Any],
    policy_row: Dict[str, Any],
    det_row: Dict[str, Any],
) -> Dict[str, Any]:
    base = base_leaf_fields("B", layer2_row, policy_row, det_row)

    semantic_support = to_float(layer2_row.get("semantic_grounding_support"), default=None)
    semantic_risk = lower(layer2_row.get("semantic_grounding_risk_label"))
    semantic_reasons = split_codes(layer2_row.get("semantic_grounding_reasons", ""))

    execution_support = to_float(layer2_row.get("execution_support"), default=None)
    execution_risk = lower(layer2_row.get("execution_risk_label"))
    execution_reasons = split_codes(layer2_row.get("execution_reasons", ""))

    routing = lower(layer2_row.get("final_routing_decision"))
    routing_reasons = split_codes(layer2_row.get("routing_reasons", ""))

    policy_action = lower(base["layer1_policy_action_hint"])
    action_category = lower(base["layer1a_action_category"])
    issue_codes = collect_layer1_codes(policy_row, det_row)

    base.update({
        "branch_b_semantic_grounding_support": "" if semantic_support is None else semantic_support,
        "branch_b_semantic_grounding_risk_label": semantic_risk,
        "branch_b_semantic_grounding_reasons": "|".join(semantic_reasons),
        "branch_b_execution_support": "" if execution_support is None else execution_support,
        "branch_b_execution_risk_label": execution_risk,
        "branch_b_execution_reasons": "|".join(execution_reasons),
        "branch_b_final_routing_decision": routing,
        "branch_b_routing_reasons": "|".join(routing_reasons),
    })

    # Priority order matters:
    # 1. deterministic structural repair/downgrade candidates
    # 2. mandatory verifier / hard semantic issues
    # 3. computability review
    # 4. accept
    # 5. optional review
    if action_category == "safe_normalization_candidate" or policy_action == "safe_normalization_candidate":
        action = {
            "layer3_action_family": "structural_repair",
            "layer3_primary_action": "safe_structural_repair_candidate",
            "layer3_priority": 20,
            "layer3_changes_ast": 1,
            "layer3_uses_llm": 0,
            "layer3_requires_reverification": 1,
            "layer3_reason": "Layer 1 detected a safe normalization candidate. Apply deterministic repair in Layer 3, then re-run verification.",
            "suggested_rescue_task": "deterministic_safe_normalization",
            "suggested_fallback": "reverify_after_repair",
        }

    elif action_category == "conservative_rewrite_candidate":
        action = {
            "layer3_action_family": "conservative_downgrade",
            "layer3_primary_action": "conservative_downgrade_candidate",
            "layer3_priority": 30,
            "layer3_changes_ast": 1,
            "layer3_uses_llm": 0,
            "layer3_requires_reverification": 1,
            "layer3_reason": "Layer 1 detected a structural inconsistency that should not be silently accepted. Downgrade to partial/non-computable only if clinically safe.",
            "suggested_rescue_task": "deterministic_conservative_downgrade",
            "suggested_fallback": "human_review_if_downgrade_changes_meaning",
        }

    elif policy_action in {"mandatory_verifier_candidate", "mandatory_rescue_or_review_candidate"} or routing == "llm_verifier":
        rescue_task = infer_rescue_task(
            issue_codes,
            semantic_reasons + routing_reasons,
            policy_action,
            branch="B",
        )
        action = {
            "layer3_action_family": "llm_rescue_or_review",
            "layer3_primary_action": "targeted_llm_rescue_candidate",
            "layer3_priority": 40,
            "layer3_changes_ast": 1,
            "layer3_uses_llm": 1,
            "layer3_requires_reverification": 1,
            "layer3_reason": "Branch B semantic grounding/routing screen selected this leaf for mandatory LLM verification or targeted rescue.",
            "suggested_rescue_task": rescue_task,
            "suggested_fallback": "mark_partial_or_human_review_if_rescue_fails",
        }

    elif policy_action == "computability_review_candidate" or routing == "computability_review":
        action = {
            "layer3_action_family": "computability_review",
            "layer3_primary_action": "computability_review_or_partial",
            "layer3_priority": 50,
            "layer3_changes_ast": 0,
            "layer3_uses_llm": 0,
            "layer3_requires_reverification": 0,
            "layer3_reason": "Branch B extraction may be semantically grounded, but execution/computability is uncertain.",
            "suggested_rescue_task": "computability_review",
            "suggested_fallback": "mark_partial_or_non_computable_with_reason",
        }

    elif routing == "keep_without_layer2_action":
        action = {
            "layer3_action_family": "keep_no_action_reference",
            "layer3_primary_action": "keep_without_layer2_action_candidate",
            "layer3_priority": 80,
            "layer3_changes_ast": 0,
            "layer3_uses_llm": 0,
            "layer3_requires_reverification": 0,
            "layer3_reason": (
                "Branch B Layer 2 routed this leaf to keep without Layer 2 action. "
                "This is a high-support no-action reference group, not a calibrated proof of correctness."
            ),
            "suggested_rescue_task": "none",
            "suggested_fallback": "none",
        }

    elif routing == "optional_llm_verifier_or_review":
        rescue_task = infer_rescue_task(
            issue_codes,
            semantic_reasons + routing_reasons,
            policy_action,
            branch="B",
        )
        action = {
            "layer3_action_family": "optional_review",
            "layer3_primary_action": "optional_targeted_rescue_or_review",
            "layer3_priority": 60,
            "layer3_changes_ast": 0,
            "layer3_uses_llm": 0,
            "layer3_requires_reverification": 1,
            "layer3_reason": "Branch B leaf is not mandatory rescue but remains uncertain enough for optional LLM verification or human review.",
            "suggested_rescue_task": rescue_task,
            "suggested_fallback": "accept_only_if_budget_or_policy_allows",
        }

    else:
        action = {
            "layer3_action_family": "human_review_or_abstain",
            "layer3_primary_action": "human_review_or_abstain",
            "layer3_priority": 100,
            "layer3_changes_ast": 0,
            "layer3_uses_llm": 0,
            "layer3_requires_reverification": 0,
            "layer3_reason": f"Unrecognized or unresolved Branch B routing decision: {routing}",
            "suggested_rescue_task": "manual_review",
            "suggested_fallback": "mark_pending_human_review",
        }

    base.update(action)
    return base


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLayer 3 decision inventory")
    print("Layer 1 inventory:", LAYER1_INVENTORY_CSV)
    print("Branch A policy:", BRANCH_A_POLICY_CSV)
    print("Branch A Layer 2:", BRANCH_A_LAYER2_CSV)
    print("Branch B policy:", BRANCH_B_POLICY_CSV)
    print("Branch B Layer 2:", BRANCH_B_LAYER2_CSV)

    layer1_rows = read_csv(LAYER1_INVENTORY_CSV)
    a_policy_rows = read_csv(BRANCH_A_POLICY_CSV)
    b_policy_rows = read_csv(BRANCH_B_POLICY_CSV)
    a_layer2_rows = read_csv(BRANCH_A_LAYER2_CSV)
    b_layer2_rows = read_csv(BRANCH_B_LAYER2_CSV)
    branch_a_validation_summary = read_json(
        BRANCH_A_VALIDATION_SUMMARY_JSON,
        required=False,
    )
    # Deterministic inventory uses A_raw / B_raw.
    det_a_index = index_by_branch_and_criterion(
        layer1_rows,
        branch_name="A",
        accepted_branch_values=["A_bert_rules", "A_raw", "A"],
    )

    det_b_index = index_by_branch_and_criterion(
        layer1_rows,
        branch_name="B",
        accepted_branch_values=["B_llm_pass2", "B_raw", "B"],
    )

    a_policy_index = index_by_branch_and_criterion(a_policy_rows, branch_name="A")
    b_policy_index = index_by_branch_and_criterion(b_policy_rows, branch_name="B")

    a_rows: List[Dict[str, Any]] = []
    b_rows: List[Dict[str, Any]] = []

    for r in a_layer2_rows:
        cid = criterion_id_from_row(r)
        key = ("A", cid)
        policy_row = a_policy_index.get(key, {})
        det_row = det_a_index.get(key, {})
        a_rows.append(decide_branch_a(r, policy_row, det_row))

    for r in b_layer2_rows:
        cid = criterion_id_from_row(r)
        key = ("B", cid)
        policy_row = b_policy_index.get(key, {})
        det_row = det_b_index.get(key, {})
        b_rows.append(decide_branch_b(r, policy_row, det_row))

    rows = a_rows + b_rows

    write_csv(OUT_CSV, rows)

    # Summary
    by_branch_action = defaultdict(Counter)
    by_branch_family = defaultdict(Counter)
    by_branch_priority = defaultdict(Counter)
    by_branch_rescue_task = defaultdict(Counter)

    for r in rows:
        branch = r.get("branch", "")
        by_branch_action[branch][r.get("layer3_primary_action", "")] += 1
        by_branch_family[branch][r.get("layer3_action_family", "")] += 1
        by_branch_priority[branch][str(r.get("layer3_priority", ""))] += 1
        by_branch_rescue_task[branch][r.get("suggested_rescue_task", "")] += 1

    summary = {
        "description": (
            "Layer 3 decision inventory. This file assigns actions but does not repair, "
            "modify ASTs, or call the LLM."
        ),
        "inputs": {
            "layer1_inventory_csv": str(LAYER1_INVENTORY_CSV),
            "branch_a_policy_csv": str(BRANCH_A_POLICY_CSV),
            "branch_a_layer2_csv": str(BRANCH_A_LAYER2_CSV),
            "branch_b_policy_csv": str(BRANCH_B_POLICY_CSV),
            "branch_b_layer2_csv": str(BRANCH_B_LAYER2_CSV),
            "branch_a_validation_summary_json": (
                str(BRANCH_A_VALIDATION_SUMMARY_JSON)
                if BRANCH_A_VALIDATION_SUMMARY_JSON.exists()
                else ""
            ),
        },  # missing closing brace was here

        "outputs": {
            "leaf_level_decision_inventory_csv": str(OUT_CSV),
            "summary_json": str(OUT_JSON),
        },
        "n_rows_total": len(rows),
        "n_rows_by_branch": {
            "A": len(a_rows),
            "B": len(b_rows),
        },
        "layer3_primary_action_counts_by_branch": {
            branch: dict(counter.most_common())
            for branch, counter in by_branch_action.items()
        },
        "layer3_action_family_counts_by_branch": {
            branch: dict(counter.most_common())
            for branch, counter in by_branch_family.items()
        },
        "layer3_priority_counts_by_branch": {
            branch: dict(counter.most_common())
            for branch, counter in by_branch_priority.items()
        },
        "suggested_rescue_task_counts_by_branch": {
            branch: dict(counter.most_common())
            for branch, counter in by_branch_rescue_task.items()
        },
        "method_notes": [
            "Branch A targeted_llm_rescue_candidate is a broad upstream candidate label. It does not mean all Branch A candidates will be sent to the LLM. Later 06f/06f2 refines this set using Branch B substitution, actionable rescue rules, and human-review fallbacks."
            "Branch A low-risk leaves are not automatically accepted because the pilot threshold analysis found no safe Branch A accept threshold.",
            "Branch B keep-without-action candidates come from the Branch B grounding/routing screen and are not calibrated automatic accepts.",
            "Any leaf changed by deterministic repair or LLM rescue must be re-verified by Layer 1 and Layer 2.",
            "This script is decision-only. It does not apply repairs.",
        ],
        "branch_a_validation_summary_loaded": bool(
            branch_a_validation_summary
        ),
    }

    write_json(OUT_JSON, summary)

    print("\nDONE")
    print("Output CSV:", OUT_CSV)
    print("Output JSON:", OUT_JSON)
    print("Rows total:", len(rows))
    print("Rows by branch:", summary["n_rows_by_branch"])

    print("\nLayer 3 action counts by branch:")
    for branch, counts in summary["layer3_primary_action_counts_by_branch"].items():
        print(f"\n--- Branch {branch} ---")
        for action, count in counts.items():
            print(f"{action}: {count}")

    print("\nLayer 3 action family counts by branch:")
    for branch, counts in summary["layer3_action_family_counts_by_branch"].items():
        print(f"\n--- Branch {branch} ---")
        for action, count in counts.items():
            print(f"{action}: {count}")


if __name__ == "__main__":
    main()

# Run from the repository root:
# python scripts/03_verification/03_layer3/01_build_decision_inventory.py