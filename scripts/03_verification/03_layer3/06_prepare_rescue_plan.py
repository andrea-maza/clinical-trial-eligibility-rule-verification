"""
06_prepare_rescue_plan.py

Prepare the final Layer 3 rescue and review plan.

The broad candidate queue contains uncertainty signals, but high risk
alone does not imply that an LLM repair is appropriate. This script
separates candidates into safer action groups:

    - Branch A cross-branch substitution candidates
    - Branch A deferred diagnostic cases
    - Branch A status or human-review cases
    - Branch B mandatory judge--repair candidates
    - Branch B safe non-LLM references
    - Branch B keep-without-change references
    - Branch B blocked or manual-review cases

Branch A does not receive a new LLM extraction call. When appropriate,
the corresponding Branch B leaf is offered as a semantic substitute.

Branch B mandatory verifier candidates enter the bounded judge--repair
process. They are not automatically rewritten.

Manual labels are used only for retrospective diagnostics after the
plan has been created. They do not influence candidate selection.

Outputs:
    outputs/verification/layer3/rescue_plan/

Run from the repository root:
python scripts/03_verification/03_layer3/06_prepare_rescue_plan.py
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------

QUEUE_06D_JSONL = (
    ROOT
    / "outputs"
    / "verification"
    / "layer3"
    / "targeted_rescue_candidates"
    / "layer3_targeted_llm_rescue_candidates.jsonl"
)

DECISION_CSV = (
    ROOT
    / "outputs"
    / "verification"
    / "layer3"
    / "decision_inventory"
    / "layer3_decision_inventory_leaf_level.csv"
)

# Optional input. The broad queue already stores conservative issue codes.
CONSERVATIVE_CSV = (
    ROOT
    / "outputs"
    / "verification"
    / "layer3"
    / "conservative_downgrade_inspection"
    / "layer3_conservative_downgrade_candidates.csv"
)

BRANCH_B_MANDATORY_CSV = (
    ROOT
    / "outputs"
    / "verification"
    / "layer3"
    / "p1_rescue_subset"
    / "p1_branch_b_mandatory_llm_judge.csv"
)


# ---------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------

OUT_DIR = (
    ROOT
    / "outputs"
    / "verification"
    / "layer3"
    / "rescue_plan"
)

OUT_MAIN_JSONL = OUT_DIR / "main_llm_rescue_plan.jsonl"
OUT_MAIN_CSV = OUT_DIR / "main_llm_rescue_plan.csv"

OUT_BRANCH_A_MAIN_CSV = OUT_DIR / "branch_a_main_llm_rescue_plan.csv"
OUT_BRANCH_A_DEFERRED_CSV = OUT_DIR / "branch_a_deferred_diagnostic_plan.csv"
OUT_BRANCH_A_STATUS_REVIEW_CSV = OUT_DIR / "branch_a_status_or_review_plan.csv"

OUT_BRANCH_B_MAIN_CSV = OUT_DIR / "branch_b_main_llm_rescue_plan.csv"
OUT_BRANCH_B_SAFE_CSV = OUT_DIR / "branch_b_safe_non_llm_reference_plan.csv"
OUT_BRANCH_B_ACCEPT_CSV = OUT_DIR / "branch_b_accept_no_change_reference_plan.csv"
OUT_BRANCH_B_BLOCKED_CSV = OUT_DIR / "branch_b_blocked_or_manual_review_plan.csv"

OUT_ALL_CSV = OUT_DIR / "layer3_literature_aligned_rescue_plan_all.csv"
OUT_ALL_JSONL = OUT_DIR / "layer3_literature_aligned_rescue_plan_all.jsonl"

OUT_SUMMARY_JSON = OUT_DIR / "literature_aligned_rescue_plan_summary.json"

MANUAL_PRE_LABELS_CSV = (
    ROOT
    / "outputs"
    / "evaluation"
    / "pre_verification"
    / "semantic_manual_pre_verification_A_B_summary"
    / "reviewed_semantic_clause_labels_A_B.csv"
)

OUT_MANUAL_DIAG_DIR = OUT_DIR / "manual_overlap_diagnostic"
OUT_MANUAL_MAIN_ANNOTATED_CSV = (
    OUT_MANUAL_DIAG_DIR / "main_rescue_plan_with_pre_manual_labels.csv"
)
OUT_MANUAL_ALL_ANNOTATED_CSV = (
    OUT_MANUAL_DIAG_DIR / "all_rescue_plan_with_pre_manual_labels.csv"
)
OUT_MANUAL_BY_BRANCH_TASK_CSV = (
    OUT_MANUAL_DIAG_DIR
    / "main_rescue_manual_label_summary_by_branch_task.csv"
)
OUT_MANUAL_BY_BRANCH_STRATEGY_CSV = (
    OUT_MANUAL_DIAG_DIR
    / "main_rescue_manual_label_summary_by_branch_strategy.csv"
)
OUT_MANUAL_ERROR_COVERAGE_CSV = (
    OUT_MANUAL_DIAG_DIR / "manual_error_coverage_by_branch.csv"
)
OUT_MANUAL_A_B_CANDIDATE_CSV = (
    OUT_MANUAL_DIAG_DIR
    / "branch_a_rescue_b_candidate_manual_diagnostic.csv"
)
OUT_MANUAL_SUMMARY_JSON = (
    OUT_MANUAL_DIAG_DIR / "rescue_manual_overlap_summary.json"
)

# ---------------------------------------------------------------------
# Policy constants
# ---------------------------------------------------------------------

CONSERVATIVE_STRUCTURAL_ISSUES = {
    "list_operator_without_list_value",
    "comparison_without_scalar_value",
    "range_with_both_bounds_missing",
}

BRANCH_A_ACTIONABLE_RESCUE_TASKS = {
    "list_value_recovery",
    "range_bounds_recovery",
    "scalar_value_recovery",
    "value_or_operator_recovery",
    "temporal_context_recovery",
    "history_context_recovery",
    "polarity_negation_repair",
}

BRANCH_A_ENTITY_RESCUE_TASKS = {
    "entity_regrounding",
    "entity_type_reclassification",
}

BRANCH_A_DEFER_TASKS = {
    "duplicate_prune_or_merge_judge",
    "branch_a_field_level_semantic_rescue",
}

BRANCH_A_CONTEXT_TASKS = {
    "condition_exception_context_recovery",
}

EXPLICIT_ENTITY_PROBLEM_TERMS = {
    "entity_not_in_evidence",
    "entity_not_grounded",
    "entity_not_grounded_in_evidence_or_item",
    "entity_text_empty",
    "generic_entity_text",
    "critical_qualifier_missing_from_entity",
    "entity_type_not_supported_by_best_anchor",
    "entity_text_not_aligned_with_any_anchor",
}

NO_BERT_ANCHOR_TERMS = {
    "no_bert_anchor",
    "missing_bert_anchor",
}

BRANCH_B_ACTIONABLE_RESCUE_TASKS = {
    "entity_regrounding",
    "entity_type_reclassification",
    "value_or_operator_recovery",
    "temporal_context_recovery",
    "history_context_recovery",
    "polarity_negation_repair",
    "list_value_recovery",
    "range_bounds_recovery",
    "scalar_value_recovery",
}

BRANCH_B_CONTEXT_TASKS = {
    "condition_exception_context_recovery",
}

BRANCH_B_DEFER_TASKS = {
    "duplicate_prune_or_merge_judge",
    "branch_b_field_level_semantic_judge",
}

BRANCH_B_CONTEXT_OR_SCOPE_TERMS = {
    "condition_context_present",
    "condition_context_present_without_handling",
    "exception_or_condition_clause_without_context_handling",
    "exception_clause_computable_despite_context",
    "computable_with_exception_context",
    "condition_exception",
    "exception_context",
}


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------

def read_jsonl(path: Path, required: bool = True) -> List[Dict[str, Any]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"JSONL not found: {path}")
        return []

    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}, line {line_no}") from exc

    return rows


def read_csv(path: Path, required: bool = True) -> List[Dict[str, str]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"CSV not found: {path}")
        return []

    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]

    for enc in encodings:
        try:
            with path.open("r", encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue

    raise RuntimeError(f"Could not decode CSV: {path}")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
        path.write_text("", encoding="utf-8-sig")
        return

    priority = [
        "plan_id",
        "branch_to_update",
        "criterion_id",
        "document_id",
        "item_uid",
        "clause_id",
        "execution_group",
        "execution_priority",
        "rescue_strategy",
        "rescue_task_type",
        "include_A_current",
        "include_B_current",
        "allow_new_llm_candidate",
        "max_llm_attempts",
        "requires_llm",
        "requires_reverification",
        "safe_to_apply_without_llm",
        "blocked_from_auto_apply",
        "b_candidate_allowed",
        "b_candidate_reason",
        "selection_policy",
        "why_selected",
        "why_deferred_or_blocked",
        "source_branch",
        "source_run_stage",
        "source_candidate_kind",
        "source_issue_codes",
        "source_risk_reasons",
        "source_diagnosis_summary",
        "branch_a_risk_label",
        "branch_a_leaf_support",
        "branch_b_final_routing_decision",
        "branch_b_semantic_grounding_risk_label",
        "branch_b_semantic_grounding_support",
        "branch_b_execution_risk_label",
        "branch_b_execution_support",
    ]

    cols = []
    seen = set()

    for c in priority:
        if any(c in row for row in rows):
            cols.append(c)
            seen.add(c)

    for row in rows:
        for c in row:
            if c not in seen:
                cols.append(c)
                seen.add(c)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()

        for row in rows:
            writer.writerow({c: serialize_cell(row.get(c, "")) for c in cols})


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def clean(x: Any) -> str:
    return str(x or "").strip()


def lower(x: Any) -> str:
    return clean(x).lower()


def to_float(x: Any, default: float | None = None) -> float | None:
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

def pct(n: int, d: int) -> float | None:
    if d == 0:
        return None

    return round(100.0 * n / d, 2)

def split_terms(x: Any) -> List[str]:
    if x is None:
        return []

    if isinstance(x, list):
        return [clean(v) for v in x if clean(v)]

    s = clean(x)

    if not s:
        return []

    if s.startswith("[") and s.endswith("]"):
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [clean(v) for v in obj if clean(v)]
        except Exception:
            pass

    return [p.strip() for p in re.split(r"[;|,]", s) if p.strip()]


def normalize_issue_code(code: str) -> str:
    return clean(code).split(":", 1)[-1].strip()


def normalize_issue_codes(x: Any) -> List[str]:
    return sorted(set(normalize_issue_code(v) for v in split_terms(x) if normalize_issue_code(v)))


def unique_join(values: List[Any]) -> str:
    out = []

    for value in values:
        s = clean(value)

        if s and s not in out:
            out.append(s)

    return ";".join(out)


def criterion_id_from_row(row: Dict[str, Any]) -> str:
    cid = clean(row.get("criterion_id"))

    if cid:
        return cid

    candidate_id = clean(row.get("candidate_id"))

    if candidate_id and "__" in candidate_id:
        parts = candidate_id.split("__")

        if len(parts) >= 3:
            return "__".join(parts[1:-1])

    item_uid = clean(row.get("item_uid"))
    clause_id = clean(row.get("clause_id"))

    if item_uid and clause_id:
        return f"{item_uid}_{clause_id}"

    return ""


def branch_from_row(row: Dict[str, Any]) -> str:
    branch = clean(row.get("branch"))

    if branch in {"A", "B"}:
        return branch

    candidate_id = clean(row.get("candidate_id"))

    if candidate_id.startswith("A__"):
        return "A"

    if candidate_id.startswith("B__"):
        return "B"

    return branch


def parse_item_clause(criterion_id: str) -> Tuple[str, str]:
    criterion_id = clean(criterion_id)

    if "_" not in criterion_id:
        return "", ""

    item_uid, clause_id = criterion_id.rsplit("_", 1)
    return item_uid, clause_id


def rescue_task_from_row(row: Dict[str, Any]) -> str:
    for key in ["rescue_task_type", "rescue_task", "suggested_rescue_task"]:
        value = clean(row.get(key))
        if value:
            return value

    return ""


def all_diagnosis_terms(row: Dict[str, Any], decision_row: Dict[str, Any] | None = None) -> List[str]:
    terms = []

    for key in [
        "all_layer1_codes",
        "diagnosis_summary",
        "branch_a_risk_reasons",
        "branch_b_semantic_grounding_reasons",
        "branch_b_execution_reasons",
        "branch_b_routing_reasons",
        "conservative_issues",
    ]:
        terms.extend(normalize_issue_codes(row.get(key, "")))

    if decision_row:
        for key in [
            "all_layer1_codes",
            "branch_a_risk_reasons",
            "branch_b_semantic_grounding_reasons",
            "branch_b_execution_reasons",
            "branch_b_routing_reasons",
            "layer3_reason",
        ]:
            terms.extend(normalize_issue_codes(decision_row.get(key, "")))

    return sorted(set(t for t in terms if t))


def has_any(terms: List[str], wanted: set[str]) -> bool:
    lower_terms = {t.lower() for t in terms}
    wanted_lower = {w.lower() for w in wanted}
    return bool(lower_terms & wanted_lower)


def only_no_bert_anchor_problem(terms: List[str]) -> bool:
    lower_terms = {t.lower() for t in terms}

    if not (lower_terms & NO_BERT_ANCHOR_TERMS):
        return False

    actionable_without_anchor = (
        lower_terms
        & (
            EXPLICIT_ENTITY_PROBLEM_TERMS
            | CONSERVATIVE_STRUCTURAL_ISSUES
            | {
                "temporal_marker_without_temporal_context",
                "history_marker_without_history_context",
                "condition_context_present_without_handling",
                "comparison_without_scalar_value",
                "value_missing_with_quantitative_cue",
                "quantitative_cue_unhandled",
                "negation_clause_with_exists_operator",
                "positive_clause_with_not_exists_operator",
                "negative_entity_with_not_exists_operator",
            }
        )
    )

    return len(actionable_without_anchor) == 0


# ---------------------------------------------------------------------
# Indices
# ---------------------------------------------------------------------

def build_decision_index(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, str]]:
    out = {}

    for row in rows:
        branch = clean(row.get("branch"))
        cid = criterion_id_from_row(row)

        if branch and cid:
            out[(branch, cid)] = row

    return out


def build_06d_group_index(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    out = defaultdict(list)

    for row in rows:
        branch = branch_from_row(row)
        cid = criterion_id_from_row(row)

        if branch and cid:
            out[(branch, cid)].append(row)

    return dict(out)


def build_conservative_index(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, str]]:
    out = {}

    for row in rows:
        branch = clean(row.get("branch"))
        cid = criterion_id_from_row(row)

        if branch and cid:
            out[(branch, cid)] = row

    return out


# ---------------------------------------------------------------------
# Branch B usability as candidate source for Branch A
# ---------------------------------------------------------------------

def branch_b_candidate_status(b_decision: Dict[str, Any] | None) -> Tuple[bool, str]:
    """
    Decide whether Branch B can be offered as a candidate source for Branch A.

    Important:
    This does NOT mean Branch B will be applied.
    It only means 06F2 may compare B_current against A_current and possible LLM candidates.

    The old version was too strict because it allowed B only when B looked very clean.
    But Branch B may be semantically useful even when its grounding score is imperfect.
    Therefore, only hard blockers exclude B here.
    """
    if not b_decision:
        return False, "no_matching_branch_b_decision_row"

    entity_text = clean(b_decision.get("entity_text"))
    evidence_text = clean(b_decision.get("evidence_text"))
    computability = lower(b_decision.get("computability"))
    routing = lower(b_decision.get("branch_b_final_routing_decision"))
    layer3_action = lower(b_decision.get("layer3_primary_action"))

    terms = all_diagnosis_terms(b_decision)

    if not entity_text:
        return False, "branch_b_missing_entity_text"

    if not evidence_text:
        return False, "branch_b_missing_evidence_text"

    soft_warnings = []

    # Keep these as hard blockers.
    # These can change the clinical meaning of the rule.
    if has_any(
        terms,
        {
            "negation_clause_with_exists_operator",
            "positive_clause_with_not_exists_operator",
            "negative_entity_with_not_exists_operator",
        },
    ):
        return False, "branch_b_polarity_problem"

    if has_any(terms, CONSERVATIVE_STRUCTURAL_ISSUES):
        return False, "branch_b_structural_value_problem"

    # Make these soft warnings, not blockers.
    # Branch B can still be offered as a candidate, but 06F2 must not auto-apply it blindly.
    if computability == "non_computable":
        soft_warnings.append("branch_b_non_computable")

    if has_any(terms, BRANCH_B_CONTEXT_OR_SCOPE_TERMS):
        soft_warnings.append("branch_b_context_or_exception_scope_problem")

    if routing:
        base_reason = f"branch_b_available_as_candidate_routing:{routing}"
    elif layer3_action:
        base_reason = f"branch_b_available_as_candidate_action:{layer3_action}"
    else:
        base_reason = "branch_b_available_as_semantic_candidate"

    if soft_warnings:
        return True, base_reason + "|soft_warning:" + ";".join(soft_warnings)

    return True, base_reason


# ---------------------------------------------------------------------
# Plan row builders
# ---------------------------------------------------------------------

def base_plan_row(
    *,
    branch_to_update: str,
    criterion_id: str,
    execution_group: str,
    execution_priority: int,
    rescue_strategy: str,
    rescue_task_type: str,
    requires_llm: int,
    requires_reverification: int,
    safe_to_apply_without_llm: int,
    blocked_from_auto_apply: int,
    why_selected: str = "",
    why_deferred_or_blocked: str = "",
    source_row: Dict[str, Any] | None = None,
    decision_row: Dict[str, Any] | None = None,
    b_decision_row: Dict[str, Any] | None = None,
    b_candidate_allowed: int = 0,
    b_candidate_reason: str = "",
    include_A_current: int = 1,
    include_B_current: int = 0,
    allow_new_llm_candidate: int = 0,
    max_llm_attempts: int = 0,
    selection_policy: str = "",
) -> Dict[str, Any]:
    item_uid, clause_id = parse_item_clause(criterion_id)

    source_row = source_row or {}
    decision_row = decision_row or {}
    b_decision_row = b_decision_row or {}

    plan_id = f"{branch_to_update}__{criterion_id}__{execution_group}__{rescue_task_type}"

    return {
        "plan_id": plan_id,
        "branch_to_update": branch_to_update,
        "criterion_id": criterion_id,
        "document_id": clean(decision_row.get("document_id") or source_row.get("document_id")),
        "item_uid": item_uid,
        "clause_id": clause_id,

        "execution_group": execution_group,
        "execution_priority": execution_priority,
        "rescue_strategy": rescue_strategy,
        "rescue_task_type": rescue_task_type,

        "include_A_current": include_A_current,
        "include_B_current": include_B_current,
        "allow_new_llm_candidate": allow_new_llm_candidate,
        "max_llm_attempts": max_llm_attempts,

        "requires_llm": requires_llm,
        "requires_reverification": requires_reverification,
        "safe_to_apply_without_llm": safe_to_apply_without_llm,
        "blocked_from_auto_apply": blocked_from_auto_apply,

        "b_candidate_allowed": b_candidate_allowed,
        "b_candidate_reason": b_candidate_reason,
        "selection_policy": selection_policy,

        "why_selected": why_selected,
        "why_deferred_or_blocked": why_deferred_or_blocked,

        "source_branch": branch_from_row(source_row),
        "source_run_stage": clean(source_row.get("run_stage")),
        "source_candidate_kind": clean(source_row.get("candidate_kind")),
        "source_issue_codes": unique_join(normalize_issue_codes(source_row.get("all_layer1_codes", ""))),
        "source_risk_reasons": unique_join(
            split_terms(source_row.get("branch_a_risk_reasons", ""))
            + split_terms(source_row.get("branch_b_semantic_grounding_reasons", ""))
            + split_terms(source_row.get("branch_b_execution_reasons", ""))
        ),
        "source_diagnosis_summary": clean(source_row.get("diagnosis_summary")),

        "branch_a_risk_label": clean(decision_row.get("branch_a_risk_label")),
        "branch_a_leaf_support": clean(decision_row.get("branch_a_leaf_support")),

        "branch_b_final_routing_decision": clean(b_decision_row.get("branch_b_final_routing_decision")),
        "branch_b_semantic_grounding_risk_label": clean(b_decision_row.get("branch_b_semantic_grounding_risk_label")),
        "branch_b_semantic_grounding_support": clean(b_decision_row.get("branch_b_semantic_grounding_support")),
        "branch_b_execution_risk_label": clean(b_decision_row.get("branch_b_execution_risk_label")),
        "branch_b_execution_support": clean(b_decision_row.get("branch_b_execution_support")),
    }


# ---------------------------------------------------------------------
# Branch A classification
# ---------------------------------------------------------------------

def classify_branch_a_candidate(
    *,
    criterion_id: str,
    rows_06d: List[Dict[str, Any]],
    a_decision: Dict[str, str],
    b_decision: Dict[str, str] | None,
    conservative_row: Dict[str, str] | None,
) -> Dict[str, Any]:
    representative = rows_06d[0] if rows_06d else {}

    rescue_tasks = sorted(set(rescue_task_from_row(r) for r in rows_06d if rescue_task_from_row(r)))
    task = rescue_tasks[0] if len(rescue_tasks) == 1 else unique_join(rescue_tasks)

    terms = []
    for r in rows_06d:
        terms.extend(all_diagnosis_terms(r, a_decision))
    terms = sorted(set(terms))

    b_allowed, b_reason = branch_b_candidate_status(b_decision)

    risk_label = lower(a_decision.get("branch_a_risk_label"))
    policy_action = lower(a_decision.get("layer1_policy_action_hint"))
    layer3_primary = lower(a_decision.get("layer3_primary_action"))

    is_conservative = conservative_row is not None or has_any(terms, CONSERVATIVE_STRUCTURAL_ISSUES)

    # 1. Conservative structural recovery is truly actionable.
    if is_conservative:
        return base_plan_row(
            branch_to_update="A",
            criterion_id=criterion_id,
            execution_group="A_main_llm_rescue",
            execution_priority=10,
            rescue_strategy="recover_structural_information_or_abstain",
            rescue_task_type="conservative_structural_recovery",
            requires_llm=1,
            requires_reverification=1,
            safe_to_apply_without_llm=0,
            blocked_from_auto_apply=0,
            source_row=representative,
            decision_row=a_decision,
            b_decision_row=b_decision,
            b_candidate_allowed=1 if b_allowed else 0,
            b_candidate_reason=b_reason,
            include_A_current=1,
            include_B_current=1 if b_allowed else 0,
            allow_new_llm_candidate=1,
            max_llm_attempts=2,
            selection_policy=(
                "Try to recover the missing list/range/scalar information from evidence. "
                "If unrecoverable, mark partial/non-computable rather than downgrade silently."
            ),
            why_selected=(
                "Branch A has a conservative structural issue. This is actionable because "
                "the missing value/list/range may be recoverable from the source text."
            ),
        )

    # 2. Context/exception handling should not be blindly repaired into computable logic.
    if any(t in BRANCH_A_CONTEXT_TASKS for t in rescue_tasks) or has_any(
        terms,
        {
            "condition_context_present_without_handling",
            "exception_or_condition_clause_without_context_handling",
            "exception_clause_computable_despite_context",
            "computable_with_exception_context",
        },
    ):
        return base_plan_row(
            branch_to_update="A",
            criterion_id=criterion_id,
            execution_group="A_status_or_review",
            execution_priority=30,
            rescue_strategy="abstain_or_human_review_for_context_logic",
            rescue_task_type="condition_exception_context_review",
            requires_llm=0,
            requires_reverification=1,
            safe_to_apply_without_llm=0,
            blocked_from_auto_apply=1,
            source_row=representative,
            decision_row=a_decision,
            b_decision_row=b_decision,
            b_candidate_allowed=1 if b_allowed else 0,
            b_candidate_reason=b_reason,
            include_A_current=1,
            include_B_current=1 if b_allowed else 0,
            allow_new_llm_candidate=0,
            max_llm_attempts=0,
            selection_policy=(
                "Do not flatten condition/exception logic into a computable leaf. "
                "Preserve evidence and mark partial or human_review."
            ),
            why_deferred_or_blocked=(
                "Condition/exception context changes clinical logic. This should not be "
                "auto-repaired by a simple leaf-level prompt."
            ),
        )

    # 3. Duplicate/merge and broad semantic rescue are not good first-pass LLM rescue.
    if any(t in BRANCH_A_DEFER_TASKS for t in rescue_tasks):
        return base_plan_row(
            branch_to_update="A",
            criterion_id=criterion_id,
            execution_group="A_deferred_diagnostic",
            execution_priority=90,
            rescue_strategy="defer_broad_or_duplicate_task",
            rescue_task_type=task or "deferred_broad_task",
            requires_llm=0,
            requires_reverification=0,
            safe_to_apply_without_llm=0,
            blocked_from_auto_apply=1,
            source_row=representative,
            decision_row=a_decision,
            b_decision_row=b_decision,
            b_candidate_allowed=1 if b_allowed else 0,
            b_candidate_reason=b_reason,
            include_A_current=1,
            include_B_current=0,
            allow_new_llm_candidate=0,
            max_llm_attempts=0,
            selection_policy="Diagnostic only. Not part of main rescue execution.",
            why_deferred_or_blocked=(
                "This is a broad duplicate/semantic task, not a safe leaf-level rescue target."
            ),
        )

    # 4. No BERT anchor alone is not enough.
    if only_no_bert_anchor_problem(terms):
        return base_plan_row(
            branch_to_update="A",
            criterion_id=criterion_id,
            execution_group="A_deferred_diagnostic",
            execution_priority=80,
            rescue_strategy="defer_no_bert_anchor_only",
            rescue_task_type=task or "no_bert_anchor_only",
            requires_llm=0,
            requires_reverification=0,
            safe_to_apply_without_llm=0,
            blocked_from_auto_apply=1,
            source_row=representative,
            decision_row=a_decision,
            b_decision_row=b_decision,
            b_candidate_allowed=1 if b_allowed else 0,
            b_candidate_reason=b_reason,
            include_A_current=1,
            include_B_current=1 if b_allowed else 0,
            allow_new_llm_candidate=0,
            max_llm_attempts=0,
            selection_policy=(
                "Do not rescue based only on missing BERT anchor. Use as uncertainty flag."
            ),
            why_deferred_or_blocked=(
                "The only clear problem is no_bert_anchor. That is uncertainty, not enough "
                "evidence for automatic LLM rescue."
            ),
        )

    # 5. Actionable Branch A field problems.
    if any(t in BRANCH_A_ACTIONABLE_RESCUE_TASKS for t in rescue_tasks):
        return base_plan_row(
            branch_to_update="A",
            criterion_id=criterion_id,
            execution_group="A_main_llm_rescue",
            execution_priority=20,
            rescue_strategy="targeted_field_recovery_with_candidate_check",
            rescue_task_type=task,
            requires_llm=1,
            requires_reverification=1,
            safe_to_apply_without_llm=0,
            blocked_from_auto_apply=0,
            source_row=representative,
            decision_row=a_decision,
            b_decision_row=b_decision,
            b_candidate_allowed=1 if b_allowed else 0,
            b_candidate_reason=b_reason,
            include_A_current=1,
            include_B_current=1 if b_allowed else 0,
            allow_new_llm_candidate=1,
            max_llm_attempts=2,
            selection_policy=(
                "Generate or choose a candidate only if it improves the actionable field "
                "without losing clinical meaning. Reverify afterward."
            ),
            why_selected=(
                "Branch A has an actionable field-level problem: value/operator, temporal, "
                "history, list/range/scalar, or polarity."
            ),
        )

    # 6. Explicit entity problem can be rescued; no-anchor-only was already excluded.
    if any(t in BRANCH_A_ENTITY_RESCUE_TASKS for t in rescue_tasks) or has_any(terms, EXPLICIT_ENTITY_PROBLEM_TERMS):
        return base_plan_row(
            branch_to_update="A",
            criterion_id=criterion_id,
            execution_group="A_main_llm_rescue",
            execution_priority=25,
            rescue_strategy="entity_candidate_selection_not_anchor_only",
            rescue_task_type=task or "entity_regrounding",
            requires_llm=1,
            requires_reverification=1,
            safe_to_apply_without_llm=0,
            blocked_from_auto_apply=0,
            source_row=representative,
            decision_row=a_decision,
            b_decision_row=b_decision,
            b_candidate_allowed=1 if b_allowed else 0,
            b_candidate_reason=b_reason,
            include_A_current=1,
            include_B_current=1 if b_allowed else 0,
            allow_new_llm_candidate=1,
            max_llm_attempts=2,
            selection_policy=(
                "Fix entity only when the problem is explicit: not in evidence, generic, "
                "wrong type, or missing critical qualifier. Do not optimize only for shortest span."
            ),
            why_selected=(
                "Branch A has an explicit entity grounding/type problem, not just weak BERT support."
            ),
        )

    # 7. Remaining Branch A high risk becomes deferred, not automatic rescue.
    return base_plan_row(
        branch_to_update="A",
        criterion_id=criterion_id,
        execution_group="A_deferred_diagnostic",
        execution_priority=95,
        rescue_strategy="defer_non_actionable_high_risk",
        rescue_task_type=task or "non_actionable_high_risk",
        requires_llm=0,
        requires_reverification=0,
        safe_to_apply_without_llm=0,
        blocked_from_auto_apply=1,
        source_row=representative,
        decision_row=a_decision,
        b_decision_row=b_decision,
        b_candidate_allowed=1 if b_allowed else 0,
        b_candidate_reason=b_reason,
        include_A_current=1,
        include_B_current=0,
        allow_new_llm_candidate=0,
        max_llm_attempts=0,
        selection_policy="Do not rescue without actionable diagnosis.",
        why_deferred_or_blocked=(
            "Branch A is high risk, but no actionable rescue target was found. "
            "High risk alone is not enough."
        ),
    )


# ---------------------------------------------------------------------
# Branch B classification
# ---------------------------------------------------------------------

def classify_branch_b_candidate(
    *,
    criterion_id: str,
    rows_06d: List[Dict[str, Any]],
    b_decision: Dict[str, str],
    a_decision: Dict[str, str] | None,
    conservative_row: Dict[str, str] | None,
) -> Dict[str, Any]:
    """
    Clean Branch B planning.

    Branch B is not pruned again inside 06f.
    If a Branch B leaf is in the 06d mandatory verifier subset
    (P1_run_now_branch_b_mandatory_judge), it is routed to 06f2
    candidate-selection rescue.

    This is a verification/check step, not forced repair:
        - B_current remains the baseline candidate.
        - Branch A is never used as a Branch B replacement.
        - 06f2 may keep B_current, generate a better candidate,
          mark partial/non-computable, or send to human_review.
        - 06f2b later applies only validated and schema-safe changes.
    """
    representative = rows_06d[0] if rows_06d else {}

    rescue_tasks = sorted(
        set(rescue_task_from_row(r) for r in rows_06d if rescue_task_from_row(r))
    )
    task = rescue_tasks[0] if len(rescue_tasks) == 1 else unique_join(rescue_tasks)

    run_stages = {clean(r.get("run_stage")) for r in rows_06d}

    is_mandatory_branch_b_candidate = (
        "P1_run_now_branch_b_mandatory_judge" in run_stages
    )

    layer3_primary = lower(b_decision.get("layer3_primary_action"))
    routing = lower(b_decision.get("branch_b_final_routing_decision"))

    # ------------------------------------------------------------
    # 1. Clean Branch B main policy:
    #    all mandatory Branch B verifier rows go to candidate selection.
    # ------------------------------------------------------------
    if is_mandatory_branch_b_candidate:
        return base_plan_row(
            branch_to_update="B",
            criterion_id=criterion_id,
            execution_group="B_main_llm_rescue",
            execution_priority=25,
            rescue_strategy="mandatory_branch_b_candidate_selection",
            rescue_task_type=task or "branch_b_mandatory_verifier_check",
            requires_llm=1,
            requires_reverification=1,
            safe_to_apply_without_llm=0,
            blocked_from_auto_apply=0,
            source_row=representative,
            decision_row=b_decision,
            b_decision_row=b_decision,
            b_candidate_allowed=0,
            b_candidate_reason="branch_b_is_target_not_candidate_source",
            include_A_current=0,
            include_B_current=1,
            allow_new_llm_candidate=1,
            max_llm_attempts=2,
            selection_policy=(
                "Branch B is the target branch. Route this mandatory verifier leaf "
                "to candidate selection, but do not force a repair. Keep B_current "
                "unless a generated candidate is clearly better, schema-valid, and "
                "evidence-grounded. If the problem is context/exception scope or is "
                "uncertain, abstain with human_review. Do not use Branch A as a "
                "replacement for Branch B."
            ),
            why_selected=(
                "This Branch B leaf belongs to the mandatory verifier subset from 06d. "
                "It is sent to 06f2 for candidate-selection checking, not automatic repair."
            ),
        )

    # ------------------------------------------------------------
    # 2. Accept rows are kept unchanged if they appear in the 06d queue.
    #    Usually Branch B accept rows are not present in 06d, so this is mostly
    #    a reference safeguard.
    # ------------------------------------------------------------
    if (
        layer3_primary == "keep_without_layer2_action_candidate"
        or routing == "keep_without_layer2_action"
    ):
        return base_plan_row(
            branch_to_update="B",
            criterion_id=criterion_id,
            execution_group="B_accept_no_change_reference",
            execution_priority=5,
            rescue_strategy="keep_without_layer2_action_reference",
            rescue_task_type="none",
            requires_llm=0,
            requires_reverification=0,
            safe_to_apply_without_llm=0,
            blocked_from_auto_apply=0,
            source_row=representative,
            decision_row=b_decision,
            b_decision_row=b_decision,
            b_candidate_allowed=0,
            b_candidate_reason="not_applicable",
            include_A_current=0,
            include_B_current=1,
            allow_new_llm_candidate=0,
            max_llm_attempts=0,
            selection_policy="Keep Branch B leaf unchanged.",
            why_selected=(
                "Branch B verification routed this leaf to the high-support no-action reference group. "
                "This is not a calibrated proof of correctness."
            ),
        )

    # ------------------------------------------------------------
    # 3. Optional Branch B rows are not part of the main LLM rescue budget.
    # ------------------------------------------------------------
    return base_plan_row(
        branch_to_update="B",
        criterion_id=criterion_id,
        execution_group="B_blocked_or_manual_review",
        execution_priority=85,
        rescue_strategy="optional_branch_b_review_not_main_rescue",
        rescue_task_type=task or "optional_branch_b_review",
        requires_llm=0,
        requires_reverification=0,
        safe_to_apply_without_llm=0,
        blocked_from_auto_apply=1,
        source_row=representative,
        decision_row=b_decision,
        b_decision_row=b_decision,
        b_candidate_allowed=0,
        b_candidate_reason="not_applicable",
        include_A_current=0,
        include_B_current=1,
        allow_new_llm_candidate=0,
        max_llm_attempts=0,
        selection_policy=(
            "Do not spend the main Branch B LLM rescue budget on optional rows. "
            "Keep as optional review/diagnostic unless a separate ablation is planned."
        ),
        why_deferred_or_blocked=(
            "This Branch B row is not in the explicit 06d2 mandatory verifier subset."
        ),
    )


# ---------------------------------------------------------------------
# Optional manual-overlap diagnostic
# ---------------------------------------------------------------------

VALID_MANUAL_LABELS = {"correct", "partial", "incorrect"}
MANUAL_LABEL_SCORE = {"incorrect": 0, "partial": 1, "correct": 2}


def normalize_col_name(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def normalize_manual_label(x: Any) -> str:
    s = lower(x).replace(" ", "_")

    if s in {"correct", "ok", "valid", "true", "c"}:
        return "correct"

    if s in {"partial", "partially_correct", "partly_correct", "p"}:
        return "partial"

    if s in {"incorrect", "wrong", "error", "unsupported", "false", "i"}:
        return "incorrect"

    return ""


def read_csv_optional(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    return read_csv(path, required=False)


def detect_manual_label_col(rows: List[Dict[str, str]], branch: str) -> str:
    if not rows:
        return ""

    branch_lower = branch.lower()
    candidates = [
        f"manual_{branch}_leaf_label",
        f"manual_{branch_lower}_leaf_label",
        f"manual_{branch}_label",
        f"manual_{branch_lower}_label",
        f"{branch}_leaf_label",
        f"{branch_lower}_leaf_label",
        f"label_{branch}",
        f"label_{branch_lower}",
    ]

    norm_to_actual = {normalize_col_name(c): c for c in rows[0].keys()}

    for cand in candidates:
        key = normalize_col_name(cand)

        if key not in norm_to_actual:
            continue

        col = norm_to_actual[key]
        if sum(1 for row in rows if normalize_manual_label(row.get(col))) > 0:
            return col

    return ""


def detect_manual_issue_col(rows: List[Dict[str, str]], branch: str) -> str:
    if not rows:
        return ""

    branch_lower = branch.lower()
    candidates = [
        f"manual_{branch}_issue_type",
        f"manual_{branch_lower}_issue_type",
        f"{branch}_issue_type",
        f"{branch_lower}_issue_type",
        f"manual_issue_type_{branch}",
        f"manual_issue_type_{branch_lower}",
    ]

    norm_to_actual = {normalize_col_name(c): c for c in rows[0].keys()}

    for cand in candidates:
        key = normalize_col_name(cand)
        if key in norm_to_actual:
            return norm_to_actual[key]

    return ""


def manual_key(row: Dict[str, Any]) -> str:
    cid = criterion_id_from_row(row)
    if cid:
        return cid
    item_uid = clean(row.get("item_uid"))
    clause_id = clean(row.get("clause_id"))
    if item_uid and clause_id:
        return f"{item_uid}_{clause_id}"
    return ""


def compare_manual_b_to_a(a_label: str, b_label: str) -> str:
    if a_label not in VALID_MANUAL_LABELS or b_label not in VALID_MANUAL_LABELS:
        return "missing"
    if MANUAL_LABEL_SCORE[b_label] > MANUAL_LABEL_SCORE[a_label]:
        return "B_pre_better_than_A_pre"
    if MANUAL_LABEL_SCORE[b_label] < MANUAL_LABEL_SCORE[a_label]:
        return "B_pre_worse_than_A_pre"
    return "B_pre_same_as_A_pre"


def build_manual_index(rows: List[Dict[str, str]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    if not rows:
        return {}, {}

    col_a = detect_manual_label_col(rows, "A")
    col_b = detect_manual_label_col(rows, "B")
    issue_a = detect_manual_issue_col(rows, "A")
    issue_b = detect_manual_issue_col(rows, "B")

    detected = {
        "manual_A_label_column_detected": col_a,
        "manual_B_label_column_detected": col_b,
        "manual_A_issue_column_detected": issue_a,
        "manual_B_issue_column_detected": issue_b,
    }

    if not col_a or not col_b:
        return {}, detected

    out = {}

    for row in rows:
        cid = manual_key(row)
        if not cid:
            continue

        a_label = normalize_manual_label(row.get(col_a))
        b_label = normalize_manual_label(row.get(col_b))

        out[cid] = {
            "manual_A_pre_label": a_label,
            "manual_B_pre_label": b_label,
            "manual_A_pre_label_raw": clean(row.get(col_a)),
            "manual_B_pre_label_raw": clean(row.get(col_b)),
            "manual_A_pre_issue_type": clean(row.get(issue_a)) if issue_a else "",
            "manual_B_pre_issue_type": clean(row.get(issue_b)) if issue_b else "",
            "manual_B_vs_A_pre": compare_manual_b_to_a(a_label, b_label),
            "manual_A_pre_is_error": 1 if a_label in {"partial", "incorrect"} else 0,
            "manual_B_pre_is_error": 1 if b_label in {"partial", "incorrect"} else 0,
        }

    return out, detected


def annotate_plan_with_manual(rows: List[Dict[str, Any]], manual_index: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []

    for row in rows:
        new = dict(row)
        cid = clean(row.get("criterion_id"))
        manual = manual_index.get(cid, {})

        for key in [
            "manual_A_pre_label",
            "manual_B_pre_label",
            "manual_A_pre_label_raw",
            "manual_B_pre_label_raw",
            "manual_A_pre_issue_type",
            "manual_B_pre_issue_type",
            "manual_B_vs_A_pre",
            "manual_A_pre_is_error",
            "manual_B_pre_is_error",
        ]:
            new[key] = manual.get(key, "")

        branch = clean(row.get("branch_to_update"))
        if branch == "A":
            target_label = manual.get("manual_A_pre_label", "")
        elif branch == "B":
            target_label = manual.get("manual_B_pre_label", "")
        else:
            target_label = ""

        new["target_branch_pre_label"] = target_label
        new["target_branch_pre_is_error"] = 1 if target_label in {"partial", "incorrect"} else 0
        new["has_any_manual_pre_label"] = 1 if (
            manual.get("manual_A_pre_label") in VALID_MANUAL_LABELS
            or manual.get("manual_B_pre_label") in VALID_MANUAL_LABELS
        ) else 0

        out.append(new)

    return out


def summarize_manual_group(rows: List[Dict[str, Any]], group_keys: List[str], label_key: str) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        key = tuple(clean(row.get(k)) for k in group_keys)
        grouped[key].append(row)

    out = []

    for key, bucket in grouped.items():
        labels = [clean(r.get(label_key)) for r in bucket if clean(r.get(label_key)) in VALID_MANUAL_LABELS]
        counts = Counter(labels)

        row = {k: v for k, v in zip(group_keys, key)}
        row.update(
            {
                "n_plan_rows": len(bucket),
                "n_with_manual_label": len(labels),
                "manual_label_coverage_pct": pct(len(labels), len(bucket)),
                "correct_n": counts.get("correct", 0),
                "partial_n": counts.get("partial", 0),
                "incorrect_n": counts.get("incorrect", 0),
                "partial_or_incorrect_n": counts.get("partial", 0) + counts.get("incorrect", 0),
                "correct_pct_among_labeled": pct(counts.get("correct", 0), len(labels)),
                "partial_or_incorrect_pct_among_labeled": pct(
                    counts.get("partial", 0) + counts.get("incorrect", 0),
                    len(labels),
                ),
            }
        )
        out.append(row)

    return sorted(out, key=lambda r: (clean(r.get("branch_to_update")), -to_int(r.get("n_plan_rows"))))


def summarize_manual_error_coverage(
    main_annotated: List[Dict[str, Any]],
    manual_index: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows = []

    for branch in ["A", "B"]:
        label_key = f"manual_{branch}_pre_label"
        planned_ids = {clean(r.get("criterion_id")) for r in main_annotated if clean(r.get("branch_to_update")) == branch}

        manual_labeled = [
            (cid, m)
            for cid, m in manual_index.items()
            if m.get(label_key) in VALID_MANUAL_LABELS
        ]

        errors = [(cid, m) for cid, m in manual_labeled if m.get(label_key) in {"partial", "incorrect"}]
        corrects = [(cid, m) for cid, m in manual_labeled if m.get(label_key) == "correct"]

        errors_in_plan = sum(1 for cid, _m in errors if cid in planned_ids)
        corrects_in_plan = sum(1 for cid, _m in corrects if cid in planned_ids)

        branch_plan = [r for r in main_annotated if clean(r.get("branch_to_update")) == branch]
        target_labels = [clean(r.get("target_branch_pre_label")) for r in branch_plan]

        rows.append(
            {
                "branch": branch,
                "manual_labeled_rows": len(manual_labeled),
                "manual_error_rows_partial_or_incorrect": len(errors),
                "manual_error_rows_in_main_rescue": errors_in_plan,
                "manual_error_coverage_pct": pct(errors_in_plan, len(errors)),
                "manual_correct_rows": len(corrects),
                "manual_correct_rows_in_main_rescue": corrects_in_plan,
                "manual_correct_routed_pct": pct(corrects_in_plan, len(corrects)),
                "main_rescue_rows_for_branch": len(branch_plan),
                "main_rescue_rows_with_manual_label": sum(1 for x in target_labels if x in VALID_MANUAL_LABELS),
                "main_rescue_rows_pre_correct": sum(1 for x in target_labels if x == "correct"),
                "main_rescue_rows_pre_partial": sum(1 for x in target_labels if x == "partial"),
                "main_rescue_rows_pre_incorrect": sum(1 for x in target_labels if x == "incorrect"),
            }
        )

    return rows


def summarize_branch_a_b_candidate_manual(main_annotated: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = Counter()

    for row in main_annotated:
        if clean(row.get("branch_to_update")) != "A":
            continue
        key = (
            clean(row.get("manual_A_pre_label")) or "missing",
            clean(row.get("manual_B_pre_label")) or "missing",
            clean(row.get("manual_B_vs_A_pre")) or "missing",
        )
        grouped[key] += 1

    out = []
    for (a_label, b_label, b_vs_a), n in grouped.most_common():
        out.append(
            {
                "manual_A_pre_label": a_label,
                "manual_B_pre_label": b_label,
                "manual_B_vs_A_pre": b_vs_a,
                "n": n,
            }
        )
    return out


def run_manual_overlap_diagnostic(
    *,
    main_rows: List[Dict[str, Any]],
    all_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Retrospective diagnostic only.
    This function is intentionally called AFTER the plan has already been built.
    It cannot influence routing decisions.
    """
    if not MANUAL_PRE_LABELS_CSV.exists():
        return {
            "enabled": False,
            "reason": "manual_pre_labels_csv_not_found",
            "manual_csv": str(MANUAL_PRE_LABELS_CSV),
        }

    OUT_MANUAL_DIAG_DIR.mkdir(parents=True, exist_ok=True)

    manual_rows = read_csv_optional(MANUAL_PRE_LABELS_CSV)
    manual_index, detected_cols = build_manual_index(manual_rows)

    if not manual_index:
        return {
            "enabled": False,
            "reason": "manual_label_columns_not_detected_or_empty",
            "manual_csv": str(MANUAL_PRE_LABELS_CSV),
            "detected_columns": detected_cols,
        }

    main_annotated = annotate_plan_with_manual(main_rows, manual_index)
    all_annotated = annotate_plan_with_manual(all_rows, manual_index)

    by_branch_task = summarize_manual_group(
        main_annotated,
        ["branch_to_update", "rescue_task_type"],
        "target_branch_pre_label",
    )
    by_branch_strategy = summarize_manual_group(
        main_annotated,
        ["branch_to_update", "rescue_strategy"],
        "target_branch_pre_label",
    )
    error_coverage = summarize_manual_error_coverage(main_annotated, manual_index)
    branch_a_b_diag = summarize_branch_a_b_candidate_manual(main_annotated)

    write_csv(OUT_MANUAL_MAIN_ANNOTATED_CSV, main_annotated)
    write_csv(OUT_MANUAL_ALL_ANNOTATED_CSV, all_annotated)
    write_csv(OUT_MANUAL_BY_BRANCH_TASK_CSV, by_branch_task)
    write_csv(OUT_MANUAL_BY_BRANCH_STRATEGY_CSV, by_branch_strategy)
    write_csv(OUT_MANUAL_ERROR_COVERAGE_CSV, error_coverage)
    write_csv(OUT_MANUAL_A_B_CANDIDATE_CSV, branch_a_b_diag)

    target_labels = [clean(r.get("target_branch_pre_label")) or "missing" for r in main_annotated]
    branch_a_rows = [r for r in main_annotated if clean(r.get("branch_to_update")) == "A"]

    summary = {
        "enabled": True,
        "important_warning": (
            "Manual labels are used only for retrospective sanity checking. "
            "They are not used to decide rescue, calibrate thresholds, or select candidates. "
            "The plan has already been created before this diagnostic runs."
        ),
        "manual_csv": str(MANUAL_PRE_LABELS_CSV),
        "detected_columns": detected_cols,
        "outputs": {
            "main_annotated": str(OUT_MANUAL_MAIN_ANNOTATED_CSV),
            "all_annotated": str(OUT_MANUAL_ALL_ANNOTATED_CSV),
            "by_branch_task": str(OUT_MANUAL_BY_BRANCH_TASK_CSV),
            "by_branch_strategy": str(OUT_MANUAL_BY_BRANCH_STRATEGY_CSV),
            "manual_error_coverage": str(OUT_MANUAL_ERROR_COVERAGE_CSV),
            "branch_a_b_candidate_diagnostic": str(OUT_MANUAL_A_B_CANDIDATE_CSV),
            "summary_json": str(OUT_MANUAL_SUMMARY_JSON),
        },
        "main_plan_counts": {
            "rows_total": len(main_annotated),
            "rows_by_branch": dict(Counter(clean(r.get("branch_to_update")) for r in main_annotated)),
            "rows_with_any_manual_label": sum(to_int(r.get("has_any_manual_pre_label")) for r in main_annotated),
            "rows_with_target_branch_manual_label": sum(1 for x in target_labels if x in VALID_MANUAL_LABELS),
            "target_branch_pre_label_counts": dict(Counter(target_labels)),
            "rescue_task_counts": dict(Counter(clean(r.get("rescue_task_type")) for r in main_annotated)),
            "rescue_strategy_counts": dict(Counter(clean(r.get("rescue_strategy")) for r in main_annotated)),
        },
        "branch_a_main_rescue_manual_diagnostic": {
            "rows_total": len(branch_a_rows),
            "rows_with_A_and_B_manual_labels": sum(
                1
                for r in branch_a_rows
                if clean(r.get("manual_A_pre_label")) in VALID_MANUAL_LABELS
                and clean(r.get("manual_B_pre_label")) in VALID_MANUAL_LABELS
            ),
            "manual_B_vs_A_pre_counts": dict(Counter(clean(r.get("manual_B_vs_A_pre")) or "missing" for r in branch_a_rows)),
            "A_pre_label_counts": dict(Counter(clean(r.get("manual_A_pre_label")) or "missing" for r in branch_a_rows)),
            "B_pre_label_counts": dict(Counter(clean(r.get("manual_B_pre_label")) or "missing" for r in branch_a_rows)),
        },
        "manual_error_coverage_by_branch": error_coverage,
    }

    write_json(OUT_MANUAL_SUMMARY_JSON, summary)
    return summary


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLayer 3 rescue plan")
    print("Targeted rescue queue:", QUEUE_06D_JSONL)
    print("Decision inventory:", DECISION_CSV)
    print("Optional conservative inspection:", CONSERVATIVE_CSV)
    print("Branch B mandatory verifier subset:", BRANCH_B_MANDATORY_CSV)

    queue_rows = read_jsonl(QUEUE_06D_JSONL)
    decision_rows = read_csv(DECISION_CSV)
    conservative_rows = read_csv(CONSERVATIVE_CSV, required=False)
    branch_b_mandatory_rows = read_csv(BRANCH_B_MANDATORY_CSV, required=True)

    decision_index = build_decision_index(decision_rows)
    queue_grouped = build_06d_group_index(queue_rows)
    conservative_index = build_conservative_index(conservative_rows)

    branch_b_mandatory_ids = {
        criterion_id_from_row(row)
        for row in branch_b_mandatory_rows
        if criterion_id_from_row(row)
    }

    branch_a_main = []
    branch_a_deferred = []
    branch_a_status_review = []

    branch_b_main = []
    branch_b_safe = []
    branch_b_accept = []
    branch_b_blocked = []

    # ------------------------------
    # Branch A from 06d grouped rows
    # ------------------------------
    for (branch, criterion_id), rows in sorted(queue_grouped.items()):
        if branch != "A":
            continue

        a_decision = decision_index.get(("A", criterion_id), {})
        b_decision = decision_index.get(("B", criterion_id), {})
        conservative_row = conservative_index.get(("A", criterion_id))

        plan_row = classify_branch_a_candidate(
            criterion_id=criterion_id,
            rows_06d=rows,
            a_decision=a_decision,
            b_decision=b_decision,
            conservative_row=conservative_row,
        )

        group = plan_row["execution_group"]

        if group == "A_main_llm_rescue":
            # Branch A is the BERT/rules branch.
            # It is not repaired with a new LLM call.
            # 06F2 may use Branch B as semantic substitute only if Branch B is locally usable.
            plan_row["requires_llm"] = 0
            plan_row["allow_new_llm_candidate"] = 0
            plan_row["max_llm_attempts"] = 0
            plan_row["include_A_current"] = 1

            if int(plan_row.get("b_candidate_allowed", 0)) == 1:
                plan_row["include_B_current"] = 1
                plan_row["selection_policy"] = (
                    "Branch A is not repaired by a new LLM call. Use the corresponding "
                    "Branch B leaf as the semantic substitute only because Branch B passed "
                    "local usability checks. If the Branch B leaf is also routed to Branch B "
                    "De Jure rescue, 06F2 should use the final best validated Branch B leaf."
                )
                branch_a_main.append(plan_row)
            else:
                plan_row["execution_group"] = "A_status_or_review"
                plan_row["rescue_strategy"] = "branch_a_needs_rescue_but_branch_b_not_usable"
                plan_row["requires_llm"] = 0
                plan_row["requires_reverification"] = 0
                plan_row["safe_to_apply_without_llm"] = 0
                plan_row["blocked_from_auto_apply"] = 1
                plan_row["include_B_current"] = 0
                plan_row["allow_new_llm_candidate"] = 0
                plan_row["max_llm_attempts"] = 0
                plan_row["why_deferred_or_blocked"] = (
                    "Branch A has an actionable issue, but the matching Branch B leaf is not "
                    "locally usable as a substitute. Do not auto-apply substitution."
                )
                branch_a_status_review.append(plan_row)
        elif group == "A_status_or_review":
            branch_a_status_review.append(plan_row)
        else:
            branch_a_deferred.append(plan_row)

    # ------------------------------
    # Branch B De Jure subset.
    # IMPORTANT:
    # Do not infer the 96 rows indirectly from grouped 06D.
    # Read the explicit 06d2 mandatory verifier CSV and route every row.
    # ------------------------------
    for row in sorted(branch_b_mandatory_rows, key=lambda r: criterion_id_from_row(r)):
        criterion_id = criterion_id_from_row(row)
        if not criterion_id:
            continue

        b_decision = decision_index.get(("B", criterion_id), {})
        task = (
            rescue_task_from_row(row)
            or clean(row.get("suggested_rescue_task"))
            or "branch_b_mandatory_verifier_check"
        )

        plan_row = base_plan_row(
            branch_to_update="B",
            criterion_id=criterion_id,
            execution_group="B_main_llm_rescue",
            execution_priority=25,
            rescue_strategy="branch_b_dejure_judge_repair_loop",
            rescue_task_type=task,
            requires_llm=1,
            requires_reverification=1,
            safe_to_apply_without_llm=0,
            blocked_from_auto_apply=0,
            source_row=row,
            decision_row=b_decision,
            b_decision_row=b_decision,
            b_candidate_allowed=0,
            b_candidate_reason="branch_b_is_target_not_candidate_source",
            include_A_current=0,
            include_B_current=1,
            allow_new_llm_candidate=1,
            max_llm_attempts=3,
            selection_policy=(
                "Branch B follows a De Jure-style judge-and-repair loop. "
                "Judge B_current first. If it passes, keep it. If it fails, "
                "repair/regenerate up to 3 attempts, judge each candidate, and keep "
                "the best validated candidate. Never use Branch A as replacement."
            ),
            why_selected=(
                "This Branch B leaf belongs to the explicit 06d2 mandatory verifier subset. "
                "It is routed to De Jure-style judge/repair, not automatic replacement."
            ),
        )
        branch_b_main.append(plan_row)

    # ------------------------------
    # Branch B optional/non-mandatory rows from 06d.
    # These stay as diagnostics and are not sent to the main LLM budget.
    # ------------------------------
    for (branch, criterion_id), rows in sorted(queue_grouped.items()):
        if branch != "B":
            continue

        if criterion_id in branch_b_mandatory_ids:
            continue

        representative = rows[0] if rows else {}
        b_decision = decision_index.get(("B", criterion_id), {})

        plan_row = base_plan_row(
            branch_to_update="B",
            criterion_id=criterion_id,
            execution_group="B_blocked_or_manual_review",
            execution_priority=85,
            rescue_strategy="optional_branch_b_review_not_main_rescue",
            rescue_task_type=rescue_task_from_row(representative) or "optional_branch_b_review",
            requires_llm=0,
            requires_reverification=0,
            safe_to_apply_without_llm=0,
            blocked_from_auto_apply=1,
            source_row=representative,
            decision_row=b_decision,
            b_decision_row=b_decision,
            b_candidate_allowed=0,
            b_candidate_reason="not_applicable",
            include_A_current=0,
            include_B_current=1,
            allow_new_llm_candidate=0,
            max_llm_attempts=0,
            selection_policy=(
                "Not part of the main Branch B De Jure rescue budget. "
                "Keep as optional review/diagnostic."
            ),
            why_deferred_or_blocked=(
                "This Branch B row is not in the explicit 06d2 mandatory verifier subset."
            ),
        )
        branch_b_blocked.append(plan_row)

    main_llm = sorted(
        branch_a_main + branch_b_main,
        key=lambda r: (
            to_int(r.get("execution_priority"), 999),
            clean(r.get("branch_to_update")),
            clean(r.get("criterion_id")),
        ),
    )

    all_rows = sorted(
        main_llm
        + branch_a_deferred
        + branch_a_status_review
        + branch_b_safe
        + branch_b_accept
        + branch_b_blocked,
        key=lambda r: (
            to_int(r.get("execution_priority"), 999),
            clean(r.get("execution_group")),
            clean(r.get("branch_to_update")),
            clean(r.get("criterion_id")),
        ),
    )

    write_jsonl(OUT_MAIN_JSONL, main_llm)
    write_csv(OUT_MAIN_CSV, main_llm)

    write_csv(OUT_BRANCH_A_MAIN_CSV, branch_a_main)
    write_csv(OUT_BRANCH_A_DEFERRED_CSV, branch_a_deferred)
    write_csv(OUT_BRANCH_A_STATUS_REVIEW_CSV, branch_a_status_review)

    write_csv(OUT_BRANCH_B_MAIN_CSV, branch_b_main)
    write_csv(OUT_BRANCH_B_SAFE_CSV, branch_b_safe)
    write_csv(OUT_BRANCH_B_ACCEPT_CSV, branch_b_accept)
    write_csv(OUT_BRANCH_B_BLOCKED_CSV, branch_b_blocked)

    write_jsonl(OUT_ALL_JSONL, all_rows)
    write_csv(OUT_ALL_CSV, all_rows)

    counts_by_group = Counter(r["execution_group"] for r in all_rows)
    counts_by_branch = Counter(r["branch_to_update"] for r in all_rows)
    main_counts_by_branch = Counter(r["branch_to_update"] for r in main_llm)
    counts_by_strategy = Counter(r["rescue_strategy"] for r in all_rows)
    counts_by_task_main = Counter(r["rescue_task_type"] for r in main_llm)

    manual_overlap_diagnostic = run_manual_overlap_diagnostic(
        main_rows=main_llm,
        all_rows=all_rows,
    )

    summary = {
        "stage": "06f_layer3_literature_aligned_rescue_plan",
        "description": (
            "Strict rescue planning. High risk alone is not enough. "
            "Manual labels are not used as the repair baseline. "
            "Branch A rescue is limited to actionable failures. "
            "Branch B routes the full mandatory verifier subset to candidate-selection checking, not forced repair."
        ),
        "inputs": {
            "queue_06d_jsonl": str(QUEUE_06D_JSONL),
            "decision_csv": str(DECISION_CSV),
            "conservative_csv": str(CONSERVATIVE_CSV),
            "branch_b_mandatory_csv": str(BRANCH_B_MANDATORY_CSV),
        },
        "outputs": {
            "main_llm_rescue_plan_jsonl": str(OUT_MAIN_JSONL),
            "main_llm_rescue_plan_csv": str(OUT_MAIN_CSV),
            "all_plan_csv": str(OUT_ALL_CSV),
            "summary_json": str(OUT_SUMMARY_JSON),
            "manual_overlap_diagnostic_summary_json": str(OUT_MANUAL_SUMMARY_JSON),
        },
        "counts": {
            "input_06d_rows": len(queue_rows),
            "input_branch_b_mandatory_rows": len(branch_b_mandatory_rows),
            "input_branch_b_mandatory_unique_criteria": len(branch_b_mandatory_ids),
            "input_06d_grouped_branch_a_criteria": len(
                [1 for (branch, _cid) in queue_grouped if branch == "A"]
            ),
            "input_06d_grouped_branch_b_criteria": len(
                [1 for (branch, _cid) in queue_grouped if branch == "B"]
            ),

            "main_rescue_plan_rows": len(main_llm),
            "main_rescue_plan_by_branch": dict(main_counts_by_branch),
            "main_rescue_plan_requires_llm_rows": sum(
                1 for r in main_llm if int(r.get("requires_llm", 0)) == 1
            ),
            "main_rescue_plan_non_llm_rows": sum(
                1 for r in main_llm if int(r.get("requires_llm", 0)) == 0
            ),
            "main_llm_execution_rows_by_branch": dict(
                Counter(
                    r["branch_to_update"]
                    for r in main_llm
                    if int(r.get("requires_llm", 0)) == 1
                )
            ),

            "branch_a_main_llm_rescue_rows": len(branch_a_main),
            "branch_a_deferred_diagnostic_rows": len(branch_a_deferred),
            "branch_a_status_or_review_rows": len(branch_a_status_review),

            "branch_b_main_llm_rescue_rows": len(branch_b_main),
            "branch_b_safe_non_llm_reference_rows": len(branch_b_safe),
            "branch_b_accept_no_change_reference_rows": len(branch_b_accept),
            "branch_b_blocked_or_manual_review_rows": len(branch_b_blocked),

            "all_plan_rows": len(all_rows),
            "counts_by_group": dict(counts_by_group),
            "counts_by_branch": dict(counts_by_branch),
            "counts_by_strategy": dict(counts_by_strategy),
            "main_rescue_task_counts": dict(counts_by_task_main),
        },
        "manual_overlap_diagnostic": manual_overlap_diagnostic,
        "method_notes": [
            "Manual labels are read only after the plan is created, for retrospective overlap diagnostics.",
            "Manual labels are not used as a baseline, driver, threshold, or candidate selector.",
            "Branch A high-risk leaves are not automatically rescued.",
            "No-BERT-anchor-only Branch A cases are deferred because that is uncertainty, not an actionable repair target.",
            "Condition/exception cases are not blindly repaired into computable leaves.",
            "Branch B rescue no longer depends on 06e/06e2. The full 06d mandatory verifier subset is routed to 06f2 candidate-selection checking.",
            "Branch B candidate selection is not forced repair: 06f2 may keep B_current, generate a better candidate, mark partial/non-computable, or abstain.",
            "06f only prepares the plan. It does not call the LLM and does not modify ASTs.",
            "All later applied changes must be re-run through Layer 1 and Layer 2.",
        ],
    }

    write_json(OUT_SUMMARY_JSON, summary)

    print("\nDONE")
    print("Main LLM rescue JSONL:", OUT_MAIN_JSONL)
    print("Main LLM rescue CSV:", OUT_MAIN_CSV)
    print("All plan CSV:", OUT_ALL_CSV)
    print("Summary JSON:", OUT_SUMMARY_JSON)

    print("\nCounts:")
    print("Input queue rows:", len(queue_rows))
    print(
        "Branch A grouped queue criteria:",
        summary["counts"]["input_06d_grouped_branch_a_criteria"],
    )
    print(
        "Branch B grouped queue criteria:",
        summary["counts"]["input_06d_grouped_branch_b_criteria"],
    )
    print("Branch B mandatory verifier rows:", len(branch_b_mandatory_rows))
    print("Branch B mandatory verifier unique criteria:", len(branch_b_mandatory_ids))
    print("Main rescue plan rows:", len(main_llm))
    print(
        "Main rescue plan rows requiring LLM:",
        sum(1 for r in main_llm if int(r.get("requires_llm", 0)) == 1),
    )
    print(
        "Main rescue plan rows not requiring LLM:",
        sum(1 for r in main_llm if int(r.get("requires_llm", 0)) == 0),
    )
    print("Main LLM rescue by branch:", dict(main_counts_by_branch))
    print("Branch A main:", len(branch_a_main))
    print("Branch A deferred:", len(branch_a_deferred))
    print("Branch A status/review:", len(branch_a_status_review))
    print("Branch B main:", len(branch_b_main))
    print("Branch B safe non-LLM reference:", len(branch_b_safe))
    print("Branch B accept/no-change reference:", len(branch_b_accept))
    print("Branch B blocked/manual review:", len(branch_b_blocked))
    print("Main rescue task counts:", dict(counts_by_task_main))

    print("\nManual-overlap diagnostic:")
    print("Enabled:", manual_overlap_diagnostic.get("enabled"))
    if manual_overlap_diagnostic.get("enabled"):
        print("Target-branch pre-label counts:", manual_overlap_diagnostic["main_plan_counts"]["target_branch_pre_label_counts"])
        print("Manual error coverage by branch:", manual_overlap_diagnostic["manual_error_coverage_by_branch"])
        print("Diagnostic summary:", OUT_MANUAL_SUMMARY_JSON)
    else:
        print("Reason:", manual_overlap_diagnostic.get("reason"))


if __name__ == "__main__":
    main()

# Run from the repository root:
# python scripts/03_verification/03_layer3/06_prepare_rescue_plan.py