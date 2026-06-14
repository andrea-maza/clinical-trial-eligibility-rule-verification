"""
04_build_targeted_rescue_candidates.py

Build the broad Layer 3 queue of leaves that may require targeted LLM
judgment, targeted recovery, or later human review.

The script combines the Layer 3 decision inventory with any available
conservative-issue inspection. When the separate conservative file is
absent, conservative issues are recovered directly from the Layer 1
issue codes stored in the decision inventory.

The script assigns:
    - candidate type
    - rescue task
    - execution stage
    - whether judgment is required before repair
    - structured prompt input

This script does not call the LLM, modify logical rule trees, apply
repairs, or accept rescued outputs.

Outputs:
    outputs/verification/layer3/targeted_rescue_candidates/
        layer3_targeted_llm_rescue_candidates.jsonl
        layer3_targeted_llm_rescue_candidates.csv
        layer3_targeted_llm_rescue_candidates_summary.json

Run from the repository root:
python scripts/03_verification/03_layer3/04_build_targeted_rescue_candidates.py
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[3]

DECISION_CSV = (
    ROOT
    / "outputs"
    / "verification"
    / "layer3"
    / "decision_inventory"
    / "layer3_decision_inventory_leaf_level.csv"
)

# Optional input. When absent, conservative issues are obtained directly
# from all_layer1_codes in the decision inventory.
CONSERVATIVE_CSV = (
    ROOT
    / "outputs"
    / "verification"
    / "layer3"
    / "conservative_downgrade_inspection"
    / "layer3_conservative_downgrade_candidates.csv"
)

OUT_DIR = (
    ROOT
    / "outputs"
    / "verification"
    / "layer3"
    / "targeted_rescue_candidates"
)

OUT_JSONL = OUT_DIR / "layer3_targeted_llm_rescue_candidates.jsonl"
OUT_CSV = OUT_DIR / "layer3_targeted_llm_rescue_candidates.csv"
OUT_SUMMARY_JSON = OUT_DIR / "layer3_targeted_llm_rescue_candidates_summary.json"

LLM_ACTIONS = {
    "targeted_llm_rescue_candidate",
    "optional_targeted_rescue_or_review",
}

CONSERVATIVE_ISSUES = {
    "list_operator_without_list_value",
    "comparison_without_scalar_value",
    "range_with_both_bounds_missing",
}

RUN_STAGE_ORDER = {
    "P1_run_now_conservative_structural_recovery": 1,
    "P1_run_now_branch_b_mandatory_judge": 2,
    "P2_budget_allows_branch_b_optional_judge": 3,
    "P3_branch_a_mandatory_diagnostic_rescue_or_ablation": 4,
    "P4_branch_a_optional_diagnostic_rescue": 5,
}


# ---------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------

def read_csv(path: Path, required: bool = True) -> List[Dict[str, str]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"CSV not found: {path}")
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


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
        path.write_text("", encoding="utf-8")
        return

    priority = [
        "candidate_id",
        "branch",
        "criterion_id",
        "document_id",
        "run_stage",
        "candidate_kind",
        "rescue_task_type",
        "requires_judge_first",
        "expected_llm_action",
        "max_attempts",
        "requires_reverification_after_llm",

        "entity_text",
        "entity_type",
        "operator",
        "value_type",
        "value",
        "unit",
        "computability",
        "evidence_text",

        "layer3_primary_action",
        "layer3_action_family",
        "branch_b_final_routing_decision",
        "branch_b_semantic_grounding_support",
        "branch_b_semantic_grounding_risk_label",
        "branch_a_leaf_support",
        "branch_a_risk_label",

        "conservative_issues",
        "all_layer1_codes",
        "diagnosis_summary",
        "prompt_input_json",
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


def split_codes(x: Any) -> List[str]:
    if x is None:
        return []

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

    return [p.strip() for p in re.split(r"[;|]", s) if p.strip()]


def normalize_issue_codes(codes: List[str]) -> List[str]:
    return [
        code.split(":", 1)[-1].strip()
        for code in codes
        if code.strip()
    ]


def criterion_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return (clean(row.get("branch")), clean(row.get("criterion_id")))


def conservative_issues_from_codes(codes_text: Any) -> List[str]:
    codes = normalize_issue_codes(split_codes(codes_text))
    return sorted(set(codes) & CONSERVATIVE_ISSUES)


def build_conservative_index(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, str]]:
    index = {}

    for r in rows:
        branch = clean(r.get("branch"))
        criterion_id = clean(r.get("criterion_id"))

        if branch and criterion_id:
            index[(branch, criterion_id)] = r

    return index


def current_leaf_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "entity_type": clean(row.get("entity_type")),
        "entity_text": clean(row.get("entity_text")),
        "operator": clean(row.get("operator")),
        "value_type": clean(row.get("value_type")),
        "value": clean(row.get("value")),
        "unit": clean(row.get("unit")),
        "computability": clean(row.get("computability")),
        "evidence_text": clean(row.get("evidence_text")),
    }


def collect_reasons(row: Dict[str, Any]) -> List[str]:
    reason_cols = [
        "layer1_policy_reasons",
        "branch_a_risk_reasons",
        "branch_b_semantic_grounding_reasons",
        "branch_b_execution_reasons",
        "branch_b_routing_reasons",
    ]

    reasons: List[str] = []

    for col in reason_cols:
        reasons.extend(split_codes(row.get(col, "")))

    return sorted(set(r for r in reasons if r))


def collect_all_diagnosis_terms(row: Dict[str, Any], conservative_issues: List[str]) -> List[str]:
    codes = normalize_issue_codes(split_codes(row.get("all_layer1_codes", "")))
    reasons = collect_reasons(row)

    all_terms = codes + reasons + conservative_issues
    return sorted(set(t for t in all_terms if t))


def contains_any(text: str, patterns: List[str]) -> bool:
    return any(p in text for p in patterns)


def infer_rescue_task_type(
    row: Dict[str, Any],
    conservative_issues: List[str],
) -> str:
    """
    Map diagnosis to a targeted rescue task.
    """
    branch = clean(row.get("branch"))
    joined = " ".join(collect_all_diagnosis_terms(row, conservative_issues)).lower()

    # Conservative structural issues first.
    if "list_operator_without_list_value" in conservative_issues:
        return "list_value_recovery"

    if "range_with_both_bounds_missing" in conservative_issues:
        return "range_bounds_recovery"

    if "comparison_without_scalar_value" in conservative_issues:
        return "scalar_value_recovery"

    # Field-specific semantic issues.
    if contains_any(joined, [
        "entity_not_grounded",
        "entity_not_in_evidence",
        "entity_not_grounded_in_evidence_or_item",
        "entity_text_empty",
        "generic_entity_text",
        "no_bert_anchor",
        "entity_partially_supported",
        "entity_supported_by_source_token_overlap",
    ]):
        return "entity_regrounding"

    if contains_any(joined, [
        "comparison_without_scalar_value",
        "operator_value_not_structurally_supported",
        "value_not_grounded",
        "value_missing",
        "quantitative_cue_not_represented",
        "quantitative_cue_unhandled",
        "equality_without_value",
        "pattern_operator_without_scalar_value",
    ]):
        return "value_or_operator_recovery"

    if contains_any(joined, [
        "temporal_marker_without_temporal_context",
        "temporal_context_missing",
        "temporal_anchor_mismatch",
        "duration_marker_missing",
    ]):
        return "temporal_context_recovery"

    if contains_any(joined, [
        "history_marker_without_history_context",
        "history_context_invalid",
        "history_context_missing",
    ]):
        return "history_context_recovery"

    if contains_any(joined, [
        "condition_context_present",
        "condition_or_exception",
        "exception_context",
        "exception_clause",
        "computable_with_exception_context",
        "condition_exception",
        "exception_or_condition_clause_without_context_handling",
        "exception_clause_computable_despite_context",
    ]):
        return "condition_exception_context_recovery"

    if contains_any(joined, [
        "negation_clause_with_exists_operator",
        "positive_clause_with_not_exists_operator",
        "negative_entity_with_not_exists_operator",
        "negation",
        "not_exists",
        "polarity",
    ]):
        return "polarity_negation_repair"

    if contains_any(joined, [
        "entity_type_invalid",
        "entity_type_other",
        "entity_type_not_supported",
        "bad_entity_type",
    ]):
        return "entity_type_reclassification"

    if contains_any(joined, [
        "duplicate_identical_leaf",
        "duplicate",
    ]):
        return "duplicate_prune_or_merge_judge"

    if branch == "B":
        return "branch_b_field_level_semantic_judge"

    return "branch_a_field_level_semantic_rescue"


def infer_candidate_kind(
    row: Dict[str, Any],
    conservative_issues: List[str],
) -> str:
    primary = lower(row.get("layer3_primary_action"))

    if conservative_issues:
        return "conservative_structural_recovery_before_downgrade"

    if primary == "targeted_llm_rescue_candidate":
        return "mandatory_llm_rescue_or_judge"

    if primary == "optional_targeted_rescue_or_review":
        return "optional_llm_rescue_or_review"

    return "other"


def infer_run_stage(
    row: Dict[str, Any],
    conservative_issues: List[str],
) -> str:
    branch = clean(row.get("branch"))
    primary = lower(row.get("layer3_primary_action"))

    if conservative_issues:
        return "P1_run_now_conservative_structural_recovery"

    if branch == "B" and primary == "targeted_llm_rescue_candidate":
        return "P1_run_now_branch_b_mandatory_judge"

    if branch == "B" and primary == "optional_targeted_rescue_or_review":
        return "P2_budget_allows_branch_b_optional_judge"

    if branch == "A" and primary == "targeted_llm_rescue_candidate":
        return "P3_branch_a_mandatory_diagnostic_rescue_or_ablation"

    if branch == "A" and primary == "optional_targeted_rescue_or_review":
        return "P4_branch_a_optional_diagnostic_rescue"

    return "P9_not_scheduled"


def infer_requires_judge_first(
    row: Dict[str, Any],
    conservative_issues: List[str],
    task_type: str,
) -> int:
    branch = clean(row.get("branch"))

    # Branch B should first be judged field-by-field, because Branch B is already LLM-generated.
    if branch == "B":
        return 1

    # Conservative structural issues can go directly to targeted recovery.
    if conservative_issues:
        return 0

    # Broad Branch A semantic failures should be judged before repair.
    if task_type in {
        "branch_a_field_level_semantic_rescue",
        "duplicate_prune_or_merge_judge",
        "condition_exception_context_recovery",
    }:
        return 1

    return 0


def infer_expected_llm_action(task_type: str, requires_judge_first: int) -> str:
    if requires_judge_first:
        return "judge_fields_then_return_repair_decision"

    if task_type in {
        "list_value_recovery",
        "range_bounds_recovery",
        "scalar_value_recovery",
        "value_or_operator_recovery",
        "temporal_context_recovery",
        "history_context_recovery",
        "entity_regrounding",
        "entity_type_reclassification",
        "polarity_negation_repair",
    }:
        return "targeted_field_recovery"

    return "targeted_leaf_rescue"


def build_task_instructions(task_type: str) -> List[str]:
    common = [
        "Use only the provided evidence_text and current leaf fields.",
        "Do not invent clinical facts.",
        "Prefer exact substrings from evidence_text for any repaired text field.",
        "If the field cannot be recovered from evidence_text, return status='no_repair_possible'.",
        "Return a structured JSON object only.",
        "Any repaired leaf must later be re-run through Layer 1 and Layer 2 verification.",
    ]

    task_specific = {
        "list_value_recovery": [
            "The current operator suggests a list relation, but the list value is missing.",
            "Recover the list values explicitly stated in evidence_text.",
            "If no list values are explicitly present, do not downgrade silently; return no_repair_possible.",
        ],
        "range_bounds_recovery": [
            "The current leaf uses a range-like value_type, but both bounds are missing.",
            "Recover lower_bound and/or upper_bound only if explicitly stated in evidence_text.",
            "If no bounds are present, return no_repair_possible.",
        ],
        "scalar_value_recovery": [
            "The current leaf has a comparison operator but no scalar value.",
            "Recover the numeric scalar value and unit from evidence_text.",
            "If no numeric threshold is present, return no_repair_possible.",
        ],
        "entity_regrounding": [
            "The current entity_text is weakly grounded or not found in the evidence.",
            "Re-extract entity_text as a verbatim span from evidence_text.",
            "If no valid entity span exists, return no_repair_possible.",
        ],
        "value_or_operator_recovery": [
            "Check whether the operator and value are supported by evidence_text.",
            "Recover the missing value/unit or propose a corrected operator only if explicitly supported.",
        ],
        "temporal_context_recovery": [
            "Recover temporal relation, value, unit, and anchor_event from evidence_text.",
            "Use null for fields that are not explicitly recoverable.",
        ],
        "history_context_recovery": [
            "Recover history context only if evidence_text explicitly refers to prior/current/history status.",
        ],
        "condition_exception_context_recovery": [
            "Identify whether evidence_text contains condition, exception, allowance, or qualifier logic.",
            "Do not flatten exception logic into a simple computable rule if the scope is ambiguous.",
            "Return mark_partial_or_non_computable if exception logic is unresolved.",
        ],
        "polarity_negation_repair": [
            "Check whether the leaf polarity/operator matches negation in evidence_text.",
            "Correct exists/not_exists only if evidence_text clearly supports the correction.",
        ],
        "entity_type_reclassification": [
            "Classify the entity into the closest valid schema entity_type.",
            "Keep entity_type='other' only if no valid type fits.",
        ],
        "duplicate_prune_or_merge_judge": [
            "Determine whether this leaf duplicates another leaf or should be retained.",
            "Do not delete; return a recommendation only.",
        ],
        "branch_b_field_level_semantic_judge": [
            "Judge whether the Branch B LLM-generated leaf is semantically faithful to evidence_text.",
            "Identify incorrect, unsupported, incomplete, or over-specific fields.",
            "Return accept, minor_fix, reextract_leaf, mark_partial_or_non_computable, or human_review.",
        ],
        "branch_a_field_level_semantic_rescue": [
            "Judge whether the Branch A BERT/rules leaf is semantically faithful to evidence_text.",
            "Because Branch A is not safe for automatic acceptance, use this as diagnostic rescue only.",
        ],
    }

    return common + task_specific.get(task_type, [])


def build_required_output_schema(task_type: str, requires_judge_first: int) -> Dict[str, Any]:
    if requires_judge_first:
        return {
            "status": "accept | minor_fix | reextract_leaf | mark_partial_or_non_computable | human_review | no_repair_possible",
            "field_judgments": {
                "entity_text": "correct | incorrect | unsupported | incomplete | not_applicable",
                "entity_type": "correct | incorrect | unsupported | incomplete | not_applicable",
                "operator": "correct | incorrect | unsupported | incomplete | not_applicable",
                "value": "correct | incorrect | unsupported | incomplete | not_applicable",
                "unit": "correct | incorrect | unsupported | incomplete | not_applicable",
                "temporal_context": "correct | incorrect | unsupported | incomplete | not_applicable",
                "history_context": "correct | incorrect | unsupported | incomplete | not_applicable",
                "condition_exception_context": "correct | incorrect | unsupported | incomplete | not_applicable",
                "computability": "correct | incorrect | unsupported | incomplete | not_applicable",
            },
            "repaired_fields": {},
            "supporting_spans": [],
            "unresolved_reasons": [],
            "verification_note": "brief explanation grounded in evidence_text",
        }

    if task_type == "list_value_recovery":
        return {
            "status": "repaired | no_repair_possible | human_review",
            "repaired_fields": {
                "value_type": "list",
                "value": ["..."],
            },
            "supporting_spans": ["exact evidence span"],
            "unresolved_reasons": [],
        }

    if task_type == "range_bounds_recovery":
        return {
            "status": "repaired | no_repair_possible | human_review",
            "repaired_fields": {
                "value_type": "range",
                "value": {
                    "lower": None,
                    "upper": None,
                },
                "unit": None,
            },
            "supporting_spans": ["exact evidence span"],
            "unresolved_reasons": [],
        }

    if task_type == "scalar_value_recovery":
        return {
            "status": "repaired | no_repair_possible | human_review",
            "repaired_fields": {
                "value_type": "scalar",
                "value": None,
                "unit": None,
            },
            "supporting_spans": ["exact evidence span"],
            "unresolved_reasons": [],
        }

    return {
        "status": "repaired | no_repair_possible | human_review | mark_partial_or_non_computable",
        "repaired_fields": {},
        "supporting_spans": ["exact evidence span if available"],
        "unresolved_reasons": [],
        "verification_note": "brief explanation grounded in evidence_text",
    }


def build_prompt_input(
    row: Dict[str, Any],
    task_type: str,
    requires_judge_first: int,
    conservative_issues: List[str],
) -> Dict[str, Any]:
    return {
        "task_type": task_type,
        "branch": clean(row.get("branch")),
        "document_id": clean(row.get("document_id")),
        "criterion_id": clean(row.get("criterion_id")),
        "evidence_text": clean(row.get("evidence_text")),
        "current_leaf": current_leaf_from_row(row),
        "diagnosis": {
            "layer3_primary_action": clean(row.get("layer3_primary_action")),
            "layer3_action_family": clean(row.get("layer3_action_family")),
            "all_layer1_codes": normalize_issue_codes(split_codes(row.get("all_layer1_codes", ""))),
            "layer1_policy_reasons": split_codes(row.get("layer1_policy_reasons", "")),
            "branch_a_risk_label": clean(row.get("branch_a_risk_label")),
            "branch_a_risk_reasons": split_codes(row.get("branch_a_risk_reasons", "")),
            "branch_b_final_routing_decision": clean(row.get("branch_b_final_routing_decision")),
            "branch_b_semantic_grounding_risk_label": clean(row.get("branch_b_semantic_grounding_risk_label")),
            "branch_b_semantic_grounding_reasons": split_codes(row.get("branch_b_semantic_grounding_reasons", "")),
            "branch_b_execution_risk_label": clean(row.get("branch_b_execution_risk_label")),
            "branch_b_execution_reasons": split_codes(row.get("branch_b_execution_reasons", "")),
            "conservative_issues": conservative_issues,
        },
        "instructions": build_task_instructions(task_type),
        "required_output_schema": build_required_output_schema(task_type, requires_judge_first),
    }


def diagnosis_summary(row: Dict[str, Any], conservative_issues: List[str]) -> str:
    terms = collect_all_diagnosis_terms(row, conservative_issues)
    return ";".join(terms)


def should_include_candidate(row: Dict[str, Any], conservative_issues: List[str]) -> bool:
    primary = lower(row.get("layer3_primary_action"))

    if conservative_issues:
        return True

    if primary in LLM_ACTIONS:
        return True

    return False


def make_candidate(row: Dict[str, Any], conservative_row: Dict[str, str] | None) -> Dict[str, Any]:
    branch = clean(row.get("branch"))
    criterion_id = clean(row.get("criterion_id"))

    conservative_issues = []

    if conservative_row:
        conservative_issues = split_codes(conservative_row.get("conservative_issues", ""))

    if not conservative_issues:
        conservative_issues = conservative_issues_from_codes(row.get("all_layer1_codes", ""))

    task_type = infer_rescue_task_type(row, conservative_issues)
    candidate_kind = infer_candidate_kind(row, conservative_issues)
    run_stage = infer_run_stage(row, conservative_issues)
    requires_judge_first = infer_requires_judge_first(row, conservative_issues, task_type)
    expected_llm_action = infer_expected_llm_action(task_type, requires_judge_first)

    prompt_input = build_prompt_input(
        row=row,
        task_type=task_type,
        requires_judge_first=requires_judge_first,
        conservative_issues=conservative_issues,
    )

    candidate_id = f"{branch}__{criterion_id}__{task_type}"

    return {
        "candidate_id": candidate_id,
        "branch": branch,
        "criterion_id": criterion_id,
        "document_id": clean(row.get("document_id")),
        "run_stage": run_stage,
        "run_stage_order": RUN_STAGE_ORDER.get(run_stage, 99),
        "candidate_kind": candidate_kind,
        "rescue_task_type": task_type,
        "requires_judge_first": requires_judge_first,
        "expected_llm_action": expected_llm_action,
        "max_attempts": 3 if branch == "B" else 2,
        "requires_reverification_after_llm": 1,

        "entity_text": clean(row.get("entity_text")),
        "entity_type": clean(row.get("entity_type")),
        "operator": clean(row.get("operator")),
        "value_type": clean(row.get("value_type")),
        "value": clean(row.get("value")),
        "unit": clean(row.get("unit")),
        "computability": clean(row.get("computability")),
        "evidence_text": clean(row.get("evidence_text")),

        "layer3_primary_action": clean(row.get("layer3_primary_action")),
        "layer3_action_family": clean(row.get("layer3_action_family")),

        "branch_b_final_routing_decision": clean(row.get("branch_b_final_routing_decision")),
        "branch_b_semantic_grounding_support": clean(row.get("branch_b_semantic_grounding_support")),
        "branch_b_semantic_grounding_risk_label": clean(row.get("branch_b_semantic_grounding_risk_label")),
        "branch_a_leaf_support": clean(row.get("branch_a_leaf_support")),
        "branch_a_risk_label": clean(row.get("branch_a_risk_label")),

        "conservative_issues": ";".join(conservative_issues),
        "all_layer1_codes": clean(row.get("all_layer1_codes")),
        "diagnosis_summary": diagnosis_summary(row, conservative_issues),
        "prompt_input": prompt_input,
        "prompt_input_json": json.dumps(prompt_input, ensure_ascii=False),
    }


def sort_candidates(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda r: (
            int(r.get("run_stage_order", 99)),
            clean(r.get("branch")),
            clean(r.get("criterion_id")),
            clean(r.get("rescue_task_type")),
        ),
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLayer 3 targeted LLM rescue candidate builder")
    print("Decision inventory:", DECISION_CSV)
    print("Conservative inspection:", CONSERVATIVE_CSV)

    decision_rows = read_csv(DECISION_CSV)
    conservative_rows = read_csv(CONSERVATIVE_CSV, required=False)
    conservative_index = build_conservative_index(conservative_rows)

    candidates: List[Dict[str, Any]] = []
    seen_keys = set()

    for row in decision_rows:
        key = criterion_key(row)
        conservative_row = conservative_index.get(key)

        conservative_issues = []
        if conservative_row:
            conservative_issues = split_codes(conservative_row.get("conservative_issues", ""))

        if not conservative_issues:
            conservative_issues = conservative_issues_from_codes(row.get("all_layer1_codes", ""))

        if not should_include_candidate(row, conservative_issues):
            continue

        candidate = make_candidate(row, conservative_row)

        candidate_key = candidate["candidate_id"]

        if candidate_key in seen_keys:
            continue

        seen_keys.add(candidate_key)
        candidates.append(candidate)

    candidates = sort_candidates(candidates)

    # JSONL keeps the full structured prompt input.
    write_jsonl(OUT_JSONL, candidates)

    # CSV is for quick inspection.
    write_csv(OUT_CSV, candidates)

    counts_by_branch = Counter(r["branch"] for r in candidates)
    counts_by_run_stage = Counter(r["run_stage"] for r in candidates)
    counts_by_task = Counter(r["rescue_task_type"] for r in candidates)
    counts_by_kind = Counter(r["candidate_kind"] for r in candidates)
    counts_by_requires_judge = Counter(str(r["requires_judge_first"]) for r in candidates)

    summary = {
        "description": (
            "Layer 3 targeted LLM rescue candidate queue. "
            "This script only builds the queue and does not call the LLM."
        ),
        "inputs": {
            "decision_inventory_csv": str(DECISION_CSV),
            "conservative_inspection_csv": str(CONSERVATIVE_CSV),
        },
        "outputs": {
            "candidate_jsonl": str(OUT_JSONL),
            "candidate_csv": str(OUT_CSV),
            "summary_json": str(OUT_SUMMARY_JSON),
        },
        "total_candidates": len(candidates),
        "counts_by_branch": dict(counts_by_branch.most_common()),
        "counts_by_run_stage": dict(counts_by_run_stage.most_common()),
        "counts_by_candidate_kind": dict(counts_by_kind.most_common()),
        "counts_by_rescue_task_type": dict(counts_by_task.most_common()),
        "counts_by_requires_judge_first": dict(counts_by_requires_judge.most_common()),
        "recommended_execution_order": [
            "P1_run_now_conservative_structural_recovery",
            "P1_run_now_branch_b_mandatory_judge",
            "P2_budget_allows_branch_b_optional_judge",
            "P3_branch_a_mandatory_diagnostic_rescue_or_ablation",
            "P4_branch_a_optional_diagnostic_rescue",
        ],
        "method_notes": [
            "Branch B is treated as the main semantic branch; mandatory Branch B candidates should be judged first.",
            "Conservative structural issues are sent to targeted recovery before any downgrade is considered.",
            "Branch A candidates are kept for diagnostic rescue or ablation, not automatic acceptance.",
            "Any future LLM repair must be re-run through Layer 1 and Layer 2 before final use.",
            "This queue uses max_attempts: 3 if branch == B else 2, by default to keep rescue bounded and auditable.",
            (
                "This queue is broad and preliminary. Later Layer 3 scripts "
                "select the runnable subset and prepare the final rescue plan."
            ),
        ],
    }

    write_json(OUT_SUMMARY_JSON, summary)

    print("\nDONE")
    print("Output JSONL:", OUT_JSONL)
    print("Output CSV:", OUT_CSV)
    print("Output JSON:", OUT_SUMMARY_JSON)

    print("\nTotal candidates:", len(candidates))
    print("Counts by branch:", summary["counts_by_branch"])
    print("Counts by run stage:", summary["counts_by_run_stage"])
    print("Counts by candidate kind:", summary["counts_by_candidate_kind"])
    print("Counts by rescue task:", summary["counts_by_rescue_task_type"])
    print("Requires judge first:", summary["counts_by_requires_judge_first"])


if __name__ == "__main__":
    main()

# Run from the repository root:
# python scripts/03_verification/03_layer3/04_build_targeted_rescue_candidates.py