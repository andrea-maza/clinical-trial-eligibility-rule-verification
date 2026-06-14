"""
08_apply_candidate_selection_rescue.py

Apply the previously generated and locally validated Layer 3 rescue
proposals to copies of the Branch A and Branch B logical rule trees.

This script does not call the LLM. It reads the existing
candidate_selection_rescue_results.jsonl file.

A proposal is applied only when:
    - final_decision is select_candidate or
      mark_partial_or_non_computable
    - local_validation_status is pass
    - selected_leaf is present and schema-valid
    - the update introduces no new document-level schema errors

Human-review, failed-validation, and no-change results are not applied.

Outputs:
    outputs/verification/layer3/applied_candidate_selection_rescue/

Run from the repository root:
python scripts/03_verification/03_layer3/08_apply_candidate_selection_rescue.py
"""

from __future__ import annotations

import csv
import json
import re
from copy import deepcopy
from collections import Counter
from pathlib import Path
from jsonschema import Draft7Validator
from typing import Any, Dict, List, Tuple


# --------------------------------------------------
# Allowed values
# --------------------------------------------------

APPLICABLE_FINAL_DECISIONS = {
    "select_candidate",
    "mark_partial_or_non_computable",
}

ALLOWED_ENTITY_TYPES = {
    "condition",
    "drug",
    "procedure",
    "lab",
    "demographic",
    "therapy",
    "biomarker",
    "vital",
    "observation",
    "stage",
    "line_of_therapy",
    "other",
}

ALLOWED_OPERATORS = {
    "exists",
    "not_exists",
    "=",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
    "between",
    "in",
    "not_in",
    "contains",
    "matches",
}

ALLOWED_VALUE_TYPES = {
    "scalar",
    "list",
    "range",
    "null",
}

ALLOWED_COMPUTABILITY = {
    "computable",
    "partial",
    "non_computable",
}

ALLOWED_HISTORY_CONTEXT = {
    "current",
    "prior",
    "previously_treated",
    "stable_dose",
    "investigational_use",
    "other",
    None,
}

ALLOWED_TEMPORAL_RELATIONS = {
    "before",
    "after",
    "during",
    "within",
    "since",
}

ALLOWED_TEMPORAL_UNITS = {
    "hour",
    "day",
    "week",
    "month",
    "year",
    None,
}

ALLOWED_ANCHOR_EVENTS = {
    "screening",
    "randomization",
    "treatment_start",
    "diagnosis",
    "index_date",
    "surgery",
    "procedure",
    "baseline",
    "other",
}

# Fields that may be changed in the target AST leaf.
# criterion_id is intentionally excluded.
APPLY_FIELDS = [
    "entity_text",
    "entity_type",
    "normalized_concept",
    "operator",
    "value",
    "value_type",
    "unit",
    "temporal_context",
    "history_context",
    "negated",
    "computability",
    "non_computable_reason",
    "evidence_text",
]


# --------------------------------------------------
# IO helpers
# --------------------------------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSONL: {path}")

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


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


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
        "final_decision",
        "selected_source",
        "apply_status",
        "apply_action",
        "skip_reason",
        "changed_fields",
        "old_entity_text",
        "new_entity_text",
        "old_operator",
        "new_operator",
        "old_value_type",
        "new_value_type",
        "old_value",
        "new_value",
        "old_unit",
        "new_unit",
        "old_computability",
        "new_computability",
        "local_validation_status",
        "local_validation_reasons",
    ]

    fieldnames = []
    seen = set()

    for col in priority:
        if any(col in row for row in rows):
            fieldnames.append(col)
            seen.add(col)

    for row in rows:
        for col in row:
            if col not in seen:
                fieldnames.append(col)
                seen.add(col)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({col: serialize_cell(row.get(col, "")) for col in fieldnames})


# --------------------------------------------------
# Basic helpers
# --------------------------------------------------

def clean(x: Any) -> str:
    return str(x or "").strip()


def norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def contains_normalized(container: Any, content: Any) -> bool:
    c = norm(container)
    x = norm(content)

    if not c or not x:
        return False

    return x in c


def normalize_temporal_context(tc: Any) -> Any:
    if tc in ["", "null", "None"]:
        return None
    return tc


def values_equal(a: Any, b: Any) -> bool:
    return json.dumps(a, sort_keys=True, ensure_ascii=False) == json.dumps(
        b,
        sort_keys=True,
        ensure_ascii=False,
    )


def list_changed_fields(old: Dict[str, Any], new: Dict[str, Any]) -> List[str]:
    changed = []

    for field in APPLY_FIELDS:
        if field in new and not values_equal(old.get(field), new.get(field)):
            changed.append(field)

    return changed


# --------------------------------------------------
# Local validation before apply
# --------------------------------------------------

def value_consistency_reasons(leaf: Dict[str, Any]) -> List[str]:
    reasons = []

    operator = leaf.get("operator")
    value_type = leaf.get("value_type")
    value = leaf.get("value")

    if value_type == "null":
        if value is not None:
            reasons.append("null_value_type_with_nonnull_value")

    elif value_type == "scalar":
        if value is None or isinstance(value, (list, dict)):
            reasons.append("scalar_value_type_with_invalid_value")

    elif value_type == "list":
        if not isinstance(value, list) or len(value) == 0:
            reasons.append("list_value_type_with_invalid_value")

    elif value_type == "range":
        if not isinstance(value, dict):
            reasons.append("range_value_type_with_invalid_value")
        elif "min" not in value or "max" not in value:
            reasons.append("range_value_missing_min_or_max_keys")
        elif value.get("min") is None and value.get("max") is None:
            reasons.append("range_with_both_bounds_missing")

    if operator in {"<", "<=", ">", ">=", "=", "!="}:
        if value_type != "scalar" or value is None:
            reasons.append("comparison_operator_without_scalar_value")

    if operator == "between" and value_type != "range":
        reasons.append("between_operator_without_range_value")

    if operator in {"in", "not_in"} and value_type != "list":
        reasons.append("list_operator_without_list_value")

    if operator in {"exists", "not_exists"} and value_type != "null":
        reasons.append("existence_operator_with_non_null_value_type")

    return reasons


def validate_candidate_leaf(candidate: Dict[str, Any]) -> List[str]:
    reasons = []

    entity_text = clean(candidate.get("entity_text"))
    evidence_text = clean(candidate.get("evidence_text"))
    entity_type = candidate.get("entity_type")
    operator = candidate.get("operator")
    value_type = candidate.get("value_type")
    computability = candidate.get("computability")
    history_context = candidate.get("history_context")
    temporal_context = normalize_temporal_context(candidate.get("temporal_context"))

    if not clean(candidate.get("criterion_id")):
        reasons.append("missing_criterion_id")

    if not entity_text:
        reasons.append("missing_entity_text")

    if not evidence_text:
        reasons.append("missing_evidence_text")

    if entity_type not in ALLOWED_ENTITY_TYPES:
        reasons.append(f"entity_type_not_allowed:{entity_type}")

    if operator not in ALLOWED_OPERATORS:
        reasons.append(f"operator_not_allowed:{operator}")

    if value_type not in ALLOWED_VALUE_TYPES:
        reasons.append(f"value_type_not_allowed:{value_type}")

    if computability not in ALLOWED_COMPUTABILITY:
        reasons.append(f"computability_not_allowed:{computability}")

    if history_context not in ALLOWED_HISTORY_CONTEXT:
        reasons.append(f"history_context_not_allowed:{history_context}")

    if temporal_context is not None:
        if not isinstance(temporal_context, dict):
            reasons.append("temporal_context_must_be_object_or_null")
        else:
            allowed_keys = {"relation", "value", "unit", "anchor_event"}
            bad_keys = set(temporal_context.keys()) - allowed_keys

            if bad_keys:
                reasons.append(f"temporal_context_has_unexpected_keys:{sorted(bad_keys)}")

            relation = temporal_context.get("relation")
            unit = temporal_context.get("unit")
            anchor_event = temporal_context.get("anchor_event")

            if relation not in ALLOWED_TEMPORAL_RELATIONS:
                reasons.append(f"temporal_relation_not_allowed:{relation}")

            if unit not in ALLOWED_TEMPORAL_UNITS:
                reasons.append(f"temporal_unit_not_allowed:{unit}")

            if anchor_event not in ALLOWED_ANCHOR_EVENTS:
                reasons.append(f"temporal_anchor_event_not_allowed:{anchor_event}")

    reasons.extend(value_consistency_reasons(candidate))

    if computability == "non_computable" and not clean(candidate.get("non_computable_reason")):
        reasons.append("non_computable_without_reason")

    return reasons


# --------------------------------------------------
# Rule-tree traversal and update
# --------------------------------------------------

def build_leaf_location_index(ast_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Return criterion_id -> metadata containing:
        - leaf reference
        - parent document row reference
        - leaf path

    This allows full document-level schema validation before accepting a repair.
    """
    out = {}

    def walk(node: Any, doc_row: Dict[str, Any], path: str):
        if not isinstance(node, dict):
            return

        if node.get("node_type") == "criterion":
            criterion = node.get("criterion")

            if isinstance(criterion, dict):
                cid = clean(criterion.get("criterion_id"))

                if cid:
                    out[cid] = {
                        "leaf": criterion,
                        "doc_row": doc_row,
                        "path": path,
                    }

            return

        if node.get("node_type") == "group":
            children = node.get("children", [])

            if isinstance(children, list):
                for i, child in enumerate(children):
                    walk(child, doc_row, f"{path}.children[{i}]")

    for row in ast_rows:
        ast = row.get("rules_v3_ast", {})

        if not isinstance(ast, dict):
            continue

        walk(ast.get("inclusion_criteria"), row, "inclusion_criteria")
        walk(ast.get("exclusion_criteria"), row, "exclusion_criteria")

    return out


def make_candidate_leaf(old_leaf: Dict[str, Any], selected_leaf: Dict[str, Any]) -> Dict[str, Any]:
    candidate = deepcopy(old_leaf)

    # Preserve original criterion_id.
    candidate["criterion_id"] = old_leaf.get("criterion_id")

    for field in APPLY_FIELDS:
        if field == "criterion_id":
            continue

        if field in selected_leaf:
            candidate[field] = deepcopy(selected_leaf.get(field))

    # Normalize common null string issue.
    candidate["temporal_context"] = normalize_temporal_context(candidate.get("temporal_context"))

    if candidate.get("value_type") == "null":
        candidate["value"] = None

    if candidate.get("computability") == "non_computable":
        if not clean(candidate.get("non_computable_reason")):
            candidate["non_computable_reason"] = "Marked non-computable by Layer 3 candidate-selection rescue."

    return candidate


def apply_candidate_to_leaf(target_leaf: Dict[str, Any], candidate_leaf: Dict[str, Any]) -> None:
    target_leaf.clear()
    target_leaf.update(candidate_leaf)


# --------------------------------------------------
# Result filtering and audit
# --------------------------------------------------

def deduplicate_results(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep the last result for each plan_id.
    """
    by_plan = {}

    for row in rows:
        plan_id = clean(row.get("plan_id"))

        if plan_id:
            by_plan[plan_id] = row

    return list(by_plan.values())


def should_attempt_apply(result: Dict[str, Any]) -> Tuple[bool, str]:
    final_decision = clean(result.get("final_decision"))
    validation_status = clean(result.get("local_validation_status"))

    if final_decision not in APPLICABLE_FINAL_DECISIONS:
        return False, f"not_applicable_final_decision:{final_decision}"

    if validation_status != "pass":
        return False, f"local_validation_not_pass:{validation_status}"

    selected_leaf = result.get("selected_leaf")

    if not isinstance(selected_leaf, dict):
        return False, "selected_leaf_missing_or_not_object"

    return True, ""


def make_audit_row(
    result: Dict[str, Any],
    apply_status: str,
    apply_action: str,
    skip_reason: str,
    old_leaf: Dict[str, Any] | None,
    new_leaf: Dict[str, Any] | None,
    changed_fields: List[str],
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    old_leaf = old_leaf or {}
    new_leaf = new_leaf or {}
    extra = extra or {}

    row = {
        "plan_id": result.get("plan_id"),
        "branch_to_update": result.get("branch_to_update"),
        "criterion_id": result.get("criterion_id"),
        "final_decision": result.get("final_decision"),
        "decision": result.get("decision"),
        "selected_source": result.get("selected_source"),
        "apply_status": apply_status,
        "apply_action": apply_action,
        "skip_reason": skip_reason,
        "changed_fields": changed_fields,

        "old_entity_text": old_leaf.get("entity_text"),
        "new_entity_text": new_leaf.get("entity_text"),
        "old_operator": old_leaf.get("operator"),
        "new_operator": new_leaf.get("operator"),
        "old_value_type": old_leaf.get("value_type"),
        "new_value_type": new_leaf.get("value_type"),
        "old_value": old_leaf.get("value"),
        "new_value": new_leaf.get("value"),
        "old_unit": old_leaf.get("unit"),
        "new_unit": new_leaf.get("unit"),
        "old_computability": old_leaf.get("computability"),
        "new_computability": new_leaf.get("computability"),
        "old_negated": old_leaf.get("negated"),
        "new_negated": new_leaf.get("negated"),

        "local_validation_status": result.get("local_validation_status"),
        "local_validation_reasons": result.get("local_validation_reasons"),
        "selection_reason": result.get("selection_reason"),
        "confidence": result.get("confidence"),
    }

    row.update(extra)

    return row

def load_schema(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def format_schema_errors(errors: List[Any], max_errors: int = 5) -> str:
    parts = []

    for e in errors[:max_errors]:
        error_path = ".".join(str(x) for x in e.absolute_path)
        parts.append(f"{error_path}: {e.message}")

    if len(errors) > max_errors:
        parts.append(f"... plus {len(errors) - max_errors} more schema errors")

    return " | ".join(parts)

def schema_error_signature(errors: List[Any]) -> set:
    """
    Convert schema errors into comparable signatures.

    This lets 06F2b distinguish:
        - schema errors already present before applying a rescue candidate
        - new schema errors introduced by the rescue candidate
    """
    return {
        (
            ".".join(str(x) for x in e.absolute_path),
            e.message,
        )
        for e in errors
    }

# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    ROOT = Path(__file__).resolve().parents[3]

    rescue_results_path = (
        ROOT
        / "outputs"
        / "verification"
        / "layer3"
        / "candidate_selection_rescue"
        / "candidate_selection_rescue_results.jsonl"
    )

    branch_a_input_ast_path = (
        ROOT
        / "outputs"
        / "verification"
        / "layer3"
        / "safe_structural_repairs"
        / "chia_text_only_200_rules_v3_ast_A_layer3_safe_structural.jsonl"
    )

    branch_b_input_ast_path = (
        ROOT
        / "outputs"
        / "verification"
        / "layer3"
        / "safe_structural_repairs"
        / "chia_text_only_200_rules_v3_ast_B_layer3_safe_structural.jsonl"
    )

    out_root = (
        ROOT
        / "outputs"
        / "verification"
        / "layer3"
        / "applied_candidate_selection_rescue"
    )

    out_root.mkdir(parents=True, exist_ok=True)

    branch_a_output_ast_path = (
        out_root
        / "chia_text_only_200_rules_v3_ast_A_layer3_candidate_selection_rescue.jsonl"
    )

    branch_b_output_ast_path = (
        out_root
        / "chia_text_only_200_rules_v3_ast_B_layer3_candidate_selection_rescue.jsonl"
    )

    audit_csv_path = (
        out_root / "layer3_candidate_selection_apply_audit.csv"
    )
    summary_json_path = (
        out_root / "layer3_candidate_selection_apply_summary.json"
    )

    schema_path = ROOT / "schemas" / "rules_v3.json"
    schema = load_schema(schema_path)
    schema_validator = Draft7Validator(schema)

    print("\nLayer 3 apply candidate-selection rescue")
    print("Rescue results:", rescue_results_path)
    print("Branch A input rule tree:", branch_a_input_ast_path)
    print("Branch B input rule tree:", branch_b_input_ast_path)

    rescue_results = deduplicate_results(load_jsonl(rescue_results_path))

    ast_by_branch = {
        "A": load_jsonl(branch_a_input_ast_path),
        "B": load_jsonl(branch_b_input_ast_path),
    }

    leaf_index_by_branch = {
        "A": build_leaf_location_index(ast_by_branch["A"]),
        "B": build_leaf_location_index(ast_by_branch["B"]),
    }

    audit_rows = []

    for result in rescue_results:
        branch = clean(result.get("branch_to_update"))
        criterion_id = clean(result.get("criterion_id"))

        if branch not in {"A", "B"}:
            audit_rows.append(
                make_audit_row(
                    result=result,
                    apply_status="skipped",
                    apply_action="no_update",
                    skip_reason=f"unknown_branch:{branch}",
                    old_leaf=None,
                    new_leaf=None,
                    changed_fields=[],
                )
            )
            continue

        target_meta = leaf_index_by_branch[branch].get(criterion_id)

        if target_meta is None:
            audit_rows.append(
                make_audit_row(
                    result=result,
                    apply_status="skipped",
                    apply_action="no_update",
                    skip_reason="criterion_id_not_found_in_target_ast",
                    old_leaf=None,
                    new_leaf=None,
                    changed_fields=[],
                )
            )
            continue

        target_leaf = target_meta["leaf"]
        target_doc_row = target_meta["doc_row"]

        old_leaf = deepcopy(target_leaf)

        can_apply, skip_reason = should_attempt_apply(result)

        if not can_apply:
            audit_rows.append(
                make_audit_row(
                    result=result,
                    apply_status="skipped",
                    apply_action="no_update",
                    skip_reason=skip_reason,
                    old_leaf=old_leaf,
                    new_leaf=old_leaf,
                    changed_fields=[],
                )
            )
            continue

        selected_leaf = result.get("selected_leaf", {})
        candidate_leaf = make_candidate_leaf(old_leaf, selected_leaf)

        # --------------------------------------------------
        # Safety rule: do not change negation unless this row
        # was explicitly routed for polarity/negation repair.
        # Negation changes can flip clinical meaning.
        # --------------------------------------------------
        rescue_task_type = clean(result.get("rescue_task_type"))

        old_negated = old_leaf.get("negated")
        new_negated = candidate_leaf.get("negated")

        if old_negated != new_negated and rescue_task_type != "polarity_negation_repair":
            candidate_leaf["negated"] = deepcopy(old_negated)

        # Extra local validation before modifying AST.
        schema_reasons = validate_candidate_leaf(candidate_leaf)

        if schema_reasons:
            audit_rows.append(
                make_audit_row(
                    result=result,
                    apply_status="skipped",
                    apply_action="no_update",
                    skip_reason="candidate_schema_invalid:" + "|".join(schema_reasons),
                    old_leaf=old_leaf,
                    new_leaf=candidate_leaf,
                    changed_fields=[],
                )
            )
            continue

        changed_fields = list_changed_fields(old_leaf, candidate_leaf)

        if not changed_fields:
            audit_rows.append(
                make_audit_row(
                    result=result,
                    apply_status="no_change",
                    apply_action="no_update",
                    skip_reason="candidate_identical_to_target_leaf",
                    old_leaf=old_leaf,
                    new_leaf=candidate_leaf,
                    changed_fields=[],
                )
            )
            continue

        # --------------------------------------------------
        # Full schema validation before accepting the update.
        #
        # Important:
        # Some AST documents may already contain schema warnings before rescue.
        # Therefore, we only block the update if the candidate introduces NEW
        # schema errors. Existing pre-rescue schema errors are not blamed on the
        # rescue candidate.
        # --------------------------------------------------
        doc_ast = target_doc_row.get("rules_v3_ast", {})

        pre_errors = sorted(
            schema_validator.iter_errors(doc_ast),
            key=lambda e: list(e.absolute_path),
        )
        pre_error_sig = schema_error_signature(pre_errors)

        backup_leaf = deepcopy(target_leaf)

        apply_candidate_to_leaf(target_leaf, candidate_leaf)

        post_errors = sorted(
            schema_validator.iter_errors(doc_ast),
            key=lambda e: list(e.absolute_path),
        )
        post_error_sig = schema_error_signature(post_errors)

        new_error_sig = post_error_sig - pre_error_sig

        new_errors = [
            e for e in post_errors
            if (
                ".".join(str(x) for x in e.absolute_path),
                e.message,
            ) in new_error_sig
        ]

        if new_errors:
            target_leaf.clear()
            target_leaf.update(backup_leaf)

            audit_rows.append(
                make_audit_row(
                    result=result,
                    apply_status="skipped",
                    apply_action="no_update",
                    skip_reason="full_ast_schema_new_invalid:" + format_schema_errors(new_errors),
                    old_leaf=old_leaf,
                    new_leaf=candidate_leaf,
                    changed_fields=[],
                )
            )
            continue

        audit_rows.append(
            make_audit_row(
                result=result,
                apply_status="applied",
                apply_action="apply_selected_candidate_leaf",
                skip_reason="",
                old_leaf=old_leaf,
                new_leaf=candidate_leaf,
                changed_fields=changed_fields,
            )
        )

    write_jsonl(branch_a_output_ast_path, ast_by_branch["A"])
    write_jsonl(branch_b_output_ast_path, ast_by_branch["B"])
    write_csv(audit_csv_path, audit_rows)

    changed_field_counts = Counter()

    for row in audit_rows:
        if row.get("apply_status") != "applied":
            continue

        for field in row.get("changed_fields", []) or []:
            changed_field_counts[field] += 1

    summary = {
        "stage": "08_apply_candidate_selection_rescue",
        "description": (
            "Applies previously generated and validated candidate-selection "
            "rescue proposals. This script does not call the LLM and does not "
            "use manual labels."
        ),
        "inputs": {
            "rescue_results": str(rescue_results_path),
            "branch_a_input_ast": str(branch_a_input_ast_path),
            "branch_b_input_ast": str(branch_b_input_ast_path),
        },
        "outputs": {
            "branch_a_output_ast": str(branch_a_output_ast_path),
            "branch_b_output_ast": str(branch_b_output_ast_path),
            "audit_csv": str(audit_csv_path),
            "summary_json": str(summary_json_path),
        },
        "counts": {
            "rescue_results": len(rescue_results),
            "audit_rows": len(audit_rows),
            "apply_status_counts": dict(Counter(row.get("apply_status") for row in audit_rows)),
            "apply_action_counts": dict(Counter(row.get("apply_action") for row in audit_rows)),
            "counts_by_branch": dict(Counter(row.get("branch_to_update") for row in audit_rows)),
            "counts_by_final_decision": dict(Counter(row.get("final_decision") for row in audit_rows)),
            "counts_by_selected_source": dict(Counter(row.get("selected_source") for row in audit_rows)),
            "changed_field_counts": dict(changed_field_counts),
            "skipped_reasons": dict(
                Counter(
                    row.get("skip_reason")
                    for row in audit_rows
                    if row.get("apply_status") == "skipped"
                )
            ),
        },
        "method_notes": [
            "Only validated selected_leaf outputs are applied.",
            "Human-review rows are not applied.",
            "Failed-validation rows are not applied.",
            "No-change rows remain unchanged and are retained as audit evidence.",
            "criterion_id is preserved from the target rule tree.",
            "Tree structure is not changed; only leaf fields can be updated.",
            "Layer 1 and Layer 2 must be rerun after this step.",
        ],
    }

    write_json(summary_json_path, summary)

    print("\nDONE")
    print("Branch A output AST:", branch_a_output_ast_path)
    print("Branch B output AST:", branch_b_output_ast_path)
    print("Audit CSV:", audit_csv_path)
    print("Summary JSON:", summary_json_path)

    print("\nRescue results:", len(rescue_results))
    print("Audit rows:", len(audit_rows))
    print("Apply status counts:", summary["counts"]["apply_status_counts"])
    print("Apply action counts:", summary["counts"]["apply_action_counts"])
    print("Counts by branch:", summary["counts"]["counts_by_branch"])
    print("Counts by selected source:", summary["counts"]["counts_by_selected_source"])
    print("Changed field counts:", summary["counts"]["changed_field_counts"])
    print("Skipped reasons:", summary["counts"]["skipped_reasons"])


if __name__ == "__main__":
    main()

# Run from the repository root:
# python scripts/03_verification/03_layer3/08_apply_candidate_selection_rescue.py