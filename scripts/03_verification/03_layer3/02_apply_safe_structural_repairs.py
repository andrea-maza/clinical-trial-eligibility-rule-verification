"""
02_apply_safe_structural_repairs.py

Apply deterministic safe structural repairs selected by the Layer 3
decision inventory.

The script reads the original Branch A and Branch B logical rule trees,
applies only repairs that do not invent new clinical information, and
writes repaired copies together with an audit trail.

Implemented repairs:
    - convert numeric strings to numeric values
    - change computable leaves with unresolved exception context to partial
    - correct computability when a non-computable reason already exists
    - add a missing reason to non-computable leaves
    - complete temporal value or unit only when already available in the leaf

Conservative structural problems are not repaired here and remain
candidates for later targeted recovery or review.

Outputs:
    outputs/verification/layer3/safe_structural_repairs/
        chia_text_only_200_rules_v3_ast_A_layer3_safe_structural.jsonl
        chia_text_only_200_rules_v3_ast_B_layer3_safe_structural.jsonl
        layer3_safe_structural_repair_audit.csv
        layer3_safe_structural_repair_summary.json

Run from the repository root:
python scripts/03_verification/03_layer3/02_apply_safe_structural_repairs.py
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[3]

DECISION_CSV = (
    ROOT
    / "outputs"
    / "verification"
    / "layer3"
    / "decision_inventory"
    / "layer3_decision_inventory_leaf_level.csv"
)

BRANCH_A_RAW_AST = (
    ROOT
    / "outputs"
    / "extraction"
    / "branch_a"
    / "rules_v3"
    / "chia_text_only_200_rules_v3_ast_A.jsonl"
)

BRANCH_B_RAW_AST = (
    ROOT
    / "outputs"
    / "extraction"
    / "branch_b"
    / "rules_v3_llm_pass2"
    / "chia_text_only_200_rules_v3_ast_B.jsonl"
)

OUT_DIR = (
    ROOT
    / "outputs"
    / "verification"
    / "layer3"
    / "safe_structural_repairs"
)

OUT_A_AST = (
    OUT_DIR
    / "chia_text_only_200_rules_v3_ast_A_layer3_safe_structural.jsonl"
)

OUT_B_AST = (
    OUT_DIR
    / "chia_text_only_200_rules_v3_ast_B_layer3_safe_structural.jsonl"
)

OUT_AUDIT_CSV = OUT_DIR / "layer3_safe_structural_repair_audit.csv"
OUT_SUMMARY_JSON = OUT_DIR / "layer3_safe_structural_repair_summary.json"

COMPARISON_OPERATORS = {">", ">=", "<", "<="}

SUPPORTED_SAFE_REPAIR_ISSUES = {
    "comparison_value_is_numeric_string",
    "computable_with_exception_context",
    "computable_with_non_computable_reason",
    "non_computable_without_reason",
    "temporal_context_missing_value",
    "temporal_context_missing_unit",
}

# These are intentionally not repaired in this script
# They reduce specificity and should be handled in a separate conservative
# downgrade or LLM rescue step.
CONSERVATIVE_REPAIR_ISSUES = {
    "comparison_without_scalar_value",
    "list_operator_without_list_value",
    "range_with_both_bounds_missing",
}


# ---------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL not found: {path}")

    rows: List[Dict[str, Any]] = []

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
        "repair_action",
        "repair_applied",
        "repair_skipped_reason",
        "field_changed",

        "operator",
        "value_type",

        "old_value",
        "new_value",
        "old_unit",
        "new_unit",
        "old_temporal_context",
        "new_temporal_context",
        "old_computability",
        "new_computability",
        "old_non_computable_reason",
        "new_non_computable_reason",

        "layer3_primary_action",
        "layer3_action_family",
        "layer1a_action_category",
        "layer1a_action_hint",
        "all_layer1_codes",
        "normalized_issue_codes",
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


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def clean(x: Any) -> str:
    return str(x or "").strip()


def lower(x: Any) -> str:
    return clean(x).lower()


def normalize_text(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()


def split_codes(x: Any) -> List[str]:
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

    return [p.strip() for p in re.split(r"[;|]", s) if p.strip()]


def normalize_issue_codes(issue_codes: List[str]) -> set:
    """
    Convert prefixed codes into raw issue codes.

    Example:
        layer1a:comparison_value_is_numeric_string
    becomes:
        comparison_value_is_numeric_string
    """
    return {
        code.split(":", 1)[-1].strip()
        for code in issue_codes
        if code.strip()
    }


def parse_numeric_string(x: Any) -> Tuple[bool, Any]:
    """
    Convert numeric strings to int or float.

    Returns:
        (success, parsed_value)
    """
    if not isinstance(x, str):
        return False, x

    s = x.strip()

    if not re.fullmatch(r"[-+]?\d+(\.\d+)?", s):
        return False, x

    value = float(s)

    if value.is_integer():
        return True, int(value)

    return True, value


def add_non_computable_reason(criterion: Dict[str, Any], reason: str) -> None:
    old_reason = normalize_text(criterion.get("non_computable_reason"))

    if not old_reason:
        criterion["non_computable_reason"] = reason
        return

    existing = {
        part.strip()
        for part in old_reason.split(";")
        if part.strip()
    }

    if reason not in existing:
        criterion["non_computable_reason"] = old_reason + "; " + reason


def is_temporal_unit(unit: Any) -> bool:
    unit_norm = str(unit or "").lower().strip().rstrip("s")
    return unit_norm in {"hour", "day", "week", "month", "year"}


def criterion_id_from_criterion(criterion: Dict[str, Any]) -> str:
    return clean(criterion.get("criterion_id"))


def iter_criterion_nodes(
    node: Any,
    path: str = "",
) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """
    Yield:
        path, criterion_dict

    Expected rules_v3 leaf:
        {"node_type": "criterion", "criterion": {...}}
    """
    if not isinstance(node, dict):
        return

    if node.get("node_type") == "criterion" and isinstance(node.get("criterion"), dict):
        yield path, node["criterion"]
        return

    children = node.get("children")

    if isinstance(children, list):
        for i, child in enumerate(children):
            child_path = f"{path}.children[{i}]" if path else f"children[{i}]"
            yield from iter_criterion_nodes(child, child_path)


def iter_document_criteria(doc: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    ast = doc.get("rules_v3_ast")

    if not isinstance(ast, dict):
        return

    for section in ["inclusion_criteria", "exclusion_criteria"]:
        root = ast.get(section)

        if isinstance(root, dict):
            yield from iter_criterion_nodes(root, section)


# ---------------------------------------------------------------------
# Decision index
# ---------------------------------------------------------------------

def load_safe_repair_decisions() -> Dict[Tuple[str, str], Dict[str, str]]:
    """
    Load Layer 3 safe structural repair candidates.

    Key:
        (branch, criterion_id)

    Important:
        A leaf may contain both:
            - a safe repair issue
            - a conservative downgrade issue

        In that case, this script should still apply the safe repair part,
        but it must NOT apply the conservative downgrade part.
    """
    rows = read_csv(DECISION_CSV)
    out: Dict[Tuple[str, str], Dict[str, str]] = {}

    for r in rows:
        branch = clean(r.get("branch"))
        criterion_id = clean(r.get("criterion_id"))

        if not branch or not criterion_id:
            continue

        primary = lower(r.get("layer3_primary_action"))
        family = lower(r.get("layer3_action_family"))

        issue_codes = split_codes(r.get("all_layer1_codes", ""))
        normalized_codes = normalize_issue_codes(issue_codes)

        has_supported_safe_issue = bool(normalized_codes & SUPPORTED_SAFE_REPAIR_ISSUES)

        is_safe_structural_decision = (
            primary == "safe_structural_repair_candidate"
            or family == "structural_repair"
        )

        # Do NOT exclude leaves that also contain conservative issues.
        # The repair function will apply only the safe part and record
        # conservative issues as pending.
        if is_safe_structural_decision and has_supported_safe_issue:
            out[(branch, criterion_id)] = r

    return out


# ---------------------------------------------------------------------
# Repair logic
# ---------------------------------------------------------------------

def apply_safe_repair_to_criterion(
    branch: str,
    document_id: str,
    criterion: Dict[str, Any],
    decision: Dict[str, str],
) -> Dict[str, Any]:
    """
    Apply safe deterministic Layer 3 repairs.

    Safe means:
        - no new clinical content is invented
        - only type normalization or metadata consistency is changed
        - temporal value/unit is copied only if already present in the leaf

    Conservative downgrades are intentionally NOT handled here.
    """
    criterion_id = criterion_id_from_criterion(criterion)

    operator = clean(criterion.get("operator"))
    value_type = clean(criterion.get("value_type"))

    old_value = deepcopy(criterion.get("value"))
    old_unit = deepcopy(criterion.get("unit"))
    old_temporal = deepcopy(criterion.get("temporal_context"))
    old_computability = deepcopy(criterion.get("computability"))
    old_reason = deepcopy(criterion.get("non_computable_reason"))

    issue_codes = split_codes(decision.get("all_layer1_codes", ""))
    issue_set = normalize_issue_codes(issue_codes)

    action_hint = clean(decision.get("layer1a_action_hint"))
    action_category = clean(decision.get("layer1a_action_category"))

    audit = {
        "branch": branch,
        "document_id": document_id,
        "criterion_id": criterion_id,

        "repair_action": "",
        "repair_applied": 0,
        "repair_skipped_reason": "",
        "field_changed": "",

        "operator": operator,
        "value_type": value_type,

        "old_value": old_value,
        "new_value": "",
        "old_unit": old_unit,
        "new_unit": "",
        "old_temporal_context": old_temporal,
        "new_temporal_context": "",
        "old_computability": old_computability,
        "new_computability": "",
        "old_non_computable_reason": old_reason,
        "new_non_computable_reason": "",

        "layer3_primary_action": decision.get("layer3_primary_action", ""),
        "layer3_action_family": decision.get("layer3_action_family", ""),
        "layer1a_action_category": action_category,
        "layer1a_action_hint": action_hint,
        "all_layer1_codes": decision.get("all_layer1_codes", ""),
        "normalized_issue_codes": ";".join(sorted(issue_set)),
        "evidence_text": criterion.get("evidence_text", ""),
    }

    applied_actions: List[str] = []
    fields_changed: List[str] = []
    skipped_reasons: List[str] = []

    # ---------------------------------------------------------
    # Safe repair 1:
    # numeric string -> numeric value
    # ---------------------------------------------------------
    if "comparison_value_is_numeric_string" in issue_set:
        if operator not in COMPARISON_OPERATORS:
            skipped_reasons.append("comparison_value_is_numeric_string:operator_is_not_comparison")

        elif value_type != "scalar":
            skipped_reasons.append("comparison_value_is_numeric_string:value_type_is_not_scalar")

        else:
            ok, parsed_value = parse_numeric_string(old_value)

            if not ok:
                skipped_reasons.append("comparison_value_is_numeric_string:value_is_not_numeric_string")

            else:
                criterion["value"] = parsed_value
                applied_actions.append("comparison_value_is_numeric_string")
                fields_changed.append("value")

    # ---------------------------------------------------------
    # Safe repair 2:
    # computable leaf has unresolved exception/condition context
    # ---------------------------------------------------------
    if "computable_with_exception_context" in issue_set:
        criterion["computability"] = "partial"
        add_non_computable_reason(criterion, "exception_context_unresolved")

        applied_actions.append("computable_with_exception_context")
        fields_changed.append("computability")
        fields_changed.append("non_computable_reason")

    # ---------------------------------------------------------
    # Safe repair 3:
    # computable but already has non-computable reason
    # ---------------------------------------------------------
    if "computable_with_non_computable_reason" in issue_set:
        criterion["computability"] = "partial"

        applied_actions.append("computable_with_non_computable_reason")
        fields_changed.append("computability")

    # ---------------------------------------------------------
    # Safe repair 4:
    # non-computable but missing reason
    # ---------------------------------------------------------
    if "non_computable_without_reason" in issue_set:
        add_non_computable_reason(criterion, "reason_missing_after_extraction")

        applied_actions.append("non_computable_without_reason")
        fields_changed.append("non_computable_reason")

    # ---------------------------------------------------------
    # Safe repair 5:
    # temporal_context missing value/unit, but value/unit already exist
    # in the same leaf and unit is clearly temporal.
    # ---------------------------------------------------------
    temporal_missing = (
        "temporal_context_missing_value" in issue_set
        or "temporal_context_missing_unit" in issue_set
    )

    if temporal_missing:
        temporal_context = criterion.get("temporal_context")

        if not isinstance(temporal_context, dict):
            skipped_reasons.append("temporal_context_missing_value_or_unit:temporal_context_is_not_dict")

        else:
            leaf_value = criterion.get("value")
            leaf_unit = criterion.get("unit")

            if leaf_value is None or not is_temporal_unit(leaf_unit):
                skipped_reasons.append("temporal_context_missing_value_or_unit:no_existing_temporal_value_unit_to_copy")

            else:
                if (
                    "temporal_context_missing_value" in issue_set
                    and temporal_context.get("value") is None
                ):
                    temporal_context["value"] = leaf_value
                    applied_actions.append("temporal_context_missing_value")
                    fields_changed.append("temporal_context.value")

                if (
                    "temporal_context_missing_unit" in issue_set
                    and temporal_context.get("unit") is None
                ):
                    temporal_context["unit"] = str(leaf_unit).lower().strip().rstrip("s")
                    applied_actions.append("temporal_context_missing_unit")
                    fields_changed.append("temporal_context.unit")

    # ---------------------------------------------------------
    # Conservative repairs are explicitly skipped in 06a.
    # ---------------------------------------------------------
    conservative_present = sorted(issue_set & CONSERVATIVE_REPAIR_ISSUES)

    if conservative_present:
        skipped_reasons.append(
            "conservative_repair_issue_present_not_applied_in_06a:"
            + "|".join(conservative_present)
        )

    if not applied_actions:
        audit["repair_action"] = "no_safe_repair_applied"
        audit["repair_skipped_reason"] = (
            ";".join(skipped_reasons)
            if skipped_reasons
            else "safe_structural_repair_candidate_but_no_supported_safe_issue_code_found"
        )
        return audit

    audit["repair_action"] = ";".join(sorted(set(applied_actions)))
    audit["repair_applied"] = 1
    audit["repair_skipped_reason"] = ";".join(skipped_reasons)
    audit["field_changed"] = ";".join(sorted(set(fields_changed)))

    audit["new_value"] = deepcopy(criterion.get("value"))
    audit["new_unit"] = deepcopy(criterion.get("unit"))
    audit["new_temporal_context"] = deepcopy(criterion.get("temporal_context"))
    audit["new_computability"] = deepcopy(criterion.get("computability"))
    audit["new_non_computable_reason"] = deepcopy(criterion.get("non_computable_reason"))

    return audit


def process_branch(
    branch: str,
    raw_ast_path: Path,
    out_ast_path: Path,
    decisions: Dict[Tuple[str, str], Dict[str, str]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = read_jsonl(raw_ast_path)
    output_rows = deepcopy(rows)

    audit_rows: List[Dict[str, Any]] = []

    candidate_count = 0
    applied_count = 0
    skipped_count = 0
    visited_leaves = 0

    decision_keys_for_branch = {
        criterion_id
        for (b, criterion_id) in decisions.keys()
        if b == branch
    }

    matched_decision_keys = set()

    for doc in output_rows:
        document_id = clean(doc.get("document_id"))

        for _path, criterion in iter_document_criteria(doc):
            visited_leaves += 1

            criterion_id = criterion_id_from_criterion(criterion)

            if not criterion_id:
                continue

            decision = decisions.get((branch, criterion_id))

            if not decision:
                continue

            matched_decision_keys.add(criterion_id)
            candidate_count += 1

            audit = apply_safe_repair_to_criterion(
                branch=branch,
                document_id=document_id,
                criterion=criterion,
                decision=decision,
            )

            audit_rows.append(audit)

            if int(audit.get("repair_applied", 0)) == 1:
                applied_count += 1
            else:
                skipped_count += 1

    unmatched_decision_keys = sorted(decision_keys_for_branch - matched_decision_keys)

    write_jsonl(out_ast_path, output_rows)

    summary = {
        "branch": branch,
        "input_ast": str(raw_ast_path),
        "output_ast": str(out_ast_path),
        "visited_leaves": visited_leaves,
        "safe_repair_decisions_for_branch": len(decision_keys_for_branch),
        "safe_repair_candidates_matched_in_ast": candidate_count,
        "repairs_applied": applied_count,
        "repairs_skipped": skipped_count,
        "unmatched_decision_keys_count": len(unmatched_decision_keys),
        "unmatched_decision_keys_examples": unmatched_decision_keys[:20],
    }

    return audit_rows, summary


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLayer 3 safe structural repairs")
    print("Decision inventory:", DECISION_CSV)
    print("Branch A raw rule tree:", BRANCH_A_RAW_AST)
    print("Branch B raw rule tree:", BRANCH_B_RAW_AST)
    
    decisions = load_safe_repair_decisions()

    print("Safe structural repair decisions loaded:", len(decisions))

    audit_a, summary_a = process_branch(
        branch="A",
        raw_ast_path=BRANCH_A_RAW_AST,
        out_ast_path=OUT_A_AST,
        decisions=decisions,
    )

    audit_b, summary_b = process_branch(
        branch="B",
        raw_ast_path=BRANCH_B_RAW_AST,
        out_ast_path=OUT_B_AST,
        decisions=decisions,
    )

    EXPECTED_LEAVES = 2402

    for branch_name, summary in {"A": summary_a, "B": summary_b}.items():
        if summary["visited_leaves"] != EXPECTED_LEAVES:
            raise RuntimeError(
                f"Branch {branch_name}: visited {summary['visited_leaves']} leaves, "
                f"expected {EXPECTED_LEAVES}. Check rule-tree input paths."
            )

        if summary["unmatched_decision_keys_count"] != 0:
            raise RuntimeError(
                f"Branch {branch_name}: {summary['unmatched_decision_keys_count']} safe repair "
                f"decision keys were not found in the rule tree. Check criterion_id alignment."
            )

    audit_rows = audit_a + audit_b

    write_csv(OUT_AUDIT_CSV, audit_rows)

    repair_action_counts = Counter(r.get("repair_action", "") for r in audit_rows)

    applied_repair_counts = Counter(
        f"{r.get('branch')}:{r.get('repair_action')}"
        for r in audit_rows
        if int(r.get("repair_applied", 0)) == 1
    )

    skipped_reasons = Counter(
        r.get("repair_skipped_reason", "")
        for r in audit_rows
        if int(r.get("repair_applied", 0)) == 0
    )

    changed_fields = Counter()

    for r in audit_rows:
        if int(r.get("repair_applied", 0)) != 1:
            continue

        for field in split_codes(r.get("field_changed", "")):
            changed_fields[field] += 1

    summary = {
        "description": (
            "Layer 3 safe structural repair. This script applies only deterministic "
            "safe repairs and writes repaired rule-tree copies. Re-verification is required."
        ),
        "inputs": {
            "decision_inventory_csv": str(DECISION_CSV),
            "branch_a_raw_ast": str(BRANCH_A_RAW_AST),
            "branch_b_raw_ast": str(BRANCH_B_RAW_AST),
        },
        "outputs": {
            "branch_a_safe_structural_ast": str(OUT_A_AST),
            "branch_b_safe_structural_ast": str(OUT_B_AST),
            "audit_csv": str(OUT_AUDIT_CSV),
            "summary_json": str(OUT_SUMMARY_JSON),
        },
        "branch_summaries": {
            "A": summary_a,
            "B": summary_b,
        },
        "repair_action_counts": dict(repair_action_counts.most_common()),
        "applied_repair_counts": dict(applied_repair_counts.most_common()),
        "skipped_repair_reasons": dict(skipped_reasons.most_common()),
        "changed_field_counts": dict(changed_fields.most_common()),
        "supported_safe_repair_issues": sorted(SUPPORTED_SAFE_REPAIR_ISSUES),
        "conservative_repair_issues_not_applied_here": sorted(CONSERVATIVE_REPAIR_ISSUES),
        "method_notes": [
            "Only safe deterministic structural repairs are applied.",
            "Conservative downgrade candidates are not applied in this script.",
            "No LLM call is made.",
            "The resulting AST files must be re-run through Layer 1 and Layer 2.",
            "Issue-code prefixes such as layer1a: are normalized before repair matching.",
        ],
    }

    write_json(OUT_SUMMARY_JSON, summary)

    print("\nDONE")
    print("Branch A output AST:", OUT_A_AST)
    print("Branch B output AST:", OUT_B_AST)
    print("Audit CSV:", OUT_AUDIT_CSV)
    print("Summary JSON:", OUT_SUMMARY_JSON)

    print("\nBranch summaries:")
    print(summary["branch_summaries"])

    print("\nRepair action counts:")
    print(summary["repair_action_counts"])

    print("\nApplied repair counts:")
    print(summary["applied_repair_counts"])

    print("\nSkipped repair reasons:")
    print(summary["skipped_repair_reasons"])

    print("\nChanged field counts:")
    print(summary["changed_field_counts"])


if __name__ == "__main__":
    main()


# Run from the repository root:
# python scripts/03_verification/03_layer3/02_apply_safe_structural_repairs.py