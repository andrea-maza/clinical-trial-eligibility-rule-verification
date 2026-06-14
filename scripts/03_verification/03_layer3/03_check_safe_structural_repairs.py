"""
03_check_safe_structural_repairs.py

Check that the safe deterministic repairs changed only the intended
fields in the logical rule trees.

The script compares the original and repaired Branch A and Branch B
rule trees. It verifies that:

    - repaired leaves changed only permitted fields
    - leaves not selected for repair remained unchanged
    - exception-context repairs changed computability to partial
    - the required non-computable reason was added
    - unresolved conservative issues remain explicitly pending

This script does not modify the rule trees, call the LLM, or apply any
additional repairs.

Outputs:
    outputs/verification/layer3/safe_structural_repair_check/
        layer3_safe_structural_repair_diff_check.csv
        layer3_safe_structural_repair_check_summary.json

Run from the repository root:
python scripts/03_verification/03_layer3/03_check_safe_structural_repairs.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[3]

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

BRANCH_A_REPAIRED_AST = (
    ROOT
    / "outputs"
    / "verification"
    / "layer3"
    / "safe_structural_repairs"
    / "chia_text_only_200_rules_v3_ast_A_layer3_safe_structural.jsonl"
)

BRANCH_B_REPAIRED_AST = (
    ROOT
    / "outputs"
    / "verification"
    / "layer3"
    / "safe_structural_repairs"
    / "chia_text_only_200_rules_v3_ast_B_layer3_safe_structural.jsonl"
)

REPAIR_AUDIT_CSV = (
    ROOT
    / "outputs"
    / "verification"
    / "layer3"
    / "safe_structural_repairs"
    / "layer3_safe_structural_repair_audit.csv"
)

OUT_DIR = (
    ROOT
    / "outputs"
    / "verification"
    / "layer3"
    / "safe_structural_repair_check"
)

OUT_DIFF_CSV = OUT_DIR / "layer3_safe_structural_repair_diff_check.csv"
OUT_SUMMARY_JSON = OUT_DIR / "layer3_safe_structural_repair_check_summary.json"


LEAF_FIELDS_TO_COMPARE = [
    "entity_type",
    "entity_text",
    "operator",
    "value_type",
    "value",
    "unit",
    "temporal_context",
    "history_context",
    "computability",
    "non_computable_reason",
    "evidence_text",
]

ALLOWED_CHANGED_FIELDS_BY_ACTION = {
    "computable_with_exception_context": {
        "computability",
        "non_computable_reason",
    },
    "computable_with_non_computable_reason": {
        "computability",
    },
    "non_computable_without_reason": {
        "non_computable_reason",
    },
    "comparison_value_is_numeric_string": {
        "value",
    },
    "temporal_context_missing_value": {
        "temporal_context",
    },
    "temporal_context_missing_unit": {
        "temporal_context",
    },
}


# ---------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL not found: {path}")

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


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    priority = [
        "branch",
        "criterion_id",
        "document_id",
        "repair_action",
        "repair_applied",
        "changed_fields",
        "unexpected_changed_fields",
        "expected_changed_fields",
        "check_status",
        "check_reason",
        "raw_computability",
        "repaired_computability",
        "raw_non_computable_reason",
        "repaired_non_computable_reason",
        "has_pending_conservative_issue",
        "pending_conservative_issues",
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
        writer.writerows(rows)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def clean(x: Any) -> str:
    return str(x or "").strip()


def as_bool_int(x: Any) -> int:
    s = clean(x).lower()
    if s in {"1", "1.0", "true", "yes", "y"}:
        return 1
    return 0


def stable_json(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, sort_keys=True)


def split_semicolon(x: Any) -> List[str]:
    s = clean(x)
    if not s:
        return []
    return [p.strip() for p in s.split(";") if p.strip()]


def criterion_id_from_criterion(criterion: Dict[str, Any]) -> str:
    return clean(criterion.get("criterion_id"))


def iter_criterion_nodes(node: Any, path: str = "") -> Iterable[Tuple[str, Dict[str, Any]]]:
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


def leaf_snapshot(criterion: Dict[str, Any]) -> Dict[str, Any]:
    return {field: criterion.get(field) for field in LEAF_FIELDS_TO_COMPARE}


def build_leaf_index(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index = {}

    for doc in rows:
        for _path, criterion in iter_document_criteria(doc):
            criterion_id = criterion_id_from_criterion(criterion)
            if criterion_id:
                index[criterion_id] = leaf_snapshot(criterion)

    return index


def build_audit_index(rows: List[Dict[str, str]], branch: str) -> Dict[str, Dict[str, str]]:
    out = {}

    for r in rows:
        if clean(r.get("branch")) != branch:
            continue

        criterion_id = clean(r.get("criterion_id"))
        if criterion_id:
            out[criterion_id] = r

    return out


def changed_fields(raw: Dict[str, Any], repaired: Dict[str, Any]) -> List[str]:
    changed = []

    for field in LEAF_FIELDS_TO_COMPARE:
        if stable_json(raw.get(field)) != stable_json(repaired.get(field)):
            changed.append(field)

    return changed


def expected_fields_for_repair_action(action: str) -> set:
    fields = set()

    for part in split_semicolon(action):
        fields.update(ALLOWED_CHANGED_FIELDS_BY_ACTION.get(part, set()))

    return fields


def pending_conservative_issues_from_audit(audit: Dict[str, str]) -> List[str]:
    codes = clean(audit.get("all_layer1_codes"))

    pending = []

    for issue in [
        "comparison_without_scalar_value",
        "list_operator_without_list_value",
        "range_with_both_bounds_missing",
    ]:
        if issue in codes:
            pending.append(issue)

    return pending


def check_repaired_leaf(
    branch: str,
    criterion_id: str,
    raw: Dict[str, Any],
    repaired: Dict[str, Any],
    audit: Dict[str, str],
) -> Dict[str, Any]:
    repair_action = clean(audit.get("repair_action"))
    repair_applied = as_bool_int(audit.get("repair_applied"))

    observed_changed = changed_fields(raw, repaired)
    expected_changed = expected_fields_for_repair_action(repair_action)
    unexpected = sorted(set(observed_changed) - expected_changed)

    pending_conservative = pending_conservative_issues_from_audit(audit)

    status = "pass"
    reasons = []

    if repair_applied != 1:
        status = "fail"
        reasons.append("audit_row_says_repair_not_applied")

    if unexpected:
        status = "fail"
        reasons.append("unexpected_fields_changed")

    if repair_action == "computable_with_exception_context":
        if clean(repaired.get("computability")) != "partial":
            status = "fail"
            reasons.append("computability_not_partial_after_repair")

        if "exception_context_unresolved" not in clean(repaired.get("non_computable_reason")):
            status = "fail"
            reasons.append("missing_exception_context_unresolved_reason")

    if not observed_changed:
        status = "fail"
        reasons.append("no_fields_changed_despite_applied_repair")

    return {
        "branch": branch,
        "criterion_id": criterion_id,
        "document_id": clean(audit.get("document_id")),
        "repair_action": repair_action,
        "repair_applied": repair_applied,
        "changed_fields": ";".join(observed_changed),
        "unexpected_changed_fields": ";".join(unexpected),
        "expected_changed_fields": ";".join(sorted(expected_changed)),
        "check_status": status,
        "check_reason": ";".join(reasons),
        "raw_computability": raw.get("computability"),
        "repaired_computability": repaired.get("computability"),
        "raw_non_computable_reason": raw.get("non_computable_reason"),
        "repaired_non_computable_reason": repaired.get("non_computable_reason"),
        "has_pending_conservative_issue": 1 if pending_conservative else 0,
        "pending_conservative_issues": ";".join(pending_conservative),
    }


def check_unrepaired_leaves(
    branch: str,
    raw_index: Dict[str, Dict[str, Any]],
    repaired_index: Dict[str, Dict[str, Any]],
    repaired_criterion_ids: set,
) -> List[Dict[str, Any]]:
    rows = []

    for criterion_id, raw_leaf in raw_index.items():
        if criterion_id in repaired_criterion_ids:
            continue

        repaired_leaf = repaired_index.get(criterion_id)

        if repaired_leaf is None:
            rows.append({
                "branch": branch,
                "criterion_id": criterion_id,
                "repair_action": "",
                "repair_applied": 0,
                "changed_fields": "",
                "unexpected_changed_fields": "leaf_missing_in_repaired_rule_tree",
                "expected_changed_fields": "",
                "check_status": "fail",
                "check_reason": "leaf_missing_in_repaired_ast",
            })
            continue

        observed_changed = changed_fields(raw_leaf, repaired_leaf)

        if observed_changed:
            rows.append({
                "branch": branch,
                "criterion_id": criterion_id,
                "repair_action": "",
                "repair_applied": 0,
                "changed_fields": ";".join(observed_changed),
                "unexpected_changed_fields": ";".join(observed_changed),
                "expected_changed_fields": "",
                "check_status": "fail",
                "check_reason": "unrepaired_leaf_changed",
                "raw_computability": raw_leaf.get("computability"),
                "repaired_computability": repaired_leaf.get("computability"),
                "raw_non_computable_reason": raw_leaf.get("non_computable_reason"),
                "repaired_non_computable_reason": repaired_leaf.get("non_computable_reason"),
                "has_pending_conservative_issue": 0,
                "pending_conservative_issues": "",
            })

    return rows


def process_branch(
    branch: str,
    raw_path: Path,
    repaired_path: Path,
    audit_rows: List[Dict[str, str]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw_index = build_leaf_index(read_jsonl(raw_path))
    repaired_index = build_leaf_index(read_jsonl(repaired_path))
    audit_index = build_audit_index(audit_rows, branch)

    check_rows = []
    repaired_ids = set(audit_index.keys())

    missing_in_raw = []
    missing_in_repaired = []

    for criterion_id, audit in audit_index.items():
        raw_leaf = raw_index.get(criterion_id)
        repaired_leaf = repaired_index.get(criterion_id)

        if raw_leaf is None:
            missing_in_raw.append(criterion_id)
            continue

        if repaired_leaf is None:
            missing_in_repaired.append(criterion_id)
            continue

        check_rows.append(
            check_repaired_leaf(
                branch=branch,
                criterion_id=criterion_id,
                raw=raw_leaf,
                repaired=repaired_leaf,
                audit=audit,
            )
        )

    # Also check that leaves not mentioned in repair audit did not change.
    check_rows.extend(
        check_unrepaired_leaves(
            branch=branch,
            raw_index=raw_index,
            repaired_index=repaired_index,
            repaired_criterion_ids=repaired_ids,
        )
    )

    status_counts = Counter(r.get("check_status", "") for r in check_rows)
    pending_count = sum(int(r.get("has_pending_conservative_issue", 0)) for r in check_rows)

    summary = {
        "branch": branch,
        "raw_ast": str(raw_path),
        "repaired_ast": str(repaired_path),
        "raw_leaf_count": len(raw_index),
        "repaired_leaf_count": len(repaired_index),
        "repair_audit_rows_for_branch": len(audit_index),
        "missing_audit_leaves_in_raw": missing_in_raw[:20],
        "missing_audit_leaves_in_repaired": missing_in_repaired[:20],
        "missing_audit_leaves_in_raw_count": len(missing_in_raw),
        "missing_audit_leaves_in_repaired_count": len(missing_in_repaired),
        "check_status_counts": dict(status_counts.most_common()),
        "pending_conservative_issue_after_safe_repair_count": pending_count,
    }

    return check_rows, summary


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLayer 3 safe structural repair check")
    print("Repair audit:", REPAIR_AUDIT_CSV)

    audit_rows = read_csv(REPAIR_AUDIT_CSV)

    check_a, summary_a = process_branch(
        branch="A",
        raw_path=BRANCH_A_RAW_AST,
        repaired_path=BRANCH_A_REPAIRED_AST,
        audit_rows=audit_rows,
    )

    check_b, summary_b = process_branch(
        branch="B",
        raw_path=BRANCH_B_RAW_AST,
        repaired_path=BRANCH_B_REPAIRED_AST,
        audit_rows=audit_rows,
    )

    EXPECTED_LEAVES = 2402

    for branch_name, summary in {"A": summary_a, "B": summary_b}.items():
        if summary["raw_leaf_count"] != EXPECTED_LEAVES:
            raise RuntimeError(
                f"Branch {branch_name}: raw leaf count is {summary['raw_leaf_count']}, "
                f"expected {EXPECTED_LEAVES}."
            )

        if summary["repaired_leaf_count"] != EXPECTED_LEAVES:
            raise RuntimeError(
                f"Branch {branch_name}: repaired leaf count is {summary['repaired_leaf_count']}, "
                f"expected {EXPECTED_LEAVES}."
            )

        if summary["missing_audit_leaves_in_raw_count"] != 0:
            raise RuntimeError(
                f"Branch {branch_name}: audit leaves missing in the original rule tree."
            )

        if summary["missing_audit_leaves_in_repaired_count"] != 0:
            raise RuntimeError(
                f"Branch {branch_name}: audit leaves missing in the repaired rule tree."
            )

    all_check_rows = check_a + check_b
    write_csv(OUT_DIFF_CSV, all_check_rows)

    total_status = Counter(r.get("check_status", "") for r in all_check_rows)
    if total_status.get("fail", 0) > 0:
        raise RuntimeError(
            f"06b found failed repair checks: {dict(total_status)}. "
            "Inspect layer3_safe_structural_repair_diff_check.csv before continuing."
        )
    total_pending = sum(int(r.get("has_pending_conservative_issue", 0)) for r in all_check_rows)

    summary = {
        "description": (
            "Checks that safe structural repairs changed only intended fields "
            "in the logical rule trees. This script does not modify them."
        ),
        "inputs": {
            "repair_audit_csv": str(REPAIR_AUDIT_CSV),
            "branch_a_raw_ast": str(BRANCH_A_RAW_AST),
            "branch_a_repaired_ast": str(BRANCH_A_REPAIRED_AST),
            "branch_b_raw_ast": str(BRANCH_B_RAW_AST),
            "branch_b_repaired_ast": str(BRANCH_B_REPAIRED_AST),
        },
        "outputs": {
            "diff_check_csv": str(OUT_DIFF_CSV),
            "summary_json": str(OUT_SUMMARY_JSON),
        },
        "branch_summaries": {
            "A": summary_a,
            "B": summary_b,
        },
        "overall_check_status_counts": dict(total_status.most_common()),
        "overall_pending_conservative_issue_after_safe_repair_count": total_pending,
        "method_notes": [
            "Pass means repaired leaves changed only expected fields.",
            "Pending conservative issues are not failures; they are intentionally left for a later Layer 3 step.",
            "After this check, repaired rule trees must still be re-run through Layer 1 and Layer 2 before final use.",
        ],
    }

    write_json(OUT_SUMMARY_JSON, summary)

    print("\nDONE")
    print("Diff check CSV:", OUT_DIFF_CSV)
    print("Summary JSON:", OUT_SUMMARY_JSON)

    print("\nBranch summaries:")
    print(summary["branch_summaries"])

    print("\nOverall check status counts:")
    print(summary["overall_check_status_counts"])

    print("\nPending conservative issues after safe repair:")
    print(total_pending)


if __name__ == "__main__":
    main()

# Run from the repository root:
# python scripts/03_verification/03_layer3/03_check_safe_structural_repairs.py