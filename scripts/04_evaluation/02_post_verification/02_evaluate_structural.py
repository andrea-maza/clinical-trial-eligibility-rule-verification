"""
02_evaluate_structural.py

Evaluate the structural validity and internal consistency of the
post-verification Branch A and Branch B logical rule trees.

The same evaluation logic is used for the pre- and post-verification
outputs, allowing direct comparison of:

    - schema-valid document rate
    - logical group and leaf counts
    - evidence and entity completeness
    - operator--value consistency
    - computability distribution
    - structural issue counts

Inputs:
    outputs/verification/layer3/
        applied_candidate_selection_rescue/

Outputs:
    outputs/evaluation/post_verification/
        structural_post_verification_A_B/

The script also compares the post-verification results with the
pre-verification structural summary.

This script does not call the LLM and does not modify predictions.

Run from the repository root:
python scripts/04_evaluation/02_post_verification/02_evaluate_structural.py
"""


from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jsonschema import Draft7Validator


# --------------------------------------------------
# IO helpers
# --------------------------------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_schema(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------
# Basic helpers
# --------------------------------------------------

COMPARISON_OPERATORS = {"<", "<=", ">", ">=", "=", "!="}
LIST_OPERATORS = {"in", "not_in"}


def safe_div(num: int, den: int) -> Optional[float]:
    if den == 0:
        return None

    return round(num / den, 4)


def is_scalar_value(value: Any) -> bool:
    return isinstance(value, (int, float, str)) and value is not None


def is_range_value(value: Any) -> bool:
    return isinstance(value, dict) and "min" in value and "max" in value


def is_list_value(value: Any) -> bool:
    return isinstance(value, list) and len(value) >= 1


def add_issue(
    issues: List[Dict[str, Any]],
    issue_type: str,
    path: str,
    detail: str,
) -> None:
    issues.append(
        {
            "issue_type": issue_type,
            "path": path,
            "detail": detail,
        }
    )


# --------------------------------------------------
# Logical rule-tree traversal
# --------------------------------------------------

def walk_node(
    node: Dict[str, Any],
    path: str,
    groups: List[Dict[str, Any]],
    criteria: List[Tuple[str, Dict[str, Any]]],
) -> None:
    node_type = node.get("node_type")

    if node_type == "group":
        groups.append({"path": path, "node": node})

        children = node.get("children", [])

        for i, child in enumerate(children):
            walk_node(child, f"{path}.children[{i}]", groups, criteria)

        return

    if node_type == "criterion":
        criteria.append((path, node.get("criterion", {})))
        return


# --------------------------------------------------
# Criterion checks
# --------------------------------------------------

def check_criterion(
    path: str,
    criterion: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    criterion_id = criterion.get("criterion_id")
    entity_text = criterion.get("entity_text")
    operator = criterion.get("operator")
    value_type = criterion.get("value_type")
    value = criterion.get("value")
    temporal_context = criterion.get("temporal_context")
    computability = criterion.get("computability")
    evidence_text = criterion.get("evidence_text")
    provenance = criterion.get("provenance")

    if not str(criterion_id or "").strip():
        add_issue(
            issues,
            "missing_criterion_id",
            path,
            "criterion_id is empty or missing.",
        )

    if not str(entity_text or "").strip():
        add_issue(
            issues,
            "missing_entity_text",
            path,
            "entity_text is empty or missing.",
        )

    if not str(evidence_text or "").strip():
        add_issue(
            issues,
            "missing_evidence_text",
            path,
            "evidence_text is empty or missing.",
        )

    # value_type consistency
    if value_type == "null":
        if value is not None:
            add_issue(
                issues,
                "null_value_type_with_nonnull_value",
                path,
                f"value_type is 'null' but value is {value!r}.",
            )

    elif value_type == "scalar":
        if not is_scalar_value(value):
            add_issue(
                issues,
                "scalar_value_type_with_invalid_value",
                path,
                f"value_type is 'scalar' but value is {value!r}.",
            )

    elif value_type == "range":
        if not is_range_value(value):
            add_issue(
                issues,
                "range_value_type_with_invalid_value",
                path,
                f"value_type is 'range' but value is {value!r}.",
            )
        else:
            if value.get("min") is None and value.get("max") is None:
                add_issue(
                    issues,
                    "range_with_both_bounds_missing",
                    path,
                    "range value has both min and max missing.",
                )

    elif value_type == "list":
        if not is_list_value(value):
            add_issue(
                issues,
                "list_value_type_with_invalid_value",
                path,
                f"value_type is 'list' but value is {value!r}.",
            )

    # operator / value consistency
    if operator in COMPARISON_OPERATORS:
        if value_type != "scalar" or value is None:
            add_issue(
                issues,
                "comparison_without_scalar_value",
                path,
                f"operator {operator!r} requires scalar value, but value_type={value_type!r}, value={value!r}.",
            )

    if operator == "between":
        if value_type != "range" or not is_range_value(value):
            add_issue(
                issues,
                "between_without_range",
                path,
                f"operator 'between' requires range value, but value_type={value_type!r}, value={value!r}.",
            )

    if operator in LIST_OPERATORS:
        if value_type != "list":
            add_issue(
                issues,
                "list_operator_without_list_value",
                path,
                f"operator {operator!r} usually expects value_type='list', but got {value_type!r}.",
            )

    # temporal consistency
    if temporal_context is not None:
        relation = temporal_context.get("relation")
        t_value = temporal_context.get("value")
        t_unit = temporal_context.get("unit")
        anchor_event = temporal_context.get("anchor_event")

        if not relation:
            add_issue(
                issues,
                "temporal_missing_relation",
                path,
                "temporal_context.relation is missing.",
            )

        if not anchor_event:
            add_issue(
                issues,
                "temporal_missing_anchor_event",
                path,
                "temporal_context.anchor_event is missing.",
            )

        if relation == "within":
            if t_value is None:
                add_issue(
                    issues,
                    "within_temporal_missing_value",
                    path,
                    "temporal relation 'within' should usually include a value.",
                )

            if t_unit is None:
                add_issue(
                    issues,
                    "within_temporal_missing_unit",
                    path,
                    "temporal relation 'within' should usually include a unit.",
                )

    # computability sanity
    if computability == "computable":
        if provenance and provenance.get("source_exception_context"):
            add_issue(
                issues,
                "computable_with_exception_context",
                path,
                "Leaf is marked computable but provenance contains exception context.",
            )

    if computability == "non_computable":
        reason = criterion.get("non_computable_reason")

        if not str(reason or "").strip():
            add_issue(
                issues,
                "non_computable_without_reason",
                path,
                "Leaf is non_computable but non_computable_reason is empty.",
            )


# --------------------------------------------------
# Document evaluation
# --------------------------------------------------

def evaluate_document(
    row: Dict[str, Any],
    validator: Draft7Validator,
) -> Dict[str, Any]:
    document_id = row.get("document_id")
    ast = row.get("rules_v3_ast")

    issues: List[Dict[str, Any]] = []
    groups: List[Dict[str, Any]] = []
    criteria: List[Tuple[str, Dict[str, Any]]] = []

    schema_errors = sorted(
        validator.iter_errors(ast),
        key=lambda e: list(e.absolute_path),
    )
    schema_valid = len(schema_errors) == 0

    schema_error_messages = []

    for e in schema_errors:
        error_path = ".".join(str(x) for x in e.absolute_path)
        schema_error_messages.append(
            {
                "message": e.message,
                "path": error_path,
                "validator": e.validator,
            }
        )

    # IMPORTANT:
    # This intentionally matches the pre-verification script.
    # Leaves/groups are counted only if the AST document is schema-valid.
    if schema_valid:
        if ast.get("inclusion_criteria") is not None:
            walk_node(
                ast["inclusion_criteria"],
                "inclusion_criteria",
                groups,
                criteria,
            )

        if ast.get("exclusion_criteria") is not None:
            walk_node(
                ast["exclusion_criteria"],
                "exclusion_criteria",
                groups,
                criteria,
            )

        # per-criterion checks
        for path, criterion in criteria:
            check_criterion(path, criterion, issues)

        # duplicate criterion_ids
        criterion_ids = [
            c.get("criterion_id")
            for _, c in criteria
            if c.get("criterion_id") is not None
        ]

        counts = Counter(criterion_ids)

        for cid, cnt in counts.items():
            if cnt > 1:
                add_issue(
                    issues,
                    "duplicate_criterion_id",
                    document_id or "",
                    f"criterion_id {cid!r} appears {cnt} times.",
                )

    criteria_only = [c for _, c in criteria]

    n_leaves = len(criteria_only)
    n_groups = len(groups)

    evidence_nonempty = sum(
        1 for c in criteria_only
        if str(c.get("evidence_text") or "").strip()
    )

    entity_nonempty = sum(
        1 for c in criteria_only
        if str(c.get("entity_text") or "").strip()
    )

    computability_counts = Counter(c.get("computability") for c in criteria_only)

    temporal_present = sum(
        1 for c in criteria_only
        if c.get("temporal_context") is not None
    )

    comparison_valid = 0
    comparison_total = 0

    for c in criteria_only:
        op = c.get("operator")

        if op in COMPARISON_OPERATORS:
            comparison_total += 1

            if c.get("value_type") == "scalar" and c.get("value") is not None:
                comparison_valid += 1

        elif op == "between":
            comparison_total += 1

            if c.get("value_type") == "range" and is_range_value(c.get("value")):
                comparison_valid += 1

    issue_counts = Counter(i["issue_type"] for i in issues)

    return {
        "document_id": document_id,
        "schema_valid": schema_valid,
        "schema_errors": schema_error_messages,
        "n_groups": n_groups,
        "n_leaves": n_leaves,
        "evidence_nonempty_rate": safe_div(evidence_nonempty, n_leaves),
        "entity_nonempty_rate": safe_div(entity_nonempty, n_leaves),
        "comparison_consistency_rate": safe_div(comparison_valid, comparison_total),
        "n_temporal_leaves": temporal_present,
        "computability_counts": dict(computability_counts),
        "n_issues": len(issues),
        "issue_counts": dict(issue_counts),
        "issues": issues,
    }


# --------------------------------------------------
# Branch-level summary
# --------------------------------------------------

def summarize_branch(
    branch_name: str,
    in_path: Path,
    schema_path: Path,
    out_root: Path,
) -> Dict[str, Any] | None:
    if not in_path.exists():
        print(f"[SKIP] {branch_name}: file not found: {in_path}")
        return None

    rows = load_jsonl(in_path)
    schema = load_schema(schema_path)
    validator = Draft7Validator(schema)

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    err_rows = [r for r in rows if r.get("status") != "ok"]

    document_details = [
        evaluate_document(r, validator)
        for r in ok_rows
    ]

    n_docs = len(rows)
    n_docs_ok = len(ok_rows)
    n_docs_err = len(err_rows)

    schema_valid_docs = sum(1 for d in document_details if d["schema_valid"])
    total_groups = sum(d["n_groups"] for d in document_details)
    total_leaves = sum(d["n_leaves"] for d in document_details)
    total_issues = sum(d["n_issues"] for d in document_details)

    all_issue_counts = Counter()
    all_computability_counts = Counter()

    evidence_num = 0
    evidence_den = 0
    entity_num = 0
    entity_den = 0

    for d in document_details:
        all_issue_counts.update(d["issue_counts"])
        all_computability_counts.update(d["computability_counts"])

        if d["evidence_nonempty_rate"] is not None:
            evidence_num += round(d["evidence_nonempty_rate"] * d["n_leaves"])
            evidence_den += d["n_leaves"]

        if d["entity_nonempty_rate"] is not None:
            entity_num += round(d["entity_nonempty_rate"] * d["n_leaves"])
            entity_den += d["n_leaves"]

    comparison_rates = [
        d["comparison_consistency_rate"]
        for d in document_details
        if d["comparison_consistency_rate"] is not None
    ]

    mean_comparison_rate = (
        round(sum(comparison_rates) / len(comparison_rates), 4)
        if comparison_rates
        else None
    )

    summary = {
        "evaluation_stage": "structural_post_verification_A_B",
        "branch": branch_name,
        "input_file": str(in_path),
        "schema_file": str(schema_path),
        "n_documents_total": n_docs,
        "n_documents_ok_rows": n_docs_ok,
        "n_documents_error_rows": n_docs_err,
        "schema_valid_document_rate": safe_div(schema_valid_docs, n_docs_ok),
        "total_groups": total_groups,
        "total_leaves": total_leaves,
        "avg_groups_per_doc": round(total_groups / n_docs_ok, 4) if n_docs_ok else None,
        "avg_leaves_per_doc": round(total_leaves / n_docs_ok, 4) if n_docs_ok else None,
        "evidence_nonempty_rate": safe_div(evidence_num, evidence_den),
        "entity_text_nonempty_rate": safe_div(entity_num, entity_den),
        "mean_document_comparison_consistency_rate": mean_comparison_rate,
        "computability_counts": dict(all_computability_counts),
        "total_issues": total_issues,
        "issue_counts": dict(all_issue_counts),
        "document_errors": [
            {
                "document_id": r.get("document_id"),
                "status": r.get("status"),
                "error": r.get("error"),
            }
            for r in err_rows
        ],
    }

    out_dir = out_root / branch_name
    out_dir.mkdir(parents=True, exist_ok=True)

    write_json(out_dir / "summary.json", summary)
    write_jsonl(out_dir / "document_details.jsonl", document_details)

    print(f"\n===== STRUCTURAL POST-VERIFICATION: {branch_name} =====")
    print(f"Input file: {in_path}")
    print(f"Documents total: {summary['n_documents_total']}")
    print(f"Documents with ok rows: {summary['n_documents_ok_rows']}")
    print(f"Schema-valid document rate: {summary['schema_valid_document_rate']}")
    print(f"Total groups: {summary['total_groups']}")
    print(f"Total leaves: {summary['total_leaves']}")
    print(f"Evidence non-empty rate: {summary['evidence_nonempty_rate']}")
    print(f"Entity text non-empty rate: {summary['entity_text_nonempty_rate']}")
    print(f"Mean comparison consistency rate: {summary['mean_document_comparison_consistency_rate']}")
    print(f"Computability counts: {summary['computability_counts']}")
    print(f"Total issues: {summary['total_issues']}")
    print(f"Issue counts: {summary['issue_counts']}")

    return summary


# --------------------------------------------------
# Pre/post comparison
# --------------------------------------------------

def compact_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "n_documents_total": summary["n_documents_total"],
        "schema_valid_document_rate": summary["schema_valid_document_rate"],
        "total_groups": summary["total_groups"],
        "total_leaves": summary["total_leaves"],
        "evidence_nonempty_rate": summary["evidence_nonempty_rate"],
        "entity_text_nonempty_rate": summary["entity_text_nonempty_rate"],
        "mean_document_comparison_consistency_rate": summary["mean_document_comparison_consistency_rate"],
        "computability_counts": summary["computability_counts"],
        "total_issues": summary["total_issues"],
        "issue_counts": summary["issue_counts"],
    }


def load_pre_summary(root: Path) -> Dict[str, Any]:
    path = (
        root
        / "outputs"
        / "evaluation"
        / "pre_verification"
        / "structural_pre_verification_A_B"
        / "all_branch_summary.json"
    )

    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_pre_post_comparison(
    root: Path,
    post_summaries: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    pre = load_pre_summary(root)

    mapping = {
        "A_post_verification": "A_bert_rules",
        "B_post_verification": "B_llm_pass2",
    }

    comparison = {}

    for post_branch, pre_branch in mapping.items():
        comparison[post_branch] = {
            "pre_branch": pre_branch,
            "post_branch": post_branch,
            "pre": pre.get(pre_branch, {}),
            "post": post_summaries.get(post_branch, {}),
            "interpretation_note": (
                "This is a strict structural diagnostic. It uses the same logic as "
                "the pre-verification script: groups/leaves are counted only for "
                "schema-valid logical rule-tree documents."
            ),
        }

    return comparison


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    ROOT = Path(__file__).resolve().parents[3]

    schema_candidates = sorted(
        (ROOT / "schemas").rglob("rules_v3.json")
    )

    if len(schema_candidates) != 1:
        raise RuntimeError(
            "Expected exactly one rules_v3.json schema, found: "
            + ", ".join(str(path) for path in schema_candidates)
        )

    schema_path = schema_candidates[0]

    branch_paths = {
        "A_post_verification": (
            ROOT
            / "outputs"
            / "verification"
            / "layer3"
            / "applied_candidate_selection_rescue"
            / "chia_text_only_200_rules_v3_ast_A_layer3_candidate_selection_rescue.jsonl"
        ),
        "B_post_verification": (
            ROOT
            / "outputs"
            / "verification"
            / "layer3"
            / "applied_candidate_selection_rescue"
            / "chia_text_only_200_rules_v3_ast_B_layer3_candidate_selection_rescue.jsonl"
        ),
    }

    required_paths = {
        "rule-tree schema": schema_path,
        **{
            f"{branch_name} rule trees": path
            for branch_name, path in branch_paths.items()
        },
    }

    for name, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    out_root = (
        ROOT
        / "outputs"
        / "evaluation"
        / "post_verification"
        / "structural_post_verification_A_B"
    )

    out_root.mkdir(parents=True, exist_ok=True)

    all_summaries = {}

    for branch_name, in_path in branch_paths.items():
        summary = summarize_branch(
            branch_name=branch_name,
            in_path=in_path,
            schema_path=schema_path,
            out_root=out_root,
        )

        if summary is not None:
            all_summaries[branch_name] = compact_summary(summary)

    all_summary_path = out_root / "all_branch_summary.json"
    write_json(all_summary_path, all_summaries)

    pre_post = build_pre_post_comparison(
        root=ROOT,
        post_summaries=all_summaries,
    )

    comparison_path = (
        out_root / "pre_post_structural_comparison.json"
    )
    write_json(comparison_path, pre_post)

    print("\n===== POST-VERIFICATION STRUCTURAL SUMMARY =====")
    print("Schema:", schema_path)
    print("Post-verification summary:", all_summary_path)
    print("Pre/post comparison:", comparison_path)


if __name__ == "__main__":
    main()


# Run from the repository root:
# python scripts/04_evaluation/02_post_verification/02_evaluate_structural.py