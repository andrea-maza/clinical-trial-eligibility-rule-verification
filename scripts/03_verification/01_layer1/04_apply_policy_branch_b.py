"""
04_apply_policy_branch_b.py

Branch B Layer 1 policy and interpretation.

This script does not detect new issues and does not modify the logical
rule tree. It reads the shared deterministic Layer 1 inventories and
assigns a Branch-B-specific interpretation to the existing issue codes.

Branch B uses LLM-based Pass 2 leaf extraction.

Run from the repository root:
python scripts/03_verification/01_layer1/04_apply_policy_branch_b.py
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[3]

LAYER1_INV_DIR = (
    ROOT
    / "outputs"
    / "verification"
    / "layer1"
    / "deterministic_inventory"
)

LAYER1D_DIR = (
    ROOT
    / "outputs"
    / "verification"
    / "layer1"
    / "pass1_pass2_consistency"
)

LEAF_INVENTORY_CSV = (
    LAYER1_INV_DIR
    / "deterministic_verification_inventory_leaf_level.csv"
)

AST_INVENTORY_CSV = (
    LAYER1_INV_DIR
    / "deterministic_verification_inventory_ast_level.csv"
)

LAYER1D_AUDIT_CSV = (
    LAYER1D_DIR
    / "layer1d_pass1_pass2_consistency_audit.csv"
)

OUT_DIR = (
    ROOT
    / "outputs"
    / "verification"
    / "layer1"
    / "policy_branch_b"
)

OUT_CSV = OUT_DIR / "layer1_policy_branch_b_leaf_level.csv"
OUT_JSON = OUT_DIR / "layer1_policy_branch_b_summary.json"

DETERMINISTIC_BRANCH_NAME = "B_llm_pass2"
LAYER1D_BRANCH_NAME = "B_llm_pass2"
POLICY_BRANCH_NAME = "B"


# ---------------------------------------------------------------------
# Branch B policy
# ---------------------------------------------------------------------
# Branch B is semantically stronger than Branch A, so not every Layer 1
# warning should force LLM verification. Hard errors go to mandatory verifier;
# execution-only issues go to computability review; soft warnings feed Layer 2.

SAFE_NORMALIZATION_ISSUES = {
    "comparison_value_is_numeric_string",
}

EXECUTION_OR_COMPUTABILITY_ISSUES = {
    "computable_with_exception_context",
    "computable_with_non_computable_reason",
    "non_computable_without_reason",
    "computability_invalid_enum",
}

HARD_LAYER1A_STRUCTURAL_ISSUES = {
    "entity_text_empty",
    "evidence_text_empty",
    "operator_missing",
    "comparison_without_scalar_value",
    "equality_without_value",
    "pattern_operator_without_scalar_value",
    "between_without_range_value",
    "range_with_both_bounds_missing",
    "range_with_missing_bound",
    "list_operator_without_list_value",
    "existence_operator_with_non_null_value",
    "null_value_type_with_non_null_value",
    "scalar_value_type_with_complex_value",
    "list_value_type_without_list_value",
    "range_value_type_without_dict_value",
    "temporal_context_missing_relation",
    "temporal_context_invalid_relation",
    "temporal_context_missing_anchor_event",
    "temporal_context_invalid_anchor_event",
    "temporal_context_invalid_unit",
    "temporal_context_missing_value",
    "temporal_context_missing_unit",
    "unit_present_without_value",
    "entity_type_invalid_enum",
    "history_context_invalid_enum",
    "criterion_id_missing",
    "criterion_id_malformed",
    "operator_invalid_enum",
    "value_type_invalid_enum",
    "normalized_concept_invalid_system",
    "normalized_concept_system_without_code",
    "range_min_greater_than_max",
}

HARD_LAYER1C_WARNINGS = {
    "comparison_entity_mismatch_from_source",
    "categorical_value_missing",
    "requirement_object_inversion",
    "condition_context_present_without_handling",
    "temporal_anchor_mismatch_from_source",
    "existence_operator_on_quantitative_phrase",
    "entity_text_too_generic",
}

SOFT_LAYER1C_WARNINGS = {
    "comparison_with_non_quantitative_entity",
    "temporal_marker_without_temporal_context",
    "duration_marker_missing_from_temporal_context",
    "critical_qualifier_missing_from_entity",
}

HARD_LAYER1D_ISSUES = {
    "pass2_leaf_without_pass1_clause",
    "evidence_text_not_substring_of_pass1_source_text",
    "negation_clause_with_exists_operator",
    "positive_clause_with_not_exists_operator",
    "negative_entity_with_not_exists_operator",
    "exception_or_condition_clause_without_context_handling",
    "exception_or_condition_clause_with_computable_status",
    "exception_clause_computable_despite_context",
}

AST_INTEGRITY_ISSUES = {
    "duplicate_criterion_id",
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def serialize(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, bool):
        return "1" if x else "0"
    if isinstance(x, (list, dict)):
        return json.dumps(x, ensure_ascii=False)
    return str(x)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    priority = [
        "branch", "document_id", "criterion_id", "item_uid", "clause_id", "path",
        "entity_type", "entity_text", "operator", "value_type", "value", "unit",
        "computability", "evidence_text",
        "layer1a_issues", "layer1b_flags", "layer1c_warnings", "layer1d_issues",
        "all_layer1_codes",
        "layer1_policy_bucket", "layer1_policy_severity", "layer1_policy_action_hint",
        "layer1_policy_reasons", "layer1_policy_score",
        "layer1_policy_hard_issue_count", "layer1_policy_soft_warning_count",
        "layer1_policy_execution_issue_count",
    ]

    cols, seen = [], set()
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
            writer.writerow({c: serialize(r.get(c, "")) for c in cols})


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def split_codes(x: Any) -> List[str]:
    if x is None:
        return []
    s = str(x).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [str(v).strip() for v in obj if str(v).strip()]
        except Exception:
            pass
    return [p.strip() for p in s.split(";") if p.strip()]


def criterion_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return (str(row.get("document_id", "")), str(row.get("criterion_id", "")))


def path_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return (str(row.get("document_id", "")), str(row.get("path", "")))


def build_ast_flag_index(ast_rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], List[str]]:
    out: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for r in ast_rows:
        if r.get("branch") != DETERMINISTIC_BRANCH_NAME:
            continue

        issue = r.get("issue_type", "").strip()
        doc = r.get("document_id", "").strip()

        for p in split_codes(r.get("paths", "")):
            out[(doc, p)].append(issue)

    return {k: sorted(set(v)) for k, v in out.items()}


def build_layer1d_index(layer1d_rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], List[str]]:
    out: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for r in layer1d_rows:
        if r.get("branch") != LAYER1D_BRANCH_NAME:
            continue
        if r.get("row_type") != "pass2_leaf_check":
            continue

        out[criterion_key(r)].extend(split_codes(r.get("issues", "")))

    return {k: sorted(set(v)) for k, v in out.items()}


def collect_missing_pass2_rows(layer1d_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    rows = []

    for r in layer1d_rows:
        if r.get("branch") != LAYER1D_BRANCH_NAME:
            continue
        if r.get("row_type") != "pass1_clause_missing_pass2_leaf":
            continue

        rows.append({
            "branch": POLICY_BRANCH_NAME,
            "document_id": r.get("document_id", ""),
            "criterion_id": "",
            "item_uid": r.get("item_uid", ""),
            "clause_id": r.get("clause_id", ""),
            "path": "",
            "entity_type": "",
            "entity_text": "",
            "operator": "",
            "value_type": "",
            "value": "",
            "unit": "",
            "computability": "",
            "evidence_text": "",
            "layer1a_issues": "",
            "layer1b_flags": "",
            "layer1c_warnings": "",
            "layer1d_issues": "pass1_clause_without_pass2_leaf",
            "all_layer1_codes": "layer1d:pass1_clause_without_pass2_leaf",
            "layer1_policy_bucket": "missing_leaf",
            "layer1_policy_severity": "high",
            "layer1_policy_action_hint": "mandatory_verifier_candidate",
            "layer1_policy_reasons": "Pass 1 clause exists but Branch B did not produce a corresponding Pass 2 leaf.",
            "layer1_policy_score": 1.0,
            "layer1_policy_hard_issue_count": 1,
            "layer1_policy_soft_warning_count": 0,
            "layer1_policy_execution_issue_count": 0,
        })

    return rows


def decide_policy(
    layer1a: List[str],
    layer1b: List[str],
    layer1c: List[str],
    layer1d: List[str],
) -> Tuple[str, str, str, str, float, int, int, int]:
    set_a, set_b, set_c, set_d = set(layer1a), set(layer1b), set(layer1c), set(layer1d)

    hard_structural = sorted((set_a & HARD_LAYER1A_STRUCTURAL_ISSUES) | (set_b & AST_INTEGRITY_ISSUES))
    hard_semantic = sorted((set_c & HARD_LAYER1C_WARNINGS) | (set_d & HARD_LAYER1D_ISSUES))
    soft_semantic = sorted(set_c & SOFT_LAYER1C_WARNINGS)
    execution = sorted(set_a & EXECUTION_OR_COMPUTABILITY_ISSUES)
    safe_norm = sorted(set_a & SAFE_NORMALIZATION_ISSUES)

    hard_count = len(hard_structural) + len(hard_semantic)
    soft_count = len(soft_semantic)
    execution_count = len(execution)

    reasons = []
    if hard_structural:
        reasons.append("hard_structural:" + "|".join(hard_structural))
    if hard_semantic:
        reasons.append("hard_semantic:" + "|".join(hard_semantic))
    if soft_semantic:
        reasons.append("soft_semantic:" + "|".join(soft_semantic))
    if execution:
        reasons.append("execution_or_computability:" + "|".join(execution))
    if safe_norm:
        reasons.append("safe_normalization:" + "|".join(safe_norm))

    if hard_count > 0:
        return (
            "hard_semantic_or_structural_issue",
            "high",
            "mandatory_verifier_candidate",
            "; ".join(reasons),
            1.00,
            hard_count,
            soft_count,
            execution_count,
        )

    if execution_count > 0:
        return (
            "execution_or_computability_issue",
            "medium",
            "computability_review_candidate",
            "; ".join(reasons),
            0.65,
            hard_count,
            soft_count,
            execution_count,
        )

    if soft_count > 0:
        return (
            "soft_semantic_warning",
            "low_to_medium",
            "continue_to_layer2_grounding_screen",
            "; ".join(reasons),
            0.40,
            hard_count,
            soft_count,
            execution_count,
        )

    if safe_norm:
        return (
            "safe_normalization_only",
            "low",
            "safe_normalization_candidate",
            "; ".join(reasons),
            0.15,
            hard_count,
            soft_count,
            execution_count,
        )

    return (
        "no_layer1_issue",
        "none",
        "continue_to_layer2",
        "",
        0.00,
        hard_count,
        soft_count,
        execution_count,
    )


def main() -> None:
    print("\nBranch B Layer 1 policy")
    print(f"Leaf inventory: {LEAF_INVENTORY_CSV}")
    print(f"Rule-tree inventory: {AST_INVENTORY_CSV}")
    print(f"Layer 1D audit: {LAYER1D_AUDIT_CSV}")
    print(f"Output CSV: {OUT_CSV}")
    print(f"Output JSON: {OUT_JSON}")

    leaf_rows = read_csv(LEAF_INVENTORY_CSV)
    ast_rows = read_csv(AST_INVENTORY_CSV)
    layer1d_rows = read_csv(LAYER1D_AUDIT_CSV)

    ast_index = build_ast_flag_index(ast_rows)
    layer1d_index = build_layer1d_index(layer1d_rows)

    out_rows: List[Dict[str, Any]] = []

    for r in leaf_rows:
        if r.get("branch") != DETERMINISTIC_BRANCH_NAME:
            continue

        l1a = split_codes(r.get("deterministic_issues", ""))
        l1b = ast_index.get(path_key(r), [])
        l1c = split_codes(r.get("layer1c_source_text_warnings", ""))
        l1d = layer1d_index.get(criterion_key(r), [])

        bucket, severity, action_hint, reasons, score, hard_count, soft_count, execution_count = decide_policy(
            l1a, l1b, l1c, l1d
        )

        all_codes = (
            [f"layer1a:{x}" for x in l1a]
            + [f"layer1b:{x}" for x in l1b]
            + [f"layer1c:{x}" for x in l1c]
            + [f"layer1d:{x}" for x in l1d]
        )

        out_rows.append({
            "branch": POLICY_BRANCH_NAME,
            "document_id": r.get("document_id", ""),
            "criterion_id": r.get("criterion_id", ""),
            "item_uid": r.get("item_uid", ""),
            "clause_id": r.get("clause_id", ""),
            "path": r.get("path", ""),
            "entity_type": r.get("entity_type", ""),
            "entity_text": r.get("entity_text", ""),
            "operator": r.get("operator", ""),
            "value_type": r.get("value_type", ""),
            "value": r.get("value", ""),
            "unit": r.get("unit", ""),
            "computability": r.get("computability", ""),
            "evidence_text": r.get("evidence_text", ""),
            "layer1a_issues": ";".join(l1a),
            "layer1b_flags": ";".join(l1b),
            "layer1c_warnings": ";".join(l1c),
            "layer1d_issues": ";".join(l1d),
            "all_layer1_codes": ";".join(all_codes),
            "layer1_policy_bucket": bucket,
            "layer1_policy_severity": severity,
            "layer1_policy_action_hint": action_hint,
            "layer1_policy_reasons": reasons,
            "layer1_policy_score": score,
            "layer1_policy_hard_issue_count": hard_count,
            "layer1_policy_soft_warning_count": soft_count,
            "layer1_policy_execution_issue_count": execution_count,
        })

    out_rows.extend(collect_missing_pass2_rows(layer1d_rows))

    write_csv(OUT_CSV, out_rows)

    action_counts = Counter(r["layer1_policy_action_hint"] for r in out_rows)
    bucket_counts = Counter(r["layer1_policy_bucket"] for r in out_rows)
    severity_counts = Counter(r["layer1_policy_severity"] for r in out_rows)

    issue_counter = Counter()
    for r in out_rows:
        for code in split_codes(r.get("all_layer1_codes", "")):
            issue_counter[code] += 1

    summary = {
        "stage": "layer1_policy_branch_b",
        "description": ( 
            "Branch-specific interpretation of common deterministic Layer 1 issue codes. " 
            "No new issues are detected and no rule-tree modification is applied. " 
            "Branch B uses LLM-based Pass 2 leaf extraction, so hard deterministic " 
            "issues are separated from soft semantic warnings and " 
            "execution/computability issues." ),
        "inputs": {
            "leaf_inventory_csv": str(LEAF_INVENTORY_CSV),
            "ast_inventory_csv": str(AST_INVENTORY_CSV),
            "layer1d_audit_csv": str(LAYER1D_AUDIT_CSV),
        },
        "outputs": {
            "leaf_policy_csv": str(OUT_CSV),
            "summary_json": str(OUT_JSON),
        },
        "branch": POLICY_BRANCH_NAME,
        "n_policy_rows": len(out_rows),
        "action_hint_counts": dict(action_counts.most_common()),
        "bucket_counts": dict(bucket_counts.most_common()),
        "severity_counts": dict(severity_counts.most_common()),
        "top_layer1_codes": dict(issue_counter.most_common(30)),
        "method_note": (
            "This is a policy layer, not a detector. Actual repair/rescue decisions "
            "should be made in Layer 3. Branch B Layer 2 can use these policy columns "
            "instead of treating every Layer 1 warning as mandatory LLM-verifier input."
        ),
    }

    write_json(OUT_JSON, summary)

    print("DONE")
    print(f"Rows written: {len(out_rows)}")
    print("Action hints:", dict(action_counts.most_common()))
    print("Buckets:", dict(bucket_counts.most_common()))
    print("Severities:", dict(severity_counts.most_common()))


if __name__ == "__main__":
    main()

# Run from the repository root: 
# # python scripts/03_verification/01_layer1/04_apply_policy_branch_b.py