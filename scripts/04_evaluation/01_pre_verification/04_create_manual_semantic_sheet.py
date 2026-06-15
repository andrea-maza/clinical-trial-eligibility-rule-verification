"""
04_create_manual_semantic_sheet.py

Create the pre-verification manual semantic review sheets for
Branch A and Branch B.

Outputs:
    - shared Pass 1 logic review sheet
    - clause-level Branch A and Branch B semantic review sheet
    - review manifest

Important:
    This script creates blank manual-label columns. It must not be run
    when completed manual review sheets already exist.

This script does not call the LLM and does not modify extraction outputs.

Run from the repository root only when creating a new review:
python scripts/04_evaluation/01_pre_verification/04_create_manual_semantic_sheet.py
"""

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List


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


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def safe_json(obj: Any) -> str:
    if obj is None:
        return ""
    return json.dumps(obj, ensure_ascii=False)


def clause_sort_key(clause_id: str) -> int:
    m = re.search(r"(\d+)$", clause_id or "")
    return int(m.group(1)) if m else 999999


def build_logic_chain(clauses: List[Dict[str, Any]]) -> str:
    parts = []

    for clause in clauses:
        clause_id = clause.get("clause_id")
        clause_text = normalize_text(clause.get("clause_text"))
        is_negated = clause.get("is_negated")
        connector = clause.get("connector_to_next")
        quantifier = clause.get("quantifier")

        token = f"{clause_id}: {clause_text}"

        if is_negated:
            token = f"NOT({token})"

        if quantifier:
            qtype = quantifier.get("quantifier_type")
            qvalue = quantifier.get("value")
            token = f"[{qtype} {qvalue}] {token}"

        parts.append(token)

        if connector is not None:
            parts.append(connector)

    return " ".join(parts)


def classify_stratum(item_text: str, clauses: List[Dict[str, Any]], criteria_by_branch: Dict[str, List[Dict[str, Any]]]) -> str:
    t = normalize_text(item_text).lower()

    if any(marker in t for marker in ["unless", "except", "with the exception of", " if "]):
        return "exception"

    if any(c.get("is_negated") for c in clauses) or any(marker in f" {t} " for marker in [" no ", " without ", " absence of "]):
        return "negation"

    if any(c.get("quantifier") is not None for c in clauses):
        return "quantifier"

    all_criteria = []
    for criteria in criteria_by_branch.values():
        all_criteria.extend(criteria)

    if any((e.get("criterion") or {}).get("temporal_context") is not None for e in all_criteria):
        return "temporal"

    if any(marker in t for marker in ["within", "prior to", "before", "after", "since", "screening", "baseline"]):
        return "temporal"

    if any((e.get("criterion") or {}).get("computability") == "non_computable" for e in all_criteria):
        return "non_computable"

    if len(clauses) > 1:
        return "multi_clause"

    return "simple"


def get_branch_criteria(pass2_by_item_uid: Dict[str, Dict[str, Any]], item_uid: str) -> List[Dict[str, Any]]:
    row = pass2_by_item_uid.get(item_uid)

    if row is None or row.get("status") != "ok":
        return []

    return (row.get("pass2_output") or {}).get("criteria", []) or []


def get_criteria_map(criteria: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        entry.get("clause_id"): entry
        for entry in criteria
        if entry.get("clause_id")
    }


def get_criterion(pass2_by_item_uid: Dict[str, Dict[str, Any]], item_uid: str, clause_id: str) -> Dict[str, Any]:
    criteria = get_branch_criteria(pass2_by_item_uid, item_uid)
    entry = get_criteria_map(criteria).get(clause_id, {})
    return entry.get("criterion", {}) or {}


def compact_leaf(criterion: Dict[str, Any]) -> str:
    if not criterion:
        return "MISSING"

    parts = [
        f"entity_type={criterion.get('entity_type', '')}",
        f"entity={normalize_text(criterion.get('entity_text', ''))}",
        f"operator={criterion.get('operator', '')}",
        f"value_type={criterion.get('value_type', '')}",
        f"value={safe_json(criterion.get('value'))}",
        f"unit={criterion.get('unit', '')}",
        f"temporal={safe_json(criterion.get('temporal_context'))}",
        f"history={criterion.get('history_context', '')}",
        f"computability={criterion.get('computability', '')}",
        f"reason={normalize_text(criterion.get('non_computable_reason', ''))}",
    ]

    return " | ".join(parts)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    ROOT = Path(__file__).resolve().parents[3]

    pass1_path = (
        ROOT
        / "outputs"
        / "extraction"
        / "pass1_flat"
        / "chia_text_only_200_pass1_flat.jsonl"
    )

    branch_paths = {
        "A": (
            ROOT
            / "outputs"
            / "extraction"
            / "branch_a"
            / "pass2_leaves"
            / "chia_text_only_200_pass2_leaves.jsonl"
        ),
        "B": (
            ROOT
            / "outputs"
            / "extraction"
            / "branch_b"
            / "pass2_leaves_llm"
            / "chia_text_only_200_pass2_leaves_llm.jsonl"
        ),
    }

    out_dir = (
        ROOT
        / "outputs"
        / "evaluation"
        / "pre_verification"
        / "semantic_manual_pre_verification_A_B"
    )

    manifest_path = out_dir / "semantic_manifest_A_B.json"
    logic_sheet_path = out_dir / "semantic_pass1_logic_sheet.csv"
    ab_sheet_path = out_dir / "semantic_clause_sheet_A_B.csv"

    existing_outputs = [
        path
        for path in [manifest_path, logic_sheet_path, ab_sheet_path]
        if path.exists()
    ]

    if existing_outputs:
        raise FileExistsError(
            "Manual review outputs already exist and will not be overwritten:\n"
            + "\n".join(str(path) for path in existing_outputs)
        )

    required_inputs = {
        "Pass 1 output": pass1_path,
        "Branch A leaves": branch_paths["A"],
        "Branch B leaves": branch_paths["B"],
    }

    for name, path in required_inputs.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    pass1_rows = [
        r for r in load_jsonl(pass1_path)
        if r.get("status") == "ok"
    ]

    pass1_rows = [r for r in load_jsonl(pass1_path) if r.get("status") == "ok"]

    pass2_by_branch = {}
    for branch, path in branch_paths.items():
        rows = [r for r in load_jsonl(path) if r.get("status") == "ok"]
        pass2_by_branch[branch] = {
            r["item_uid"]: r
            for r in rows
            if r.get("item_uid")
        }

    records = []

    for p1 in pass1_rows:
        item_uid = p1["item_uid"]
        item_text = p1.get("item_text", "")
        clauses = (p1.get("parsed_pass1_with_ids") or {}).get("clauses", [])

        criteria_by_branch = {
            branch: get_branch_criteria(index, item_uid)
            for branch, index in pass2_by_branch.items()
        }

        records.append({
            "document_id": p1.get("document_id"),
            "chia_id": p1.get("chia_id"),
            "item_uid": item_uid,
            "item_index": p1.get("item_index"),
            "criterion_type": p1.get("criterion_type_hint"),
            "item_text": item_text,
            "clauses": clauses,
            "stratum": classify_stratum(item_text, clauses, criteria_by_branch),
        })

    records = sorted(
        records,
        key=lambda r: (
            str(r["document_id"]),
            0 if r["criterion_type"] == "inclusion" else 1,
            int(r["item_index"]) if r["item_index"] is not None else 999999,
        ),
    )

    logic_rows = []
    ab_rows = []

    for i, record in enumerate(records, start=1):
        review_id = f"I{i:05d}"
        item_uid = record["item_uid"]
        clauses = record["clauses"]

        logic_rows.append({
            "review_id": review_id,
            "document_id": record["document_id"],
            "chia_id": record["chia_id"],
            "item_uid": item_uid,
            "item_index": record["item_index"],
            "criterion_type": record["criterion_type"],
            "stratum": record["stratum"],
            "item_text": record["item_text"],
            "n_clauses": len(clauses),
            "logic_chain": build_logic_chain(clauses),
            "manual_logic_label": "",
            "manual_logic_issue_type": "",
            "manual_logic_notes": "",
        })

        clause_map = {
            c["clause_id"]: c
            for c in clauses
            if c.get("clause_id")
        }

        for clause_id in sorted(clause_map.keys(), key=clause_sort_key):
            clause = clause_map[clause_id]

            common = {
                "review_id": review_id,
                "document_id": record["document_id"],
                "chia_id": record["chia_id"],
                "item_uid": item_uid,
                "item_index": record["item_index"],
                "criterion_type": record["criterion_type"],
                "stratum": record["stratum"],
                "item_text": record["item_text"],
                "clause_id": clause_id,
                "clause_text": normalize_text(clause.get("clause_text")),
                "evidence_text": normalize_text(clause.get("evidence_text")),
                "is_negated": clause.get("is_negated"),
                "connector_to_next": clause.get("connector_to_next"),
                "quantifier": safe_json(clause.get("quantifier")),
            }

            a_leaf = compact_leaf(get_criterion(pass2_by_branch["A"], item_uid, clause_id))
            b_leaf = compact_leaf(get_criterion(pass2_by_branch["B"], item_uid, clause_id))

            ab_rows.append({
                **common,
                "A_leaf": a_leaf,
                "B_leaf": b_leaf,
                "manual_A_leaf_label": "",
                "manual_B_leaf_label": "",
                "manual_best_branch_A_B": "",
                "manual_A_issue_type": "",
                "manual_B_issue_type": "",
                "manual_notes": "",
            })

    common_logic_fields = [
        "review_id",
        "document_id",
        "chia_id",
        "item_uid",
        "item_index",
        "criterion_type",
        "stratum",
        "item_text",
        "n_clauses",
        "logic_chain",
        "manual_logic_label",
        "manual_logic_issue_type",
        "manual_logic_notes",
    ]

    common_clause_fields = [
        "review_id",
        "document_id",
        "chia_id",
        "item_uid",
        "item_index",
        "criterion_type",
        "stratum",
        "item_text",
        "clause_id",
        "clause_text",
        "evidence_text",
        "is_negated",
        "connector_to_next",
        "quantifier",
    ]

    ab_fields = [
        *common_clause_fields,
        "A_leaf",
        "B_leaf",
        "manual_A_leaf_label",
        "manual_B_leaf_label",
        "manual_best_branch_A_B",
        "manual_A_issue_type",
        "manual_B_issue_type",
        "manual_notes",
    ]

    manifest = {
        "stage": "semantic_manual_pre_verification_A_B",
        "description": (
            "Manual semantic review sheets for Branch A and Branch B. "
            "Pass 1 logic is shared and reviewed separately."
        ),
        "n_items": len(logic_rows),
        "n_clauses": len(ab_rows),
        "branch_paths": {k: str(v) for k, v in branch_paths.items()},
        "outputs": {
            "pass1_logic_sheet": str(logic_sheet_path),
            "A_B_clause_sheet": str(ab_sheet_path),
        },
        "manual_leaf_label_values": ["correct", "partial", "incorrect"],
        "manual_best_values": ["A", "B", "tie", "none"],
        "important_note": (
            "Branch comparison is clause/leaf-level because Branch A and Branch B share Pass 1. "
            "Pass 1 logic should be evaluated once using the logic sheet."
        ),
    }

    write_json(manifest_path, manifest)
    write_csv(logic_sheet_path, logic_rows, common_logic_fields)
    write_csv(ab_sheet_path, ab_rows, ab_fields)

    print("Wrote manifest:", manifest_path)
    print("Wrote Pass 1 logic sheet:", logic_sheet_path)
    print("Wrote A/B clause sheet:", ab_sheet_path)
    print("Items exported:", len(logic_rows))
    print("Clauses exported:", len(ab_rows))

    # Missingness check
    for branch in ["A", "B"]:
        missing = 0
        for record in records:
            item_uid = record["item_uid"]
            for clause in record["clauses"]:
                clause_id = clause.get("clause_id")
                crit = get_criterion(pass2_by_branch[branch], item_uid, clause_id)
                if not crit:
                    missing += 1
        print(f"{branch} missing clauses:", missing)


if __name__ == "__main__":
    main()

# Run from the repository root only when creating a new manual review:
# python scripts/04_evaluation/01_pre_verification/04_create_manual_semantic_sheet.py