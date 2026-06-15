"""
05_create_manual_semantic_sheet.py

Create the post-verification manual semantic review sheet for
Branch A and Branch B.

Existing pre-verification labels are reused only when the corresponding
branch-specific leaf is unchanged. Leaves changed by verification or
rescue remain blank for independent post-verification review.

The sheet also includes:
    - Branch A and Branch B final decisions
    - cross-branch decisions
    - rescue application metadata

Important:
    This script creates a manual-review sheet. It must not be run when
    the completed post-verification review sheet already exists.

This script does not call the LLM or modify predictions.

Run from the repository root only when creating a new review sheet:
python scripts/04_evaluation/02_post_verification/05_create_manual_semantic_sheet.py
"""


from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


# --------------------------------------------------
# IO helpers
# --------------------------------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []

    if not path.exists():
        raise FileNotFoundError(f"Missing JSONL file: {path}")

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


def load_csv_if_exists(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []

    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]

    last_error = None

    for enc in encodings:
        try:
            with path.open("r", encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError as exc:
            last_error = exc

    raise RuntimeError(f"Could not decode {path}. Last error: {last_error}")


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


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({c: serialize_cell(row.get(c, "")) for c in fieldnames})


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


def parse_criterion_id(criterion_id: str) -> Tuple[str, str]:
    cid = normalize_text(criterion_id)

    if "_" not in cid:
        return "", ""

    item_uid, clause_id = cid.rsplit("_", 1)
    return item_uid, clause_id


def make_clause_key(item_uid: str, clause_id: str) -> Tuple[str, str]:
    return normalize_text(item_uid), normalize_text(clause_id)


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


def classify_stratum(
    item_text: str,
    clauses: List[Dict[str, Any]],
    criteria_by_branch: Dict[str, List[Dict[str, Any]]],
) -> str:
    t = normalize_text(item_text).lower()

    if any(marker in t for marker in ["unless", "except", "with the exception of", " if "]):
        return "exception"

    if any(c.get("is_negated") for c in clauses) or any(
        marker in f" {t} "
        for marker in [" no ", " without ", " absence of "]
    ):
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


# --------------------------------------------------
# Index builders
# --------------------------------------------------

def build_pass1_records(pass1_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records = []

    for p1 in pass1_rows:
        if p1.get("status") != "ok":
            continue

        item_uid = p1.get("item_uid")
        clauses = (p1.get("parsed_pass1_with_ids") or {}).get("clauses", [])

        records.append(
            {
                "document_id": p1.get("document_id"),
                "chia_id": p1.get("chia_id"),
                "item_uid": item_uid,
                "item_index": p1.get("item_index"),
                "criterion_type": p1.get("criterion_type_hint"),
                "item_text": p1.get("item_text", ""),
                "clauses": clauses,
            }
        )

    records = sorted(
        records,
        key=lambda r: (
            str(r.get("document_id")),
            0 if r.get("criterion_type") == "inclusion" else 1,
            int(r.get("item_index")) if r.get("item_index") is not None else 999999,
        ),
    )

    return records


def build_pass2_by_item_uid(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = {}

    for row in rows:
        if row.get("status") != "ok":
            continue

        item_uid = row.get("item_uid")

        if not item_uid:
            payload = row.get("pass2_output", {})
            item_uid = payload.get("item_uid")

        if item_uid:
            out[item_uid] = row

    return out


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


def build_final_decision_index(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, str]]:
    out = {}

    for row in rows:
        item_uid = normalize_text(row.get("item_uid"))
        clause_id = normalize_text(row.get("clause_id"))
        criterion_id = normalize_text(row.get("criterion_id"))

        if item_uid and clause_id:
            out[make_clause_key(item_uid, clause_id)] = row

        if criterion_id:
            parsed_item_uid, parsed_clause_id = parse_criterion_id(criterion_id)

            if parsed_item_uid and parsed_clause_id:
                out[make_clause_key(parsed_item_uid, parsed_clause_id)] = row

    return out


def build_cross_decision_index(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, str]]:
    out = {}

    for row in rows:
        criterion_id = normalize_text(row.get("criterion_id"))

        if criterion_id:
            item_uid, clause_id = parse_criterion_id(criterion_id)

            if item_uid and clause_id:
                out[make_clause_key(item_uid, clause_id)] = row

    return out


def build_rescue_audit_index(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str, str], Dict[str, str]]:
    out = {}

    for row in rows:
        branch = normalize_text(row.get("branch_to_update") or row.get("branch"))
        criterion_id = normalize_text(row.get("criterion_id"))

        if not branch or not criterion_id:
            continue

        item_uid, clause_id = parse_criterion_id(criterion_id)

        if item_uid and clause_id:
            out[(branch, item_uid, clause_id)] = row

    return out




def build_pre_manual_label_index(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, str]]:
    out = {}

    for row in rows:
        item_uid = normalize_text(row.get("item_uid"))
        clause_id = normalize_text(row.get("clause_id"))

        if item_uid and clause_id:
            out[make_clause_key(item_uid, clause_id)] = row

    return out


def maybe_copy_old_label_if_leaf_unchanged(
    *,
    pre_row: Dict[str, str],
    pre_leaf_col: str,
    post_leaf: str,
    old_label_col: str,
    old_issue_col: str,
) -> Tuple[str, str, str]:
    """
    Returns:
        post_label, post_issue_type, status

    status values:
        copied
        changed
        unchanged_but_old_label_missing
        no_pre_row
    """
    if not pre_row:
        return "", "", "no_pre_row"

    old_leaf = normalize_text(pre_row.get(pre_leaf_col))
    new_leaf = normalize_text(post_leaf)

    if old_leaf != new_leaf:
        return "", "", "changed"

    old_label = normalize_text(pre_row.get(old_label_col))
    old_issue = normalize_text(pre_row.get(old_issue_col))

    if not old_label:
        return "", "", "unchanged_but_old_label_missing"

    return old_label, old_issue, "copied"


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

    post_a_path = (
        ROOT
        / "outputs"
        / "evaluation"
        / "post_verification"
        / "post_verification_pass2_leaves"
        / "chia_text_only_200_post_verification_pass2_leaves_A.jsonl"
    )

    post_b_path = (
        ROOT
        / "outputs"
        / "evaluation"
        / "post_verification"
        / "post_verification_pass2_leaves"
        / "chia_text_only_200_post_verification_pass2_leaves_B.jsonl"
    )

    final_a_path = (
        ROOT
        / "outputs"
        / "verification"
        / "layer3"
        / "final_decisions"
        / "layer3_final_decision_branch_a.csv"
    )

    final_b_path = (
        ROOT
        / "outputs"
        / "verification"
        / "layer3"
        / "final_decisions"
        / "layer3_final_decision_branch_b.csv"
    )

    cross_path = (
        ROOT
        / "outputs"
        / "verification"
        / "layer3"
        / "final_decisions"
        / "layer3_final_decision_cross_branch_by_criterion.csv"
    )

    rescue_audit_path = (
        ROOT
        / "outputs"
        / "verification"
        / "layer3"
        / "applied_candidate_selection_rescue"
        / "layer3_candidate_selection_apply_audit.csv"
    )

    pre_reviewed_path = (
        ROOT
        / "outputs"
        / "evaluation"
        / "pre_verification"
        / "semantic_manual_pre_verification_A_B_summary"
        / "reviewed_semantic_clause_labels_A_B.csv"
    )

    out_dir = (
        ROOT
        / "outputs"
        / "evaluation"
        / "post_verification"
        / "semantic_manual_post_verification_A_B"
    )

    manifest_path = (
        out_dir / "semantic_manifest_post_verification_A_B.json"
    )
    sheet_path = (
        out_dir / "semantic_clause_sheet_A_B_post_verification.csv"
    )

    input_paths = {
        "Pass 1 output": pass1_path,
        "Branch A post-verification leaves": post_a_path,
        "Branch B post-verification leaves": post_b_path,
        "Branch A final decisions": final_a_path,
        "Branch B final decisions": final_b_path,
        "cross-branch final decisions": cross_path,
        "rescue application audit": rescue_audit_path,
        "pre-verification reviewed labels": pre_reviewed_path,
    }

    for name, path in input_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    existing_outputs = [
        path
        for path in [manifest_path, sheet_path]
        if path.exists()
    ]

    if existing_outputs:
        raise FileExistsError(
            "Manual post-verification review outputs already exist "
            "and will not be overwritten:\n"
            + "\n".join(str(path) for path in existing_outputs)
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    pass1_rows = load_jsonl(pass1_path)
    post_a_rows = load_jsonl(post_a_path)
    post_b_rows = load_jsonl(post_b_path)

    final_a_rows = load_csv_if_exists(final_a_path)
    final_b_rows = load_csv_if_exists(final_b_path)
    cross_rows = load_csv_if_exists(cross_path)
    rescue_rows = load_csv_if_exists(rescue_audit_path)
    pre_reviewed_rows = load_csv_if_exists(pre_reviewed_path)

    records = build_pass1_records(pass1_rows)

    pass2_by_branch = {
        "A": build_pass2_by_item_uid(post_a_rows),
        "B": build_pass2_by_item_uid(post_b_rows),
    }

    final_a_index = build_final_decision_index(final_a_rows)
    final_b_index = build_final_decision_index(final_b_rows)
    cross_index = build_cross_decision_index(cross_rows)
    rescue_index = build_rescue_audit_index(rescue_rows)
    pre_manual_index = build_pre_manual_label_index(pre_reviewed_rows)

    rows = []
    missing_counts = Counter()
    final_decision_counts_a = Counter()
    final_decision_counts_b = Counter()
    rescue_apply_counts = Counter()
    prefill_counts = Counter()

    for i, record in enumerate(records, start=1):
        review_id = f"I{i:05d}"
        item_uid = record["item_uid"]
        item_text = record["item_text"]
        clauses = record["clauses"]

        criteria_by_branch = {
            branch: get_branch_criteria(index, item_uid)
            for branch, index in pass2_by_branch.items()
        }

        stratum = classify_stratum(item_text, clauses, criteria_by_branch)

        clause_map = {
            c["clause_id"]: c
            for c in clauses
            if c.get("clause_id")
        }

        for clause_id in sorted(clause_map.keys(), key=clause_sort_key):
            clause = clause_map[clause_id]
            key = make_clause_key(item_uid, clause_id)

            a_criterion = get_criterion(pass2_by_branch["A"], item_uid, clause_id)
            b_criterion = get_criterion(pass2_by_branch["B"], item_uid, clause_id)

            a_post_leaf = compact_leaf(a_criterion)
            b_post_leaf = compact_leaf(b_criterion)

            pre_manual = pre_manual_index.get(key, {})

            manual_a_post_label, manual_a_post_issue, a_copy_status = maybe_copy_old_label_if_leaf_unchanged(
                pre_row=pre_manual,
                pre_leaf_col="A_leaf",
                post_leaf=a_post_leaf,
                old_label_col="manual_A_leaf_label",
                old_issue_col="manual_A_issue_type",
            )

            manual_b_post_label, manual_b_post_issue, b_copy_status = maybe_copy_old_label_if_leaf_unchanged(
                pre_row=pre_manual,
                pre_leaf_col="B_leaf",
                post_leaf=b_post_leaf,
                old_label_col="manual_B_leaf_label",
                old_issue_col="manual_B_issue_type",
            )

            prefill_counts[f"A_{a_copy_status}"] += 1
            prefill_counts[f"B_{b_copy_status}"] += 1

            if not a_criterion:
                missing_counts["A_missing_post_leaf"] += 1

            if not b_criterion:
                missing_counts["B_missing_post_leaf"] += 1

            final_a = final_a_index.get(key, {})
            final_b = final_b_index.get(key, {})
            cross = cross_index.get(key, {})

            rescue_a = rescue_index.get(("A", item_uid, clause_id), {})
            rescue_b = rescue_index.get(("B", item_uid, clause_id), {})

            a_final_decision = final_a.get("final_decision", "")
            b_final_decision = final_b.get("final_decision", "")

            if a_final_decision:
                final_decision_counts_a[a_final_decision] += 1

            if b_final_decision:
                final_decision_counts_b[b_final_decision] += 1

            for audit_row in [rescue_a, rescue_b]:
                if audit_row.get("apply_status"):
                    rescue_apply_counts[audit_row.get("apply_status")] += 1

            rows.append(
                {
                    "review_id": review_id,
                    "document_id": record["document_id"],
                    "chia_id": record["chia_id"],
                    "item_uid": item_uid,
                    "item_index": record["item_index"],
                    "criterion_type": record["criterion_type"],
                    "stratum": stratum,
                    "item_text": item_text,
                    "logic_chain": build_logic_chain(clauses),

                    "clause_id": clause_id,
                    "clause_text": normalize_text(clause.get("clause_text")),
                    "evidence_text": normalize_text(clause.get("evidence_text")),
                    "is_negated": clause.get("is_negated"),
                    "connector_to_next": clause.get("connector_to_next"),
                    "quantifier": safe_json(clause.get("quantifier")),

                    "A_post_leaf": a_post_leaf,
                    "B_post_leaf": b_post_leaf,

                    "A_final_decision": a_final_decision,
                    "A_final_decision_rule": final_a.get("final_decision_rule", ""),
                    "A_final_decision_reasons": final_a.get("final_decision_reasons", ""),

                    "B_final_decision": b_final_decision,
                    "B_final_decision_rule": final_b.get("final_decision_rule", ""),
                    "B_final_decision_reasons": final_b.get("final_decision_reasons", ""),

                    "cross_chosen_branch": cross.get("chosen_branch", ""),
                    "cross_chosen_decision": cross.get("chosen_branch_decision", ""),
                    "cross_branch_reason": cross.get("cross_branch_reason", ""),

                    "A_rescue_apply_status": rescue_a.get("apply_status", ""),
                    "A_rescue_final_decision": rescue_a.get("final_decision", ""),
                    "A_rescue_changed_fields": rescue_a.get("changed_fields", ""),
                    "A_rescue_skip_reason": rescue_a.get("skip_reason", ""),

                    "B_rescue_apply_status": rescue_b.get("apply_status", ""),
                    "B_rescue_final_decision": rescue_b.get("final_decision", ""),
                    "B_rescue_changed_fields": rescue_b.get("changed_fields", ""),
                    "B_rescue_skip_reason": rescue_b.get("skip_reason", ""),

                    "manual_A_post_leaf_label": manual_a_post_label,
                    "manual_B_post_leaf_label": manual_b_post_label,
                    "manual_A_post_issue_type": manual_a_post_issue,
                    "manual_B_post_issue_type": manual_b_post_issue,
                    "manual_post_notes": "",
                }
            )

    fieldnames = [
        "review_id",
        "document_id",
        "chia_id",
        "item_uid",
        "item_index",
        "criterion_type",
        "stratum",
        "item_text",
        "logic_chain",
        "clause_id",
        "clause_text",
        "evidence_text",
        "is_negated",
        "connector_to_next",
        "quantifier",

        "A_post_leaf",
        "B_post_leaf",

        "A_final_decision",
        "A_final_decision_rule",
        "A_final_decision_reasons",
        "B_final_decision",
        "B_final_decision_rule",
        "B_final_decision_reasons",

        "cross_chosen_branch",
        "cross_chosen_decision",
        "cross_branch_reason",

        "A_rescue_apply_status",
        "A_rescue_final_decision",
        "A_rescue_changed_fields",
        "A_rescue_skip_reason",
        "B_rescue_apply_status",
        "B_rescue_final_decision",
        "B_rescue_changed_fields",
        "B_rescue_skip_reason",

        "manual_A_post_leaf_label",
        "manual_B_post_leaf_label",
        "manual_A_post_issue_type",
        "manual_B_post_issue_type",
        "manual_post_notes",
    ]

    manifest = {
        "stage": "semantic_manual_post_verification_A_B",
        "description": (
            "Manual semantic review sheet for post-verification Branch A and Branch B. "
            "Pass 1 logic is not relabeled because Pass 1 did not change. "
            "Pre-verification labels are reused only for unchanged branch-specific leaves. "
            "Changed leaves remain blank for independent post-verification review."
        ),
        "n_items": len(records),
        "n_clauses": len(rows),
        "inputs": {
            "pass1_flat": str(pass1_path),
            "post_A_pass2_leaves": str(post_a_path),
            "post_B_pass2_leaves": str(post_b_path),
            "branch_A_final_decisions": str(final_a_path),
            "branch_B_final_decisions": str(final_b_path),
            "cross_branch_final_decisions": str(cross_path),
            "rescue_audit": str(rescue_audit_path),
            "pre_reviewed_manual_labels": str(pre_reviewed_path),
        },
        "outputs": {
            "post_manual_sheet": str(sheet_path),
            "manifest": str(manifest_path),
        },
        "manual_leaf_label_values": ["correct", "partial", "incorrect"],
        "manual_issue_type_examples": [
            "wrong_entity",
            "wrong_entity_type",
            "wrong_operator",
            "wrong_value",
            "wrong_unit",
            "wrong_temporal_context",
            "wrong_history_context",
            "wrong_negation",
            "wrong_computability",
            "missing_context",
            "overbroad",
            "underspecified",
            "unsupported",
        ],
        "counts": {
            "missing_counts": dict(missing_counts),
            "A_final_decision_counts": dict(final_decision_counts_a),
            "B_final_decision_counts": dict(final_decision_counts_b),
            "rescue_apply_status_counts": dict(rescue_apply_counts),
            "manual_label_prefill_counts": dict(prefill_counts),
        },
        "important_notes": [
            "Pre-verification labels are carried forward only when the branch-specific compact leaf is unchanged.",
            "Changed leaves remain blank and must be manually labeled.",
            "Branch C is intentionally excluded.",
            "The post sheet is clause/leaf-level because Branch A and Branch B share Pass 1.",
        ],
    }

    write_csv(sheet_path, rows, fieldnames)
    write_json(manifest_path, manifest)

    print("\nPost-verification manual semantic sheet")
    print("Wrote sheet:", sheet_path)
    print("Wrote manifest:", manifest_path)
    print("Items exported:", len(records))
    print("Clauses exported:", len(rows))
    print("Missing counts:", dict(missing_counts))
    print("A final decision counts:", dict(final_decision_counts_a))
    print("B final decision counts:", dict(final_decision_counts_b))
    print("Rescue apply status counts:", dict(rescue_apply_counts))
    print("Manual label prefill counts:", dict(prefill_counts))


if __name__ == "__main__":
    main()

# Run from the repository root only when creating a new review sheet:
# python scripts/04_evaluation/02_post_verification/05_create_manual_semantic_sheet.py