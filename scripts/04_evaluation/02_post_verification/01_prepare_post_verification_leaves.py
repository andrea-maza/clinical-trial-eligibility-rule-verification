"""
01_prepare_post_verification_leaves.py

Convert the post-verification Branch A and Branch B logical rule trees
into the Pass 2 leaf format required by the evaluation scripts.

Inputs:
    outputs/verification/layer3/applied_candidate_selection_rescue/
    outputs/extraction/pass1_flat/
    outputs/extraction/pass2_inputs/

Outputs:
    outputs/evaluation/post_verification/
        post_verification_pass2_leaves/

This is a format conversion only. It does not change rule content,
call the LLM, or use manual labels.

Run from the repository root:
python scripts/04_evaluation/02_post_verification/01_prepare_post_verification_leaves.py
"""


from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[3]

A_POST_AST = (
    ROOT
    / "outputs"
    / "verification"
    / "layer3"
    / "applied_candidate_selection_rescue"
    / "chia_text_only_200_rules_v3_ast_A_layer3_candidate_selection_rescue.jsonl"
)

B_POST_AST = (
    ROOT
    / "outputs"
    / "verification"
    / "layer3"
    / "applied_candidate_selection_rescue"
    / "chia_text_only_200_rules_v3_ast_B_layer3_candidate_selection_rescue.jsonl"
)

PASS1_FLAT = (
    ROOT
    / "outputs"
    / "extraction"
    / "pass1_flat"
    / "chia_text_only_200_pass1_flat.jsonl"
)

PASS2_INPUTS = (
    ROOT
    / "outputs"
    / "extraction"
    / "pass2_inputs"
    / "chia_text_only_200_pass2_inputs.jsonl"
)

OUT_DIR = (
    ROOT
    / "outputs"
    / "evaluation"
    / "post_verification"
    / "post_verification_pass2_leaves"
)

OUT_A = (
    OUT_DIR
    / "chia_text_only_200_post_verification_pass2_leaves_A.jsonl"
)
OUT_B = (
    OUT_DIR
    / "chia_text_only_200_post_verification_pass2_leaves_B.jsonl"
)
OUT_SUMMARY = (
    OUT_DIR
    / "post_verification_pass2_leaves_summary.json"
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

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


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def clean(x: Any) -> str:
    return str(x or "").strip()


def parse_criterion_id(criterion_id: str) -> Tuple[str, str]:
    """
    Expected criterion_id:
        NCT00050349_exc__item2_C4

    item_uid:
        NCT00050349_exc__item2

    clause_id:
        C4
    """
    cid = clean(criterion_id)

    if "_" not in cid:
        return "", ""

    item_uid, clause_id = cid.rsplit("_", 1)

    return item_uid, clause_id


def clause_sort_key(clause_id: str) -> int:
    m = re.search(r"(\d+)$", clean(clause_id))
    return int(m.group(1)) if m else 999999


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


def iter_doc_criteria(doc: Dict[str, Any]) -> Iterable[Tuple[str, str, Dict[str, Any]]]:
    ast = doc.get("rules_v3_ast")

    if not isinstance(ast, dict):
        return

    for criterion_type, section in [
        ("inclusion", "inclusion_criteria"),
        ("exclusion", "exclusion_criteria"),
    ]:
        root = ast.get(section)

        if isinstance(root, dict):
            for path, criterion in iter_criterion_nodes(root, section):
                yield criterion_type, path, criterion


def build_pass1_item_index(pass1_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index = {}

    for r in pass1_rows:
        if r.get("status") != "ok":
            continue

        item_uid = clean(r.get("item_uid"))

        if item_uid:
            index[item_uid] = r

    return index


def build_pass2_input_index(pass2_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index = {}

    for r in pass2_rows:
        payload = r.get("pass2_input", r)

        if not isinstance(payload, dict):
            continue

        item_uid = clean(payload.get("item_uid"))

        if item_uid:
            index[item_uid] = payload

    return index


def convert_ast_to_pass2_leaves(
    ast_path: Path,
    out_path: Path,
    branch_name: str,
    pass1_index: Dict[str, Dict[str, Any]],
    pass2_input_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    ast_rows = read_jsonl(ast_path)

    grouped: Dict[str, Dict[str, Any]] = {}

    missing_item_uid = 0
    missing_clause_id = 0
    leaves_total = 0

    for doc in ast_rows:
        document_id = clean(doc.get("document_id"))
        chia_id = clean(doc.get("chia_id"))
        trial_id = clean(doc.get("trial_id"))

        for criterion_type, path, criterion in iter_doc_criteria(doc):
            leaves_total += 1

            criterion_id = clean(criterion.get("criterion_id"))
            item_uid, clause_id = parse_criterion_id(criterion_id)

            if not item_uid:
                missing_item_uid += 1
                continue

            if not clause_id:
                missing_clause_id += 1
                continue

            pass1_item = pass1_index.get(item_uid, {})
            pass2_input = pass2_input_index.get(item_uid, {})

            key = item_uid

            if key not in grouped:
                grouped[key] = {
                    "status": "ok",
                    "document_id": document_id or clean(pass1_item.get("document_id")),
                    "chia_id": chia_id or clean(pass1_item.get("chia_id")),
                    "item_uid": item_uid,
                    "trial_id": trial_id,
                    "criterion_type": criterion_type,
                    "pass2_output": {
                        "document_id": document_id or clean(pass1_item.get("document_id")),
                        "chia_id": chia_id or clean(pass1_item.get("chia_id")),
                        "item_uid": item_uid,
                        "trial_id": trial_id,
                        "criterion_type": criterion_type,
                        "item_text": pass2_input.get("item_text") or pass1_item.get("item_text"),
                        "criteria": [],
                    },
                }

            grouped[key]["pass2_output"]["criteria"].append(
                {
                    "clause_id": clause_id,
                    "criterion_id": criterion_id,
                    "criterion": criterion,
                    "source_ast_path": path,
                    "post_verification_branch": branch_name,
                }
            )

    out_rows = []

    for item_uid, row in sorted(grouped.items()):
        row["pass2_output"]["criteria"] = sorted(
            row["pass2_output"]["criteria"],
            key=lambda x: clause_sort_key(x.get("clause_id", "")),
        )
        out_rows.append(row)

    write_jsonl(out_path, out_rows)

    return {
        "branch": branch_name,
        "input_ast": str(ast_path),
        "output_pass2_leaves": str(out_path),
        "input_ast_documents": len(ast_rows),
        "output_items": len(out_rows),
        "leaves_total": leaves_total,
        "missing_item_uid": missing_item_uid,
        "missing_clause_id": missing_clause_id,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nPrepare post-verification Pass 2 leaf files")
    print("Branch A verified rule trees:", A_POST_AST)
    print("Branch B verified rule trees:", B_POST_AST)

    input_paths = {
        "A_POST_AST": A_POST_AST,
        "B_POST_AST": B_POST_AST,
        "PASS1_FLAT": PASS1_FLAT,
        "PASS2_INPUTS": PASS2_INPUTS,
    }

    for name, path in input_paths.items():
        if "100" in str(path):
            raise RuntimeError(f"Old 100-run path detected in {name}: {path}")

        if "200" not in str(path):
            raise RuntimeError(f"Expected CHIA-200 path in {name}, but got: {path}")

        if not path.exists():
            raise FileNotFoundError(f"Missing input file for {name}: {path}")

    pass1_rows = read_jsonl(PASS1_FLAT)
    pass2_rows = read_jsonl(PASS2_INPUTS)

    pass1_index = build_pass1_item_index(pass1_rows)
    pass2_input_index = build_pass2_input_index(pass2_rows)

    summary_a = convert_ast_to_pass2_leaves(
        ast_path=A_POST_AST,
        out_path=OUT_A,
        branch_name="A_post_verification",
        pass1_index=pass1_index,
        pass2_input_index=pass2_input_index,
    )

    summary_b = convert_ast_to_pass2_leaves(
        ast_path=B_POST_AST,
        out_path=OUT_B,
        branch_name="B_post_verification",
        pass1_index=pass1_index,
        pass2_input_index=pass2_input_index,
    )

    for label, branch_summary in [("A", summary_a), ("B", summary_b)]:
        if branch_summary["missing_item_uid"] != 0:
            raise RuntimeError(
                f"Branch {label} has {branch_summary['missing_item_uid']} leaves "
                "with missing item_uid parsed from criterion_id."
            )

        if branch_summary["missing_clause_id"] != 0:
            raise RuntimeError(
                f"Branch {label} has {branch_summary['missing_clause_id']} leaves "
                "with missing clause_id parsed from criterion_id."
            )

        if branch_summary["leaves_total"] == 0:
            raise RuntimeError(
                f"Branch {label} produced zero leaves. Check the AST traversal."
            )
    
    summary = {
        "stage": "prepare_post_verification_pass2_leaves",
        "description": (
            "Converts post-rescue logical rule trees into Pass 2 leaf JSONL "
            "files so that pre- and post-verification evaluations use the "
            "same input format."
        ),
        "inputs": {
            "branch_a_post_ast": str(A_POST_AST),
            "branch_b_post_ast": str(B_POST_AST),
            "pass1_flat": str(PASS1_FLAT),
            "pass2_inputs": str(PASS2_INPUTS),
        },
        "outputs": {
            "branch_a_post_pass2_leaves": str(OUT_A),
            "branch_b_post_pass2_leaves": str(OUT_B),
            "summary_json": str(OUT_SUMMARY),
        },
        "branch_summaries": {
            "A": summary_a,
            "B": summary_b,
        },
        "method_notes": [
            "This is only a format conversion.",
            "No rule content is changed.",
            "This script does not call the LLM.",
            "This script does not use manual labels.",
        ],
    }

    write_json(OUT_SUMMARY, summary)

    print("\nDONE")
    print("A post Pass2 leaves:", OUT_A)
    print("B post Pass2 leaves:", OUT_B)
    print("Summary:", OUT_SUMMARY)
    print("\nBranch summaries:")
    print(summary["branch_summaries"])


if __name__ == "__main__":
    main()

# Run from the repository root:
# python scripts/04_evaluation/02_post_verification/01_prepare_post_verification_leaves.py