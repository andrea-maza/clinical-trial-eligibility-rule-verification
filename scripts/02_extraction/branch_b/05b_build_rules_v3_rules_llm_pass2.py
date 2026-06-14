import importlib.util
from pathlib import Path
from typing import Any, Dict, List

from jsonschema import Draft7Validator, ValidationError


# --------------------------------------------------
# Config
# --------------------------------------------------

REQUIRE_COMPLETE_PASS2 = True
# Keep True for final Branch B evaluation.
# Set False only to debug rule-tree assembly with a partial Pass 2 file.


# --------------------------------------------------
# Import the shared Branch A rule-tree builder
# --------------------------------------------------

def import_branch_a_builder(root: Path):
    builder_path = root / "scripts" / "02_extraction" / "branch_a" / "05_build_rules_v3_rules_from_pass1_and_pass2.py"

    spec = importlib.util.spec_from_file_location( "branch_a_rule_tree_builder", builder_path, )
    if spec is None or spec.loader is None:
        raise RuntimeError( f"Could not import the Branch A rule-tree builder: {builder_path}" )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------
# Completeness check
# --------------------------------------------------

def check_pass2_completeness(
    pass1_rows: List[Dict[str, Any]],
    pass2_index: Dict[str, Dict[str, Any]],
) -> None:
    expected_item_uids = []

    for row in pass1_rows:
        if row.get("status") != "ok":
            continue

        item_uid = row.get("item_uid")
        if item_uid:
            expected_item_uids.append(item_uid)

    missing = []
    bad_status = []

    for item_uid in expected_item_uids:
        pass2_row = pass2_index.get(item_uid)

        if pass2_row is None:
            missing.append(item_uid)
        elif pass2_row.get("status") != "ok":
            bad_status.append(
                {
                    "item_uid": item_uid,
                    "status": pass2_row.get("status"),
                    "error": pass2_row.get("error"),
                }
            )

    if missing or bad_status:
        msg = (
            f"Incomplete Branch B Pass 2 outputs. "
            f"Expected {len(expected_item_uids)} item-level Pass 2 rows. "
            f"Missing={len(missing)}, bad_status={len(bad_status)}. "
            f"First missing={missing[:10]}, first bad={bad_status[:5]}"
        )
        raise RuntimeError(msg)

def build_branch_b_pass2_index(
    rows: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Index Branch B Pass 2 rows by item_uid.

    When repeated attempts exist, prefer a successful row.
    If multiple rows have the same status, keep the latest one.
    """
    index: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        item_uid = row.get("item_uid")
        if not item_uid:
            continue

        previous = index.get(item_uid)

        if previous is None:
            index[item_uid] = row
            continue

        previous_ok = previous.get("status") == "ok"
        current_ok = row.get("status") == "ok"

        if current_ok and not previous_ok:
            index[item_uid] = row
        elif current_ok == previous_ok:
            previous_time = previous.get("timestamp") or 0
            current_time = row.get("timestamp") or 0

            if current_time >= previous_time:
                index[item_uid] = row

    return index


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    ROOT = Path(__file__).resolve().parents[3]

    builder = import_branch_a_builder(ROOT)

    pass1_path = (
        ROOT
        / "outputs"
        / "extraction"
        / "pass1_flat"
        / "chia_text_only_200_pass1_flat.jsonl"
    )

    pass2_path = (
        ROOT
        / "outputs"
        / "extraction"
        / "branch_b"
        / "pass2_leaves_llm"
        / "chia_text_only_200_pass2_leaves_llm.jsonl"
    )

    schema_path = ROOT / "schemas" / "rules_v3.json"

    out_dir = ROOT / "outputs" / "extraction" / "branch_b" / "rules_v3_llm_pass2"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "chia_text_only_200_rules_v3_ast_B.jsonl"

    pass1_rows = builder.load_jsonl(pass1_path)
    pass2_rows = builder.load_jsonl(pass2_path)
    pass2_index = build_branch_b_pass2_index(pass2_rows)

    if REQUIRE_COMPLETE_PASS2:
        check_pass2_completeness(pass1_rows, pass2_index)

    schema = builder.load_schema(schema_path)
    validator = Draft7Validator(schema)

    rows_by_doc = builder.group_by_document(pass1_rows)

    output_rows: List[Dict[str, Any]] = []
    n_ok = 0
    n_err = 0

    for document_id, doc_rows in rows_by_doc.items():
        try:
            ast = builder.build_document_ast(
                document_id=document_id,
                pass1_rows_for_doc=doc_rows,
                pass2_index=pass2_index,
            )

            errors = sorted(validator.iter_errors(ast), key=lambda e: list(e.absolute_path))
            if errors:
                e = errors[0]
                path = ".".join(str(x) for x in e.absolute_path)
                schema_path_str = ".".join(str(x) for x in e.absolute_schema_path)
                raise ValidationError(
                    f"{e.message} | path={path} | schema_path={schema_path_str} | validator={e.validator}"
                )

            n_inclusion_items = sum(
                1
                for r in doc_rows
                if r.get("status") == "ok" and r.get("criterion_type_hint") == "inclusion"
            )

            n_exclusion_items = sum(
                1
                for r in doc_rows
                if r.get("status") == "ok" and r.get("criterion_type_hint") == "exclusion"
            )

            output_rows.append(
                {
                    "dataset": "CHIA",
                    "stage": "rules_v3_ast_assembly",
                    "branch": "B_llm_pass2",
                    "document_id": document_id,
                    "status": "ok",
                    "error": None,
                    "n_pass1_rows": len(doc_rows),
                    "n_inclusion_items": n_inclusion_items,
                    "n_exclusion_items": n_exclusion_items,
                    "pass1_source": "chia_text_only_200_pass1_flat.jsonl",
                    "pass2_source": "chia_text_only_200_pass2_leaves_llm.jsonl",
                    "rules_v3_ast": ast,
                }
            )
            n_ok += 1

        except Exception as e:
            output_rows.append(
                {
                    "dataset": "CHIA",
                    "stage": "rules_v3_ast_assembly",
                    "branch": "B_llm_pass2",
                    "document_id": document_id,
                    "status": "error",
                    "error": str(e),
                    "pass1_source": "chia_text_only_200_pass1_flat.jsonl",
                    "pass2_source": "chia_text_only_200_pass2_leaves_llm.jsonl",
                }
            )
            n_err += 1

    builder.write_jsonl(out_path, output_rows)

    print("Wrote:", out_path)
    print("OK:", n_ok)
    print("ERR:", n_err)


if __name__ == "__main__":
    main()

# Run from the repository root: 
# # python scripts/02_extraction/branch_b/05b_build_rules_v3_rules_llm_pass2.py

