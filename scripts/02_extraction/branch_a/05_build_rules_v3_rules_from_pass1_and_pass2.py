import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from jsonschema import Draft7Validator, ValidationError


# ----------------------------
# IO helpers
# ----------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_schema(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------
# Small utilities
# ----------------------------

def group_by_document(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        doc_id = row.get("document_id")
        if not doc_id:
            continue
        out.setdefault(doc_id, []).append(row)
    return out


def build_pass2_index(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Index pass2 rows by item_uid.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        item_uid = row.get("item_uid")
        if item_uid:
            if item_uid in out:
                raise ValueError(f"Duplicate item_uid in Pass 2 rows: {item_uid}")
            out[item_uid] = row
    return out


# ----------------------------
# Logical rule-tree node builders
# ----------------------------

def make_group_node(
    operator: str,
    children: List[Dict[str, Any]],
    polarity: Optional[str] = None,
    group_quantifier: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    node = {
        "node_type": "group",
        "group_operator": operator,
        "children": children,
    }
    if polarity is not None:
        node["polarity"] = polarity
    if group_quantifier is not None:
        node["group_quantifier"] = group_quantifier
    return node


def make_criterion_node(criterion: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "node_type": "criterion",
        "criterion": criterion,
    }


# ----------------------------
# Fallback leaf if Pass 2 is missing
# ----------------------------

def make_fallback_partial_criterion(item_uid: str, clause: Dict[str, Any]) -> Dict[str, Any]:
    clause_id = clause["clause_id"]
    return {
        "criterion_id": f"{item_uid}_{clause_id}",
        "entity_type": "other",
        "entity_text": clause.get("clause_text", ""),
        "normalized_concept": None,
        "operator": "exists",
        "value_type": "null",
        "value": None,
        "unit": None,
        "temporal_context": None,
        "computability": "partial",
        "non_computable_reason": "Missing Pass 2 leaf for this clause during rule-tree assembly.",
        "evidence_text": clause.get("evidence_text", ""),
        "provenance": {
            "source_modifier_text": None,
            "source_condition_text": None,
            "source_exception_context": None,
            "history_context_hint": None,
        },
        "history_context": None,
    }


# ----------------------------
# Item-level logical rule-tree construction
# ----------------------------

def build_clause_criterion_map(
    item_uid: str,
    pass1_clauses: List[Dict[str, Any]],
    pass2_row: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Build a map: clause_id -> filled criterion leaf
    """
    by_clause: Dict[str, Dict[str, Any]] = {}

    if pass2_row is not None and pass2_row.get("status") == "ok":
        pass2_output = pass2_row.get("pass2_output", {})
        for entry in pass2_output.get("criteria", []):
            clause_id = entry.get("clause_id")
            criterion = entry.get("criterion")
            if clause_id and criterion:
                by_clause[clause_id] = criterion

    # fallback for any missing clauses
    for clause in pass1_clauses:
        clause_id = clause["clause_id"]
        if clause_id not in by_clause:
            by_clause[clause_id] = make_fallback_partial_criterion(item_uid, clause)

    return by_clause


def build_leaf_node_from_clause(
    clause: Dict[str, Any],
    criterion_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    clause_id = clause["clause_id"]
    criterion = criterion_map[clause_id]
    # Negation is represented at the leaf level through operator="not_exists".
    # We do not wrap negated clauses in NOT here, to avoid double negation.
    node = make_criterion_node(criterion)

    return node


def build_item_ast(
    item_uid: str,
    pass1_clauses: List[Dict[str, Any]],
    pass2_row: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Build an item-level logical rule tree deterministically from the
    flat Pass 1 clauses, using the Pass 2 criteria as leaves.

    Logic:
    - single clause -> single criterion node
    - all AND chain -> AND group
    - all OR chain -> OR group
    - mixed chain -> split by OR, AND binds tighter than OR
    - quantifier -> wrap all child leaves under OR group with group_quantifier
    """
    if not pass1_clauses:
        return None

    criterion_map = build_clause_criterion_map(item_uid, pass1_clauses, pass2_row)

    leaf_nodes = [build_leaf_node_from_clause(clause, criterion_map) for clause in pass1_clauses]

    # If any quantifier is present, use it at the item root
    first_quantifier = None
    for clause in pass1_clauses:
        if clause.get("quantifier") is not None:
            first_quantifier = clause["quantifier"]
            break

    if first_quantifier is not None:
        if len(leaf_nodes) == 1:
            return make_group_node("OR", leaf_nodes, group_quantifier=first_quantifier)
        return make_group_node("OR", leaf_nodes, group_quantifier=first_quantifier)

    if len(leaf_nodes) == 1:
        return leaf_nodes[0]

    connectors = [c.get("connector_to_next") for c in pass1_clauses[:-1]]

    # Split by OR; AND binds tighter
    runs: List[List[Dict[str, Any]]] = []
    current_run: List[Dict[str, Any]] = [leaf_nodes[0]]

    for i, conn in enumerate(connectors):
        next_node = leaf_nodes[i + 1]

        if conn == "AND":
            current_run.append(next_node)
        elif conn == "OR":
            runs.append(current_run)
            current_run = [next_node]
        else:
            # unexpected mid-chain null: treat as boundary
            runs.append(current_run)
            current_run = [next_node]

    runs.append(current_run)

    run_nodes: List[Dict[str, Any]] = []
    for run in runs:
        if len(run) == 1:
            run_nodes.append(run[0])
        else:
            run_nodes.append(make_group_node("AND", run))

    if len(run_nodes) == 1:
        return run_nodes[0]

    return make_group_node("OR", run_nodes)


# ----------------------------
# Document-level logical rule-tree construction
# ----------------------------

def combine_item_trees(
    item_trees: List[Dict[str, Any]],
    criterion_type: str,
) -> Optional[Dict[str, Any]]:
    if not item_trees:
        return None

    if len(item_trees) == 1:
        return item_trees[0]

    if criterion_type == "inclusion":
        return make_group_node("AND", item_trees, polarity="include")

    return make_group_node("OR", item_trees, polarity="exclude")


def build_document_ast(
    document_id: str,
    pass1_rows_for_doc: List[Dict[str, Any]],
    pass2_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    # sort items in source order
    pass1_rows_for_doc = sorted(
        pass1_rows_for_doc,
        key=lambda r: (
            0 if r.get("criterion_type_hint") == "inclusion" else 1,
            int(r.get("item_index", 0)),
        ),
    )

    inclusion_item_trees: List[Dict[str, Any]] = []
    exclusion_item_trees: List[Dict[str, Any]] = []

    for row in pass1_rows_for_doc:
        if row.get("status") != "ok":
            continue

        item_uid = row.get("item_uid")
        criterion_type = row.get("criterion_type_hint")
        pass1_clauses = (row.get("parsed_pass1_with_ids") or {}).get("clauses", [])
        pass2_row = pass2_index.get(item_uid)

        item_ast = build_item_ast(
            item_uid=item_uid,
            pass1_clauses=pass1_clauses,
            pass2_row=pass2_row,
        )

        if item_ast is None:
            continue

        if criterion_type == "inclusion":
            inclusion_item_trees.append(item_ast)
        elif criterion_type == "exclusion":
            exclusion_item_trees.append(item_ast)

    ast = {
        "trial_id": document_id,
        "inclusion_criteria": combine_item_trees(inclusion_item_trees, "inclusion"),
        "exclusion_criteria": combine_item_trees(exclusion_item_trees, "exclusion"),
    }

    return ast


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ROOT = Path(__file__).resolve().parents[3]

    pass1_path = ROOT / "outputs" / "extraction" / "pass1_flat" / "chia_text_only_200_pass1_flat.jsonl"
    pass2_path = ROOT / "outputs" / "extraction" / "branch_a" / "pass2_leaves" / "chia_text_only_200_pass2_leaves.jsonl"
    schema_path = ROOT / "schemas" / "rules_v3.json"

    out_dir = ROOT / "outputs" / "extraction" / "branch_a" / "rules_v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "chia_text_only_200_rules_v3_ast_A.jsonl"

    pass1_rows = load_jsonl(pass1_path)
    pass2_rows = load_jsonl(pass2_path)
    pass2_index = build_pass2_index(pass2_rows)

    schema = load_schema(schema_path)
    validator = Draft7Validator(schema)

    rows_by_doc = group_by_document(pass1_rows)

    output_rows: List[Dict[str, Any]] = []
    n_ok = 0
    n_err = 0

    for document_id, doc_rows in rows_by_doc.items():
        try:
            ast = build_document_ast(
                document_id=document_id,
                pass1_rows_for_doc=doc_rows,
                pass2_index=pass2_index,
            )

            errors = sorted(validator.iter_errors(ast), key=lambda e: list(e.absolute_path))
            if errors:
                e = errors[0]
                path = ".".join(str(x) for x in e.absolute_path)
                schema_path = ".".join(str(x) for x in e.absolute_schema_path)
                raise ValidationError(
                    f"{e.message} | path={path} | schema_path={schema_path} | validator={e.validator}"
                )

            n_inclusion_items = sum(
                1 for r in doc_rows if r.get("status") == "ok" and r.get("criterion_type_hint") == "inclusion"
            )
            n_exclusion_items = sum(
                1 for r in doc_rows if r.get("status") == "ok" and r.get("criterion_type_hint") == "exclusion"
            )

            output_rows.append(
                {
                    "dataset": "CHIA",
                    "stage": "rules_v3_ast_assembly",
                    "document_id": document_id,
                    "status": "ok",
                    "error": None,
                    "n_pass1_rows": len(doc_rows),
                    "n_inclusion_items": n_inclusion_items,
                    "n_exclusion_items": n_exclusion_items,
                    "rules_v3_ast": ast,
                }
            )
            n_ok += 1

        except Exception as e:
            output_rows.append(
                {
                    "dataset": "CHIA",
                    "stage": "rules_v3_ast_assembly",
                    "document_id": document_id,
                    "status": "error",
                    "error": str(e),
                }
            )
            n_err += 1

    write_jsonl(out_path, output_rows)

    print("Wrote:", out_path)
    print("OK:", n_ok)
    print("ERR:", n_err)


if __name__ == "__main__":
    main()

# Run from the repository root:
# python scripts/02_extraction/branch_a/05_build_rules_v3_rules_from_pass1_and_pass2.py
