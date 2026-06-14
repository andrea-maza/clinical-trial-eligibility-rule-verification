import csv
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


# ============================================================
# Purpose
# ============================================================
#
# Layer 1D deterministic verification inventory.
#
# This script checks consistency between:
#   - Pass 1 clause metadata
#   - Pass 2 raw rules_v3 rules tree leaves
#
# It DOES NOT:
#   - repair leaves
#   - call the LLM
#   - use probabilistic scores
#   - modify the rule tree
#
# Checks:
#   1. Pass 1 clause exists but no Pass 2 leaf exists.
#   2. Pass 2 leaf exists but no Pass 1 clause exists.
#   3. evidence_text is not grounded in Pass 1 source text
#      using clause_text OR item_text.
#   4. negation clause but operator = exists.
#   5. exception/conditional clause without provenance condition/exception context.
#
# Output:
#   - audit CSV
#   - summary JSON
# ============================================================


# ------------------------------------------------------------
# Import Layer 1 inventory helpers
# ------------------------------------------------------------

def import_inventory_module():
    here = Path(__file__).resolve().parent
    inventory_path = here / "01_build_deterministic_inventory.py"

    spec = importlib.util.spec_from_file_location(
        "inventory_layer1",
        inventory_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


inv = import_inventory_module()


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]

# If auto-discovery picks the wrong Pass 1 file, set this manually.
# Example:
# PASS1_PATH_OVERRIDE = ROOT / "outputs" / "extraction" / "chen" / "pass1_flat" / "chia_text_only_100_pass1_flat.jsonl"
PASS1_PATH_OVERRIDE = (
    ROOT
    / "outputs"
    / "extraction"
    / "pass1_flat"
    / "chia_text_only_200_pass1_flat.jsonl"
)

BRANCH_AST_PATHS = {
    "A_bert_rules": (
        ROOT
        / "outputs"
        / "extraction"
        / "branch_a"
        / "rules_v3"
        / "chia_text_only_200_rules_v3_ast_A.jsonl"
    ),
    "B_llm_pass2": (
        ROOT
        / "outputs"
        / "extraction"
        / "branch_b"
        / "rules_v3_llm_pass2"
        / "chia_text_only_200_rules_v3_ast_B.jsonl"
    ),
}

OUT_DIR = ROOT / "outputs" / "verification" / "layer1" / "pass1_pass2_consistency"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIT_CSV_PATH = OUT_DIR / "layer1d_pass1_pass2_consistency_audit.csv"
SUMMARY_JSON_PATH = OUT_DIR / "layer1d_pass1_pass2_consistency_summary.json"


# ------------------------------------------------------------
# IO helpers
# ------------------------------------------------------------

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


def pct(n: int, d: int) -> float:
    if d == 0:
        return 0.0
    return round(100.0 * n / d, 2)


def normalize_for_substring(x: Any) -> str:
    text = inv.normalize_text(x).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_set(text: Any) -> set:
    text = normalize_for_substring(text)
    text = re.sub(r"[^a-z0-9]+", " ", text)

    stop = {
        "a", "an", "the", "of", "with", "without", "and", "or",
        "to", "in", "on", "for", "by", "at", "from", "that",
        "which", "patients", "patient", "subjects", "subject",
        "must", "should", "be", "been", "are", "is",
    }

    return {
        t for t in text.split()
        if len(t) >= 2 and t not in stop
    }


def token_overlap(a: Any, b: Any) -> float:
    a_tokens = token_set(a)
    b_tokens = token_set(b)

    if not a_tokens:
        return 0.0

    return len(a_tokens & b_tokens) / len(a_tokens)

def infer_item_text(row: Dict[str, Any], item: Optional[Dict[str, Any]] = None) -> str:
    if item:
        value = get_first(
            item,
            [
                "item_text",
                "criterion_text",
                "original_text",
                "source_item_text",
                "full_text",
                "text",
            ],
            "",
        )
        if value:
            return inv.normalize_text(value)

    value = get_first(
        row,
        [
            "item_text",
            "criterion_text",
            "original_text",
            "source_item_text",
            "full_text",
            "text",
        ],
        "",
    )
    return inv.normalize_text(value)

def entity_text_has_negation(entity_text: Any) -> bool:
    text = normalize_for_substring(entity_text)

    patterns = [
        r"\bnot\b",
        r"\bwithout\b",
        r"\bunable to\b",
        r"\binability to\b",
        r"\bunability to\b",
        r"\bunwilling\b",
        r"\bunwillingness\b",
        r"\black of\b",
        r"\bfailure to\b",
        r"\bfailed to\b",
        r"\bcan not\b",
        r"\bcannot\b",
        r"\bnot amenable to\b",
        r"\bnot controlled\b",
        r"\bnot recovered\b",
        r"\basymptomatic\b",
        r"\bsymptom[- ]free\b",
        r"\bnon[- ]diabetic\b",
        r"\bnon[- ]pregnant\b",
        r"\bnon[- ]smoker\b",
        r"\bnon[- ]lactating\b",
    ]

    return any(re.search(pattern, text) for pattern in patterns)

# ------------------------------------------------------------
# Pass 1 loading / extraction
# ------------------------------------------------------------

def discover_pass1_path(root: Path) -> Path:
    if PASS1_PATH_OVERRIDE is not None:
        if not PASS1_PATH_OVERRIDE.exists():
            raise FileNotFoundError(f"Manual PASS1_PATH_OVERRIDE does not exist: {PASS1_PATH_OVERRIDE}")
        return PASS1_PATH_OVERRIDE

    search_root = root / "outputs" / "extraction"

    candidates = [
        p for p in search_root.rglob("*.jsonl")
        if "pass1" in p.name.lower() or "pass1" in str(p.parent).lower()
    ]

    if not candidates:
        raise FileNotFoundError(
            "Could not auto-discover a Pass 1 JSONL file. "
            "Set PASS1_PATH_OVERRIDE manually at the top of this script."
        )

    scored = []

    for path in candidates:
        try:
            rows = load_jsonl(path)
            clauses = extract_pass1_clauses(rows)
            scored.append((len(clauses), len(rows), path))
        except Exception:
            continue

    scored = [x for x in scored if x[0] > 0]

    if not scored:
        raise RuntimeError(
            "Found Pass 1-like JSONL files, but none yielded extractable clauses. "
            "Set PASS1_PATH_OVERRIDE manually and inspect the Pass 1 format."
        )

    scored.sort(reverse=True, key=lambda x: (x[0], x[1]))
    return scored[0][2]


def get_first(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for k in keys:
        if k in d and d.get(k) is not None:
            return d.get(k)
    return default


def infer_document_id(row: Dict[str, Any], item: Optional[Dict[str, Any]] = None) -> str:
    if item:
        value = get_first(item, ["document_id", "doc_id", "trial_id", "nct_id"])
        if value:
            return str(value)

    value = get_first(row, ["document_id", "doc_id", "trial_id", "nct_id", "id"])
    return str(value or "")


def infer_item_uid(row: Dict[str, Any], item: Dict[str, Any]) -> str:
    value = get_first(item, ["item_uid", "item_id", "uid", "criterion_uid"])
    if value:
        return str(value)

    value = get_first(row, ["item_uid", "item_id", "uid", "criterion_uid"])
    if value:
        return str(value)

    return ""


def infer_clause_id(clause: Dict[str, Any], fallback_index: int) -> str:
    value = get_first(clause, ["clause_id", "leaf_id", "id"])
    if value:
        value = str(value)

        # Normalize "1" -> "C1"
        if re.fullmatch(r"\d+", value):
            return f"C{value}"

        return value

    return f"C{fallback_index}"


def infer_clause_text(clause: Dict[str, Any]) -> str:
    return inv.normalize_text(
        get_first(
            clause,
            [
                "clause_text",
                "text",
                "evidence_text",
                "source_text",
                "source_clause_text",
                "criterion_text",
                "sentence",
            ],
            "",
        )
    )


def infer_clause_type(clause: Dict[str, Any], item: Optional[Dict[str, Any]] = None) -> str:
    value = get_first(
        clause,
        ["clause_type", "type", "logic_type", "semantic_type"],
        None,
    )

    if value is None and item:
        value = get_first(item, ["clause_type", "type", "logic_type"], None)

    return str(value or "").lower().strip()


def infer_is_negated(clause: Dict[str, Any]) -> bool:
    value = get_first(
        clause,
        ["is_negated", "negated", "is_negation", "negation"],
        False,
    )

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.lower().strip() in {"true", "yes", "1", "negated", "negation"}

    return bool(value)


def infer_connector(clause: Dict[str, Any]) -> str:
    value = get_first(
        clause,
        ["connector", "next_connector", "parent_connector", "logic_connector"],
        "",
    )
    return str(value or "").upper().strip()


def collect_clause_dicts_from_node(node: Any) -> List[Dict[str, Any]]:
    """
    Recursively collects dictionaries that look like Pass 1 clause objects.
    Used as a fallback for unknown Pass 1 formats.
    """
    found = []

    if isinstance(node, dict):
        has_text = any(
            k in node for k in [
                "clause_text",
                "source_clause_text",
                "source_text",
                "text",
                "evidence_text",
            ]
        )
        has_id = any(k in node for k in ["clause_id", "leaf_id", "id"])

        if has_text and has_id:
            found.append(node)

        for v in node.values():
            found.extend(collect_clause_dicts_from_node(v))

    elif isinstance(node, list):
        for x in node:
            found.extend(collect_clause_dicts_from_node(x))

    return found


def extract_pass1_clauses(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Flexible extractor for Pass 1 clauses.

    Expected output fields:
      document_id, item_uid, clause_id, clause_text,
      clause_type, is_negated, connector
    """
    out = []

    for row in rows:
        # Case 1: JSONL row is already one clause.
        if row.get("item_uid") and row.get("clause_id") and (
            row.get("clause_text") or row.get("text") or row.get("source_clause_text")
        ):
            document_id = infer_document_id(row)
            item_uid = str(row.get("item_uid"))
            clause_id = infer_clause_id(row, 1)

            out.append({
                "document_id": document_id,
                "item_uid": item_uid,
                "clause_id": clause_id,
                "clause_text": infer_clause_text(row),
                "item_text": infer_item_text(row),
                "clause_type": infer_clause_type(row),
                "is_negated": infer_is_negated(row),
                "connector": infer_connector(row),
            })
            continue

        # Case 2: JSONL row is one item with a clauses list.
        row_item_uid = get_first(row, ["item_uid", "item_id", "uid", "criterion_uid"])

        row_clauses = get_first(
            row,
            ["clauses", "leaf_clauses", "pass1_clauses", "atomic_clauses"],
            None,
        )

        if row_item_uid and isinstance(row_clauses, list):
            document_id = infer_document_id(row)
            item_uid = str(row_item_uid)

            for i, clause in enumerate(row_clauses, start=1):
                if not isinstance(clause, dict):
                    continue

                out.append({
                    "document_id": document_id,
                    "item_uid": item_uid,
                    "clause_id": infer_clause_id(clause, i),
                    "clause_text": infer_clause_text(clause),
                    "item_text": infer_item_text(row),
                    "clause_type": infer_clause_type(clause, row),
                    "is_negated": infer_is_negated(clause),
                    "connector": infer_connector(clause),
                })
            continue

        # Case 3: JSONL row is one document with items.
        item_lists = []

        for key in ["items", "criteria", "criterion_items", "pass1_items"]:
            if isinstance(row.get(key), list):
                item_lists.extend(row.get(key))

        if item_lists:
            for item in item_lists:
                if not isinstance(item, dict):
                    continue

                document_id = infer_document_id(row, item)
                item_uid = infer_item_uid(row, item)

                clauses = get_first(
                    item,
                    ["clauses", "leaf_clauses", "pass1_clauses", "atomic_clauses"],
                    None,
                )

                if isinstance(clauses, list):
                    for i, clause in enumerate(clauses, start=1):
                        if not isinstance(clause, dict):
                            continue

                        out.append({
                            "document_id": document_id,
                            "item_uid": item_uid,
                            "clause_id": infer_clause_id(clause, i),
                            "clause_text": infer_clause_text(clause),
                            "item_text": infer_item_text(row, item),
                            "clause_type": infer_clause_type(clause, item),
                            "is_negated": infer_is_negated(clause),
                            "connector": infer_connector(clause),
                        })
                else:
                    # Fallback: recursively search inside item.
                    found = collect_clause_dicts_from_node(item)

                    for i, clause in enumerate(found, start=1):
                        out.append({
                            "document_id": document_id,
                            "item_uid": item_uid,
                            "clause_id": infer_clause_id(clause, i),
                            "clause_text": infer_clause_text(clause),
                            "item_text": infer_item_text(row, item),
                            "clause_type": infer_clause_type(clause, item),
                            "is_negated": infer_is_negated(clause),
                            "connector": infer_connector(clause),
                        })

            continue

        # Case 4: Unknown row format; recursive fallback.
        found = collect_clause_dicts_from_node(row)

        if found:
            document_id = infer_document_id(row)
            item_uid = infer_item_uid(row, row)

            for i, clause in enumerate(found, start=1):
                out.append({
                    "document_id": document_id,
                    "item_uid": item_uid,
                    "clause_id": infer_clause_id(clause, i),
                    "clause_text": infer_clause_text(clause),
                    "item_text": infer_item_text(row),
                    "clause_type": infer_clause_type(clause, row),
                    "is_negated": infer_is_negated(clause),
                    "connector": infer_connector(clause),
                })

    # Remove empty / invalid keys.
    cleaned = []

    for c in out:
        if not c["document_id"] or not c["item_uid"] or not c["clause_id"]:
            continue

        cleaned.append(c)

    return cleaned


def build_pass1_index(pass1_clauses: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    index = {}

    for c in pass1_clauses:
        key = (c["document_id"], c["item_uid"], c["clause_id"])

        # If duplicates exist, keep first but preserve duplicate count.
        if key not in index:
            index[key] = c
        else:
            index[key].setdefault("_duplicate_count", 1)
            index[key]["_duplicate_count"] += 1

    return index


# ------------------------------------------------------------
# Pass 2 logical rule-tree leaves
# ------------------------------------------------------------

def extract_ast_leaves(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    document_id = row.get("document_id", "")
    ast = row.get("rules_v3_ast")

    if row.get("status") != "ok" or not isinstance(ast, dict):
        return []

    nodes = []
    nodes.extend(inv.walk_ast(ast.get("inclusion_criteria"), path="inclusion_criteria"))
    nodes.extend(inv.walk_ast(ast.get("exclusion_criteria"), path="exclusion_criteria"))

    out = []

    for path, node, criterion in nodes:
        criterion_id = criterion.get("criterion_id", "")
        item_uid, clause_id = inv.parse_item_and_clause_from_criterion_id(criterion_id)

        out.append({
            "document_id": document_id,
            "path": path,
            "criterion_id": criterion_id,
            "item_uid": item_uid,
            "clause_id": clause_id,
            "criterion": criterion,
        })

    return out


def has_exception_or_condition_context(criterion: Dict[str, Any]) -> bool:
    provenance = criterion.get("provenance")

    if not isinstance(provenance, dict):
        return False

    return bool(
        inv.normalize_text(provenance.get("source_condition_text"))
        or inv.normalize_text(provenance.get("source_exception_context"))
    )


# ------------------------------------------------------------
# Layer 1D checks
# ------------------------------------------------------------

def is_negation_clause(pass1_clause: Dict[str, Any]) -> bool:
    clause_type = str(pass1_clause.get("clause_type") or "").lower()
    clause_text = normalize_for_substring(pass1_clause.get("clause_text"))

    if pass1_clause.get("is_negated"):
        return True

    if "negation" in clause_type or clause_type == "negative":
        return True

    negation_patterns = [
        r"\bno\b",
        r"\bnot\b",
        r"\bwithout\b",
        r"\babsence of\b",
        r"\bfree of\b",
        r"\bnegative for\b",
        r"\bunable to\b",
        r"\binability to\b",
        r"\bunability to\b",
        r"\black of\b",
        r"\bfailure to\b",
        r"\bfailed to\b",
        r"\bnot agreed to\b",
        r"\bnot amenable to\b",
        r"\bnot available\b",
        r"\bnon[- ]diabetic\b",
        r"\bnon[- ]pregnant\b",
        r"\bnon[- ]smoker\b",
        r"\bnon[- ]lactating\b",
        r"\basymptomatic\b",
        r"\bsymptom[- ]free\b",
    ]

    return any(re.search(pattern, clause_text) for pattern in negation_patterns)


def is_exception_or_condition_clause(pass1_clause: Dict[str, Any]) -> bool:
    clause_type = str(pass1_clause.get("clause_type") or "").lower()
    clause_text = normalize_for_substring(pass1_clause.get("clause_text"))

    if "exception" in clause_type or "condition" in clause_type:
        return True

    if re.search(r"\b(if|unless|except|provided that|only if|if applicable)\b", clause_text):
        return True

    return False


def evidence_is_substring(evidence_text: str, clause_text: str) -> bool:
    e = normalize_for_substring(evidence_text)
    c = normalize_for_substring(clause_text)

    if not e:
        return False

    if not c:
        return False

    return e in c


def detect_layer1d_leaf_issues(
    leaf: Dict[str, Any],
    pass1_clause: Optional[Dict[str, Any]],
) -> List[str]:
    issues = []

    criterion = leaf["criterion"]

    if pass1_clause is None:
        issues.append("pass2_leaf_without_pass1_clause")
        return issues

    evidence_text = inv.normalize_text(criterion.get("evidence_text"))
    clause_text = inv.normalize_text(pass1_clause.get("clause_text"))

    item_text = inv.normalize_text(pass1_clause.get("item_text"))

    evidence_in_clause = evidence_is_substring(evidence_text, clause_text)
    evidence_in_item = evidence_is_substring(evidence_text, item_text)

    if not evidence_in_clause and not evidence_in_item:
        best_overlap = max(
            token_overlap(evidence_text, clause_text),
            token_overlap(evidence_text, item_text),
        )
        if best_overlap >= 0.70:
            pass 
        else:
            issues.append("evidence_text_not_substring_of_pass1_source_text")

    operator = criterion.get("operator")
    negation_clause = is_negation_clause(pass1_clause)

    entity_has_negative_concept = entity_text_has_negation(
        criterion.get("entity_text")
    )

    if negation_clause:
        # Do not flag cases where the negative concept is itself the entity.
        # Example: entity="inability to complete questionnaires", operator="exists"
        # is acceptable because the criterion is the presence of inability.
        if operator == "exists" and not entity_has_negative_concept:
            issues.append("negation_clause_with_exists_operator")

        # Avoid double negation.
        # Example: entity="inability to consent", operator="not_exists"
        # would mean absence of inability, which reverses the criterion.
        if operator == "not_exists" and entity_has_negative_concept:
            issues.append("negative_entity_with_not_exists_operator")
    else:
        if operator == "not_exists":
            issues.append("positive_clause_with_not_exists_operator")

    if is_exception_or_condition_clause(pass1_clause):
        has_context = has_exception_or_condition_context(criterion)
        
        if not has_context:
            issues.append("exception_or_condition_clause_without_context_handling")
        
        # Only flag as hard issue if BOTH: no context AND computability = computable
        if not has_context and criterion.get("computability") == "computable":
            issues.append("exception_or_condition_clause_with_computable_status")
        # Softer case: has context but still marked computable
        elif has_context and criterion.get("computability") == "computable":
            issues.append("exception_clause_computable_despite_context")

    return sorted(set(issues))


# ------------------------------------------------------------
# Branch processing
# ------------------------------------------------------------



def process_branch(
    branch_name: str,
    ast_path: Path,
    pass1_index: Dict[Tuple[str, str, str], Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = load_jsonl(ast_path)

    audit_rows = []

    total_documents = len(rows)
    total_pass2_leaves = 0

    pass2_keys = set()

    issue_counter = Counter()
    flagged_leaf_keys = set()

    pass2_leaf_without_pass1_count = 0
    negation_operator_mismatch_count = 0
    exception_context_missing_count = 0

    positive_not_exists_mismatch_count = 0
    exception_computable_status_count = 0
    negative_entity_not_exists_count = 0
    exception_computable_despite_context_count = 0

    evidence_not_in_source_count = 0
    evidence_in_item_only_count = 0

    for row in rows:
        leaves = extract_ast_leaves(row)

        for leaf in leaves:
            total_pass2_leaves += 1

            key = (
                leaf["document_id"],
                leaf["item_uid"],
                leaf["clause_id"],
            )

            leaf_instance_key = (
                leaf["document_id"],
                leaf["item_uid"],
                leaf["clause_id"],
                leaf["path"],
            )

            pass2_keys.add(key)
            pass1_clause = pass1_index.get(key)

            issues = detect_layer1d_leaf_issues(
                leaf=leaf,
                pass1_clause=pass1_clause,
            )

            issue_counter.update(issues)

            if issues:
                flagged_leaf_keys.add(leaf_instance_key)

            if "pass2_leaf_without_pass1_clause" in issues:
                pass2_leaf_without_pass1_count += 1

            if "evidence_text_not_substring_of_pass1_source_text" in issues:
                evidence_not_in_source_count += 1

            if "negation_clause_with_exists_operator" in issues:
                negation_operator_mismatch_count += 1

            if "positive_clause_with_not_exists_operator" in issues:
                positive_not_exists_mismatch_count += 1
            
            if "negative_entity_with_not_exists_operator" in issues:
                negative_entity_not_exists_count += 1

            if "exception_or_condition_clause_without_context_handling" in issues:
                exception_context_missing_count += 1

            if "exception_or_condition_clause_with_computable_status" in issues:
                exception_computable_status_count += 1

            if "exception_clause_computable_despite_context" in issues:
                exception_computable_despite_context_count += 1

            criterion = leaf["criterion"]

            evidence_in_clause = False
            evidence_in_item = False
            evidence_alignment_note = ""

            if pass1_clause is not None:
                evidence_in_clause = evidence_is_substring(
                    criterion.get("evidence_text", ""),
                    pass1_clause.get("clause_text", ""),
                )
                evidence_in_item = evidence_is_substring(
                    criterion.get("evidence_text", ""),
                    pass1_clause.get("item_text", ""),
                )

                if (not evidence_in_clause) and evidence_in_item:
                    evidence_in_item_only_count += 1
                    evidence_alignment_note = "evidence_in_item_text_only"

            audit_rows.append({
                "branch": branch_name,
                "row_type": "pass2_leaf_check",
                "document_id": leaf["document_id"],
                "item_uid": leaf["item_uid"],
                "clause_id": leaf["clause_id"],
                "criterion_id": leaf["criterion_id"],
                "path": leaf["path"],
                "issues": ";".join(issues),

                "pass1_clause_text": inv.normalize_text(pass1_clause.get("clause_text")) if pass1_clause else "",
                "pass1_item_text": inv.normalize_text(pass1_clause.get("item_text")) if pass1_clause else "",
                "pass1_clause_type": pass1_clause.get("clause_type", "") if pass1_clause else "",
                "pass1_is_negated": pass1_clause.get("is_negated", "") if pass1_clause else "",
                "pass1_connector": pass1_clause.get("connector", "") if pass1_clause else "",
                "evidence_in_pass1_clause_text": evidence_in_clause,
                "evidence_in_pass1_item_text": evidence_in_item,
                "evidence_alignment_note": evidence_alignment_note,

                "entity_type": criterion.get("entity_type"),
                "entity_text": inv.normalize_text(criterion.get("entity_text")),
                "operator": criterion.get("operator"),
                "value_type": criterion.get("value_type"),
                "value": json.dumps(criterion.get("value"), ensure_ascii=False),
                "temporal_context": json.dumps(criterion.get("temporal_context"), ensure_ascii=False),
                "evidence_text": inv.normalize_text(criterion.get("evidence_text")),
                
            })

    # Pass 1 clauses missing in Pass 2 for this branch
    pass1_keys = set(pass1_index.keys())
    missing_pass2_keys = sorted(pass1_keys - pass2_keys)

    for key in missing_pass2_keys:
        document_id, item_uid, clause_id = key
        c = pass1_index[key]

        issue_counter.update(["pass1_clause_without_pass2_leaf"])

        audit_rows.append({
            "branch": branch_name,
            "row_type": "pass1_clause_missing_pass2_leaf",
            "document_id": document_id,
            "item_uid": item_uid,
            "clause_id": clause_id,
            "criterion_id": "",
            "path": "",
            "issues": "pass1_clause_without_pass2_leaf",

            "pass1_clause_text": inv.normalize_text(c.get("clause_text")),
            "pass1_item_text": inv.normalize_text(c.get("item_text")),
            "pass1_clause_type": c.get("clause_type", ""),
            "pass1_is_negated": c.get("is_negated", ""),
            "pass1_connector": c.get("connector", ""),
            "evidence_in_pass1_clause_text": "",
            "evidence_in_pass1_item_text": "",
            "evidence_alignment_note": "",

            "entity_type": "",
            "entity_text": "",
            "operator": "",
            "value_type": "",
            "value": "",
            "temporal_context": "",
            "evidence_text": "",
        })

    total_pass1_clauses = len(pass1_index)

    summary = {
        "branch": branch_name,
        "input_raw_ast_jsonl": str(ast_path),

        "total_documents": total_documents,
        "total_pass1_clauses": total_pass1_clauses,
        "total_pass2_leaves": total_pass2_leaves,

        "pass1_clauses_without_pass2_leaf": len(missing_pass2_keys),
        "pass2_leaves_without_pass1_clause": pass2_leaf_without_pass1_count,
        "evidence_text_not_substring_of_pass1_source_text": evidence_not_in_source_count,
        "evidence_text_not_substring_of_atomic_clause_but_in_item_text": evidence_in_item_only_count,
        "negation_clause_with_exists_operator": negation_operator_mismatch_count,
        "exception_or_condition_clause_without_context_handling": exception_context_missing_count,
        "positive_clause_with_not_exists_operator": positive_not_exists_mismatch_count,
        "exception_or_condition_clause_with_computable_status": exception_computable_status_count,
        "negative_entity_with_not_exists_operator": negative_entity_not_exists_count,
        "exception_clause_computable_despite_context": exception_computable_despite_context_count,

        "layer1d_flagged_pass2_leaves": len(flagged_leaf_keys),

        "percentages": {
            "pass1_clause_missing_pass2_leaf_rate": pct(len(missing_pass2_keys), total_pass1_clauses),
            "pass2_leaf_without_pass1_clause_rate": pct(pass2_leaf_without_pass1_count, total_pass2_leaves),
            "evidence_text_not_in_pass1_source_rate": pct(evidence_not_in_source_count, total_pass2_leaves),
            "evidence_text_in_item_only_rate": pct(evidence_in_item_only_count, total_pass2_leaves),
            "negation_operator_mismatch_leaf_rate": pct(negation_operator_mismatch_count, total_pass2_leaves),
            "exception_context_missing_leaf_rate": pct(exception_context_missing_count, total_pass2_leaves),
            "positive_not_exists_mismatch_leaf_rate": pct(positive_not_exists_mismatch_count, total_pass2_leaves),
            "exception_computable_status_leaf_rate": pct(exception_computable_status_count, total_pass2_leaves),
            "negative_entity_not_exists_leaf_rate": pct(negative_entity_not_exists_count, total_pass2_leaves),
            "exception_computable_despite_context_leaf_rate": pct(exception_computable_despite_context_count, total_pass2_leaves),
            "layer1d_flagged_pass2_leaf_rate": pct(len(flagged_leaf_keys), total_pass2_leaves),
        },

        "issue_counts": dict(issue_counter.most_common()),
    }

    return audit_rows, summary


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    pass1_path = discover_pass1_path(ROOT)
    pass1_rows = load_jsonl(pass1_path)
    pass1_clauses = extract_pass1_clauses(pass1_rows)
    pass1_index = build_pass1_index(pass1_clauses)

    all_audit_rows = []
    branch_summaries = {}

    for branch_name, ast_path in BRANCH_AST_PATHS.items():
        audit_rows, summary = process_branch(
            branch_name=branch_name,
            ast_path=ast_path,
            pass1_index=pass1_index,
        )

        all_audit_rows.extend(audit_rows)
        branch_summaries[branch_name] = summary

    fieldnames = [
        "branch",
        "row_type",
        "document_id",
        "item_uid",
        "clause_id",
        "criterion_id",
        "path",
        "issues",

        "pass1_clause_text",
        "pass1_item_text",
        "pass1_clause_type",
        "pass1_is_negated",
        "pass1_connector",
        "evidence_in_pass1_clause_text",
        "evidence_in_pass1_item_text",
        "evidence_alignment_note",

        "entity_type",
        "entity_text",
        "operator",
        "value_type",
        "value",
        "temporal_context",
        "evidence_text",
    ]

    write_csv(AUDIT_CSV_PATH, all_audit_rows, fieldnames)

    combined = {
        "total_pass1_clauses": sum(s["total_pass1_clauses"] for s in branch_summaries.values()),
        "total_pass2_leaves": sum(s["total_pass2_leaves"] for s in branch_summaries.values()),
        "pass1_clauses_without_pass2_leaf": sum(s["pass1_clauses_without_pass2_leaf"] for s in branch_summaries.values()),
        "pass2_leaves_without_pass1_clause": sum(s["pass2_leaves_without_pass1_clause"] for s in branch_summaries.values()),

        "evidence_text_not_substring_of_pass1_source_text": sum(
            s["evidence_text_not_substring_of_pass1_source_text"]
            for s in branch_summaries.values()
        ),
        "evidence_text_not_substring_of_atomic_clause_but_in_item_text": sum(
            s["evidence_text_not_substring_of_atomic_clause_but_in_item_text"]
            for s in branch_summaries.values()
        ),

        "negation_clause_with_exists_operator": sum(s["negation_clause_with_exists_operator"] for s in branch_summaries.values()),
        "exception_or_condition_clause_without_context_handling": sum(s["exception_or_condition_clause_without_context_handling"] for s in branch_summaries.values()),
        "positive_clause_with_not_exists_operator": sum(
            s["positive_clause_with_not_exists_operator"]
            for s in branch_summaries.values()
        ),
        "negative_entity_with_not_exists_operator": sum(
            s["negative_entity_with_not_exists_operator"]
            for s in branch_summaries.values()
        ),
        "exception_or_condition_clause_with_computable_status": sum(
            s["exception_or_condition_clause_with_computable_status"]
            for s in branch_summaries.values()
        ),
        "exception_clause_computable_despite_context": sum(
            s["exception_clause_computable_despite_context"]
            for s in branch_summaries.values()
        ),
        "layer1d_flagged_pass2_leaves": sum(s["layer1d_flagged_pass2_leaves"] for s in branch_summaries.values()),
    }

    combined["percentages"] = {
        "pass1_clause_missing_pass2_leaf_rate": pct(
            combined["pass1_clauses_without_pass2_leaf"],
            combined["total_pass1_clauses"],
        ),
        "pass2_leaf_without_pass1_clause_rate": pct(
            combined["pass2_leaves_without_pass1_clause"],
            combined["total_pass2_leaves"],
        ),
        "evidence_text_not_in_pass1_source_rate": pct(
            combined["evidence_text_not_substring_of_pass1_source_text"],
            combined["total_pass2_leaves"],
        ),
        "evidence_text_in_item_only_rate": pct(
            combined["evidence_text_not_substring_of_atomic_clause_but_in_item_text"],
            combined["total_pass2_leaves"],
        ),
        "negation_operator_mismatch_leaf_rate": pct(
            combined["negation_clause_with_exists_operator"],
            combined["total_pass2_leaves"],
        ),
        "exception_context_missing_leaf_rate": pct(
            combined["exception_or_condition_clause_without_context_handling"],
            combined["total_pass2_leaves"],
        ),
        "positive_not_exists_mismatch_leaf_rate": pct(
            combined["positive_clause_with_not_exists_operator"],
            combined["total_pass2_leaves"],
        ),
        "exception_computable_status_leaf_rate": pct(
            combined["exception_or_condition_clause_with_computable_status"],
            combined["total_pass2_leaves"],
        ),
        "negative_entity_not_exists_leaf_rate": pct(
            combined["negative_entity_with_not_exists_operator"],
            combined["total_pass2_leaves"],
        ),
        "exception_computable_despite_context_leaf_rate": pct(
            combined["exception_clause_computable_despite_context"],
            combined["total_pass2_leaves"],
        ),
        "layer1d_flagged_pass2_leaf_rate": pct(
            combined["layer1d_flagged_pass2_leaves"],
            combined["total_pass2_leaves"],
        ),
    }

    full_summary = {
        "stage": "layer1d_pass1_pass2_consistency_inventory",
        "description": (
            "Deterministic Pass 1 to Pass 2 consistency inventory. "
            "No repair is applied. Checks missing/extra leaves, evidence grounding "
            "against Pass 1 clause text and item text, negation/operator consistency, and "
            "exception/condition context handling."
        ),
        "inputs": {
            "pass1_path": str(pass1_path),
            "raw_ast_branches": {k: str(v) for k, v in BRANCH_AST_PATHS.items()},
        },
        "outputs": {
            "audit_csv": str(AUDIT_CSV_PATH),
            "summary_json": str(SUMMARY_JSON_PATH),
        },
        "pass1_extraction": {
            "pass1_rows": len(pass1_rows),
            "pass1_clauses_extracted": len(pass1_clauses),
            "pass1_unique_clause_keys": len(pass1_index),
        },
        "branches": branch_summaries,
        "combined": combined,
    }

    write_json(SUMMARY_JSON_PATH, full_summary)

    print("\n===== LAYER 1D PASS1/PASS2 CONSISTENCY INVENTORY =====")
    print("Selected Pass 1 file:", pass1_path)
    print("Pass 1 rows:", len(pass1_rows))
    print("Pass 1 clauses extracted:", len(pass1_clauses))
    print("Pass 1 unique clause keys:", len(pass1_index))
    print("Wrote audit CSV:", AUDIT_CSV_PATH)
    print("Wrote summary JSON:", SUMMARY_JSON_PATH)

    for branch_name, s in branch_summaries.items():
        print(f"\n--- {branch_name} ---")
        print("Total Pass 1 clauses:", s["total_pass1_clauses"])
        print("Total Pass 2 leaves:", s["total_pass2_leaves"])
        print("Pass 1 clauses without Pass 2 leaf:", s["pass1_clauses_without_pass2_leaf"])
        print("Pass 2 leaves without Pass 1 clause:", s["pass2_leaves_without_pass1_clause"])
        print("Evidence text not in Pass 1 source text:", s["evidence_text_not_substring_of_pass1_source_text"])
        print("Evidence text in item text only:", s["evidence_text_not_substring_of_atomic_clause_but_in_item_text"])
        print("Negation clause with exists operator:", s["negation_clause_with_exists_operator"])
        print("Positive clause with not_exists operator:", s["positive_clause_with_not_exists_operator"])
        print("Negative entity with not_exists operator:", s["negative_entity_with_not_exists_operator"])
        print("Exception/condition clause with computable status:", s["exception_or_condition_clause_with_computable_status"])
        print("Exception clause computable despite context:", s["exception_clause_computable_despite_context"])
        print("Exception/condition clause without context handling:", s["exception_or_condition_clause_without_context_handling"])
        print("Layer 1D flagged Pass 2 leaves:", s["layer1d_flagged_pass2_leaves"])
        print("Percentages:", s["percentages"])
        print("Issue counts:", s["issue_counts"])

    print("\n--- Combined ---")
    print(combined)


if __name__ == "__main__":
    main()

# Run from the repository root:
# python scripts/03_verification/01_layer1/02_check_pass1_pass2_consistency.py