"""
05_inventory_branch_b_support_signals.py

Build the Layer 2 support-signal inventory for Branch B.

Branch B uses LLM-based Pass 2 leaf extraction. This script combines
the Branch B logical rule-tree leaves with source-grounding signals,
context signals, the Branch B Layer 1 policy, and Branch A agreement
information used only as a diagnostic comparison.

This script does not assign final risk labels and does not modify the
logical rule tree.

Outputs:
    outputs/verification/layer2/branch_b/
        layer2_branch_b_support_inventory_leaf_level.csv
        layer2_branch_b_support_inventory_summary.json

Run from the repository root:
python scripts/03_verification/02_layer2/05_inventory_branch_b_support_signals.py
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]

BRANCH_B_AST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "extraction"
    / "branch_b"
    / "rules_v3_llm_pass2"
    / "chia_text_only_200_rules_v3_ast_B.jsonl"
)

PASS2_INPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "extraction"
    / "pass2_inputs"
)

BRANCH_A_SCORE_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "verification"
    / "layer2"
    / "branch_a"
    / "layer2_branch_a_leaf_risk_scores.csv"
)

BRANCH_B_LAYER1_POLICY_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "verification"
    / "layer1"
    / "policy_branch_b"
    / "layer1_policy_branch_b_leaf_level.csv"
)

OUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "verification"
    / "layer2"
    / "branch_b"
)

OUT_CSV = OUT_DIR / "layer2_branch_b_support_inventory_leaf_level.csv"
OUT_JSON = OUT_DIR / "layer2_branch_b_support_inventory_summary.json"


# ---------------------------------------------------------------------
# Input auto-detection
# ---------------------------------------------------------------------

def find_one_file(candidates: List[Path], glob_patterns: List[str], folder: Path) -> Path:
    """
    Finds exactly one useful file.

    Priority:
    1. explicit candidates
    2. glob patterns
    """
    for path in candidates:
        if path.exists():
            return path

    found: List[Path] = []
    for pattern in glob_patterns:
        found.extend(sorted(folder.glob(pattern)))

    found = [p for p in found if p.is_file()]

    if not found:
        raise FileNotFoundError(
            "Could not find input file.\n"
            f"Folder searched: {folder}\n"
            f"Explicit candidates: {[str(p) for p in candidates]}\n"
            f"Glob patterns: {glob_patterns}"
        )

    if len(found) > 1:
        print("WARNING: multiple possible files found. Using the first one:")
        for p in found:
            print("  ", p)

    return found[0]


def detect_pass2_inputs() -> Path:
    candidates = [
        PASS2_INPUT_DIR / "chia_text_only_200_pass2_inputs.jsonl",
    ]

    patterns = [
        "chia_text_only_200_pass2_inputs*.jsonl",
    ]

    return find_one_file(candidates, patterns, PASS2_INPUT_DIR)


# ---------------------------------------------------------------------
# Basic readers / writers
# ---------------------------------------------------------------------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
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


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    # Stable column order: important columns first, then the rest.
    priority_cols = [
        "document_id",
        "criterion_id",
        "branch",
        "leaf_path",
        "node_type",
        "eligibility_type",
        "status",
        "entity_type",
        "entity_text",
        "operator",
        "value",
        "unit",
        "value_type",
        "evidence_text",
        "item_text",
        "source_text",
        "layer1_needs_review",
        "layer1_issue_count",
        "layer1_issue_codes_json",
        "layer1_policy_action_hint",
        "layer1_policy_bucket",
        "layer1_policy_severity",
        "layer1_policy_reasons",
        "layer1_policy_score",
        "layer1_policy_hard_issue_count",
        "layer1_policy_soft_warning_count",
        "layer1_policy_execution_issue_count",
        "branch_a_match_found",
        "branch_a_entity_type",
        "branch_a_entity_text",
        "branch_a_operator",
        "branch_a_value",
        "branch_a_unit",
        "branch_a_entity_type_agrees",
        "branch_a_operator_agrees",
        "branch_a_value_agrees",
        "branch_a_unit_agrees",
        "branch_a_entity_text_token_overlap",
        "branch_a_agreement_n",
        "entity_text_exact_in_evidence",
        "entity_text_ci_in_evidence",
        "entity_text_normalized_in_evidence",
        "entity_text_exact_in_item",
        "entity_text_ci_in_item",
        "value_text_found_in_evidence",
        "value_text_found_in_item",
        "unit_text_found_in_evidence",
        "operator_value_structurally_supported",
        "quantitative_cue_present",
        "quantitative_cue_unhandled",
        "exists_with_quantitative_cue",
        "value_missing_with_quantitative_cue",
        "temporal_marker_in_evidence",
        "temporal_context_present",
        "temporal_marker_missing_context",
        "history_marker_in_evidence",
        "history_context_present",
        "history_marker_missing_context",
        "condition_or_exception_marker_in_evidence",
        "condition_or_exception_context_present",
        "condition_or_exception_marker_missing_context",
        "generic_entity_text",
    ]

    all_cols = []
    seen = set()

    for col in priority_cols:
        if any(col in row for row in rows):
            all_cols.append(col)
            seen.add(col)

    for row in rows:
        for col in row.keys():
            if col not in seen:
                all_cols.append(col)
                seen.add(col)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: serialize_cell(row.get(col, "")) for col in all_cols})


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def serialize_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


# ---------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------

def normalize_space(text: Any) -> str:
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_text(text: Any) -> str:
    """
    Normalization for matching clinical phrases.

    Keeps letters, numbers, %, <, >, = because they matter for thresholds.
    """
    text = normalize_space(text).lower()
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = text.replace("×", "x")
    text = text.replace("≤", "<=")
    text = text.replace("≥", ">=")
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[^a-z0-9%<>=./+\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def token_set(text: Any) -> set:
    norm = normalize_text(text)
    if not norm:
        return set()
    return set(re.findall(r"[a-z0-9]+", norm))


def text_contains(haystack: Any, needle: Any, case_insensitive: bool = False) -> bool:
    hay = normalize_space(haystack)
    nee = normalize_space(needle)

    if not hay or not nee:
        return False

    if case_insensitive:
        return nee.lower() in hay.lower()

    return nee in hay


def normalized_contains(haystack: Any, needle: Any) -> bool:
    hay = normalize_text(haystack)
    nee = normalize_text(needle)

    if not hay or not nee:
        return False

    return nee in hay


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def to_bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}

def is_unique_leaf_key(value: Any) -> bool:
    """
    Accept only keys that are safe for leaf-level joins.

    Reject plain clause IDs such as C1, C2, because those repeat across
    many eligibility items and can silently create wrong matches.
    """
    key = normalize_space(value)

    if not key:
        return False

    if re.fullmatch(r"C\d+", key):
        return False

    return True

# ---------------------------------------------------------------------
# Rule-tree / leaf extraction helpers
# ---------------------------------------------------------------------

CHILD_KEYS = [
    "children",
    "clauses",
    "operands",
    "args",
    "rules",
    "items",
]

LEAF_FIELD_KEYS = [
    "entity_type",
    "entity_text",
    "operator",
    "value",
    "unit",
    "evidence_text",
    "value_type",
    "temporal_context",
    "history_context",
    "computability",
]


def looks_like_leaf(node: Dict[str, Any]) -> bool:
    node_type = str(node.get("node_type") or node.get("type") or "").lower()

    if node_type in {"leaf", "criterion", "atomic", "atom"}:
        return True

    if any(k in node for k in LEAF_FIELD_KEYS):
        # Avoid treating group nodes as leaves if they have children.
        has_children = any(isinstance(node.get(k), list) and node.get(k) for k in CHILD_KEYS)
        return not has_children

    return False


def iter_child_nodes(node: Dict[str, Any]) -> Iterable[Tuple[str, int, Dict[str, Any]]]:
    for key in CHILD_KEYS:
        children = node.get(key)
        if isinstance(children, list):
            for i, child in enumerate(children):
                if isinstance(child, dict):
                    yield key, i, child


def extract_leaves_from_doc(doc: Dict[str, Any]) -> List[Tuple[Dict[str, Any], str]]:
    """
    Returns list of (leaf_node_or_wrapper, leaf_path).

    Important for rules_v3:
    - top-level roots are inclusion_criteria and exclusion_criteria
    - criterion leaves are wrapped as:
        {"node_type": "criterion", "criterion": {...}}
    """
    roots = []

    # Actual Layer 1A verified row structure:
    # row["rules_v3_ast"]["inclusion_criteria"]
    # row["rules_v3_ast"]["exclusion_criteria"]
    ast = doc.get("rules_v3_ast")
    if isinstance(ast, dict):
        for key in ["inclusion_criteria", "exclusion_criteria"]:
            if isinstance(ast.get(key), dict):
                roots.append((f"rules_v3_ast.{key}", ast[key]))

    # Fallback: direct rules_v3 object, if a future file stores the AST directly
    for key in ["inclusion_criteria", "exclusion_criteria"]:
        if isinstance(doc.get(key), dict):
            roots.append((key, doc[key]))

    # Extra fallback: if this is a pass2_output-style row
    if isinstance(doc.get("pass2_output"), dict):
        pass2_output = doc["pass2_output"]
        criteria = pass2_output.get("criteria")
        if isinstance(criteria, list):
            leaves = []
            for i, entry in enumerate(criteria):
                if isinstance(entry, dict):
                    leaves.append((entry, f"pass2_output.criteria[{i}]"))
            return leaves

    # If no known roots, try the full document as a root.
    if not roots and isinstance(doc, dict):
        roots.append(("doc", doc))

    leaves: List[Tuple[Dict[str, Any], str]] = []

    def rec(node: Dict[str, Any], path: str) -> None:
        node_type = str(node.get("node_type") or node.get("type") or "").lower()

        if node_type in {"criterion", "leaf", "atomic", "atom"}:
            leaves.append((node, path))
            return

        if looks_like_leaf(node):
            leaves.append((node, path))
            return

        for key, i, child in iter_child_nodes(node):
            rec(child, f"{path}.{key}[{i}]")

    for root_name, root in roots:
        rec(root, root_name)

    return leaves


def get_first(d: Dict[str, Any], keys: List[str], default: Any = "") -> Any:
    for key in keys:
        if key in d and d[key] not in [None, ""]:
            return d[key]
    return default


def get_nested(d: Dict[str, Any], key: str) -> Any:
    """
    Get a field from:
    1. the object itself
    2. a nested 'criterion' object
    3. a nested 'fields' object

    This is necessary because rules_v3 stores leaves as:
        {"node_type": "criterion", "criterion": {...}}
    """
    if key in d:
        return d.get(key)

    criterion = d.get("criterion")
    if isinstance(criterion, dict) and key in criterion:
        return criterion.get(key)

    fields = d.get("fields")
    if isinstance(fields, dict) and key in fields:
        return fields.get(key)

    return None


def get_leaf_value(leaf: Dict[str, Any], key: str, default: Any = "") -> Any:
    value = get_nested(leaf, key)
    return default if value is None else value


def extract_temporal_context(leaf: Dict[str, Any]) -> Any:
    return get_leaf_value(leaf, "temporal_context", "")


def extract_history_context(leaf: Dict[str, Any]) -> Any:
    return get_leaf_value(leaf, "history_context", "")


def context_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return any(v not in [None, "", [], {}] for v in value.values())
    if isinstance(value, list):
        return len(value) > 0
    return bool(str(value).strip())


# ---------------------------------------------------------------------
# Layer 1 issue extraction
# ---------------------------------------------------------------------

ISSUE_KEYS = [
    "layer1_issues",
    "layer1a_issues",
    "verification_issues",
    "issues",
    "issue_codes",
    "layer1_issue_codes",
]


def collect_issue_codes(obj: Any) -> List[str]:
    """
    Robustly collect issue codes from lists/dicts/strings.
    """
    codes: List[str] = []

    def add_code(x: Any) -> None:
        if x is None:
            return
        if isinstance(x, str):
            if x.strip():
                codes.append(x.strip())
            return
        if isinstance(x, dict):
            for k in ["code", "issue_code", "type", "name"]:
                if k in x and x[k]:
                    codes.append(str(x[k]).strip())
                    return
            # fallback: collect nested
            for v in x.values():
                add_code(v)
            return
        if isinstance(x, list):
            for item in x:
                add_code(item)

    if isinstance(obj, dict):
        for key in ISSUE_KEYS:
            if key in obj:
                add_code(obj[key])

        # Sometimes needs_review reasons are stored separately.
        for key in ["review_reasons", "needs_review_reasons"]:
            if key in obj:
                add_code(obj[key])

    return sorted(set(codes))


def extract_layer1_needs_review(leaf: Dict[str, Any], doc: Dict[str, Any]) -> bool:
    for obj in [leaf, doc]:
        for key in ["needs_review", "layer1_needs_review", "layer1a_needs_review"]:
            if key in obj:
                return to_bool_value(obj.get(key))
    return False

def as_string_list(value: Any) -> List[str]:
    """
    Convert Layer 1 metadata values to a clean list of strings.
    Handles lists and semicolon-separated strings.
    """
    if value is None:
        return []

    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    if isinstance(value, str):
        parts = []
        for chunk in value.split(";"):
            chunk = chunk.strip()
            if chunk:
                parts.append(chunk)
        return parts

    return [str(value).strip()] if str(value).strip() else []


def build_layer1_metadata_index(doc: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Build index from:
        row["verification_layer1a"]["leaf_metadata"]

    Keys:
        ("path", path)
        ("criterion_id", criterion_id)
    """
    index: Dict[Tuple[str, str], Dict[str, Any]] = {}

    block = doc.get("verification_layer1a", {})
    metadata = block.get("leaf_metadata", [])

    if not isinstance(metadata, list):
        return index

    for m in metadata:
        if not isinstance(m, dict):
            continue

        path = m.get("path")
        criterion_id = m.get("criterion_id")

        if path:
            index[("path", str(path))] = m

        if criterion_id:
            index[("criterion_id", str(criterion_id))] = m

    return index


def get_layer1_metadata(
    doc: Dict[str, Any],
    leaf_path: str,
    criterion_id: str,
) -> Dict[str, Any]:
    index = build_layer1_metadata_index(doc)

    return (
        index.get(("path", leaf_path))
        or index.get(("criterion_id", criterion_id))
        or {}
    )


def extract_layer1_signals_from_metadata(layer1_meta: Dict[str, Any]) -> Tuple[bool, bool, List[str]]:
    """
    Returns:
        layer1_needs_review
        layer1_needs_llm_rescue_candidate
        layer1_issue_codes
    """
    codes: List[str] = []

    for key in [
        "layer1a_post_issues",
        "layer1b_flags",
        "layer1c_flags",
        "layer1a_conservative_repair_issues",
    ]:
        codes.extend(as_string_list(layer1_meta.get(key)))

    codes = sorted(set(codes))

    needs_review = to_bool_value(layer1_meta.get("needs_review")) or bool(codes)
    needs_rescue = to_bool_value(layer1_meta.get("needs_llm_rescue_candidate")) or bool(codes)

    return needs_review, needs_rescue, codes

# ---------------------------------------------------------------------
# Pass2 context index
# ---------------------------------------------------------------------

def build_pass2_context_index(pass2_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Build index from pass2_inputs.

    The actual pass2 input file usually has:
        row["pass2_input"]["item_uid"]
        row["pass2_input"]["clauses"][i]["clause_id"]

    Branch B criterion_id is usually:
        {item_uid}_{clause_id}
    """
    index: Dict[str, Dict[str, Any]] = {}

    for row in pass2_rows:
        payload = row.get("pass2_input", row)

        if not isinstance(payload, dict):
            continue

        item_uid = payload.get("item_uid")

        clauses = payload.get("clauses", [])
        if isinstance(clauses, list):
            for clause in clauses:
                if not isinstance(clause, dict):
                    continue

                clause_id = clause.get("clause_id")
                criterion_id = f"{item_uid}_{clause_id}" if item_uid and clause_id else None

                context = dict(payload)
                context.update(clause)
                context["pass2_payload"] = payload

                if criterion_id:
                    index[str(criterion_id)] = context

    return index


def find_pass2_context(
    leaf: Dict[str, Any],
    doc: Dict[str, Any],
    pass2_index: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    possible_keys = [
        get_leaf_value(leaf, "criterion_id", ""),
        get_leaf_value(leaf, "leaf_id", ""),
        doc.get("criterion_id"),
        doc.get("leaf_id"),
    ]

    for key in possible_keys:
        if is_unique_leaf_key(key) and str(key) in pass2_index:
            return pass2_index[str(key)]

    return None


# ---------------------------------------------------------------------
# Branch A index
# ---------------------------------------------------------------------

def build_branch_a_index(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    index: Dict[str, Dict[str, str]] = {}

    for row in rows:
        possible_keys = [
            row.get("criterion_id"),
            row.get("leaf_id"),
        ]

        for key in possible_keys:
            if is_unique_leaf_key(key):
                index[str(key)] = row

    return index

# ---------------------------------------------------------------------
# Branch B Layer 1 policy index
# ---------------------------------------------------------------------

def build_branch_b_layer1_policy_index(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    index: Dict[str, Dict[str, str]] = {}

    for row in rows:
        possible_keys = [
            row.get("criterion_id"),
            row.get("leaf_id"),
        ]

        for key in possible_keys:
            if is_unique_leaf_key(key):
                index[str(key)] = row

    return index


def find_branch_b_layer1_policy_row(
    leaf: Dict[str, Any],
    doc: Dict[str, Any],
    policy_index: Dict[str, Dict[str, str]],
) -> Optional[Dict[str, str]]:
    possible_keys = [
        get_leaf_value(leaf, "criterion_id", ""),
        get_leaf_value(leaf, "leaf_id", ""),
        doc.get("criterion_id"),
        doc.get("leaf_id"),
    ]

    for key in possible_keys:
        if is_unique_leaf_key(key) and str(key) in policy_index:
            return policy_index[str(key)]

    return None

def find_branch_a_row(
    leaf: Dict[str, Any],
    doc: Dict[str, Any],
    branch_a_index: Dict[str, Dict[str, str]],
) -> Optional[Dict[str, str]]:
    possible_keys = [
        get_leaf_value(leaf, "criterion_id", ""),
        get_leaf_value(leaf, "leaf_id", ""),
        doc.get("criterion_id"),
        doc.get("leaf_id"),
    ]

    for key in possible_keys:
        if is_unique_leaf_key(key) and str(key) in branch_a_index:
            return branch_a_index[str(key)]

    return None


def compare_text_field(a: Any, b: Any) -> int:
    if not normalize_text(a) and not normalize_text(b):
        return 1
    if normalize_text(a) == normalize_text(b):
        return 1
    return 0


def token_overlap_ratio(a: Any, b: Any) -> float:
    ta = token_set(a)
    tb = token_set(b)

    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0

    return len(ta & tb) / len(ta | tb)


def compute_branch_a_agreement(
    leaf: Dict[str, Any],
    branch_a_row: Optional[Dict[str, str]],
) -> Dict[str, Any]:
    if branch_a_row is None:
        return {
            "branch_a_match_found": 0,
            "branch_a_entity_type": "",
            "branch_a_entity_text": "",
            "branch_a_operator": "",
            "branch_a_value": "",
            "branch_a_unit": "",
            "branch_a_entity_type_agrees": "",
            "branch_a_operator_agrees": "",
            "branch_a_value_agrees": "",
            "branch_a_unit_agrees": "",
            "branch_a_entity_text_token_overlap": "",
            "branch_a_agreement_n": 0,
        }

    b_entity_type = get_leaf_value(leaf, "entity_type", "")
    b_entity_text = get_leaf_value(leaf, "entity_text", "")
    b_operator = get_leaf_value(leaf, "operator", "")
    b_value = get_leaf_value(leaf, "value", "")
    b_unit = get_leaf_value(leaf, "unit", "")

    a_entity_type = branch_a_row.get("entity_type", "")
    a_entity_text = branch_a_row.get("entity_text", "")
    a_operator = branch_a_row.get("operator", "")
    a_value = branch_a_row.get("value", "")
    a_unit = branch_a_row.get("unit", "")

    entity_type_agrees = compare_text_field(a_entity_type, b_entity_type)
    operator_agrees = compare_text_field(a_operator, b_operator)
    value_agrees = compare_text_field(a_value, b_value)
    unit_agrees = compare_text_field(a_unit, b_unit)
    entity_overlap = token_overlap_ratio(a_entity_text, b_entity_text)

    agreement_n = sum([
        entity_type_agrees,
        operator_agrees,
        value_agrees,
        unit_agrees,
        1 if entity_overlap >= 0.5 else 0,
    ])

    return {
        "branch_a_match_found": 1,
        "branch_a_entity_type": a_entity_type,
        "branch_a_entity_text": a_entity_text,
        "branch_a_operator": a_operator,
        "branch_a_value": a_value,
        "branch_a_unit": a_unit,
        "branch_a_entity_type_agrees": entity_type_agrees,
        "branch_a_operator_agrees": operator_agrees,
        "branch_a_value_agrees": value_agrees,
        "branch_a_unit_agrees": unit_agrees,
        "branch_a_entity_text_token_overlap": round(entity_overlap, 6),
        "branch_a_agreement_n": agreement_n,
    }


# ---------------------------------------------------------------------
# Clinical cue detection
# ---------------------------------------------------------------------

TEMPORAL_MARKER_RE = re.compile(
    r"("
    r"\bwithin\b|\bduring\b|\bbefore\b|\bafter\b|\bprior to\b|"
    r"\bsince\b|\buntil\b|"
    r"\bfollowing\s+(surgery|treatment|therapy|randomization|enrollment|procedure|transplantation)\b|"
    r"\bbaseline\b|\bscreening\b|\benrollment\b|\brandomization\b|"
    r"\bstudy entry\b|\bstudy start\b|\btreatment start\b|"
    r"\blast\s+\d+\s+(days?|weeks?|months?|years?)\b|"
    r"\bwithin\s+\d+\s+(days?|weeks?|months?|years?)\b|"
    r"\bprior\s+to\s+study\s+start\b|"
    r"\bprior\s+to\s+randomization\b"
    r")",
    flags=re.IGNORECASE,
)

HISTORY_MARKER_RE = re.compile(
    r"\b("
    r"history of|prior|previous|previously|past|recurrent|active|current|currently|"
    r"ongoing|known|documented|diagnosed|treated|received|receiving"
    r")\b",
    flags=re.IGNORECASE,
)

CONDITION_EXCEPTION_MARKER_RE = re.compile(
    r"\b("
    r"if|unless|except|except for|other than|provided that|"
    r"in case of|only if|as long as|if applicable"
    r")\b",
    flags=re.IGNORECASE,
)

QUANTITATIVE_CUE_RE = re.compile(
    r"("
    r"<=|>=|<|>|=|≤|≥|"
    r"\b(at least|at most|less than|more than|greater than|no more than|no less than|"
    r"minimum|maximum|above|below|under|over|between)\b|"
    r"\b\d+(\.\d+)?\b|"
    r"\b(uln|lln|mg|ml|dl|kg|g|mmol|mol|iu|units?|%|percent|grade|score|bpm)\b"
    r")",
    flags=re.IGNORECASE,
)

GENERIC_ENTITY_TEXTS = {
    "",
    "patient",
    "patients",
    "subject",
    "subjects",
    "participant",
    "participants",
    "individual",
    "individuals",
    "disease",
    "condition",
    "therapy",
    "treatment",
    "test",
    "value",
    "result",
    "measurement",
    "criteria",
    "criterion",
    "other",
    "unknown",
    "none",
    "null",
}


def has_temporal_marker(text: Any) -> bool:
    return bool(TEMPORAL_MARKER_RE.search(normalize_space(text)))


def has_history_marker(text: Any) -> bool:
    return bool(HISTORY_MARKER_RE.search(normalize_space(text)))


def has_condition_exception_marker(text: Any) -> bool:
    return bool(CONDITION_EXCEPTION_MARKER_RE.search(normalize_space(text)))


def has_true_threshold_cue(text: Any) -> bool:
    t = normalize_space(text).lower()

    if re.search(r"(<=|>=|<|>|≤|≥)", t):
        return True

    if re.search(
        r"\b(at least|at most|less than|more than|greater than|"
        r"no more than|no less than|minimum|maximum|above|below|"
        r"under|over|between)\b",
        t,
    ):
        return True

    if re.search(r"\b\d+(\.\d+)?\s*-\s*\d+(\.\d+)?\b", t):
        return True

    if re.search(r"\bgrade\s+\d+", t):
        return True

    return False


def is_numeric_label_context(text: Any) -> bool:
    t = normalize_space(text).lower()

    label_patterns = [
        r"\btype\s+\d+\b",
        r"\bstage\s+\d+\b",
        r"\bphase\s+\d+\b",
        r"\bdsm[- ]?iv\b",
        r"\bm\d+\b",
        r"\bil[- ]?\d+\b",
        r"\bher2\b",
    ]

    return any(re.search(p, t) for p in label_patterns)


def is_temporal_duration_context(text: Any) -> bool:
    t = normalize_space(text).lower()

    return bool(
        re.search(
            r"\b(within|prior to|after|before|during|for|last)\s+"
            r"\d+(\.\d+)?\s+(days?|weeks?|months?|years?)\b",
            t,
        )
    )


def has_quantitative_cue(text: Any) -> bool:
    """
    Detect true threshold/comparison cues.

    Avoid false positives from numeric labels:
      type 2 diabetic
      stage 4 NSCLC
      M1
      IL-2
      DSM-IV

    Avoid treating pure temporal durations as quantitative extraction errors.
    Those should be handled by temporal_context checks.
    """
    if is_numeric_label_context(text) and not has_true_threshold_cue(text):
        return False

    if is_temporal_duration_context(text) and not has_true_threshold_cue(text):
        return False

    return has_true_threshold_cue(text)


def operator_is_quantitative(operator: Any) -> bool:
    op = normalize_text(operator)
    return op in {
        "<",
        ">",
        "<=",
        ">=",
        "=",
        "less_than",
        "less_equal",
        "greater_than",
        "greater_equal",
        "equal",
        "between",
        "range",
        "not_equal",
        "lt",
        "le",
        "gt",
        "ge",
        "eq",
    }


def operator_is_exists(operator: Any) -> bool:
    op = normalize_text(operator)
    return op in {
        "exists",
        "present",
        "is_present",
        "has",
        "have",
        "any",
        "yes",
    }


# ---------------------------------------------------------------------
# Support signal computation
# ---------------------------------------------------------------------

def extract_value_text(value: Any) -> str:
    """
    Convert value to searchable text.
    """
    if value is None:
        return ""
    if isinstance(value, dict):
        for key in ["value", "text", "raw", "normalized"]:
            if key in value and value[key] not in [None, ""]:
                return normalize_space(value[key])
        return normalize_space(json.dumps(value, ensure_ascii=False))
    if isinstance(value, list):
        return " ".join(normalize_space(v) for v in value if v not in [None, ""])
    return normalize_space(value)


def compute_operator_value_supported(operator: Any, value: Any) -> bool:
    op = normalize_text(operator)
    val = extract_value_text(value)

    if not op:
        return False

    if operator_is_exists(op):
        return True

    if operator_is_quantitative(op):
        return bool(val)

    # Other operators should usually have a value or explicit entity.
    if op in {"in", "not_in", "contains", "excludes", "equal", "not_equal"}:
        return bool(val)

    return True

def get_provenance_field(leaf: Dict[str, Any], key: str) -> Any:
    provenance = get_leaf_value(leaf, "provenance", None)

    if isinstance(provenance, dict):
        return provenance.get(key, "")

    return ""

def compute_leaf_inventory_row(
    doc: Dict[str, Any],
    leaf: Dict[str, Any],
    leaf_path: str,
    pass2_context: Optional[Dict[str, Any]],
    branch_a_row: Optional[Dict[str, str]],
    layer1_policy_row: Optional[Dict[str, str]],
    layer1_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    document_id = get_first(
        doc,
        ["document_id", "doc_id", "nct_id", "trial_id"],
        default="",
    )

    criterion_id = (
        get_leaf_value(leaf, "criterion_id", "")
        or get_first(doc, ["criterion_id", "leaf_id", "clause_id", "pass1_clause_id", "id"], "")
    )

    entity_type = get_leaf_value(leaf, "entity_type", "")
    entity_text = get_leaf_value(leaf, "entity_text", "")
    operator = get_leaf_value(leaf, "operator", "")
    value = get_leaf_value(leaf, "value", "")
    unit = get_leaf_value(leaf, "unit", "")
    value_type = get_leaf_value(leaf, "value_type", "")

    evidence_text = get_leaf_value(leaf, "evidence_text", "")
    item_text = get_leaf_value(leaf, "item_text", "")
    source_text = get_leaf_value(leaf, "source_text", "")

    if pass2_context is not None:
        if not evidence_text:
            evidence_text = get_first(pass2_context, ["evidence_text", "clause_text", "text"], "")
        if not item_text:
            item_text = get_first(pass2_context, ["item_text", "criterion_text", "source_item_text"], "")
        if not source_text:
            source_text = get_first(pass2_context, ["source_text", "full_text", "raw_text"], "")

    # Fallback: evidence is the most important text for verification.
    if not source_text:
        source_text = item_text or evidence_text
    if not item_text:
        item_text = source_text or evidence_text

    value_text = extract_value_text(value)

    # Layer 1 issues from verification_layer1a.leaf_metadata.
    # This is the correct source for your Layer 1A verified files.
    layer1_meta = layer1_meta or {}
    layer1_needs_review, layer1_needs_rescue, issue_codes = (
        extract_layer1_signals_from_metadata(layer1_meta)
    )

    # Fallback for older files where issue codes may be embedded directly.
    if not issue_codes:
        issue_codes = sorted(set(collect_issue_codes(leaf) + collect_issue_codes(doc)))
        if issue_codes:
            layer1_needs_review = True
            layer1_needs_rescue = True

    # Branch-B-specific Layer 1 policy from 04_apply_policy_branch_b.py.
    layer1_policy_row = layer1_policy_row or {}

    layer1_policy_action_hint = layer1_policy_row.get("layer1_policy_action_hint", "")
    layer1_policy_bucket = layer1_policy_row.get("layer1_policy_bucket", "")
    layer1_policy_severity = layer1_policy_row.get("layer1_policy_severity", "")
    layer1_policy_reasons = layer1_policy_row.get("layer1_policy_reasons", "")
    layer1_policy_score = layer1_policy_row.get("layer1_policy_score", "")
    layer1_policy_hard_issue_count = layer1_policy_row.get("layer1_policy_hard_issue_count", "")
    layer1_policy_soft_warning_count = layer1_policy_row.get("layer1_policy_soft_warning_count", "")
    layer1_policy_execution_issue_count = layer1_policy_row.get("layer1_policy_execution_issue_count", "")

    policy_codes = as_string_list(layer1_policy_row.get("all_layer1_codes", ""))

    if layer1_policy_action_hint and layer1_policy_action_hint != "continue_to_layer2":
        issue_codes = sorted(set(policy_codes))
        layer1_needs_review = True
        layer1_needs_rescue = layer1_policy_action_hint == "mandatory_verifier_candidate"
    else:
        issue_codes = []
        layer1_needs_review = False
        layer1_needs_rescue = False

    # Grounding.
    entity_exact_evidence = text_contains(evidence_text, entity_text, case_insensitive=False)
    entity_ci_evidence = text_contains(evidence_text, entity_text, case_insensitive=True)
    entity_norm_evidence = normalized_contains(evidence_text, entity_text)

    entity_exact_item = text_contains(item_text, entity_text, case_insensitive=False)
    entity_ci_item = text_contains(item_text, entity_text, case_insensitive=True)

    value_found_evidence = normalized_contains(evidence_text, value_text)
    value_found_item = normalized_contains(item_text, value_text)

    unit_found_evidence = normalized_contains(evidence_text, unit)

    generic_entity_text = normalize_text(entity_text) in GENERIC_ENTITY_TEXTS

    # Operator / value.
    operator_value_supported = compute_operator_value_supported(operator, value)

    # Quantitative completeness.
    quantitative_cue_present = has_quantitative_cue(evidence_text)
    quantitative_operator = operator_is_quantitative(operator)
    exists_operator = operator_is_exists(operator)

    value_missing = not bool(value_text)
    quantitative_cue_unhandled = bool(
        quantitative_cue_present
        and not quantitative_operator
        and value_missing
    )
    exists_with_quantitative_cue = bool(
        quantitative_cue_present
        and exists_operator
        and not quantitative_operator
    )
    value_missing_with_quantitative_cue = bool(
        quantitative_cue_present
        and value_missing
    )

    # Temporal / history / context support.
    temporal_context = extract_temporal_context(leaf)
    history_context = extract_history_context(leaf)

    temporal_marker = has_temporal_marker(evidence_text)
    temporal_present = context_present(temporal_context)
    temporal_missing_context = bool(temporal_marker and not temporal_present)

    history_marker = has_history_marker(evidence_text)
    history_present = context_present(history_context)
    history_missing_context = bool(history_marker and not history_present)

    condition_exception_marker = has_condition_exception_marker(evidence_text)
    condition_context = (
        get_leaf_value(leaf, "condition_context", "")
        or get_provenance_field(leaf, "source_condition_text")
    )

    exception_context = (
        get_leaf_value(leaf, "exception_context", "")
        or get_provenance_field(leaf, "source_exception_context")
    )
    condition_exception_present = context_present(condition_context) or context_present(exception_context)
    condition_exception_missing = bool(condition_exception_marker and not condition_exception_present)

    branch_a_agreement = compute_branch_a_agreement(leaf, branch_a_row)

    row: Dict[str, Any] = {
        "document_id": document_id,
        "criterion_id": criterion_id,
        "branch": "B_llm_pass2",
        "leaf_path": leaf_path,
        "node_type": get_first(leaf, ["node_type", "type"], ""),
        "eligibility_type": get_first({**doc, **leaf}, ["eligibility_type", "section", "criterion_type"], ""),
        "status": get_leaf_value(leaf, "status", ""),
        "entity_type": entity_type,
        "entity_text": entity_text,
        "operator": operator,
        "value": value,
        "unit": unit,
        "value_type": value_type,
        "evidence_text": evidence_text,
        "item_text": item_text,
        "source_text": source_text,
        "temporal_context_json": temporal_context,
        "history_context_json": history_context,
        "condition_context": condition_context,
        "exception_context": exception_context,
        "computability": get_leaf_value(leaf, "computability", ""),
        "layer1_needs_review": layer1_needs_review,
        "layer1_needs_llm_rescue_candidate": layer1_needs_rescue,
        "layer1_issue_count": len(issue_codes),
        "layer1_issue_codes_json": issue_codes,
        "layer1_policy_action_hint": layer1_policy_action_hint,
        "layer1_policy_bucket": layer1_policy_bucket,
        "layer1_policy_severity": layer1_policy_severity,
        "layer1_policy_reasons": layer1_policy_reasons,
        "layer1_policy_score": layer1_policy_score,
        "layer1_policy_hard_issue_count": layer1_policy_hard_issue_count,
        "layer1_policy_soft_warning_count": layer1_policy_soft_warning_count,
        "layer1_policy_execution_issue_count": layer1_policy_execution_issue_count,
        "missing_pass2_clause_context": pass2_context is None,
        "entity_text_exact_in_evidence": entity_exact_evidence,
        "entity_text_ci_in_evidence": entity_ci_evidence,
        "entity_text_normalized_in_evidence": entity_norm_evidence,
        "entity_text_exact_in_item": entity_exact_item,
        "entity_text_ci_in_item": entity_ci_item,
        "value_text": value_text,
        "value_text_found_in_evidence": value_found_evidence,
        "value_text_found_in_item": value_found_item,
        "unit_text_found_in_evidence": unit_found_evidence,
        "operator_value_structurally_supported": operator_value_supported,
        "quantitative_cue_present": quantitative_cue_present,
        "quantitative_cue_unhandled": quantitative_cue_unhandled,
        "exists_with_quantitative_cue": exists_with_quantitative_cue,
        "value_missing_with_quantitative_cue": value_missing_with_quantitative_cue,
        "temporal_marker_in_evidence": temporal_marker,
        "temporal_context_present": temporal_present,
        "temporal_marker_missing_context": temporal_missing_context,
        "history_marker_in_evidence": history_marker,
        "history_context_present": history_present,
        "history_marker_missing_context": history_missing_context,
        "condition_or_exception_marker_in_evidence": condition_exception_marker,
        "condition_or_exception_context_present": condition_exception_present,
        "condition_or_exception_marker_missing_context": condition_exception_missing,
        "generic_entity_text": generic_entity_text,
    }

    row.update(branch_a_agreement)

    return row


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def rate(rows: List[Dict[str, Any]], col: str) -> float:
    if not rows:
        return 0.0
    return round(sum(1 for r in rows if to_bool_value(r.get(col))) / len(rows), 6)

def conditional_rate(
    rows: List[Dict[str, Any]],
    condition_col: str,
    outcome_col: str,
) -> float:
    eligible = [r for r in rows if normalize_space(r.get(condition_col))]
    if not eligible:
        return 0.0
    return round(
        sum(1 for r in eligible if to_bool_value(r.get(outcome_col))) / len(eligible),
        6,
    )

def make_summary(
    rows: List[Dict[str, Any]],
    ast_path: Path,
    pass2_path: Path,
    branch_a_path: Path,
) -> Dict[str, Any]:
    issue_counter: Counter[str] = Counter()

    for row in rows:
        try:
            codes = json.loads(serialize_cell(row.get("layer1_issue_codes_json", "[]")))
            if isinstance(codes, list):
                issue_counter.update(str(c) for c in codes)
        except json.JSONDecodeError:
            pass

    support_signal_counter = Counter()

    signal_cols = [
        "missing_pass2_clause_context",
        "generic_entity_text",
        "branch_a_match_found",
        "entity_text_normalized_in_evidence",
        "value_text_found_in_evidence",
        "operator_value_structurally_supported",
        "quantitative_cue_unhandled",
        "exists_with_quantitative_cue",
        "value_missing_with_quantitative_cue",
        "temporal_marker_missing_context",
        "history_marker_missing_context",
        "condition_or_exception_marker_missing_context",
    ]

    for row in rows:
        for col in signal_cols:
            if to_bool_value(row.get(col)):
                support_signal_counter[col] += 1

    return {
        "description": "Layer 2 Branch B support-signal inventory. This is Tier 1 only; no final risk score is assigned here.",
        "branch": "B",
        "ast_input": str(ast_path),
        "pass2_input": str(pass2_path),
        "branch_a_score_input": str(branch_a_path) if branch_a_path.exists() else None,
        "outputs": {
            "leaf_level_csv": str(OUT_CSV),
            "summary_json": str(OUT_JSON),
        },
        "n_leaves": len(rows),
        "missing_pass2_clause_context": sum(1 for r in rows if to_bool_value(r.get("missing_pass2_clause_context"))),
        "branch_a_match_rate": rate(rows, "branch_a_match_found"),
        "entity_exact_in_evidence_rate": rate(rows, "entity_text_exact_in_evidence"),
        "entity_ci_in_evidence_rate": rate(rows, "entity_text_ci_in_evidence"),
        "entity_normalized_in_evidence_rate": rate(rows, "entity_text_normalized_in_evidence"),
        "value_found_in_evidence_rate": rate(rows, "value_text_found_in_evidence"),
        "unit_found_in_evidence_rate": rate(rows, "unit_text_found_in_evidence"),
        "value_nonempty_count": sum(1 for r in rows if normalize_space(r.get("value_text"))),
        "value_found_in_evidence_rate_among_nonempty_values": conditional_rate(
            rows,
            "value_text",
            "value_text_found_in_evidence",
        ),
        "unit_nonempty_count": sum(1 for r in rows if normalize_space(r.get("unit"))),
        "unit_found_in_evidence_rate_among_nonempty_units": conditional_rate(
            rows,
            "unit",
            "unit_text_found_in_evidence",
        ),
        "operator_value_supported_rate": rate(rows, "operator_value_structurally_supported"),
        "quantitative_cue_present_rate": rate(rows, "quantitative_cue_present"),
        "quantitative_cue_unhandled_count": sum(1 for r in rows if to_bool_value(r.get("quantitative_cue_unhandled"))),
        "exists_with_quantitative_cue_count": sum(1 for r in rows if to_bool_value(r.get("exists_with_quantitative_cue"))),
        "temporal_marker_missing_context_count": sum(1 for r in rows if to_bool_value(r.get("temporal_marker_missing_context"))),
        "history_marker_missing_context_count": sum(1 for r in rows if to_bool_value(r.get("history_marker_missing_context"))),
        "condition_or_exception_marker_missing_context_count": sum(
            1 for r in rows if to_bool_value(r.get("condition_or_exception_marker_missing_context"))
        ),
        "leaves_with_layer1_issues": sum(1 for r in rows if int(r.get("layer1_issue_count") or 0) > 0),
        "top_layer1_issue_codes": dict(issue_counter.most_common(20)),
        "support_signal_counts": dict(support_signal_counter.most_common()),
        "method_note": (
            "This inventory supports Branch B Layer 2 external verification. "
            "It does not use manual labels and does not use LLM logprobs. "
            "Manual labels should only be used later for validation."
        ),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    ast_path = BRANCH_B_AST_PATH
    pass2_path = detect_pass2_inputs()

    if not ast_path.exists():
        raise FileNotFoundError(f"Branch B logical rule tree not found: {ast_path}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Layer 2 Branch B support-signal inventory")
    print(f"Rule-tree input: {ast_path}")
    print(f"Pass2 input: {pass2_path}")
    print(f"Branch A score input: {BRANCH_A_SCORE_CSV if BRANCH_A_SCORE_CSV.exists() else 'NOT FOUND'}")
    print(f"Output CSV: {OUT_CSV}")
    print(f"Output JSON: {OUT_JSON}")

    ast_docs = read_jsonl(ast_path)
    pass2_rows = read_jsonl(pass2_path)
    pass2_index = build_pass2_context_index(pass2_rows)

    branch_a_rows = read_csv(BRANCH_A_SCORE_CSV)
    branch_a_index = build_branch_a_index(branch_a_rows)

    layer1_policy_rows = read_csv(BRANCH_B_LAYER1_POLICY_CSV)
    layer1_policy_index = build_branch_b_layer1_policy_index(layer1_policy_rows)

    inventory_rows: List[Dict[str, Any]] = []

    for doc in ast_docs:
        leaves = extract_leaves_from_doc(doc)

        for leaf, leaf_path in leaves:
            criterion_id = get_leaf_value(leaf, "criterion_id", "")

            layer1_meta = get_layer1_metadata(
                doc=doc,
                leaf_path=leaf_path,
                criterion_id=criterion_id,
            )

            pass2_context = find_pass2_context(leaf, doc, pass2_index)
            branch_a_row = find_branch_a_row(leaf, doc, branch_a_index)

            layer1_policy_row = find_branch_b_layer1_policy_row(
                leaf=leaf,
                doc=doc,
                policy_index=layer1_policy_index,
            )

            row = compute_leaf_inventory_row(
                doc=doc,
                leaf=leaf,
                leaf_path=leaf_path,
                pass2_context=pass2_context,
                branch_a_row=branch_a_row,
                layer1_policy_row=layer1_policy_row,
                layer1_meta=layer1_meta,
            )
            inventory_rows.append(row)
    # Hard sanity checks for the current CHIA-200 A/B run.
    expected_n = 2402

    if len(inventory_rows) != expected_n:
        raise RuntimeError(
            f"Unexpected number of Branch B leaves: {len(inventory_rows)}. "
            f"Expected {expected_n}. Check that the script is reading the current 200-run rule tree."
        )

    criterion_ids = [str(r.get("criterion_id", "")) for r in inventory_rows if r.get("criterion_id")]

    if len(criterion_ids) != len(set(criterion_ids)):
        duplicates = [
            x for x, c in Counter(criterion_ids).items()
            if c > 1
        ][:10]
        raise RuntimeError(
            f"Duplicate criterion_id values found in Branch B inventory. "
            f"First examples: {duplicates}"
        )

    missing_pass2 = sum(1 for r in inventory_rows if to_bool_value(r.get("missing_pass2_clause_context")))
    if missing_pass2 != 0:
        raise RuntimeError(
            f"{missing_pass2} Branch B leaves are missing Pass 2 clause context. "
            "Do not continue until criterion_id alignment is fixed."
        )

    missing_branch_a = sum(1 for r in inventory_rows if not to_bool_value(r.get("branch_a_match_found")))
    if missing_branch_a != 0:
        raise RuntimeError(
            f"{missing_branch_a} Branch B leaves did not match Branch A scores. "
            "Do not continue until A/B criterion_id alignment is fixed."
        )
    write_csv(OUT_CSV, inventory_rows)

    summary = make_summary(
        rows=inventory_rows,
        ast_path=ast_path,
        pass2_path=pass2_path,
        branch_a_path=BRANCH_A_SCORE_CSV,
    )
    write_json(OUT_JSON, summary)

    print("DONE")
    print(f"Leaves inventoried: {summary['n_leaves']}")
    print(f"Missing Pass2 clause context: {summary['missing_pass2_clause_context']}")
    print(f"Branch A match rate: {summary['branch_a_match_rate']}")
    print(f"Entity exact in evidence rate: {summary['entity_exact_in_evidence_rate']}")
    print(f"Entity normalized in evidence rate: {summary['entity_normalized_in_evidence_rate']}")
    print(f"Value found in evidence rate: {summary['value_found_in_evidence_rate']}")
    print(f"Operator-value supported rate: {summary['operator_value_supported_rate']}")
    print(f"Quantitative cue unhandled count: {summary['quantitative_cue_unhandled_count']}")
    print(f"Leaves with Layer 1 issues: {summary['leaves_with_layer1_issues']}")


if __name__ == "__main__":
    main()

# Run from the repository root:
# python scripts/03_verification/02_layer2/05_inventory_branch_b_support_signals.py