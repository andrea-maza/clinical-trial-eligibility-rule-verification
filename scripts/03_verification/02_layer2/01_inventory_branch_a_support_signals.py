"""
01_inventory_branch_a_support_signals.py

Build the Layer 2 support-signal inventory for Branch A.

This script does not score, repair, or modify leaves. It combines the
Branch A extraction evidence with the deterministic Layer 1 outputs.

Branch A uses:
    Pass 1 LLM decomposition
    PubMedBERT-supported Pass 2 leaf completion
    Deterministic completion rules

The resulting support inventory is used for post-hoc risk verification.
Its signals are not calibrated probabilities of correctness.

Outputs:
    outputs/verification/layer2/branch_a/
        layer2_branch_a_support_inventory_leaf_level.csv
        layer2_branch_a_support_inventory_summary.json

Run from the repository root:
python scripts/03_verification/02_layer2/01_inventory_branch_a_support_signals.py
"""

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------
# Constants kept consistent with Branch A Pass 2 extraction
# ---------------------------------------------------------------------

ANCHOR_LABEL_TO_ENTITY_TYPE = {
    "Condition": "condition",
    "Drug": "drug",
    "Procedure": "procedure",
    "Measurement": "lab",
    "Device": "other",
}

COMPARISON_OPERATORS = {">", ">=", "<", "<=", "=", "!="}
LIST_OPERATORS = {"in", "not_in"}
EXISTENCE_OPERATORS = {"exists", "not_exists"}
RANGE_OPERATORS = {"between"}

TEMPORAL_MARKER_RE = re.compile(
    r"\b("
    r"within|before|after|since|during|prior to|previously|recent|recently|"
    r"screening|baseline|randomization|diagnosis|surgery|treatment start|first dose"
    r")\b"
    r"|"
    r"\b(last|past)\s+\d+\s+(days?|weeks?|months?|years?)\b"
    r"|"
    r"\bwithin\s+(?:the\s+last\s+)?\d+\s+(days?|weeks?|months?|years?)\b",
    flags=re.IGNORECASE,
)

HISTORY_MARKER_RE = re.compile(
    r"\b("
    r"history of|prior|previous|previously|current|currently|stable dose|"
    r"investigational|recurrent|recurrence|previously treated|prior treatment"
    r")\b",
    flags=re.IGNORECASE,
)

CONDITION_MARKER_RE = re.compile(
    r"\b(if|when|unless|except|with the exception of|provided that|in case of)\b",
    flags=re.IGNORECASE,
)

EXCEPTION_MARKER_RE = re.compile(
    r"\b(unless|except|with the exception of)\b",
    flags=re.IGNORECASE,
)

CRITICAL_QUALIFIER_RE = re.compile(
    r"\b(active|uncontrolled|severe|recurrent|symptomatic|measurable|metastatic|advanced)\b",
    flags=re.IGNORECASE,
)

# Layer 1 files are optional. The script should still run if they do not exist yet.
LAYER1_OPTIONAL_FILES = [
    (
        "layer1_inventory",
        Path(
            "outputs/verification/layer1/deterministic_inventory/"
            "deterministic_verification_inventory_leaf_level.csv"
        ),
    ),
    (
        "layer1d_pass1_consistency",
        Path(
            "outputs/verification/layer1/pass1_pass2_consistency/"
            "layer1d_pass1_pass2_consistency_audit.csv"
        ),
    ),
    (
        "layer1_policy_branch_a",
        Path(
            "outputs/verification/layer1/policy_branch_a/"
            "layer1_policy_branch_a_leaf_level.csv"
        ),
    ),
]

QUANTITATIVE_CUE_RE = re.compile(
    r"(>=|<=|>|<|≥|≤|=)"
    r"|"
    r"\b("
    r"greater than|less than|more than|at least|at most|"
    r"no more than|no less than|equal to|or more|or less|"
    r"minimum|maximum|between"
    r")\b"
    r"|"
    r"\b\d+(?:,\d{3})*(?:\.\d+)?\b"
    r"|"
    r"\b(%|uln|x\s*10|mm3|mm\^3|mg/dl|mg|ml|score|grade)\b",
    flags=re.IGNORECASE,
)

# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    # Keep a stable column order. Any extra keys are appended at the end.
    preferred_cols = [
        "dataset",
        "branch",
        "source_ast_file",
        "document_id",
        "criterion_type",
        "item_uid",
        "clause_id",
        "criterion_id",
        "tree_path",
        "entity_type",
        "entity_text",
        "operator",
        "value_type",
        "value_json",
        "unit",
        "computability",
        "non_computable_reason",
        "evidence_text",
        "clause_text",
        "item_text",
        "n_bert_anchors",
        "n_bert_supports",
        "best_anchor_text",
        "best_anchor_label",
        "best_anchor_score",
        "best_anchor_start",
        "best_anchor_end",
        "best_anchor_mapped_entity_type",
        "has_bert_anchor",
        "entity_type_matches_best_anchor",
        "entity_text_exact_in_evidence",
        "entity_text_ci_in_evidence",
        "entity_text_exact_in_item",
        "entity_text_ci_in_item",
        "entity_text_overlaps_best_anchor",
        "best_anchor_token_overlap",
        "any_anchor_token_overlap",
        "entity_text_token_overlaps_best_anchor",
        "entity_text_token_overlaps_any_anchor",
        "entity_text_equals_best_anchor_text",
        "entity_text_contains_best_anchor_text",
        "best_anchor_text_contains_entity_text",
        "support_labels_json",
        "n_qualifier_supports",
        "n_value_supports",
        "n_temporal_supports",
        "n_negation_supports",
        "operator_category",
        "operator_requires_value",
        "scalar_value_present",
        "range_has_min",
        "range_has_max",
        "range_has_any_bound",
        "list_value_present",
        "operator_value_structurally_supported",
        "unit_present",
        "value_text_found_in_evidence",
        "unit_text_found_in_evidence",
        "temporal_marker_in_evidence",
        "temporal_context_present",
        "temporal_relation",
        "temporal_value_present",
        "temporal_unit_present",
        "temporal_anchor_event",
        "temporal_marker_missing_context",
        "history_marker_in_evidence",
        "history_context_present",
        "history_context",
        "history_marker_missing_context",
        "condition_marker_in_evidence",
        "exception_marker_in_evidence",
        "condition_context_present",
        "exception_context_present",
        "condition_marker_missing_context",
        "exception_marker_missing_context",
        "condition_or_exception_missing_context",
        "computable_with_unhandled_condition_or_exception",
        "provenance_present",
        "source_modifier_text",
        "source_condition_text",
        "source_exception_context",
        "history_context_hint",
        "critical_qualifier_in_evidence",
        "generic_entity_text",
        "layer1_issue_count",
        "layer1_issue_codes",
        "layer1_issue_sources",
        "layer1_policy_action_hint",
        "layer1_policy_bucket",
        "layer1_policy_severity",
        "layer1_policy_score",
        "layer1_policy_hard_issue_count",
        "layer1_policy_execution_issue_count",
        "has_layer1_inventory_issue",
        "has_layer1_policy_issue",
        "has_layer1d_issue",
    ]

    keys = []
    seen = set()
    for k in preferred_cols:
        if k in rows[0]:
            keys.append(k)
            seen.add(k)
    for row in rows:
        for k in row.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------

def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def norm_lower(text: Any) -> str:
    return normalize_text(text).lower()


def safe_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None

def normalize_number_string(x: Any) -> str:
    s = normalize_text(x)
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return str(f)
    except Exception:
        return s

def bool_to_int(x: bool) -> int:
    return 1 if x else 0


def json_dumps_compact(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, sort_keys=True)


def text_contains_exact(haystack: str, needle: str) -> bool:
    haystack_n = normalize_text(haystack)
    needle_n = normalize_text(needle)
    if not needle_n:
        return False
    return needle_n in haystack_n


def text_contains_ci(haystack: str, needle: str) -> bool:
    haystack_n = norm_lower(haystack)
    needle_n = norm_lower(needle)
    if not needle_n:
        return False
    return needle_n in haystack_n

def tokenize_for_overlap(text: Any) -> List[str]:
    return re.findall(r"[a-z0-9\+\-]+", norm_lower(text))


def token_overlap_ratio(a: Any, b: Any) -> float:
    ta = set(tokenize_for_overlap(a))
    tb = set(tokenize_for_overlap(b))

    if not ta or not tb:
        return 0.0

    return len(ta & tb) / len(ta | tb)


def find_span_ci(text: str, target: str) -> Optional[Tuple[int, int]]:
    """Case-insensitive approximate span search after whitespace normalization.

    This is used only as a support signal. It does not modify offsets.
    """
    text_n = normalize_text(text)
    target_n = normalize_text(target)
    if not text_n or not target_n:
        return None

    # Exact case-sensitive first.
    idx = text_n.find(target_n)
    if idx >= 0:
        return idx, idx + len(target_n)

    # Case-insensitive fallback.
    idx = text_n.lower().find(target_n.lower())
    if idx >= 0:
        return idx, idx + len(target_n)

    return None


def overlap_len(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def is_generic_entity_text(text: str) -> bool:
    t = norm_lower(text)
    if not t:
        return True
    generic = {
        "condition",
        "disease",
        "procedure",
        "treatment",
        "therapy",
        "drug",
        "medication",
        "test",
        "measurement",
        "laboratory test",
        "criteria",
        "patient",
        "patients",
        "subject",
        "subjects",
        "treated",
        "active",
        "suspected",
        "acute",
        "chronic",
        "history",
        "dose",
        "only",
        "stable",
        "severe",
        "uncontrolled",
        "unresolved",
        "of",
        "the",
        "and",
        "or",
        "with",
        "without",
        "other",
        "any",
    }
    return t in generic


# ---------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------

def iter_criterion_nodes(
    node: Optional[Dict[str, Any]],
    criterion_type: str,
    tree_path: str,
) -> Iterable[Dict[str, Any]]:
    """Yield criterion leaves from a rules_v3 tree."""
    if node is None:
        return

    node_type = node.get("node_type")

    if node_type == "criterion":
        criterion = node.get("criterion") or {}
        yield {
            "criterion_type": criterion_type,
            "tree_path": tree_path,
            "criterion": criterion,
        }
        return

    if node_type == "group":
        group_operator = node.get("group_operator", "GROUP")
        children = node.get("children") or []
        for i, child in enumerate(children):
            child_path = f"{tree_path}/{group_operator}[{i}]"
            yield from iter_criterion_nodes(child, criterion_type, child_path)


def extract_item_uid_and_clause_id(criterion_id: str) -> Tuple[Optional[str], Optional[str]]:
    m = re.match(r"^(?P<item_uid>.+)_(?P<clause_id>C\d+)$", str(criterion_id or ""))
    if not m:
        return None, None
    return m.group("item_uid"), m.group("clause_id")


def load_branch_a_leaves_from_ast(ast_path: Path, source_ast_file: str) -> List[Dict[str, Any]]:
    rows = load_jsonl(ast_path)
    out: List[Dict[str, Any]] = []

    for row in rows:
        if row.get("status") != "ok":
            continue

        document_id = row.get("document_id")
        ast = row.get("rules_v3_ast") or {}

        for section_name, criterion_type in [
            ("inclusion_criteria", "inclusion"),
            ("exclusion_criteria", "exclusion"),
        ]:
            root = ast.get(section_name)
            for leaf in iter_criterion_nodes(root, criterion_type, section_name):
                criterion = leaf["criterion"]
                criterion_id = criterion.get("criterion_id")
                item_uid, clause_id = extract_item_uid_and_clause_id(criterion_id)
                out.append(
                    {
                        "dataset": row.get("dataset", "CHIA"),
                        "branch": "A_bert_rules",
                        "source_ast_file": source_ast_file,
                        "document_id": document_id,
                        "criterion_type": criterion_type,
                        "tree_path": leaf["tree_path"],
                        "item_uid": item_uid,
                        "clause_id": clause_id,
                        "criterion": criterion,
                    }
                )

    return out


# ---------------------------------------------------------------------
# Pass 2 input / BERT candidate helpers
# ---------------------------------------------------------------------

def build_pass2_clause_index(pass2_input_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Index Pass 2 clauses by criterion_id = item_uid + '_' + clause_id."""
    index: Dict[str, Dict[str, Any]] = {}

    for row in pass2_input_rows:
        if row.get("status") != "ok":
            continue

        payload = row.get("pass2_input") or {}
        item_uid = payload.get("item_uid") or row.get("item_uid")
        item_text = payload.get("item_text", "")
        criterion_type = payload.get("criterion_type")
        chia_id = payload.get("chia_id") or row.get("chia_id")
        document_id = payload.get("document_id") or row.get("document_id")

        for clause in payload.get("clauses", []) or []:
            clause_id = clause.get("clause_id")
            if not item_uid or not clause_id:
                continue

            criterion_id = f"{item_uid}_{clause_id}"
            index[criterion_id] = {
                "item_uid": item_uid,
                "clause_id": clause_id,
                "chia_id": chia_id,
                "document_id": document_id,
                "criterion_type": criterion_type,
                "item_text": item_text,
                "clause_text": clause.get("clause_text", ""),
                "pass1_evidence_text": clause.get("evidence_text", ""),
                "is_negated": clause.get("is_negated"),
                "connector_to_next": clause.get("connector_to_next"),
                "quantifier": clause.get("quantifier"),
                "clause_start_char": clause.get("clause_start_char"),
                "clause_end_char": clause.get("clause_end_char"),
                "bert_candidates": clause.get("bert_candidates") or {"anchors": [], "supports": []},
            }

    return index


def choose_best_anchor_from_clause_context(clause_context: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if clause_context is None:
        return None

    anchors = (clause_context.get("bert_candidates") or {}).get("anchors", []) or []
    if not anchors:
        return None

    clause_text = norm_lower(clause_context.get("clause_text", ""))
    evidence_text = norm_lower(
        clause_context.get("pass1_evidence_text")
        or clause_context.get("clause_text", "")
    )

    label_priority = {
        "Measurement": 0,
        "Condition": 1,
        "Drug": 2,
        "Procedure": 3,
        "Device": 4,
    }

    def sort_key(anchor: Dict[str, Any]):
        txt = norm_lower(anchor.get("text", ""))
        appears_in_clause = txt and txt in clause_text
        appears_in_evidence = txt and txt in evidence_text

        score = safe_float(anchor.get("score"))
        if score is None:
            score = -1.0

        try:
            length = int(anchor.get("end", 0)) - int(anchor.get("start", 0))
        except Exception:
            length = 0

        return (
            label_priority.get(anchor.get("label"), 9),
            0 if appears_in_clause else 1,
            0 if appears_in_evidence else 1,
            -score,
            -length,
        )

    return sorted(anchors, key=sort_key)[0]


def anchor_mapped_entity_type(anchor: Optional[Dict[str, Any]]) -> Optional[str]:
    if anchor is None:
        return None
    return ANCHOR_LABEL_TO_ENTITY_TYPE.get(anchor.get("label"))


def support_label_counts(clause_context: Optional[Dict[str, Any]]) -> Counter:
    counter: Counter = Counter()
    if clause_context is None:
        return counter
    supports = (clause_context.get("bert_candidates") or {}).get("supports", []) or []
    for s in supports:
        label = s.get("label") or "UNKNOWN"
        counter[label] += 1
    return counter


# ---------------------------------------------------------------------
# Layer 1 issue helpers
# ---------------------------------------------------------------------

def detect_column(fieldnames: List[str], candidates: List[str]) -> Optional[str]:
    lower_to_original = {c.lower(): c for c in fieldnames}
    for cand in candidates:
        if cand.lower() in lower_to_original:
            return lower_to_original[cand.lower()]
    return None


def row_is_branch_a(row: Dict[str, str]) -> bool:
    branch_value = " ".join(
        str(row.get(c, ""))
        for c in ["branch", "branch_name", "source_branch", "input_branch", "ast_branch"]
        if c in row
    ).lower()

    if not branch_value:
        return True

    if "branch b" in branch_value or branch_value == "b" or "_b" in branch_value:
        return False

    return "a" in branch_value or "branch a" in branch_value or "bert" in branch_value


def read_layer1_issue_index(root: Path) -> Dict[str, Dict[str, Any]]:
    """
    Read Layer 1 outputs and collect real issue/flag codes by criterion_id.

    Sources used:
      - common Layer 1 inventory
      - Layer 1D Pass1/Pass2 consistency audit
      - Branch A Layer 1 policy

    This function does not repair anything.
    It only exposes Layer 1 signals to Layer 2.
    """
    issue_index: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "codes": set(),
            "sources": set(),
            "policy_action_hint": "",
            "policy_bucket": "",
            "policy_severity": "",
            "policy_score": "",
            "policy_hard_issue_count": "",
            "policy_execution_issue_count": "",
        }
    )

    id_cols = [
        "criterion_id",
        "leaf_id",
        "leaf_uid",
        "criterion_uid",
        "id",
    ]

    code_cols = [
        # 00 common deterministic inventory
        "deterministic_issues",
        "layer1c_source_text_warnings",

        # 01 Layer 1D
        "issues",

        # 02a Branch A policy
        "all_layer1_codes",
    ]

    clean_values = {
        "",
        "none",
        "null",
        "nan",
        "false",
        "0",
        "ok",
        "clean",
        "no_issue",
        "no issues",
        "no_flag",
        "valid",
    }

    def split_issue_codes(x: Any) -> List[str]:
        s = normalize_text(x)
        if not s or s.lower() in clean_values:
            return []

        # Support JSON list if it appears.
        if s.startswith("[") and s.endswith("]"):
            try:
                obj = json.loads(s)
                if isinstance(obj, list):
                    return [
                        normalize_text(v)
                        for v in obj
                        if normalize_text(v) and normalize_text(v).lower() not in clean_values
                    ]
            except Exception:
                pass

        # Most current files use semicolon. Some Layer 2 files use pipes.
        parts = re.split(r"[;|]", s)
        return [
            p.strip()
            for p in parts
            if p.strip() and p.strip().lower() not in clean_values
        ]

    for source_name, rel_path in LAYER1_OPTIONAL_FILES:
        path = root / rel_path
        if not path.exists():
            continue

        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                continue

            id_col = detect_column(reader.fieldnames, id_cols)
            if id_col is None:
                continue

            for row in reader:
                if not row_is_branch_a(row):
                    continue

                criterion_id = normalize_text(row.get(id_col, ""))
                if not criterion_id:
                    continue

                entry = issue_index[criterion_id]

                # Collect issue codes from actual issue columns.
                row_codes: List[str] = []

                # For Branch A policy, only treat policy codes as active Layer 1 signals
                # when the policy action is not simply continue_to_layer2.
                # This prevents duplicate-only / audit-only codes from inflating Layer 2 risk.
                if source_name == "layer1_policy_branch_a":
                    action_hint = normalize_text(row.get("layer1_policy_action_hint", ""))

                    if action_hint and action_hint != "continue_to_layer2":
                        row_codes.extend(split_issue_codes(row.get("all_layer1_codes", "")))
                else:
                    for col in code_cols:
                        if col in row:
                            row_codes.extend(split_issue_codes(row.get(col, "")))

                for code in row_codes:
                    entry["codes"].add(code)

                if row_codes:
                    entry["sources"].add(source_name)

                # Store Branch A policy metadata.
                if source_name == "layer1_policy_branch_a":
                    action_hint = normalize_text(row.get("layer1_policy_action_hint", ""))
                    bucket = normalize_text(row.get("layer1_policy_bucket", ""))
                    severity = normalize_text(row.get("layer1_policy_severity", ""))
                    score = normalize_text(row.get("layer1_policy_score", ""))
                    hard_count = normalize_text(row.get("layer1_policy_hard_issue_count", ""))
                    execution_count = normalize_text(row.get("layer1_policy_execution_issue_count", ""))

                    entry["policy_action_hint"] = action_hint
                    entry["policy_bucket"] = bucket
                    entry["policy_severity"] = severity
                    entry["policy_score"] = score
                    entry["policy_hard_issue_count"] = hard_count
                    entry["policy_execution_issue_count"] = execution_count

                    if action_hint and action_hint != "continue_to_layer2":
                        entry["codes"].add(f"policy:{action_hint}")
                        entry["sources"].add(source_name)

    return issue_index


# ---------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------

def operator_category(operator: str) -> str:
    if operator in COMPARISON_OPERATORS:
        return "comparison"
    if operator in RANGE_OPERATORS:
        return "range"
    if operator in LIST_OPERATORS:
        return "list"
    if operator in EXISTENCE_OPERATORS:
        return "existence"
    return "other"


def range_flags(value: Any) -> Tuple[bool, bool, bool]:
    if not isinstance(value, dict):
        return False, False, False
    has_min = value.get("min") is not None
    has_max = value.get("max") is not None
    return has_min, has_max, has_min or has_max


def list_value_present(value: Any) -> bool:
    return isinstance(value, list) and len(value) > 0


def value_text_found_in_evidence(value: Any, evidence_text: str) -> bool:
    if value is None:
        return False

    values_to_check: List[str] = []
    if isinstance(value, dict):
        for k in ["min", "max"]:
            if value.get(k) is not None:
                values_to_check.append(str(value.get(k)))
    elif isinstance(value, list):
        values_to_check.extend(str(v) for v in value if v is not None)
    else:
        values_to_check.append(str(value))

    if not values_to_check:
        return False

    ev = norm_lower(evidence_text)
    return any(
        norm_lower(v) in ev or norm_lower(normalize_number_string(v)) in ev
        for v in values_to_check
        if normalize_text(v)
    )


def operator_value_structurally_supported(operator: str, value_type: str, value: Any) -> bool:
    if operator in COMPARISON_OPERATORS:
        return value_type == "scalar" and value is not None

    if operator == "between":
        _, _, has_any = range_flags(value)
        return value_type == "range" and has_any

    if operator in LIST_OPERATORS:
        return value_type == "list" and list_value_present(value)

    if operator in EXISTENCE_OPERATORS:
        return True

    return False


def compute_leaf_inventory_row(
    leaf: Dict[str, Any],
    pass2_clause_index: Dict[str, Dict[str, Any]],
    layer1_issue_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    criterion = leaf["criterion"]
    criterion_id = criterion.get("criterion_id")
    clause_context = pass2_clause_index.get(criterion_id)

    entity_text = normalize_text(criterion.get("entity_text"))
    entity_type = criterion.get("entity_type")
    operator = criterion.get("operator")
    value_type = criterion.get("value_type")
    value = criterion.get("value")
    unit = criterion.get("unit")
    temporal_context = criterion.get("temporal_context")
    history_context = criterion.get("history_context")
    computability = criterion.get("computability")
    non_computable_reason = criterion.get("non_computable_reason")
    evidence_text = normalize_text(criterion.get("evidence_text"))
    provenance = criterion.get("provenance") or {}

    item_text = normalize_text((clause_context or {}).get("item_text", ""))
    clause_text = normalize_text((clause_context or {}).get("clause_text", ""))
    quantitative_cue = (
        QUANTITATIVE_CUE_RE.search(evidence_text) is not None
        or QUANTITATIVE_CUE_RE.search(clause_text) is not None
    )

    exists_with_quantitative_cue = (
        operator in {"exists", "not_exists"}
        and quantitative_cue
    )

    value_missing_with_quantitative_cue = (
        quantitative_cue
        and value is None
        and operator not in {"exists", "not_exists"}
    )

    quantitative_cue_unhandled = (
        exists_with_quantitative_cue
        or value_missing_with_quantitative_cue
    )

    anchors = ((clause_context or {}).get("bert_candidates") or {}).get("anchors", []) or []
    supports = ((clause_context or {}).get("bert_candidates") or {}).get("supports", []) or []
    best_anchor = choose_best_anchor_from_clause_context(clause_context)
    mapped_type = anchor_mapped_entity_type(best_anchor)
    best_anchor_score = safe_float(best_anchor.get("score")) if best_anchor else None
    best_anchor_text = normalize_text(best_anchor.get("text")) if best_anchor else ""
    best_anchor_token_overlap = token_overlap_ratio(entity_text, best_anchor_text)

    any_anchor_token_overlap = 0.0
    if anchors:
        any_anchor_token_overlap = max(
            token_overlap_ratio(entity_text, a.get("text", ""))
            for a in anchors
        )
    best_anchor_label = best_anchor.get("label") if best_anchor else None
    best_anchor_start = best_anchor.get("start") if best_anchor else None
    best_anchor_end = best_anchor.get("end") if best_anchor else None

    entity_exact_in_evidence = text_contains_exact(evidence_text, entity_text)
    entity_ci_in_evidence = text_contains_ci(evidence_text, entity_text)

    entity_exact_in_item = text_contains_exact(item_text, entity_text)
    entity_ci_in_item = text_contains_ci(item_text, entity_text)

    entity_span = find_span_ci(item_text, entity_text)
    entity_overlaps_best_anchor = False
    if entity_span is not None and best_anchor_start is not None and best_anchor_end is not None:
        try:
            entity_overlaps_best_anchor = overlap_len(
                int(entity_span[0]),
                int(entity_span[1]),
                int(best_anchor_start),
                int(best_anchor_end),
            ) > 0
        except Exception:
            entity_overlaps_best_anchor = False

    # Text-level fallback for overlap support.
    entity_equals_anchor = bool(entity_text and best_anchor_text and norm_lower(entity_text) == norm_lower(best_anchor_text))
    entity_contains_anchor = bool(entity_text and best_anchor_text and norm_lower(best_anchor_text) in norm_lower(entity_text))
    anchor_contains_entity = bool(entity_text and best_anchor_text and norm_lower(entity_text) in norm_lower(best_anchor_text))
    if not entity_overlaps_best_anchor:
        entity_overlaps_best_anchor = entity_equals_anchor or entity_contains_anchor or anchor_contains_entity

    support_counts = support_label_counts(clause_context)

    op_category = operator_category(operator)
    op_requires_value = operator in COMPARISON_OPERATORS or operator in RANGE_OPERATORS or operator in LIST_OPERATORS
    scalar_value_present = value is not None and value_type == "scalar"
    has_min, has_max, has_any_bound = range_flags(value)
    has_list_value = list_value_present(value)
    op_val_supported = operator_value_structurally_supported(operator, value_type, value)
    unit_present = unit is not None and normalize_text(unit) != ""
    value_in_evidence = value_text_found_in_evidence(value, evidence_text)
    unit_in_evidence = text_contains_ci(evidence_text, str(unit)) if unit_present else False

    temporal_marker = TEMPORAL_MARKER_RE.search(evidence_text) is not None or TEMPORAL_MARKER_RE.search(clause_text) is not None
    temporal_present = isinstance(temporal_context, dict)
    temporal_relation = temporal_context.get("relation") if temporal_present else None
    temporal_value_present = temporal_present and temporal_context.get("value") is not None
    temporal_unit_present = temporal_present and temporal_context.get("unit") is not None
    temporal_anchor_event = temporal_context.get("anchor_event") if temporal_present else None
    temporal_marker_missing_context = temporal_marker and not temporal_present

    history_marker = HISTORY_MARKER_RE.search(evidence_text) is not None or HISTORY_MARKER_RE.search(clause_text) is not None
    history_present = history_context is not None and normalize_text(history_context) != ""
    history_marker_missing_context = history_marker and not history_present

    condition_marker = CONDITION_MARKER_RE.search(evidence_text) is not None or CONDITION_MARKER_RE.search(clause_text) is not None
    exception_marker = EXCEPTION_MARKER_RE.search(evidence_text) is not None or EXCEPTION_MARKER_RE.search(clause_text) is not None

    source_modifier_text = provenance.get("source_modifier_text")
    source_condition_text = provenance.get("source_condition_text")
    source_exception_context = provenance.get("source_exception_context")
    history_context_hint = provenance.get("history_context_hint")

    condition_context_present = source_condition_text is not None and normalize_text(source_condition_text) != ""
    exception_context_present = source_exception_context is not None and normalize_text(source_exception_context) != ""

    condition_marker_missing_context = condition_marker and not condition_context_present
    exception_marker_missing_context = exception_marker and not exception_context_present

    condition_or_exception_marker = condition_marker or exception_marker
    condition_or_exception_missing_context = (
        condition_marker_missing_context or exception_marker_missing_context
    )

    computable_with_unhandled_condition_or_exception = (
        computability == "computable"
        and condition_or_exception_missing_context
    )

    critical_qualifier = CRITICAL_QUALIFIER_RE.search(evidence_text) is not None or CRITICAL_QUALIFIER_RE.search(clause_text) is not None

    layer1 = layer1_issue_index.get(criterion_id, {"codes": set(), "sources": set()})
    layer1_codes = sorted(layer1.get("codes", set()))
    layer1_sources = sorted(layer1.get("sources", set()))

    layer1_policy_action_hint = layer1.get("policy_action_hint", "")
    layer1_policy_bucket = layer1.get("policy_bucket", "")
    layer1_policy_severity = layer1.get("policy_severity", "")
    layer1_policy_score = layer1.get("policy_score", "")
    layer1_policy_hard_issue_count = layer1.get("policy_hard_issue_count", "")
    layer1_policy_execution_issue_count = layer1.get("policy_execution_issue_count", "")

    row = {
        "dataset": leaf.get("dataset", "CHIA"),
        "branch": leaf.get("branch", "A"),
        "source_ast_file": leaf.get("source_ast_file"),
        "document_id": leaf.get("document_id"),
        "criterion_type": leaf.get("criterion_type"),
        "item_uid": leaf.get("item_uid"),
        "clause_id": leaf.get("clause_id"),
        "criterion_id": criterion_id,
        "tree_path": leaf.get("tree_path"),
        "entity_type": entity_type,
        "entity_text": entity_text,
        "operator": operator,
        "value_type": value_type,
        "value_json": json_dumps_compact(value),
        "unit": unit,
        "computability": computability,
        "non_computable_reason": non_computable_reason,
        "evidence_text": evidence_text,
        "clause_text": clause_text,
        "item_text": item_text,
        "n_bert_anchors": len(anchors),
        "n_bert_supports": len(supports),
        "best_anchor_text": best_anchor_text,
        "best_anchor_label": best_anchor_label,
        "best_anchor_score": best_anchor_score,
        "best_anchor_start": best_anchor_start,
        "best_anchor_end": best_anchor_end,
        "best_anchor_mapped_entity_type": mapped_type,
        "has_bert_anchor": bool_to_int(best_anchor is not None),
        "entity_type_matches_best_anchor": bool_to_int(mapped_type is not None and mapped_type == entity_type),
        "entity_text_exact_in_evidence": bool_to_int(entity_exact_in_evidence),
        "entity_text_ci_in_evidence": bool_to_int(entity_ci_in_evidence),
        "entity_text_exact_in_item": bool_to_int(entity_exact_in_item),
        "entity_text_ci_in_item": bool_to_int(entity_ci_in_item),
        "entity_text_overlaps_best_anchor": bool_to_int(entity_overlaps_best_anchor),
        "best_anchor_token_overlap": best_anchor_token_overlap,
        "any_anchor_token_overlap": any_anchor_token_overlap,
        "entity_text_token_overlaps_best_anchor": bool_to_int(best_anchor_token_overlap > 0),
        "entity_text_token_overlaps_any_anchor": bool_to_int(any_anchor_token_overlap > 0),
        "entity_text_equals_best_anchor_text": bool_to_int(entity_equals_anchor),
        "entity_text_contains_best_anchor_text": bool_to_int(entity_contains_anchor),
        "best_anchor_text_contains_entity_text": bool_to_int(anchor_contains_entity),
        "support_labels_json": json_dumps_compact(dict(support_counts)),
        "n_qualifier_supports": support_counts.get("Qualifier", 0),
        "n_value_supports": support_counts.get("Value", 0),
        "n_temporal_supports": support_counts.get("Temporal", 0),
        "n_negation_supports": support_counts.get("Negation", 0),
        "operator_category": op_category,
        "operator_requires_value": bool_to_int(op_requires_value),
        "scalar_value_present": bool_to_int(scalar_value_present),
        "range_has_min": bool_to_int(has_min),
        "range_has_max": bool_to_int(has_max),
        "range_has_any_bound": bool_to_int(has_any_bound),
        "list_value_present": bool_to_int(has_list_value),
        "operator_value_structurally_supported": bool_to_int(op_val_supported),
        "unit_present": bool_to_int(unit_present),
        "value_text_found_in_evidence": bool_to_int(value_in_evidence),
        "unit_text_found_in_evidence": bool_to_int(unit_in_evidence),
        "temporal_marker_in_evidence": bool_to_int(temporal_marker),
        "temporal_context_present": bool_to_int(temporal_present),
        "temporal_relation": temporal_relation,
        "temporal_value_present": bool_to_int(temporal_value_present),
        "temporal_unit_present": bool_to_int(temporal_unit_present),
        "temporal_anchor_event": temporal_anchor_event,
        "temporal_marker_missing_context": bool_to_int(temporal_marker_missing_context),
        "history_marker_in_evidence": bool_to_int(history_marker),
        "history_context_present": bool_to_int(history_present),
        "history_context": history_context,
        "history_marker_missing_context": bool_to_int(history_marker_missing_context),
        "condition_marker_in_evidence": bool_to_int(condition_marker),
        "exception_marker_in_evidence": bool_to_int(exception_marker),
        "condition_context_present": bool_to_int(condition_context_present),
        "exception_context_present": bool_to_int(exception_context_present),
        "condition_marker_missing_context": bool_to_int(condition_marker_missing_context),
        "exception_marker_missing_context": bool_to_int(exception_marker_missing_context),
        "condition_or_exception_missing_context": bool_to_int(condition_or_exception_missing_context),
        "computable_with_unhandled_condition_or_exception": bool_to_int(computable_with_unhandled_condition_or_exception),
        "provenance_present": bool_to_int(bool(provenance)),
        "source_modifier_text": source_modifier_text,
        "source_condition_text": source_condition_text,
        "source_exception_context": source_exception_context,
        "history_context_hint": history_context_hint,
        "critical_qualifier_in_evidence": bool_to_int(critical_qualifier),
        "generic_entity_text": bool_to_int(is_generic_entity_text(entity_text)),
        "layer1_issue_count": len(layer1_codes),
        "layer1_issue_codes": "|".join(layer1_codes),
        "layer1_issue_sources": "|".join(layer1_sources),

        "layer1_policy_action_hint": layer1_policy_action_hint,
        "layer1_policy_bucket": layer1_policy_bucket,
        "layer1_policy_severity": layer1_policy_severity,
        "layer1_policy_score": layer1_policy_score,
        "layer1_policy_hard_issue_count": layer1_policy_hard_issue_count,
        "layer1_policy_execution_issue_count": layer1_policy_execution_issue_count,

        "has_layer1_inventory_issue": bool_to_int("layer1_inventory" in layer1_sources),
        "has_layer1_policy_issue": bool_to_int("layer1_policy_branch_a" in layer1_sources),
        "has_layer1d_issue": bool_to_int("layer1d_pass1_consistency" in layer1_sources),
        "quantitative_cue_in_evidence": bool_to_int(quantitative_cue),
        "exists_with_quantitative_cue": bool_to_int(exists_with_quantitative_cue),
        "value_missing_with_quantitative_cue": bool_to_int(value_missing_with_quantitative_cue),
        "quantitative_cue_unhandled": bool_to_int(quantitative_cue_unhandled),
    }

    return row


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def rate(rows: List[Dict[str, Any]], col: str, denom_filter: Optional[str] = None) -> Optional[float]:
    if denom_filter is None:
        denom_rows = rows
    else:
        denom_rows = [r for r in rows if int(r.get(denom_filter, 0) or 0) == 1]

    if not denom_rows:
        return None
    return sum(int(r.get(col, 0) or 0) for r in denom_rows) / len(denom_rows)


def make_summary(rows: List[Dict[str, Any]], source_ast_path: Path, pass2_input_path: Path, layer1_issue_index: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    op_counts = Counter(r.get("operator") for r in rows)
    ent_counts = Counter(r.get("entity_type") for r in rows)
    comp_counts = Counter(r.get("computability") for r in rows)

    layer1_code_counts: Counter = Counter()
    for r in rows:
        for code in str(r.get("layer1_issue_codes", "")).split("|"):
            if code:
                layer1_code_counts[code] += 1

    layer1_source_counts: Counter = Counter()
    for r in rows:
        for source in str(r.get("layer1_issue_sources", "")).split("|"):
            if source:
                layer1_source_counts[source] += 1

    return {
        "description": "Layer 2 Branch A support-signal inventory only. No scoring or repair is applied here.",
        "source_ast_path": str(source_ast_path),
        "pass2_input_path": str(pass2_input_path),
        "total_leaves": total,
        "unique_criterion_ids": len(set(r.get("criterion_id") for r in rows)),
        "duplicate_criterion_id_count": total - len(set(r.get("criterion_id") for r in rows)),
        "has_bert_anchor_rate": rate(rows, "has_bert_anchor"),
        "entity_text_exact_in_evidence_rate": rate(rows, "entity_text_exact_in_evidence"),
        "entity_text_ci_in_evidence_rate": rate(rows, "entity_text_ci_in_evidence"),
        "entity_text_overlaps_best_anchor_rate_among_all": rate(rows, "entity_text_overlaps_best_anchor"),
        "entity_text_overlaps_best_anchor_rate_among_anchor_present": rate(rows, "entity_text_overlaps_best_anchor", denom_filter="has_bert_anchor"),
        "entity_type_matches_best_anchor_rate_among_anchor_present": rate(rows, "entity_type_matches_best_anchor", denom_filter="has_bert_anchor"),
        "entity_text_token_overlaps_best_anchor_rate_among_all": rate(rows, "entity_text_token_overlaps_best_anchor"),
        "entity_text_token_overlaps_any_anchor_rate_among_all": rate(rows, "entity_text_token_overlaps_any_anchor"),
        "operator_value_structurally_supported_rate": rate(rows, "operator_value_structurally_supported"),
        "temporal_marker_rate": rate(rows, "temporal_marker_in_evidence"),
        "temporal_marker_missing_context_rate_among_all": rate(rows, "temporal_marker_missing_context"),
        "history_marker_rate": rate(rows, "history_marker_in_evidence"),
        "history_marker_missing_context_rate_among_all": rate(rows, "history_marker_missing_context"),
        "condition_marker_rate": rate(rows, "condition_marker_in_evidence"),
        "exception_marker_rate": rate(rows, "exception_marker_in_evidence"),
        "condition_marker_missing_context_rate": rate(rows, "condition_marker_missing_context"),
        "exception_marker_missing_context_rate": rate(rows, "exception_marker_missing_context"),
        "condition_or_exception_missing_context_rate": rate(rows, "condition_or_exception_missing_context"),
        "computable_with_unhandled_condition_or_exception_rate": rate(rows, "computable_with_unhandled_condition_or_exception"),
        "generic_entity_rate": rate(rows, "generic_entity_text"),
        "leaves_with_any_layer1_issue": sum(1 for r in rows if int(r.get("layer1_issue_count", 0) or 0) > 0),
        "layer1_issue_index_size": len(layer1_issue_index),
        "operator_counts": dict(op_counts),
        "entity_type_counts": dict(ent_counts),
        "computability_counts": dict(comp_counts),
        "layer1_issue_code_counts": dict(layer1_code_counts),
        "layer1_issue_source_counts": dict(layer1_source_counts),
        "quantitative_cue_rate": rate(rows, "quantitative_cue_in_evidence"),
        "exists_with_quantitative_cue_rate": rate(rows, "exists_with_quantitative_cue"),
        "value_missing_with_quantitative_cue_rate": rate(rows, "value_missing_with_quantitative_cue"),
        "quantitative_cue_unhandled_rate": rate(rows, "quantitative_cue_unhandled"),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    ROOT = Path(__file__).resolve().parents[3]

    ast_path = (
        ROOT
        / "outputs"
        / "extraction"
        / "branch_a"
        / "rules_v3"
        / "chia_text_only_200_rules_v3_ast_A.jsonl"
    )

    pass2_input_path = (
        ROOT
        / "outputs"
        / "extraction"
        / "pass2_inputs"
        / "chia_text_only_200_pass2_inputs.jsonl"
    )

    out_dir = (
        ROOT
        / "outputs"
        / "verification"
        / "layer2"
        / "branch_a"
    )

    out_csv = (
        out_dir
        / "layer2_branch_a_support_inventory_leaf_level.csv"
    )

    out_json = (
        out_dir
        / "layer2_branch_a_support_inventory_summary.json"
    )

    if not ast_path.exists():
        raise FileNotFoundError(f"Branch A AST not found: {ast_path}")
    if not pass2_input_path.exists():
        raise FileNotFoundError(f"Pass 2 input file not found: {pass2_input_path}")

    print("Layer 2 Branch A support-signal inventory")
    print("Rule-tree input:", ast_path)
    print("Pass2 input:", pass2_input_path)
    print("Output CSV:", out_csv)
    print("Output JSON:", out_json)

    leaves = load_branch_a_leaves_from_ast(ast_path, ast_path.name)
    pass2_rows = load_jsonl(pass2_input_path)
    pass2_clause_index = build_pass2_clause_index(pass2_rows)
    leaf_ids = {leaf["criterion"].get("criterion_id") for leaf in leaves}
    pass2_ids = set(pass2_clause_index.keys())

    missing_in_pass2 = sorted(x for x in leaf_ids if x not in pass2_ids)
    extra_in_pass2 = sorted(x for x in pass2_ids if x not in leaf_ids)

    if missing_in_pass2:
        raise RuntimeError(
            f"Rule-tree leaves are not compatible with Pass 2 inputs. "
            f"Missing in Pass2 index: {len(missing_in_pass2)}. "
            f"First examples: {missing_in_pass2[:5]}"
        )

    if extra_in_pass2:
        print(
            "WARNING: Pass 2 index contains extra criterion_ids "
            "not found in the rule tree:",
            len(extra_in_pass2),
        )

    layer1_issue_index = read_layer1_issue_index(ROOT)

    inventory_rows: List[Dict[str, Any]] = []
    missing_pass2_context = 0

    for leaf in leaves:
        criterion_id = leaf["criterion"].get("criterion_id")
        if criterion_id not in pass2_clause_index:
            missing_pass2_context += 1
        inventory_rows.append(
            compute_leaf_inventory_row(
                leaf=leaf,
                pass2_clause_index=pass2_clause_index,
                layer1_issue_index=layer1_issue_index,
            )
        )

    summary = make_summary(
        rows=inventory_rows,
        source_ast_path=ast_path,
        pass2_input_path=pass2_input_path,
        layer1_issue_index=layer1_issue_index,
    )
    summary["missing_pass2_clause_context"] = missing_pass2_context

    write_csv(out_csv, inventory_rows)
    write_json(out_json, summary)

    print("DONE")
    print("Leaves inventoried:", len(inventory_rows))
    print("Missing Pass2 clause context:", missing_pass2_context)
    print("Has BERT anchor rate:", summary.get("has_bert_anchor_rate"))
    print("Entity exact in evidence rate:", summary.get("entity_text_exact_in_evidence_rate"))
    print("Entity normalized in evidence rate:", summary.get("entity_text_ci_in_evidence_rate"))
    print("Operator-value supported rate:", summary.get("operator_value_structurally_supported_rate"))
    print("Leaves with Layer 1 issues:", summary.get("leaves_with_any_layer1_issue"))


if __name__ == "__main__":
    main()

# Run from the repository root:
# python scripts/03_verification/02_layer2/01_inventory_branch_a_support_signals.py