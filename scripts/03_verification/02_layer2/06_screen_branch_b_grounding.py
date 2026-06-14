"""
06_screen_branch_b_grounding.py

Apply the Layer 2 grounding and routing screen for Branch B.

Branch B uses LLM-based Pass 2 leaf extraction. The screen evaluates
two separate forms of support:

1. Semantic grounding support
   Whether the extracted leaf is supported by the source text.

2. Execution support
   Whether the extracted leaf is sufficiently computable and usable
   as a logical rule.

The resulting values are support and routing signals, not calibrated
probabilities of correctness. Branch A comparison columns remain
available as diagnostics but are not used in the Branch B scores.

Inputs:
    outputs/verification/layer2/branch_b/
        layer2_branch_b_support_inventory_leaf_level.csv

Outputs:
    outputs/verification/layer2/branch_b/
        layer2_branch_b_grounding_screen_leaf_level.csv
        layer2_branch_b_grounding_screen_summary.json

Run from the repository root:
python scripts/03_verification/02_layer2/06_screen_branch_b_grounding.py
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]

INVENTORY_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "verification"
    / "layer2"
    / "branch_b"
    / "layer2_branch_b_support_inventory_leaf_level.csv"
)

OUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "verification"
    / "layer2"
    / "branch_b"
)

OUT_CSV = OUT_DIR / "layer2_branch_b_grounding_screen_leaf_level.csv"
OUT_JSON = OUT_DIR / "layer2_branch_b_grounding_screen_summary.json"


CRITICAL_MODIFIER_PATTERNS = [
    "active",
    "current",
    "currently",
    "ongoing",
    "uncontrolled",
    "severe",
    "symptomatic",
    "recurrent",
    "refractory",
    "relapsed",
    "residual",
    "comorbid",
    "planned",
    "planned need",
    "need for",
    "requiring",
    "adequate trial",
    "previous adequate trial",
    "prior adequate trial",
    "evident",
    "measurable",
]


ALLOWANCE_OR_POLARITY_PATTERNS = [
    "not excluded",
    "are not excluded",
    "will not be excluded",
    "not exclusionary",
    "is allowed",
    "are allowed",
    "permitted",
    "eligible despite",
]



# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------

def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Input inventory CSV not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def serialize_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    priority_cols = [
        "document_id",
        "criterion_id",
        "branch",
        "leaf_path",
        "eligibility_type",
        "entity_type",
        "entity_text",
        "operator",
        "value_type",
        "value",
        "unit",
        "computability",
        "evidence_text",
        "item_text",

        # Stage 1 semantic grounding components
        "entity_source_support",
        "value_source_support",
        "unit_source_support",
        "operator_value_support",
        "quantitative_representation_support",
        "temporal_context_support",
        "history_context_support",
        "condition_exception_context_support",
        "layer1_semantic_support",

        # Stage 1 semantic result
        "semantic_grounding_support",
        "semantic_grounding_risk_score",
        "semantic_grounding_risk_label",
        "semantic_grounding_reasons",
        "semantic_bottleneck_components",

        # Execution/computability result
        "computability_support",
        "layer1_execution_support",
        "execution_support",
        "execution_risk_score",
        "execution_risk_label",
        "execution_reasons",
        "execution_bottleneck_components",

        # Final routing
        "final_routing_decision",
        "llm_verifier_candidate",
        "computability_review_candidate",
        "routing_reasons",

        # Layer 1 metadata
        "layer1_issue_count",
        "layer1_needs_review",
        "layer1_needs_llm_rescue_candidate",
        "layer1_issue_codes_json",
        "layer1_semantic_issue_count",
        "layer1_semantic_issue_codes",
        "layer1_execution_issue_count",
        "layer1_execution_issue_codes",

        # Raw inventory signals used
        "entity_text_exact_in_evidence",
        "entity_text_ci_in_evidence",
        "entity_text_normalized_in_evidence",
        "entity_text_exact_in_item",
        "entity_text_ci_in_item",
        "value_text",
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

        # Branch A kept only as diagnostics, not used in scores
        "branch_a_match_found",
        "branch_a_entity_type_agrees",
        "branch_a_operator_agrees",
        "branch_a_value_agrees",
        "branch_a_unit_agrees",
        "branch_a_entity_text_token_overlap",
    ]

    cols: List[str] = []
    seen = set()

    for col in priority_cols:
        if any(col in row for row in rows):
            cols.append(col)
            seen.add(col)

    for row in rows:
        for col in row.keys():
            if col not in seen:
                cols.append(col)
                seen.add(col)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: serialize_cell(row.get(col, "")) for col in cols})


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_numeric_text(text: Any) -> str:
    text = clean_text(text).lower()
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r"\bthe\b", " ", text)
    text = text.replace("×", "x")
    text = text.replace("≤", "<=").replace("≥", ">=")

    # Convert written numbers that often appear in criteria
    replacements = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }

    for word, digit in replacements.items():
        text = re.sub(rf"\b{word}\b", digit, text)

    # Normalize spacing around x, comparators, and units
    text = re.sub(r"(\d)\s*x\s*(uln|lln)", r"\1x \2", text)
    text = re.sub(r"(\d)x\s*(uln|lln)", r"\1x \2", text)
    text = re.sub(r"(\d)\s+(x\s+uln|x\s+lln)", r"\1\2", text)

    text = re.sub(r"[^a-z0-9%<>=./+\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalized_value_in_source(value_text: Any, source_text: Any) -> bool:
    value_norm = normalize_numeric_text(value_text)
    source_norm = normalize_numeric_text(source_text)

    if not value_norm or not source_norm:
        return False

    return value_norm in source_norm

def phrase_in_text(phrase: str, text: Any) -> bool:
    phrase = phrase.lower().strip()
    text = clean_text(text).lower()

    if not phrase or not text:
        return False

    return phrase in text


def compute_critical_modifier_support(row: Dict[str, Any], reasons: List[str]) -> float:
    """
    Detects cases where the entity_text is grounded but too narrow because
    a clinically important modifier from the source text was dropped.

    Example:
      source: Comorbid major depressive disorder diagnosis
      entity: major depressive disorder
      missing modifier: comorbid
    """
    evidence_text = clean_text(row.get("evidence_text", ""))
    entity_text = clean_text(row.get("entity_text", ""))

    source = evidence_text.lower()
    entity = entity_text.lower()

    missing = []

    for modifier in CRITICAL_MODIFIER_PATTERNS:
        if phrase_in_text(modifier, source) and not phrase_in_text(modifier, entity):
            missing.append(modifier)

    if missing:
        add_reason(reasons, "critical_modifier_missing:" + "|".join(sorted(set(missing))))
        return 0.45

    return 1.00


def compute_allowance_polarity_support(row: Dict[str, Any], reasons: List[str]) -> float:
    """
    Detects allowance/exemption clauses that are risky for inclusion/exclusion logic.

    Example:
      source: patients with prior surgically-cured malignancies are not excluded
      extracted: not_exists(prior surgically-cured malignancies)
    """
    evidence_text = clean_text(row.get("evidence_text", ""))
    source = evidence_text.lower()

    for phrase in ALLOWANCE_OR_POLARITY_PATTERNS:
        if phrase_in_text(phrase, source):
            add_reason(reasons, "allowance_or_polarity_clause:" + phrase)
            return 0.35

    return 1.00

# ---------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------

def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_label(value: Any) -> str:
    return clean_text(value).lower()


def to_bool(row: Dict[str, Any], col: str) -> bool:
    value = row.get(col, "")

    if isinstance(value, bool):
        return value

    if value is None:
        return False

    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def to_int(row: Dict[str, Any], col: str, default: int = 0) -> int:
    value = row.get(col, "")

    try:
        if value is None or str(value).strip() == "":
            return default
        return int(float(str(value).strip()))
    except ValueError:
        return default


def parse_json_or_semicolon_list(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    text = str(value).strip()

    if not text:
        return []

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except json.JSONDecodeError:
        pass

    return [x.strip() for x in text.split(";") if x.strip()]


def add_reason(reasons: List[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


B_REVIEW_PRIORITY_REFERENCE_THRESHOLD = 0.60
B_HIGH_SUPPORT_REFERENCE_THRESHOLD = 0.85


def assign_support_bin(support: float) -> str:
    """
    Predefined Branch B reference bins.

    These are not calibrated probabilities and are not validated thresholds yet.
    They are evaluated later against manual_B_leaf_label in 04b.
    """
    if support < B_REVIEW_PRIORITY_REFERENCE_THRESHOLD:
        return "reference_review_priority"
    if support >= B_HIGH_SUPPORT_REFERENCE_THRESHOLD:
        return "reference_high_support_not_auto_accept"
    return "reference_intermediate_support"


def bottlenecks(components: Dict[str, float]) -> List[str]:
    if not components:
        return []

    min_value = min(components.values())

    if min_value >= 0.85:
        return []

    return [
        name
        for name, value in components.items()
        if abs(value - min_value) <= 1e-9
    ]


# ---------------------------------------------------------------------
# Source-overlap helpers
# ---------------------------------------------------------------------

STOPWORDS = {
    "a", "an", "the", "of", "with", "without", "and", "or",
    "to", "in", "on", "for", "by", "at", "from", "that",
    "which", "patients", "patient", "subjects", "subject",
    "participant", "participants", "must", "should", "be",
    "been", "are", "is", "have", "has", "had", "any",
}


def token_set_for_grounding(text: Any) -> set:
    text = clean_text(text).lower()
    text = text.replace("≤", "<=").replace("≥", ">=").replace("×", "x")
    text = re.sub(r"[^a-z0-9%<>=./+\-]+", " ", text)
    tokens = set()

    for tok in text.split():
        tok = tok.strip()
        if len(tok) < 2:
            continue
        if tok in STOPWORDS:
            continue
        tokens.add(tok)

    return tokens


def token_recall_in_source(query_text: Any, source_text: Any) -> float:
    """
    Recall-style token overlap:
      among tokens in query_text, how many are present in source_text?
    """
    q = token_set_for_grounding(query_text)
    s = token_set_for_grounding(source_text)

    if not q:
        return 0.0

    return len(q & s) / len(q)


# ---------------------------------------------------------------------
# Layer 1 issue classification
# ---------------------------------------------------------------------

SEMANTIC_LAYER1_KEYWORDS = {
    # direct extraction or field consistency issues
    "comparison_without_scalar_value",
    "comparison_value_missing",
    "range_with_both_bounds_missing",
    "list_operator_without_list_value",
    "categorical_value_missing",
    "existence_operator_with_non_null_value",
    "existence_operator_on_quantitative_phrase",

    # source/field mismatch issues
    "critical_qualifier_missing_from_entity",
    "comparison_entity_mismatch_from_source",
    "comparison_with_non_quantitative_entity",
    "requirement_object_inversion",

    # temporal/context issues that change the meaning
    "temporal_marker_without_temporal_context",
    "temporal_anchor_mismatch_from_source",
    "duration_marker_missing_from_temporal_context",
    "condition_context_present_without_handling",
    "exception_or_condition_clause_without_context_handling",
    "exception_or_condition_clause_with_computable_status",
    "exception_clause_computable_despite_context",

    # grounding/negation issues
    "evidence_text_not_substring_of_pass1_source_text",
    "negation_clause_with_exists_operator",
    "positive_clause_with_not_exists_operator",
    "negative_entity_with_not_exists_operator",
}


EXECUTION_LAYER1_KEYWORDS = {
    # these affect rule usability/computability but do not always mean the
    # semantic extraction is wrong
    "computable_with_exception_context",
    "computable_with_non_computable_reason",
    "non_computable_without_reason",
    "non_computable_reason",
    "partial",
    "non_computable",
}


def split_layer1_issues(issue_codes: List[str]) -> Tuple[List[str], List[str]]:
    semantic: List[str] = []
    execution: List[str] = []

    for code in issue_codes:
        c = code.strip()
        lc = c.lower()

        if c in SEMANTIC_LAYER1_KEYWORDS:
            semantic.append(c)
            continue

        if any(k in lc for k in EXECUTION_LAYER1_KEYWORDS):
            execution.append(c)
            continue

        # Unknown Layer 1 issue: treat as semantic caution because it came from
        # deterministic verification, but keep the specific code for audit.
        semantic.append(c)

    return sorted(set(semantic)), sorted(set(execution))


# ---------------------------------------------------------------------
# Semantic grounding component scores
# ---------------------------------------------------------------------

def compute_entity_source_support(row: Dict[str, Any], reasons: List[str]) -> float:
    """
    Branch B entity grounding without Branch A.

    Uses:
      - direct evidence_text matches
      - item_text fallback
      - token-overlap fallback against evidence_text/item_text

    No Branch A/BERT signal is used here.
    """
    entity_text = clean_text(row.get("entity_text", ""))
    evidence_text = clean_text(row.get("evidence_text", ""))
    item_text = clean_text(row.get("item_text", ""))

    if to_bool(row, "generic_entity_text"):
        add_reason(reasons, "generic_entity_text")
        return 0.45

    if not entity_text:
        add_reason(reasons, "entity_text_missing")
        return 0.35

    if to_bool(row, "entity_text_exact_in_evidence"):
        return 1.00

    if to_bool(row, "entity_text_ci_in_evidence") or to_bool(row, "entity_text_normalized_in_evidence"):
        return 0.95

    if to_bool(row, "entity_text_exact_in_item") or to_bool(row, "entity_text_ci_in_item"):
        add_reason(reasons, "entity_not_in_evidence_but_in_item")
        return 0.85

    evidence_recall = token_recall_in_source(entity_text, evidence_text)
    item_recall = token_recall_in_source(entity_text, item_text)
    best_recall = max(evidence_recall, item_recall)

    if best_recall >= 0.75:
        add_reason(reasons, "entity_supported_by_source_token_overlap")
        return 0.80

    if best_recall >= 0.50:
        add_reason(reasons, "entity_partially_supported_by_source_token_overlap")
        return 0.65

    if best_recall >= 0.25:
        add_reason(reasons, "entity_weakly_supported_by_source_token_overlap")
        return 0.55

    add_reason(reasons, "entity_not_grounded_in_evidence_or_item")
    return 0.35

def try_parse_json_value(text: Any) -> Any:
    raw = clean_text(text)

    if not raw:
        return ""

    try:
        return json.loads(raw)
    except Exception:
        return raw


def flatten_value_atoms(value_obj: Any) -> List[str]:
    """
    Convert scalar/list/range values into searchable atoms.

    Examples:
      [3, 4] -> ["3", "4"]
      {"min": 0, "max": 2} -> ["0", "2"]
      "2.5 x ULN" -> ["2.5 x ULN"]
    """
    if value_obj is None:
        return []

    if isinstance(value_obj, list):
        atoms = []
        for x in value_obj:
            atoms.extend(flatten_value_atoms(x))
        return atoms

    if isinstance(value_obj, dict):
        atoms = []
        for key in ["min", "max", "lower", "upper", "value"]:
            if key in value_obj:
                atoms.extend(flatten_value_atoms(value_obj[key]))
        return atoms

    text = clean_text(value_obj)
    return [text] if text else []


def structured_value_grounded(value_text: Any, evidence_text: Any, item_text: Any) -> bool:
    value_obj = try_parse_json_value(value_text)
    atoms = flatten_value_atoms(value_obj)

    if not atoms:
        return True

    evidence = normalize_numeric_text(evidence_text)
    item = normalize_numeric_text(item_text)
    source = evidence + " " + item

    if normalized_value_in_source(value_text, evidence_text):
        return True

    if normalized_value_in_source(value_text, item_text):
        return True

    atoms_norm = [normalize_numeric_text(a) for a in atoms if normalize_numeric_text(a)]

    if not atoms_norm:
        return True

    if all(atom in source for atom in atoms_norm):
        return True

    # Range/list source patterns:
    # 0-2, 0 to 2, 3 or 4
    if len(atoms_norm) == 2:
        a, b = atoms_norm[0], atoms_norm[1]

        range_patterns = [
            rf"\b{re.escape(a)}\s*-\s*{re.escape(b)}\b",
            rf"\b{re.escape(a)}\s+to\s+{re.escape(b)}\b",
            rf"\b{re.escape(a)}\s+or\s+{re.escape(b)}\b",
        ]

        if any(re.search(p, source) for p in range_patterns):
            return True

    # non-smoker vs non-smoking
    value_norm = normalize_numeric_text(value_text)
    if value_norm in {"non smoker", "non-smoker"}:
        if re.search(r"\bnon[- ]?smoking\b", source):
            return True

    return False

def compute_value_source_support(row: Dict[str, Any], reasons: List[str]) -> float:
    """
    Value grounding is conditional.
    Null/empty value is not a problem if no value was extracted.

    This version handles scalar, list, and range values.
    """
    value_text = clean_text(row.get("value_text", ""))
    value_type = normalize_label(row.get("value_type", ""))
    evidence_text = clean_text(row.get("evidence_text", ""))
    item_text = clean_text(row.get("item_text", ""))

    if not value_text or value_type in {"", "null", "none", "nan"}:
        return 1.00

    if to_bool(row, "value_text_found_in_evidence"):
        return 1.00

    if normalized_value_in_source(value_text, evidence_text):
        return 1.00

    if structured_value_grounded(value_text, evidence_text, item_text):
        return 1.00

    if to_bool(row, "value_text_found_in_item"):
        add_reason(reasons, "value_not_in_evidence_but_in_item")
        return 0.85

    add_reason(reasons, "value_not_grounded_in_evidence_or_item")
    return 0.45


def compute_unit_source_support(row: Dict[str, Any], reasons: List[str]) -> float:
    unit = clean_text(row.get("unit", ""))
    unit_norm = normalize_label(unit)
    value_text = clean_text(row.get("value_text", ""))
    evidence_text = clean_text(row.get("evidence_text", ""))
    item_text = clean_text(row.get("item_text", ""))

    if not unit or unit_norm in {"null", "none", "nan"}:
        return 1.00

    # "count" is usually a normalized unit, not a literal source unit.
    # It is supported when the source has a numeric/count cue such as one, two,
    # at least one, number of, etc.
    if unit_norm in {"count", "number"}:
        if normalized_value_in_source(value_text, evidence_text) or normalized_value_in_source(value_text, item_text):
            return 1.00

        if re.search(r"\b(one|two|three|four|five|at least|at most|number of)\b", evidence_text, flags=re.IGNORECASE):
            return 0.95

        add_reason(reasons, "count_unit_inferred_but_value_not_clearly_grounded")
        return 0.75

    if to_bool(row, "unit_text_found_in_evidence"):
        return 1.00

    best_recall = max(
        token_recall_in_source(unit, evidence_text),
        token_recall_in_source(unit, item_text),
    )

    if best_recall >= 0.75:
        add_reason(reasons, "unit_supported_by_source_token_overlap")
        return 0.85

    add_reason(reasons, "unit_not_grounded_in_evidence_or_item")
    return 0.65


def compute_operator_value_support(row: Dict[str, Any], reasons: List[str]) -> float:
    if to_bool(row, "operator_value_structurally_supported"):
        return 1.00

    add_reason(reasons, "operator_value_not_structurally_supported")
    return 0.40


def compute_quantitative_representation_support(row: Dict[str, Any], reasons: List[str]) -> float:
    """
    Checks whether quantitative cues in the source are represented by operator/value.
    """
    if to_bool(row, "quantitative_cue_unhandled"):
        add_reason(reasons, "quantitative_cue_not_represented")
        return 0.35

    if to_bool(row, "value_missing_with_quantitative_cue"):
        add_reason(reasons, "value_missing_despite_quantitative_cue")
        return 0.40

    if to_bool(row, "exists_with_quantitative_cue"):
        add_reason(reasons, "exists_operator_with_quantitative_cue")
        return 0.45

    return 1.00


def compute_temporal_context_support(row: Dict[str, Any], reasons: List[str]) -> float:
    if to_bool(row, "temporal_marker_missing_context"):
        add_reason(reasons, "temporal_marker_without_temporal_context")
        return 0.40

    if to_bool(row, "temporal_marker_in_evidence") and to_bool(row, "temporal_context_present"):
        return 0.95

    if (not to_bool(row, "temporal_marker_in_evidence")) and to_bool(row, "temporal_context_present"):
        add_reason(reasons, "temporal_context_without_clear_marker")
        return 0.85

    return 1.00


def compute_history_context_support(row: Dict[str, Any], reasons: List[str]) -> float:
    if to_bool(row, "history_marker_missing_context"):
        add_reason(reasons, "history_marker_without_history_context")
        return 0.50

    if to_bool(row, "history_marker_in_evidence") and to_bool(row, "history_context_present"):
        return 0.95

    if (not to_bool(row, "history_marker_in_evidence")) and to_bool(row, "history_context_present"):
        return 0.90

    return 1.00


def compute_condition_exception_context_support(row: Dict[str, Any], reasons: List[str]) -> float:
    if to_bool(row, "condition_or_exception_marker_missing_context"):
        add_reason(reasons, "condition_or_exception_marker_without_context")
        return 0.40

    if (
        to_bool(row, "condition_or_exception_marker_in_evidence")
        and to_bool(row, "condition_or_exception_context_present")
    ):
        return 0.95

    if (
        (not to_bool(row, "condition_or_exception_marker_in_evidence"))
        and to_bool(row, "condition_or_exception_context_present")
    ):
        return 0.90

    return 1.00


def compute_layer1_semantic_support(
    semantic_layer1_issues: List[str],
    reasons: List[str],
) -> float:
    if not semantic_layer1_issues:
        return 1.00

    if "layer1_policy_hard_issue" in semantic_layer1_issues:
        add_reason(reasons, "layer1_policy:hard_semantic_or_structural_issue")
        return 0.50

    if "layer1_policy_soft_warning" in semantic_layer1_issues:
        add_reason(reasons, "layer1_policy:soft_semantic_warning")
        return 0.80

    # Fallback for old inventories without policy columns.
    for issue in semantic_layer1_issues:
        add_reason(reasons, f"layer1_semantic:{issue}")

    n = len(semantic_layer1_issues)

    if n == 1:
        return 0.65

    if n == 2:
        return 0.50

    return 0.35

def to_float_or_none(value: Any):
    try:
        text = clean_text(value)
        if not text:
            return None
        return float(text)
    except Exception:
        return None




# ---------------------------------------------------------------------
# Execution/computability component scores
# ---------------------------------------------------------------------

def compute_computability_support(row: Dict[str, Any], reasons: List[str]) -> float:
    comp = normalize_label(row.get("computability", ""))

    if comp == "computable":
        return 1.00

    if comp == "partial":
        add_reason(reasons, "computability_partial")
        return 0.60

    if comp == "non_computable":
        add_reason(reasons, "computability_non_computable")
        return 0.35

    add_reason(reasons, "computability_missing_or_unknown")
    return 0.50


def compute_layer1_execution_support(
    row: Dict[str, Any],
    execution_layer1_issues: List[str],
    reasons: List[str],
) -> float:
    n_total = to_int(row, "layer1_issue_count", default=0)
    n_exec = len(execution_layer1_issues)

    if n_total == 0 and n_exec == 0:
        return 1.00

    for issue in execution_layer1_issues:
        add_reason(reasons, f"layer1_execution:{issue}")

    if n_exec == 0:
        # Layer 1 issues exist, but they were semantic rather than execution
        # issues. Do not double-penalize here.
        return 0.90

    if n_exec == 1:
        return 0.70

    if n_exec == 2:
        return 0.55

    return 0.40

HARD_SEMANTIC_REASON_PREFIXES = (
    "allowance_or_polarity_clause",
    "entity_not_grounded_in_evidence_or_item",
    "generic_entity_text",
    "value_not_grounded_in_evidence_or_item",
    "quantitative_cue_not_represented",
    "value_missing_despite_quantitative_cue",
    "exists_operator_with_quantitative_cue",
    "condition_or_exception_marker_without_context",
    "temporal_marker_without_temporal_context",
    "history_marker_without_history_context",
)

SPECIFIC_CRITICAL_MODIFIER_REASONS = (
    "critical_modifier_missing:adequate trial",
    "critical_modifier_missing:previous adequate trial",
    "critical_modifier_missing:prior adequate trial",
    "critical_modifier_missing:planned",
    "critical_modifier_missing:planned need",
    "critical_modifier_missing:need for",
    "critical_modifier_missing:requiring",
    "critical_modifier_missing:comorbid",
    "critical_modifier_missing:evident",
    "critical_modifier_missing:measurable",
)


def has_hard_semantic_reason(reasons: List[str]) -> bool:
    for reason in reasons:
        reason = str(reason).strip()

        if not reason:
            continue

        if reason.startswith(HARD_SEMANTIC_REASON_PREFIXES):
            return True

        if any(reason.startswith(prefix) for prefix in SPECIFIC_CRITICAL_MODIFIER_REASONS):
            return True

    return False


def get_layer1_policy_action(row: Dict[str, Any]) -> str:
    return normalize_label(row.get("layer1_policy_action_hint", ""))


def get_policy_adjusted_layer1_issues(row: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Prefer the Branch-B-specific Layer 1 policy from
    04_apply_policy_branch_b.py.

    If policy columns are absent, fall back to raw Layer 1 issue codes.
    """
    action = get_layer1_policy_action(row)

    if action:
        if action == "mandatory_verifier_candidate":
            return ["layer1_policy_hard_issue"], []

        if action == "continue_to_layer2_grounding_screen":
            return ["layer1_policy_soft_warning"], []

        if action == "computability_review_candidate":
            return [], ["layer1_policy_execution_issue"]

        return [], []

    layer1_issues = parse_json_or_semicolon_list(row.get("layer1_issue_codes_json", ""))
    return split_layer1_issues(layer1_issues)

# ---------------------------------------------------------------------
# Final routing
# ---------------------------------------------------------------------

def decide_final_routing(
    semantic_label: str,
    execution_label: str,
    semantic_layer1_issues: List[str],
    execution_layer1_issues: List[str],
    layer1_policy_action_hint: str,
    semantic_reasons: List[str],
) -> Tuple[str, bool, bool, List[str]]:
    """
    Returns:
      final_routing_decision
      llm_verifier_candidate
      computability_review_candidate
      routing_reasons
    """
    reasons: List[str] = []

    policy_action = normalize_label(layer1_policy_action_hint)
    hard_semantic = has_hard_semantic_reason(semantic_reasons)

    if policy_action == "mandatory_verifier_candidate":
        reasons.append("layer1_policy_mandatory_verifier")
        return "llm_verifier", True, False, reasons

    if hard_semantic:
        reasons.append("hard_semantic_grounding_uncertainty")
        return "llm_verifier", True, False, reasons

    if policy_action == "continue_to_layer2_grounding_screen":
        reasons.append("layer1_policy_soft_warning")
        return "optional_llm_verifier_or_review", False, False, reasons

    if semantic_label in {
        "reference_intermediate_support",
        "reference_review_priority",
    }:
        reasons.append(f"semantic_grounding_{semantic_label}")
        return "optional_llm_verifier_or_review", False, False, reasons

    if policy_action == "computability_review_candidate":
        reasons.append("layer1_policy_computability_review")
        return "computability_review", False, True, reasons

    if execution_label in {
        "reference_intermediate_support",
        "reference_review_priority",
    } or execution_layer1_issues:
        if execution_label in {
            "reference_intermediate_support",
            "reference_review_priority",
        }:
            reasons.append(f"execution_support_{execution_label}")
        if execution_layer1_issues:
            reasons.append("layer1_execution_issue_present")
        return "computability_review", False, True, reasons

    return "keep_without_layer2_action", False, False, ["high_semantic_and_execution_support_reference"]


# ---------------------------------------------------------------------
# Main scoring
# ---------------------------------------------------------------------

def score_row(row: Dict[str, Any]) -> Dict[str, Any]:
    semantic_layer1_issues, execution_layer1_issues = get_policy_adjusted_layer1_issues(row)
    layer1_policy_action_hint = get_layer1_policy_action(row)

    semantic_reasons: List[str] = []

    entity_source_support = compute_entity_source_support(row, semantic_reasons)
    value_source_support = compute_value_source_support(row, semantic_reasons)
    unit_source_support = compute_unit_source_support(row, semantic_reasons)
    operator_value_support = compute_operator_value_support(row, semantic_reasons)
    quantitative_representation_support = compute_quantitative_representation_support(row, semantic_reasons)
    temporal_context_support = compute_temporal_context_support(row, semantic_reasons)
    history_context_support = compute_history_context_support(row, semantic_reasons)
    condition_exception_context_support = compute_condition_exception_context_support(row, semantic_reasons)
    critical_modifier_support = compute_critical_modifier_support(row, semantic_reasons)
    allowance_polarity_support = compute_allowance_polarity_support(row, semantic_reasons)
    layer1_semantic_support = compute_layer1_semantic_support(semantic_layer1_issues, semantic_reasons)

    semantic_components = {
        "entity_source_support": entity_source_support,
        "value_source_support": value_source_support,
        "unit_source_support": unit_source_support,
        "operator_value_support": operator_value_support,
        "quantitative_representation_support": quantitative_representation_support,
        "temporal_context_support": temporal_context_support,
        "history_context_support": history_context_support,
        "condition_exception_context_support": condition_exception_context_support,
        "critical_modifier_support": critical_modifier_support,
        "allowance_polarity_support": allowance_polarity_support,
        "layer1_semantic_support": layer1_semantic_support,
    }

    semantic_grounding_support = min(semantic_components.values())
    semantic_grounding_risk_score = 1.0 - semantic_grounding_support
    semantic_grounding_risk_label = assign_support_bin(semantic_grounding_support)

    execution_reasons: List[str] = []
    computability_support = compute_computability_support(row, execution_reasons)
    layer1_execution_support = compute_layer1_execution_support(row, execution_layer1_issues, execution_reasons)

    execution_components = {
        "computability_support": computability_support,
        "layer1_execution_support": layer1_execution_support,
    }

    execution_support = min(execution_components.values())
    execution_risk_score = 1.0 - execution_support
    execution_risk_label = assign_support_bin(execution_support)

    final_routing_decision, llm_candidate, comp_review_candidate, routing_reasons = decide_final_routing(
        semantic_label=semantic_grounding_risk_label,
        execution_label=execution_risk_label,
        semantic_layer1_issues=semantic_layer1_issues,
        execution_layer1_issues=execution_layer1_issues,
        layer1_policy_action_hint=layer1_policy_action_hint,
        semantic_reasons=semantic_reasons,
    )

    scored = dict(row)

    scored.update({
        "entity_source_support": round(entity_source_support, 6),
        "value_source_support": round(value_source_support, 6),
        "unit_source_support": round(unit_source_support, 6),
        "operator_value_support": round(operator_value_support, 6),
        "quantitative_representation_support": round(quantitative_representation_support, 6),
        "temporal_context_support": round(temporal_context_support, 6),
        "history_context_support": round(history_context_support, 6),
        "condition_exception_context_support": round(condition_exception_context_support, 6),
        "layer1_semantic_support": round(layer1_semantic_support, 6),

        "semantic_grounding_support": round(semantic_grounding_support, 6),
        "semantic_grounding_risk_score": round(semantic_grounding_risk_score, 6),
        "semantic_grounding_risk_label": semantic_grounding_risk_label,
        "semantic_grounding_reasons": ";".join(semantic_reasons),
        "semantic_bottleneck_components": ";".join(bottlenecks(semantic_components)),

        "computability_support": round(computability_support, 6),
        "layer1_execution_support": round(layer1_execution_support, 6),
        "execution_support": round(execution_support, 6),
        "execution_risk_score": round(execution_risk_score, 6),
        "execution_risk_label": execution_risk_label,
        "execution_reasons": ";".join(execution_reasons),
        "execution_bottleneck_components": ";".join(bottlenecks(execution_components)),

        "critical_modifier_support": round(critical_modifier_support, 6),
        "allowance_polarity_support": round(allowance_polarity_support, 6),

        "layer1_semantic_issue_count": len(semantic_layer1_issues),
        "layer1_semantic_issue_codes": ";".join(semantic_layer1_issues),
        "layer1_execution_issue_count": len(execution_layer1_issues),
        "layer1_execution_issue_codes": ";".join(execution_layer1_issues),

        "layer1_policy_action_hint": layer1_policy_action_hint,
        "layer1_policy_adjusted_semantic_issue_count": len(semantic_layer1_issues),
        "layer1_policy_adjusted_execution_issue_count": len(execution_layer1_issues),

        "final_routing_decision": final_routing_decision,
        "llm_verifier_candidate": llm_candidate,
        "computability_review_candidate": comp_review_candidate,
        "routing_reasons": ";".join(routing_reasons),
    })

    return scored


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def count_by(rows: List[Dict[str, Any]], col: str) -> Dict[str, int]:
    c = Counter(str(row.get(col, "")) for row in rows)
    return dict(c.most_common())


def rate(n: int, d: int) -> float:
    if d == 0:
        return 0.0
    return round(n / d, 6)


def summarize_scores(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)

    semantic_supports = [float(r["semantic_grounding_support"]) for r in rows]
    execution_supports = [float(r["execution_support"]) for r in rows]

    semantic_reason_counter: Counter[str] = Counter()
    execution_reason_counter: Counter[str] = Counter()
    routing_reason_counter: Counter[str] = Counter()
    semantic_bottleneck_counter: Counter[str] = Counter()
    execution_bottleneck_counter: Counter[str] = Counter()

    for row in rows:
        for reason in str(row.get("semantic_grounding_reasons", "")).split(";"):
            if reason.strip():
                semantic_reason_counter[reason.strip()] += 1

        for reason in str(row.get("execution_reasons", "")).split(";"):
            if reason.strip():
                execution_reason_counter[reason.strip()] += 1

        for reason in str(row.get("routing_reasons", "")).split(";"):
            if reason.strip():
                routing_reason_counter[reason.strip()] += 1

        for comp in str(row.get("semantic_bottleneck_components", "")).split(";"):
            if comp.strip():
                semantic_bottleneck_counter[comp.strip()] += 1

        for comp in str(row.get("execution_bottleneck_components", "")).split(";"):
            if comp.strip():
                execution_bottleneck_counter[comp.strip()] += 1

    semantic_components = [
        "entity_source_support",
        "value_source_support",
        "unit_source_support",
        "operator_value_support",
        "quantitative_representation_support",
        "temporal_context_support",
        "history_context_support",
        "condition_exception_context_support",
        "layer1_semantic_support",
        "critical_modifier_support",
        "allowance_polarity_support",
    ]

    execution_components = [
        "computability_support",
        "layer1_execution_support",
    ]

    semantic_component_means = {}
    execution_component_means = {}

    for col in semantic_components:
        vals = [float(r[col]) for r in rows]
        semantic_component_means[col] = round(mean(vals), 6) if vals else 0.0

    for col in execution_components:
        vals = [float(r[col]) for r in rows]
        execution_component_means[col] = round(mean(vals), 6) if vals else 0.0

    llm_n = sum(1 for r in rows if str(r.get("llm_verifier_candidate")) in {"1", "True", "true"})
    comp_review_n = sum(1 for r in rows if str(r.get("computability_review_candidate")) in {"1", "True", "true"})

    return {
        "description": (
            "Branch B Layer 2 Stage 1 grounding screen. "
            "The screen separates semantic grounding risk from execution/computability risk. "
            "Branch A/BERT diagnostics are retained in the CSV but are not used in the main scores."
        ),
        "branch": "B",
        "input_inventory_csv": str(INVENTORY_CSV),
        "outputs": {
            "leaf_level_csv": str(OUT_CSV),
            "summary_json": str(OUT_JSON),
        },
        "n_leaves": n,

        "semantic_grounding_risk_label_counts": count_by(rows, "semantic_grounding_risk_label"),
        "execution_risk_label_counts": count_by(rows, "execution_risk_label"),
        "final_routing_decision_counts": count_by(rows, "final_routing_decision"),

        "mean_semantic_grounding_support": round(mean(semantic_supports), 6) if semantic_supports else 0.0,
        "median_semantic_grounding_support": round(median(semantic_supports), 6) if semantic_supports else 0.0,
        "mean_execution_support": round(mean(execution_supports), 6) if execution_supports else 0.0,
        "median_execution_support": round(median(execution_supports), 6) if execution_supports else 0.0,

        "llm_verifier_candidate_count": llm_n,
        "llm_verifier_candidate_rate": rate(llm_n, n),
        "computability_review_candidate_count": comp_review_n,
        "computability_review_candidate_rate": rate(comp_review_n, n),

        "semantic_component_means": semantic_component_means,
        "execution_component_means": execution_component_means,

        "top_semantic_grounding_reasons": dict(semantic_reason_counter.most_common(30)),
        "top_execution_reasons": dict(execution_reason_counter.most_common(30)),
        "top_routing_reasons": dict(routing_reason_counter.most_common(30)),
        "top_semantic_bottleneck_components": dict(semantic_bottleneck_counter.most_common(30)),
        "top_execution_bottleneck_components": dict(execution_bottleneck_counter.most_common(30)),

        "reference_bins": {
            "reference_review_priority_if_support_below": B_REVIEW_PRIORITY_REFERENCE_THRESHOLD,
            "reference_high_support_if_support_at_least": B_HIGH_SUPPORT_REFERENCE_THRESHOLD,
            "reference_intermediate_support_if_between": [
                B_REVIEW_PRIORITY_REFERENCE_THRESHOLD,
                B_HIGH_SUPPORT_REFERENCE_THRESHOLD,
            ],
        },
        "threshold_source": {
            "type": "predefined_reference_bins_not_branch_b_calibrated",
            "not_learned_from_branch_a": True,
            "not_learned_from_manual_B_labels": True,
            "required_validation": "Evaluate against manual_B_leaf_label in using error enrichment and threshold sensitivity.",
        },
        "method_note": (
            "This is not a calibrated probability. It is an auditable support/routing screen. "
            "Semantic grounding risk should be validated against manual_B_leaf_label. "
            "Execution risk should be interpreted separately as computability/usability risk, "
            "not as semantic extraction error."
        ),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print("\nBranch B Layer 2 grounding screen")
    print(f"Input inventory: {INVENTORY_CSV}")
    print(f"Output CSV: {OUT_CSV}")
    print(f"Output JSON: {OUT_JSON}")

    inventory_rows = read_csv(INVENTORY_CSV)

    if not inventory_rows:
        raise RuntimeError(
            "Input inventory is empty. Run "
            "05_inventory_branch_b_support_signals.py first."
        )

    scored_rows = [score_row(row) for row in inventory_rows]

    write_csv(OUT_CSV, scored_rows)

    summary = summarize_scores(scored_rows)
    write_json(OUT_JSON, summary)

    print("DONE")
    print(f"Leaves screened: {summary['n_leaves']}")
    print(f"Semantic grounding risk counts: {summary['semantic_grounding_risk_label_counts']}")
    print(f"Execution risk counts: {summary['execution_risk_label_counts']}")
    print(f"Final routing counts: {summary['final_routing_decision_counts']}")
    print(f"Mean semantic grounding support: {summary['mean_semantic_grounding_support']}")
    print(f"Median semantic grounding support: {summary['median_semantic_grounding_support']}")
    print(f"Mean execution support: {summary['mean_execution_support']}")
    print(f"Median execution support: {summary['median_execution_support']}")
    print(f"LLM verifier candidates: {summary['llm_verifier_candidate_count']}")
    print(f"Computability review candidates: {summary['computability_review_candidate_count']}")

    print("Top semantic grounding reasons:")
    for reason, count in list(summary["top_semantic_grounding_reasons"].items())[:10]:
        print(f"   {reason}: {count}")

    print("Top execution reasons:")
    for reason, count in list(summary["top_execution_reasons"].items())[:10]:
        print(f"   {reason}: {count}")


if __name__ == "__main__":
    main()

# Run from the repository root:
# python scripts/03_verification/02_layer2/06_screen_branch_b_grounding.py