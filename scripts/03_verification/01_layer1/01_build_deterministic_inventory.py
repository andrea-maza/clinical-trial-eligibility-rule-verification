import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from jsonschema import Draft7Validator


# ============================================================
# Purpose
# ============================================================
#
# Layer 1 deterministic verification inventory.
#
# This script DOES NOT:
#   - use BERT anchors
#   - use probabilistic risk scores
#   - call the LLM
#   - modify the rule tree
#
# It identifies:
#   Layer 1A: hard leaf-level field/schema consistency issues
#   Layer 1B: rule tree-level integrity warnings
#   Layer 1C: deterministic source-text consistency warnings
#
#Layer 1A may assign safe/conservative action hints.
#Layer 1B and Layer 1C are flag-only at this stage.
# ============================================================


COMPARISON_OPERATORS = {">", ">=", "<", "<="}
EQUALITY_OPERATORS = {"=", "!="}
LIST_OPERATORS = {"in", "not_in"}
EXISTENCE_OPERATORS = {"exists", "not_exists"}

PATTERN_OPERATORS = {"contains", "matches"}

VALID_TEMPORAL_RELATIONS = {
    "before", "after", "during", "within", "since"
}

VALID_TEMPORAL_UNITS = {
    "hour", "day", "week", "month", "year"
}

VALID_ANCHOR_EVENTS = {
    "screening",
    "randomization",
    "treatment_start",
    "diagnosis",
    "index_date",
    "surgery",
    "procedure",
    "baseline",
    "other",  
}

VALID_ENTITY_TYPES = {
    "condition", "drug", "procedure", "lab", "demographic",
    "therapy", "biomarker", "vital", "observation",
    "stage", "line_of_therapy", "other"
}

VALID_HISTORY_CONTEXT = {
    "current", "prior", "previously_treated",
    "stable_dose", "investigational_use", "other", None
}

VALID_OPERATORS = {
    "exists", "not_exists", "=", "!=", "<", "<=", ">", ">=",
    "between", "in", "not_in", "contains", "matches"
}

VALID_VALUE_TYPES = {"scalar", "list", "range", "null"}

VALID_COMPUTABILITY = {"computable", "partial", "non_computable"}

VALID_CONCEPT_SYSTEMS = {
    "SNOMED-CT", "RxNorm", "LOINC", "ICD-10-CM", "UCUM", None
}

ALL_LAYER1A_CHECKS = [
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
    "computable_with_non_computable_reason",
    "non_computable_without_reason",
    "computable_with_exception_context",
    "unit_present_without_value",
    "entity_type_invalid_enum",
    "history_context_invalid_enum",
    "criterion_id_missing",
    "criterion_id_malformed",
    "operator_invalid_enum",
    "value_type_invalid_enum",
    "computability_invalid_enum",
    "normalized_concept_invalid_system",
    "normalized_concept_system_without_code",
    "range_min_greater_than_max",
    "comparison_value_is_numeric_string",
]

ALL_LAYER1B_CHECKS = [
    "duplicate_criterion_id",
    "duplicate_identical_leaf",
]

ALL_LAYER1C_CHECKS = [
    "comparison_with_non_quantitative_entity",
    "comparison_entity_mismatch_from_source",
    "categorical_value_missing",
    "temporal_marker_without_temporal_context",
    "duration_marker_missing_from_temporal_context",
    "critical_qualifier_missing_from_entity",
    "requirement_object_inversion",
    "condition_context_present_without_handling",
    "temporal_anchor_mismatch_from_source",
    "existence_operator_on_quantitative_phrase",
    "entity_text_too_generic",
]

QUANTITATIVE_ENTITY_TYPES = {
    "lab",
    "vital",
    "observation",
    "demographic",
    "stage",
    "line_of_therapy",
}

CRITICAL_QUALIFIERS = [
    "active",
    "uncontrolled",
    "compromised",
    "refusal",
    "refuse",
    "unresolved",
    "severe",
    "stable",
]

TEMPORAL_TEXT_MARKERS = [
    "at screening",
    "during screening",
    "prior to",
    "before",
    "after",
    "within",
    "since",
    "baseline",
    "randomization",
    "study start",
    "study entry",
    "pre-procedure",
    "post-procedure",
]

ANCHOR_KEYWORD_MAP = {
    "randomization": "randomization",
    "screening": "screening",
    "baseline": "baseline",
    "study start": "screening",
    "study entry": "screening",
    "treatment start": "treatment_start",
    "prior to treatment": "treatment_start",
    "before treatment": "treatment_start",
    "diagnosis": "diagnosis",
    "surgery": "surgery",
    "procedure": "procedure",
    "pre-procedure": "procedure",
    "post-procedure": "procedure",
}

CONDITION_TEXT_MARKERS = [
    "if",
    "unless",
    "except",
    "if applicable",
    "provided that",
    "in case",
    "only if",
]
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


def load_schema(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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

def complete_counts(counter: Counter, all_keys: List[str]) -> Dict[str, int]:
    return {k: counter.get(k, 0) for k in all_keys}

def pct(num: int, den: int, digits: int = 2) -> float:
    if den == 0:
        return 0.0
    return round(100.0 * num / den, digits)

def contains_word_or_phrase(text: str, phrase: str) -> bool:
    text_l = lower_norm(text)
    phrase_l = phrase.lower().strip()

    if " " in phrase_l:
        return phrase_l in text_l

    return bool(re.search(r"\b" + re.escape(phrase_l) + r"\b", text_l))

def has_quantitative_or_normality_phrase(text: str) -> bool:
    markers = [
        "normal",
        "abnormal",
        "elevated",
        "positive",
        "negative",
        "returned to normal",
        "level",
        "levels",
        "mg/dl",
        "ng/ml",
        "mmol/l",
        "x uln",
        "upper limit",
        "grade",
        "%",
    ]

    if contains_any(text, markers):
        return True

    if re.search(r"(<=|>=|<|>|=)\s*\d", lower_norm(text)):
        return True

    if re.search(r"\b\d+(\.\d+)?\s*(mg/dl|ng/ml|mmol/l|x\s*uln|%)\b", lower_norm(text)):
        return True

    return False

# ------------------------------------------------------------
# Text helpers
# ------------------------------------------------------------

def normalize_text(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()

def lower_norm(x: Any) -> str:
    return normalize_text(x).lower()


def token_set(text: Any) -> set:
    stop_words = {
        "a", "an", "the", "of", "with", "without", "and", "or", "to",
        "in", "on", "for", "by", "at", "from", "that", "which", "has",
        "have", "had", "is", "are", "be", "been", "patients", "patient",
        "subjects", "subject", "must",
    }

    text = lower_norm(text)
    text = re.sub(r"[^a-z0-9]+", " ", text)

    return {
        tok
        for tok in text.split()
        if len(tok) >= 2 and tok not in stop_words
    }


def token_overlap_ratio(a: Any, b: Any) -> float:
    a_tokens = token_set(a)
    b_tokens = token_set(b)

    if not a_tokens:
        return 0.0

    return len(a_tokens & b_tokens) / len(a_tokens)


def contains_any(text: str, markers: List[str]) -> bool:
    t = lower_norm(text)

    for marker in markers:
        m = marker.lower().strip()

        if not m:
            continue

        if " " in m:
            if m in t:
                return True
        else:
            if re.search(r"\b" + re.escape(m) + r"\b", t):
                return True

    return False


def has_duration_phrase(text: str) -> bool:
    """
    Detects simple duration expressions:
      - 30 days
      - 6 months
      - one month
      - six months
      - at least six months
    """
    t = lower_norm(text)

    number_words = (
        "one|two|three|four|five|six|seven|eight|nine|ten|"
        "eleven|twelve"
    )

    numeric_pattern = r"\b\d+(\.\d+)?\s*(hour|hours|day|days|week|weeks|month|months|year|years)\b"
    word_pattern = rf"\b({number_words})\s*(hour|hours|day|days|week|weeks|month|months|year|years)\b"

    return bool(re.search(numeric_pattern, t) or re.search(word_pattern, t))

def value_as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False).lower()
    return str(value).lower()


def leaf_captures_marker(entity_text: str, value: Any, marker: str) -> bool:
    marker_l = marker.lower().strip()
    entity_l = lower_norm(entity_text)
    value_l = value_as_text(value)

    return (
        contains_word_or_phrase(entity_l, marker_l)
        or contains_word_or_phrase(value_l, marker_l)
    )


def source_has_temporal_relation(text: str) -> bool:
    markers = [
        "within",
        "prior to",
        "before",
        "after",
        "since",
        "following",
        "pre-procedure",
        "post-procedure",
        "study entry",
        "study start",
        "randomization",
        "screening",
    ]
    return contains_any(text, markers)


def leaf_has_scalar_duration(criterion: Dict[str, Any]) -> bool:
    value = criterion.get("value")
    unit = criterion.get("unit")

    if value is None:
        return False

    if not unit:
        return False

    return str(unit).lower().strip().rstrip("s") in {
        "hour", "day", "week", "month", "year"
    }

def is_numeric_like(x: Any) -> bool:
    """
    True only for numeric temporal values.
    This avoids treating qualitative values like 'recent' or 'morning'
    as missing a unit.
    """
    if isinstance(x, (int, float)):
        return True

    if isinstance(x, str):
        return bool(re.fullmatch(r"\d+(\.\d+)?", x.strip()))

    return False

def extract_comparison_head_from_source(source_text: str) -> str:
    """
    Finds the likely left-hand side of a comparison in source text.
    Examples:
      AST <2.5X ULN -> AST
      total bilirubin <1.5 X ULN -> total bilirubin
      Karnofsky Performance Status > 60 -> Karnofsky Performance Status
    """
    text = normalize_text(source_text)

    pattern = r"([A-Za-z][A-Za-z0-9 /\-]{1,80}?)\s*(<=|>=|<|>|=|!=)\s*[-+]?\d"

    m = re.search(pattern, text)

    if not m:
        return ""

    head = m.group(1)
    head = re.sub(r"^(and|or|with|patients with|subjects with)\s+", "", head, flags=re.I)
    return normalize_text(head)

def safe_json(x: Any) -> str:
    if x is None:
        return ""
    return json.dumps(x, ensure_ascii=False)


def parse_item_and_clause_from_criterion_id(criterion_id: str) -> Tuple[str, str]:
    criterion_id = str(criterion_id or "")

    m = re.match(r"^(.*)_(C\d+)$", criterion_id)

    if not m:
        return "", ""

    return m.group(1), m.group(2)


# ------------------------------------------------------------
# Rule tree traversal
# ------------------------------------------------------------

def walk_ast(node: Any, path: str = "") -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """
    Returns:
        [(path, criterion_node, criterion_dict), ...]
    """
    out = []

    if isinstance(node, dict):
        if node.get("node_type") == "criterion" and isinstance(node.get("criterion"), dict):
            out.append((path, node, node["criterion"]))

        children = node.get("children")

        if isinstance(children, list):
            for i, child in enumerate(children):
                child_path = f"{path}.children[{i}]" if path else f"children[{i}]"
                out.extend(walk_ast(child, child_path))

    return out


def has_exception_or_condition_context(criterion: Dict[str, Any]) -> bool:
    provenance = criterion.get("provenance")

    if not isinstance(provenance, dict):
        return False

    return bool(
        provenance.get("source_exception_context")
        or provenance.get("source_condition_text")
    )

def has_quantitative_surface(text: str) -> bool:
    t = lower_norm(text)

    # Attached numeric-unit patterns: >1cm, 50%, 100ml, 2x ULN
    if re.search(r"\b\d+(\.\d+)?\s*(cm|mm|ml|mg|kg|%)\b", t):
        return True

    if re.search(r"\b\d+(\.\d+)?\s*x\s*(uln|upper limit)", t):
        return True

    markers = [
        "count",
        "number",
        "volume",
        "diameter",
        "percentage",
        "percent",
        "grade",
        "class",
        "score",
        "regimen",
        "regimens",
        "lesion",
        "lesions",
        "stenosis",
        "age",
        "size",
        "cm",
        "mm",
        "ml",
        "fetuses",
        "x uln",
        "upper limit",
    ]

    return contains_any(text, markers)

# ------------------------------------------------------------
# Deterministic issue detection
# ------------------------------------------------------------

def detect_deterministic_issues(criterion: Dict[str, Any]) -> List[str]:
    issues = []

    operator = criterion.get("operator")
    value_type = criterion.get("value_type")
    value = criterion.get("value")
    unit = criterion.get("unit")
    temporal_context = criterion.get("temporal_context")
    computability = criterion.get("computability")
    non_computable_reason = normalize_text(criterion.get("non_computable_reason"))

    entity_text = normalize_text(criterion.get("entity_text"))
    evidence_text = normalize_text(criterion.get("evidence_text"))

    criterion_id = criterion.get("criterion_id")
    entity_type = criterion.get("entity_type")
    history_context = criterion.get("history_context")

    if not criterion_id:
        issues.append("criterion_id_missing")
    elif not re.match(r"^.+_C\d+$", str(criterion_id)):
        issues.append("criterion_id_malformed")

    if entity_type not in VALID_ENTITY_TYPES:
        issues.append("entity_type_invalid_enum")

    if history_context not in VALID_HISTORY_CONTEXT:
        issues.append("history_context_invalid_enum")

    # Required text fields
    if not entity_text:
        issues.append("entity_text_empty")

    if not evidence_text:
        issues.append("evidence_text_empty")

    if operator is None:
        issues.append("operator_missing")

    # Operator-value consistency
    if operator in COMPARISON_OPERATORS:
        if value is None or value_type != "scalar":
            issues.append("comparison_without_scalar_value")

    if operator in EQUALITY_OPERATORS:
        if value is None or value_type == "null":
            issues.append("equality_without_value")

    if operator in PATTERN_OPERATORS:
        if value is None or value_type != "scalar":
            issues.append("pattern_operator_without_scalar_value")

    if operator == "between":
        if value_type != "range" or not isinstance(value, dict):
            issues.append("between_without_range_value")
        else:
            min_value = value.get("min")
            max_value = value.get("max")

            if min_value is None and max_value is None:
                issues.append("range_with_both_bounds_missing")
            elif min_value is None or max_value is None:
                issues.append("range_with_missing_bound")

            if isinstance(min_value, (int, float)) and isinstance(max_value, (int, float)):
                if min_value > max_value:
                    issues.append("range_min_greater_than_max")

    if operator in LIST_OPERATORS:
        if value_type != "list" or not isinstance(value, list) or len(value) == 0:
            issues.append("list_operator_without_list_value")

    if operator in EXISTENCE_OPERATORS:
        if value_type != "null" or value is not None:
            issues.append("existence_operator_with_non_null_value")

    if operator in COMPARISON_OPERATORS and value_type == "scalar" and isinstance(value, str):
        stripped_value = value.strip()

        if re.fullmatch(r"[-+]?\d+(\.\d+)?", stripped_value):
            issues.append("comparison_value_is_numeric_string")


    # value_type consistency
    if value_type == "null" and value is not None:
        issues.append("null_value_type_with_non_null_value")

    if value_type == "scalar" and isinstance(value, (list, dict)):
        issues.append("scalar_value_type_with_complex_value")

    if value_type == "list" and not isinstance(value, list):
        issues.append("list_value_type_without_list_value")

    if value_type == "range" and not isinstance(value, dict):
        issues.append("range_value_type_without_dict_value")

    # Temporal completeness
    if isinstance(temporal_context, dict):
        relation = temporal_context.get("relation")
        t_value = temporal_context.get("value")
        t_unit = temporal_context.get("unit")
        anchor_event = temporal_context.get("anchor_event")

        if not relation:
            issues.append("temporal_context_missing_relation")
        elif relation not in VALID_TEMPORAL_RELATIONS:
            issues.append("temporal_context_invalid_relation")

        if not anchor_event:
            issues.append("temporal_context_missing_anchor_event")
        elif anchor_event not in VALID_ANCHOR_EVENTS:
            issues.append("temporal_context_invalid_anchor_event")

        if t_unit is not None and t_unit not in VALID_TEMPORAL_UNITS:
            issues.append("temporal_context_invalid_unit")


        # If unit is present, value should also be present.
        if t_value is None and t_unit is not None:
            issues.append("temporal_context_missing_value")

        # Only numeric temporal values require a unit.
        # Qualitative values like "recent" or "morning" should not be hard-invalid.
        if t_value is not None and is_numeric_like(t_value) and t_unit is None:
            issues.append("temporal_context_missing_unit")

        # "within" should have some temporal value.
        # But the unit is required only when the value is numeric.
        if relation == "within":
            if t_value is None:
                issues.append("temporal_context_missing_value")
            elif is_numeric_like(t_value) and t_unit is None:
                issues.append("temporal_context_missing_unit")

        # For before/after/since, value/unit are not always required.
        # They are only required if the temporal_context already claims a duration partially.
        # Later, a text-based verifier can check whether the source text contains "4 weeks", "12 hours", etc.

    # Computability consistency
    if computability == "computable" and non_computable_reason:
        issues.append("computable_with_non_computable_reason")

    if computability == "non_computable" and not non_computable_reason:
        issues.append("non_computable_without_reason")

    if computability == "computable" and has_exception_or_condition_context(criterion):
        issues.append("computable_with_exception_context")

    # Unit without a real value
    if unit and value_type == "null" and operator not in EXISTENCE_OPERATORS:
        issues.append("unit_present_without_value")

    normalized_concept = criterion.get("normalized_concept")

    if operator is not None and operator not in VALID_OPERATORS:
        issues.append("operator_invalid_enum")

    if value_type is not None and value_type not in VALID_VALUE_TYPES:
        issues.append("value_type_invalid_enum")

    if computability is not None and computability not in VALID_COMPUTABILITY:
        issues.append("computability_invalid_enum")

    if isinstance(normalized_concept, dict):
        system = normalized_concept.get("system")
        code = normalized_concept.get("code")

        if system not in VALID_CONCEPT_SYSTEMS:
            issues.append("normalized_concept_invalid_system")

        if system is not None and not code:
            issues.append("normalized_concept_system_without_code")

    return sorted(set(issues))

def detect_source_text_consistency_warnings(criterion: Dict[str, Any]) -> List[str]:
    """
    Layer 1C:
    Deterministic source-text consistency warnings.

    These checks are deterministic to detect, but they are NOT safe to repair
    without semantic extraction or LLM rescue.

    They should be used as:
      - flag_only
      - possible later LLM rescue signals
    """
    warnings = []

    entity_type = criterion.get("entity_type")
    entity_text = normalize_text(criterion.get("entity_text"))
    entity_l = lower_norm(entity_text)

    operator = criterion.get("operator")
    value_type = criterion.get("value_type")
    value = criterion.get("value")
    temporal_context = criterion.get("temporal_context")
    evidence_text = normalize_text(criterion.get("evidence_text"))

    provenance = criterion.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}

    source_condition_text = normalize_text(provenance.get("source_condition_text"))
    source_exception_context = normalize_text(provenance.get("source_exception_context"))

    source_text = evidence_text
    source_l = lower_norm(source_text)

    # --------------------------------------------------------
    # 0. Very generic entity text
    # Example: entity_text = "of"
    # This is not a hard schema error because the field is non-empty,
    # but it is a useful deterministic source-text warning.
    # --------------------------------------------------------
    generic_entity_texts = {
        "of", "and", "or", "with", "without", "other", "any",
        "patients", "subjects", "patient", "subject",
        "condition", "disease", "history"
    }

    if entity_l in generic_entity_texts or len(token_set(entity_text)) == 0:
        warnings.append("entity_text_too_generic")
        
    # --------------------------------------------------------
    # 1. Comparisons attached to non-quantitative entities
    # --------------------------------------------------------
    if operator in COMPARISON_OPERATORS:
        if entity_type not in QUANTITATIVE_ENTITY_TYPES:
            if not has_quantitative_surface(entity_text) and not has_quantitative_surface(source_text):
                warnings.append("comparison_with_non_quantitative_entity")

    # --------------------------------------------------------
    # 2. Comparison head mismatch
    # Example: source says AST <2.5, leaf entity says ALT.
    # --------------------------------------------------------
    if operator in COMPARISON_OPERATORS:
        comparison_head = extract_comparison_head_from_source(source_text)

        if comparison_head:
            overlap = token_overlap_ratio(comparison_head, entity_text)

            if overlap < 0.50:
                warnings.append("comparison_entity_mismatch_from_source")

    # --------------------------------------------------------
    # 3. Categorical value missing
    # Example: negative serum pregnancy test encoded as exists/null.
    # --------------------------------------------------------
    categorical_markers = ["negative", "positive", "normal", "abnormal"]

    for marker in categorical_markers:
        if contains_word_or_phrase(source_text, marker):
            captured = leaf_captures_marker(entity_text, value, marker)

            if operator in EXISTENCE_OPERATORS and value_type == "null" and not captured:
                warnings.append("categorical_value_missing")

            break

    # --------------------------------------------------------
    # 4. Temporal marker present but temporal_context missing
    # --------------------------------------------------------
    if contains_any(source_text, TEMPORAL_TEXT_MARKERS):
        if temporal_context is None:
            warnings.append("temporal_marker_without_temporal_context")

    # --------------------------------------------------------
    # 5. Duration phrase present but temporal_context incomplete
    # Example: "30 days prior" but temporal value/unit missing.
    # --------------------------------------------------------
    if has_duration_phrase(source_text) and source_has_temporal_relation(source_text):
        if not isinstance(temporal_context, dict):
            # Do not flag if the duration is already captured as the main scalar value.
            if not leaf_has_scalar_duration(criterion):
                warnings.append("duration_marker_missing_from_temporal_context")
        else:
            if temporal_context.get("value") is None or temporal_context.get("unit") is None:
                # Again, avoid false positives where the duration is the main value.
                if not leaf_has_scalar_duration(criterion):
                    warnings.append("duration_marker_missing_from_temporal_context")

    # --------------------------------------------------------
    # 5b. Temporal anchor keyword mismatch
    # Example: source says "study entry" but anchor_event is not compatible.
    # --------------------------------------------------------
    if isinstance(temporal_context, dict):
        extracted_anchor = temporal_context.get("anchor_event")

        for phrase, expected_anchor in ANCHOR_KEYWORD_MAP.items():
            if contains_word_or_phrase(source_text, phrase):
                if extracted_anchor and extracted_anchor != expected_anchor:
                    warnings.append("temporal_anchor_mismatch_from_source")
                break

    # --------------------------------------------------------
    # 6. Critical qualifier present in source but dropped from entity_text
    # Example:
    #   source: compromised ability to consent
    #   entity: ability to consent
    # --------------------------------------------------------
    for qualifier in CRITICAL_QUALIFIERS:
        if contains_word_or_phrase(source_l, qualifier) and not contains_word_or_phrase(entity_l, qualifier):
            warnings.append("critical_qualifier_missing_from_entity")
            break

    # --------------------------------------------------------
    # 7. Requirement-object inversion
    # Example:
    #   source: infection requiring antibiotics
    #   entity: antibiotics
    # Main condition is infection requiring antibiotics, not only antibiotics.
    # --------------------------------------------------------
    requiring_pattern = r"\b([a-z][a-z\s\-]+?)\s+requiring\s+([a-z][a-z\s\-]+)\b"
    m = re.search(requiring_pattern, source_l)

    if m:
        head = normalize_text(m.group(1))
        required_object = normalize_text(m.group(2))

        if required_object and required_object in entity_l and head not in entity_l:
            warnings.append("requirement_object_inversion")

    # --------------------------------------------------------
    # 8. Condition/exception context present but not represented
    # --------------------------------------------------------
    if contains_any(source_text, CONDITION_TEXT_MARKERS):
        if not source_condition_text and not source_exception_context:
            warnings.append("condition_context_present_without_handling")

    # --------------------------------------------------------
    # 9. Existence operator on quantitative/normality phrase
    # Example: CK has not returned to normal encoded as exists/not_exists.
    # --------------------------------------------------------
    if operator in EXISTENCE_OPERATORS and value_type == "null":
        if entity_type in {"lab", "vital", "observation"}:
            if has_quantitative_or_normality_phrase(source_text):
                warnings.append("existence_operator_on_quantitative_phrase")

    return sorted(set(warnings))

# ------------------------------------------------------------
# Repair classification
# ------------------------------------------------------------

def classify_layer1a_action_hint(
    issues: List[str],
    criterion: Dict[str, Any],
) -> Tuple[str, str]:
    """
    Returns:
        layer1a_action_category, layer1a_action_hint

    Categories:
        no_action
        safe_normalization_candidate
        conservative_rewrite_candidate
        flag_only

    Important:
        Layer 1 does not modify the AST.
        These are only action hints for Layer 3.
    """
    issue_set = set(issues)

    if not issues:
        return "no_action", ""

    if "computable_with_exception_context" in issue_set:
        return (
            "safe_normalization_candidate",
            "Set computability='partial' and non_computable_reason='exception_context_unresolved'."
        )

    if "computable_with_non_computable_reason" in issue_set:
        return (
            "safe_normalization_candidate",
            "Set computability='partial' because a non_computable_reason is already present."
        )

    if "non_computable_without_reason" in issue_set:
        return (
            "safe_normalization_candidate",
            "Keep computability='non_computable' and add non_computable_reason='reason_missing_after_extraction'."
        )

    if "range_with_missing_bound" in issue_set:
        return (
            "flag_only",
            "One range bound is present but the other is missing; cannot safely infer the missing bound."
        )

    if "equality_without_value" in issue_set:
        return (
            "flag_only",
            "Equality operator requires a value; cannot safely infer the target value without clinical knowledge."
        )

    if "pattern_operator_without_scalar_value" in issue_set:
        return (
            "flag_only",
            "Pattern operator requires a scalar string value; cannot safely infer the missing pattern."
        )

    if "temporal_context_invalid_anchor_event" in issue_set:
        return (
            "flag_only",
            "Temporal anchor_event is outside the allowed schema values; cannot safely remap without semantic interpretation."
        )

    if "temporal_context_invalid_relation" in issue_set:
        return (
            "flag_only",
            "Temporal relation is outside the allowed schema values; cannot safely remap without semantic interpretation."
        )

    if "temporal_context_invalid_unit" in issue_set:
        return (
            "flag_only",
            "Temporal unit is outside the allowed schema values; cannot safely remap without semantic interpretation."
        )
    
    if "operator_missing" in issue_set:
        return (
            "flag_only",
            "Operator is missing; cannot safely infer the operator deterministically."
        )
    
    if "entity_type_invalid_enum" in issue_set:
        return (
            "flag_only",
            "entity_type is outside the allowed schema enum; cannot safely remap deterministically."
        )

    if "history_context_invalid_enum" in issue_set:
        return (
            "flag_only",
            "history_context is outside the allowed schema enum; cannot safely remap deterministically."
        )

    if "criterion_id_missing" in issue_set:
        return (
            "flag_only",
            "criterion_id is missing; cannot safely reconstruct it without checking the source clause mapping."
        )

    if "criterion_id_malformed" in issue_set:
        return (
            "flag_only",
            "criterion_id does not follow the expected *_C[number] format."
        )
    
    if "null_value_type_with_non_null_value" in issue_set:
        return (
            "flag_only",
            "value_type is null but value is non-null. Do not clear value deterministically because it may contain clinical information."
        )
    
    if "operator_invalid_enum" in issue_set:
        return (
            "flag_only",
            "operator is outside the allowed schema enum; cannot safely remap deterministically."
        )

    if "value_type_invalid_enum" in issue_set:
        return (
            "flag_only",
            "value_type is outside the allowed schema enum; cannot safely remap deterministically."
        )

    if "computability_invalid_enum" in issue_set:
        return (
            "flag_only",
            "computability is outside the allowed schema enum; cannot safely remap deterministically."
        )

    if "normalized_concept_invalid_system" in issue_set:
        return (
            "flag_only",
            "normalized_concept.system is outside the allowed terminology enum."
        )

    if "normalized_concept_system_without_code" in issue_set:
        return (
            "flag_only",
            "normalized_concept.system is present but code is missing."
        )

    if "range_min_greater_than_max" in issue_set:
        return (
            "flag_only",
            "Range minimum is greater than range maximum; cannot safely correct bounds deterministically."
        )

    if "comparison_value_is_numeric_string" in issue_set:
        return (
            "safe_normalization_candidate",
            "Layer 3 may convert numeric string comparison value to numeric type."
        )

    # If an existence operator contains a non-null value, do not classify the row
    # as safe even if another temporal field could be normalized.
    # The value may contain clinically important threshold/timing information.
    if "existence_operator_with_non_null_value" in issue_set:
        return (
            "flag_only",
            "Existence operator has a non-null value. This may contain useful threshold information, so do not clear or relocate it deterministically."
        )
   
    # Temporal repair if value/unit are already present in main leaf fields
    temporal_missing = (
        "temporal_context_missing_value" in issue_set
        or "temporal_context_missing_unit" in issue_set
    )

    if temporal_missing:
        value = criterion.get("value")
        unit = criterion.get("unit")

        if value is not None and unit:
            return (
                "safe_normalization_candidate",
                "Fill temporal_context.value and temporal_context.unit from the leaf value/unit already extracted."
            )

        return (
            "flag_only",
            "Temporal context is incomplete and value/unit cannot be safely recovered deterministically."
        )

    # Conservative repairs: make structure valid but mark as partial.
    # These do not invent information, but they reduce specificity.
    if "list_operator_without_list_value" in issue_set:
        return (
            "conservative_rewrite_candidate",
            "Change operator to 'exists', set value_type='null', value=null, computability='partial', reason='list_value_missing'."
        )

    if "comparison_without_scalar_value" in issue_set:
        return (
            "conservative_rewrite_candidate",
            "Layer 3 may rewrite to a less specific exists/null leaf, but this should remain flagged because clinical specificity is lost."
        )

    if "range_with_both_bounds_missing" in issue_set:
        return (
            "conservative_rewrite_candidate",
            "Change operator to 'exists', set value_type='null', value=null, computability='partial', reason='range_bounds_missing'."
        )

    if "existence_operator_with_non_null_value" in issue_set:
        return (
            "flag_only",
            "Existence operator has a non-null value. This may contain useful threshold information, so do not clear it deterministically."
        )

    # These need review or later LLM rescue, not deterministic repair.
    return (
        "flag_only",
        "Cannot safely repair deterministically without adding or removing clinical meaning."
    )

def criterion_signature(criterion: Dict[str, Any]) -> Tuple[Any, ...]:
    """
    Conservative exact signature for duplicate-leaf detection.
    This does not try to decide semantic equivalence.
    """
    value = criterion.get("value")

    try:
        value_key = json.dumps(value, sort_keys=True, ensure_ascii=False)
    except Exception:
        value_key = str(value)

    temporal = criterion.get("temporal_context")

    try:
        temporal_key = json.dumps(temporal, sort_keys=True, ensure_ascii=False)
    except Exception:
        temporal_key = str(temporal)

    return (
        criterion.get("entity_type"),
        normalize_text(criterion.get("entity_text")).lower(),
        criterion.get("operator"),
        criterion.get("value_type"),
        value_key,
        criterion.get("unit"),
        temporal_key,
        criterion.get("history_context"),
    )


def detect_ast_level_deterministic_issues(
    document_id: str,
    criterion_nodes: List[Tuple[str, Dict[str, Any], Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    Deterministic document/AST-level checks.

    These are not semantic contradiction checks yet.
    They only check structural integrity:
      - duplicate criterion_id
      - duplicate identical leaf
    """
    issues = []

    id_to_paths = {}
    signature_to_paths = {}

    for path, node, criterion in criterion_nodes:
        criterion_id = criterion.get("criterion_id")

        if criterion_id:
            id_to_paths.setdefault(criterion_id, []).append(path)

        criterion_id = criterion.get("criterion_id", "")
        item_uid, clause_id = parse_item_and_clause_from_criterion_id(criterion_id)

        sig = criterion_signature(criterion)
        signature_key = (item_uid, sig)

        signature_to_paths.setdefault(signature_key, []).append(path)

    for criterion_id, paths in id_to_paths.items():
        if len(paths) > 1:
            issues.append({
                "issue_type": "duplicate_criterion_id",
                "document_id": document_id,
                "criterion_id": criterion_id,
                "paths": paths,
                "repair_category": "flag_only",
                "detail": "Same criterion_id appears more than once in the document AST."
            })

    for sig, paths in signature_to_paths.items():
        if len(paths) > 1:
            issues.append({
                "issue_type": "duplicate_identical_leaf",
                "document_id": document_id,
                "criterion_id": "",
                "paths": paths,
                "repair_category": "flag_only",
                "detail": "Two or more leaves have identical structured content."
            })

    return issues

# ------------------------------------------------------------
# Branch scan
# ------------------------------------------------------------

def scan_branch(
    branch_name: str,
    ast_path: Path,
    schema_validator: Draft7Validator,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    rows = load_jsonl(ast_path)

    leaf_rows = []
    ast_rows = []

    document_counter = 0
    schema_error_docs = 0
    total_leaves = 0

    leaf_issue_counter = Counter()
    ast_issue_counter = Counter()

    leaf_action_counter = Counter()
    ast_action_counter = Counter()

    source_warning_counter = Counter()

    source_action_counter = Counter()

    n_layer1a_hard_issue_leaves = 0
    n_layer1c_warning_leaves = 0
    n_any_layer1a_or_layer1c_flagged_leaves = 0
    n_both_layer1a_and_layer1c_flagged_leaves = 0

    for row in rows:
        document_counter += 1

        document_id = row.get("document_id")
        ast = row.get("rules_v3_ast")

        if row.get("status") != "ok" or not isinstance(ast, dict):
            continue

        schema_errors = sorted(
            schema_validator.iter_errors(ast),
            key=lambda e: list(e.absolute_path),
        )

        if schema_errors:
            schema_error_docs += 1

        criterion_nodes = []
        criterion_nodes.extend(
            walk_ast(ast.get("inclusion_criteria"), path="inclusion_criteria")
        )
        criterion_nodes.extend(
            walk_ast(ast.get("exclusion_criteria"), path="exclusion_criteria")
        )

        # ====================================================
        # Layer 1B: Rule tree - level integrity checks
        # ====================================================
        ast_level_issues = detect_ast_level_deterministic_issues(
            document_id=document_id,
            criterion_nodes=criterion_nodes,
        )

        for ast_issue in ast_level_issues:
            ast_issue_counter.update([ast_issue["issue_type"]])
            ast_action_counter.update([ast_issue["repair_category"]])

            ast_rows.append({
                "branch": branch_name,
                "document_id": document_id,
                "issue_type": ast_issue["issue_type"],
                "criterion_id": ast_issue.get("criterion_id", ""),
                "paths": ";".join(ast_issue.get("paths", [])),
                "layer1b_action_category": ast_issue["repair_category"],
                "detail": ast_issue["detail"],
            })

        # ====================================================
        # Layer 1A: leaf-level deterministic checks
        # ====================================================
        for path, node, criterion in criterion_nodes:
            total_leaves += 1

            criterion_id = criterion.get("criterion_id", "")
            item_uid, clause_id = parse_item_and_clause_from_criterion_id(criterion_id)

            issues = detect_deterministic_issues(criterion)
            layer1a_action_category, layer1a_action_hint = classify_layer1a_action_hint(
                issues,
                criterion,
            )

            source_warnings = detect_source_text_consistency_warnings(criterion)
            source_warning_counter.update(source_warnings)

            source_action = "flag_only" if source_warnings else "no_action"
            source_action_counter.update([source_action])

            leaf_issue_counter.update(issues)
            leaf_action_counter.update([layer1a_action_category])

            has_layer1a_issue = bool(issues)
            has_layer1c_warning = bool(source_warnings)

            if has_layer1a_issue:
                n_layer1a_hard_issue_leaves += 1

            if has_layer1c_warning:
                n_layer1c_warning_leaves += 1

            if has_layer1a_issue or has_layer1c_warning:
                n_any_layer1a_or_layer1c_flagged_leaves += 1

            if has_layer1a_issue and has_layer1c_warning:
                n_both_layer1a_and_layer1c_flagged_leaves += 1

            leaf_rows.append({
                "branch": branch_name,
                "document_id": document_id,
                "path": path,
                "criterion_id": criterion_id,
                "item_uid": item_uid,
                "clause_id": clause_id,

                "entity_type": criterion.get("entity_type"),
                "entity_text": normalize_text(criterion.get("entity_text")),
                "operator": criterion.get("operator"),
                "value_type": criterion.get("value_type"),
                "value": safe_json(criterion.get("value")),
                "unit": criterion.get("unit"),
                "temporal_context": safe_json(criterion.get("temporal_context")),
                "history_context": criterion.get("history_context"),
                "computability": criterion.get("computability"),
                "non_computable_reason": normalize_text(criterion.get("non_computable_reason")),
                "evidence_text": normalize_text(criterion.get("evidence_text")),

                "deterministic_issues": ";".join(issues),
                "layer1a_action_category": layer1a_action_category,
                "layer1a_action_hint": layer1a_action_hint,

                "layer1c_source_text_warnings": ";".join(source_warnings),
                "layer1c_action": source_action,
            })

    combined_action_counter = leaf_action_counter + ast_action_counter

    summary = {
        "branch": branch_name,
        "ast_path": str(ast_path),
        "n_documents": document_counter,
        "n_documents_with_schema_errors": schema_error_docs,
        "n_leaves": total_leaves,

        "leaf_flag_summary": {
            "n_total_leaves": total_leaves,

            "n_layer1a_hard_issue_leaves": n_layer1a_hard_issue_leaves,
            "pct_layer1a_hard_issue_leaves": pct(
                n_layer1a_hard_issue_leaves,
                total_leaves,
            ),

            "n_layer1c_warning_leaves": n_layer1c_warning_leaves,
            "pct_layer1c_warning_leaves": pct(
                n_layer1c_warning_leaves,
                total_leaves,
            ),

            "n_any_layer1a_or_layer1c_flagged_leaves": n_any_layer1a_or_layer1c_flagged_leaves,
            "pct_any_layer1a_or_layer1c_flagged_leaves": pct(
                n_any_layer1a_or_layer1c_flagged_leaves,
                total_leaves,
            ),

            "n_both_layer1a_and_layer1c_flagged_leaves": n_both_layer1a_and_layer1c_flagged_leaves,
            "pct_both_layer1a_and_layer1c_flagged_leaves": pct(
                n_both_layer1a_and_layer1c_flagged_leaves,
                total_leaves,
            ),

            "n_layer1b_ast_warning_rows": sum(ast_issue_counter.values()),
        },

        "layer1a_hard_leaf_issue_counts": complete_counts(
            leaf_issue_counter,
            ALL_LAYER1A_CHECKS,
        ),
        "layer1b_ast_integrity_issue_counts": complete_counts(
            ast_issue_counter,
            ALL_LAYER1B_CHECKS,
        ),

        "layer1c_source_text_warning_counts": complete_counts(
            source_warning_counter,
            ALL_LAYER1C_CHECKS,
        ),

        "leaf_action_category_counts": dict(leaf_action_counter.most_common()),
        "ast_action_category_counts": dict(ast_action_counter.most_common()),
        "action_category_counts": dict(combined_action_counter.most_common()),
        "layer1c_action_counts": dict(source_action_counter.most_common()),
    }

    return leaf_rows, ast_rows, summary


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    ROOT = Path(__file__).resolve().parents[3]

    all_ast_rows = []
    schema_path = ROOT / "schemas" / "rules_v3.json"

    branch_paths = {
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

    out_dir = ROOT / "outputs" / "verification" / "layer1" / "deterministic_inventory"
    out_dir.mkdir(parents=True, exist_ok=True)

    leaf_csv_path = out_dir / "deterministic_verification_inventory_leaf_level.csv"
    summary_json_path = out_dir / "deterministic_verification_inventory_summary.json"
    ast_csv_path = out_dir / "deterministic_verification_inventory_ast_level.csv"

    schema = load_schema(schema_path)
    validator = Draft7Validator(schema)

    all_leaf_rows = []
    summaries = {}

    for branch_name, path in branch_paths.items():
        leaf_rows, ast_rows, summary = scan_branch(
            branch_name=branch_name,
            ast_path=path,
            schema_validator=validator,
        )

        all_leaf_rows.extend(leaf_rows)
        all_ast_rows.extend(ast_rows)
        summaries[branch_name] = summary

    fieldnames = [
        "branch",
        "document_id",
        "path",
        "criterion_id",
        "item_uid",
        "clause_id",

        "entity_type",
        "entity_text",
        "operator",
        "value_type",
        "value",
        "unit",
        "temporal_context",
        "history_context",
        "computability",
        "non_computable_reason",
        "evidence_text",

        "deterministic_issues",
        "layer1a_action_category",
        "layer1a_action_hint",

        "layer1c_source_text_warnings",
        "layer1c_action",
    ]

    ast_fieldnames = [
        "branch",
        "document_id",
        "issue_type",
        "criterion_id",
        "paths",
        "layer1b_action_category",
        "detail",
    ]

    write_csv(leaf_csv_path, all_leaf_rows, fieldnames)
    write_csv(ast_csv_path, all_ast_rows, ast_fieldnames)

    combined_leaf_issue_counter = Counter()
    combined_ast_issue_counter = Counter()
    combined_action_counter = Counter()
    combined_source_warning_counter = Counter()
    combined_source_action_counter = Counter()

    for s in summaries.values():
        combined_leaf_issue_counter.update(s["layer1a_hard_leaf_issue_counts"])
        combined_ast_issue_counter.update(s["layer1b_ast_integrity_issue_counts"])
        combined_action_counter.update(s["action_category_counts"])
        combined_source_warning_counter.update(s["layer1c_source_text_warning_counts"])
        combined_source_action_counter.update(s["layer1c_action_counts"])

    combined_total_leaves = sum(
        s["leaf_flag_summary"]["n_total_leaves"]
        for s in summaries.values()
    )

    combined_n_layer1a_hard_issue_leaves = sum(
        s["leaf_flag_summary"]["n_layer1a_hard_issue_leaves"]
        for s in summaries.values()
    )

    combined_n_layer1c_warning_leaves = sum(
        s["leaf_flag_summary"]["n_layer1c_warning_leaves"]
        for s in summaries.values()
    )

    combined_n_any_layer1a_or_layer1c_flagged_leaves = sum(
        s["leaf_flag_summary"]["n_any_layer1a_or_layer1c_flagged_leaves"]
        for s in summaries.values()
    )

    combined_n_both_layer1a_and_layer1c_flagged_leaves = sum(
        s["leaf_flag_summary"]["n_both_layer1a_and_layer1c_flagged_leaves"]
        for s in summaries.values()
    )

    combined_n_layer1b_ast_warning_rows = sum(
        s["leaf_flag_summary"]["n_layer1b_ast_warning_rows"]
        for s in summaries.values()
    )

    combined_leaf_flag_summary = {
        "n_total_leaves": combined_total_leaves,

        "n_layer1a_hard_issue_leaves": combined_n_layer1a_hard_issue_leaves,
        "pct_layer1a_hard_issue_leaves": pct(
            combined_n_layer1a_hard_issue_leaves,
            combined_total_leaves,
        ),

        "n_layer1c_warning_leaves": combined_n_layer1c_warning_leaves,
        "pct_layer1c_warning_leaves": pct(
            combined_n_layer1c_warning_leaves,
            combined_total_leaves,
        ),

        "n_any_layer1a_or_layer1c_flagged_leaves": combined_n_any_layer1a_or_layer1c_flagged_leaves,
        "pct_any_layer1a_or_layer1c_flagged_leaves": pct(
            combined_n_any_layer1a_or_layer1c_flagged_leaves,
            combined_total_leaves,
        ),

        "n_both_layer1a_and_layer1c_flagged_leaves": combined_n_both_layer1a_and_layer1c_flagged_leaves,
        "pct_both_layer1a_and_layer1c_flagged_leaves": pct(
            combined_n_both_layer1a_and_layer1c_flagged_leaves,
            combined_total_leaves,
        ),

        "n_layer1b_ast_warning_rows": combined_n_layer1b_ast_warning_rows,
    }

    summary = {
        "stage": "deterministic_verification_inventory",
        "description": (
            "Layer 1 deterministic verification inventory. "
            "Layer 1A checks hard rules_v3 field consistency constraints. "
            "Layer 1B checks rule tree-level integrity warnings. "
            "Layer 1C checks deterministic source-text consistency warnings. "
            "No BERT support signals, probabilistic risk scores, LLM rescue, "
            "or rule-tree modification are used."
        ),
        "inputs": {
            "schema": str(schema_path),
            "branches": {k: str(v) for k, v in branch_paths.items()},
        },
        "outputs": {
            "leaf_csv": str(leaf_csv_path),
            "ast_csv": str(ast_csv_path),
            "summary_json": str(summary_json_path),
        },
        "branches": summaries,
        "combined": {
            "leaf_flag_summary": combined_leaf_flag_summary,
            "layer1a_hard_leaf_issue_counts": dict(combined_leaf_issue_counter.most_common()),
            "layer1b_ast_integrity_issue_counts": dict(combined_ast_issue_counter.most_common()),
            "layer1c_source_text_warning_counts": dict(combined_source_warning_counter.most_common()),
            "action_category_counts": dict(combined_action_counter.most_common()),
            "layer1c_action_counts": dict(combined_source_action_counter.most_common()),
        },
    }

    write_json(summary_json_path, summary)

    print("\n===== DETERMINISTIC VERIFICATION INVENTORY =====")
    print("Wrote leaf CSV:", leaf_csv_path)
    print("Wrote rule tree CSV:", ast_csv_path)
    print("Wrote summary JSON:", summary_json_path)

    for branch_name, s in summaries.items():
        print(f"\n--- {branch_name} ---")
        print("Documents:", s["n_documents"])
        print("Leaves:", s["n_leaves"])
        print("Documents with schema errors:", s["n_documents_with_schema_errors"])
        print("Layer 1 leaf flag summary:")
        print(s["leaf_flag_summary"])

        print("Layer 1A hard leaf issues:")
        print(s["layer1a_hard_leaf_issue_counts"])

        print("Layer 1B rule tree integrity issues:")
        print(s["layer1b_ast_integrity_issue_counts"])

        print("Layer 1C source-text warnings:")
        print(s["layer1c_source_text_warning_counts"])

        print("Layer 1C actions:")
        print(s["layer1c_action_counts"])

        print("Layer 1 action categories:")
        print(s["action_category_counts"])

    print("\n--- Combined ---")
    print("Layer 1A hard leaf issues:")
    print(dict(combined_leaf_issue_counter.most_common()))

    print("Layer 1B rule tree integrity issues:")
    print(dict(combined_ast_issue_counter.most_common()))

    print("Layer 1C source-text warnings:")
    print(dict(combined_source_warning_counter.most_common()))

    print("Layer 1 action categories:")
    print(dict(combined_action_counter.most_common()))

    print("Layer 1 combined leaf flag summary:")
    print(combined_leaf_flag_summary)


if __name__ == "__main__":
    main()

# Run from the repository root:
# python scripts/03_verification/01_layer1/01_build_deterministic_inventory.py