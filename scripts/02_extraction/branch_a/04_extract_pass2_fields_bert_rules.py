import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def lower(text: str) -> str:
    return normalize_text(text).lower()


def first_or_none(xs: List[Any]) -> Any:
    return xs[0] if xs else None


# ----------------------------
# BERT helpers
# ----------------------------

ANCHOR_LABEL_TO_ENTITY_TYPE = {
    "Condition": "condition",
    "Drug": "drug",
    "Procedure": "procedure",
    "Measurement": "lab",
    "Device": "other",
}

UNIT_PATTERN = r"(?:mg/dl|mg/kg/d|mg/kg|ml/min|cc/min|cells/mm\^3|mmhg|mm hg|meq/l|iu/l|u/l|seconds?|minutes?|hours?|months?|weeks?|years?|yrs?|days?|grade|mg|kg|g|ml|mm|cm|l|%|m\b)"
WORD_NUMBER_MAP = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


def choose_best_anchor(clause: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    candidates = clause.get("bert_candidates") or {}

    anchors = candidates.get("anchors", [])
    others = candidates.get("others", [])

    usable = list(anchors)

    # Backup only if no real anchor exists.
    # Do not use Value/Temporal/Qualifier as main anchors.
    if not usable:
        usable = [
            x for x in others
            if x.get("label") in {"Condition", "Drug", "Procedure", "Measurement", "Device"}
        ]

    if not usable:
        return None

    clause_text = lower(clause.get("clause_text", ""))
    evidence_text = lower(clause.get("evidence_text", ""))
    combined_text = clause_text + " " + evidence_text

    has_comparison = bool(
        re.search(
            r"(<=|>=|(?<![<>=!])<(?!=)|(?<![<>=!])>(?!=)|(?<![<>=!])=(?![=])|"
            r"\bbetween\b|\babove\b|\bbelow\b|\bless than\b|\bgreater than\b|"
            r"\bat least\b|\bat most\b)",
            combined_text,
        )
    )

    has_value_support = any(
        s.get("label") == "Value"
        for s in candidates.get("supports", [])
    )

    # If there is a numeric/value comparison, Measurement is often the correct anchor.
    # Otherwise, Condition is usually more clinically central than Procedure.
    if has_comparison or has_value_support:
        label_priority = {
            "Measurement": 0,
            "Condition": 1,
            "Drug": 2,
            "Procedure": 3,
            "Device": 4,
        }
    else:
        label_priority = {
            "Condition": 0,
            "Drug": 1,
            "Procedure": 2,
            "Measurement": 3,
            "Device": 4,
        }

    def token_overlap_score(anchor_text: str, target_text: str) -> int:
        anchor_tokens = set(re.findall(r"[a-zA-Z0-9]+", lower(anchor_text)))
        target_tokens = set(re.findall(r"[a-zA-Z0-9]+", lower(target_text)))
        return len(anchor_tokens & target_tokens)

    def score_anchor(x):
        txt = lower(x.get("text", ""))
        appears_in_clause = txt and txt in clause_text
        appears_in_evidence = txt and txt in evidence_text
        overlap = token_overlap_score(txt, clause_text)

        return (
            label_priority.get(x.get("label"), 9),
            0 if appears_in_clause else 1,
            0 if appears_in_evidence else 1,
            -overlap,
            -(float(x.get("score")) if x.get("score") is not None else -1.0),
            -(int(x.get("end", 0)) - int(x.get("start", 0))),
        )

    return sorted(usable, key=score_anchor)[0]


def map_entity_type_from_anchor(anchor: Optional[Dict[str, Any]]) -> Optional[str]:
    if anchor is None:
        return None
    label = anchor.get("label")
    return ANCHOR_LABEL_TO_ENTITY_TYPE.get(label)

OPERATIONAL_MARKERS = [
    "preference",
    "willing",
    "unwilling",
    "able to",
    "unable to",
    "informed consent",
    "questionnaire",
    "follow-up appointment",
    "attend follow-up",
    "included in a study",
    "participate",
]


def is_operational_criterion_text(text: str) -> bool:
    t = lower(text)
    return any(marker in t for marker in OPERATIONAL_MARKERS)


def fallback_entity_type_from_text(clause_text: str) -> str:
    t = lower(clause_text)

    if is_operational_criterion_text(t):
        return "other"

    if re.search(r"\b(ct|mri|scan|biopsy|surgery|transplant|therapy|treatment|chemoembolization)\b", t):
        return "procedure"
    if any(x in t for x in ["drug", "warfarin", "coumadin", "epothilone", "gemcitabine", "topotecan"]):
        return "drug"
    if re.search(r"\bage\b|\byears old\b|\bmale\b|\bfemale\b|\bpregnant\b|\blactating\b", t):
        return "demographic"
    if any(x in t for x in ["grade", "ecog", "hemoglobin", "platelet", "creatinine", "bilirubin"]):
        return "lab"
    if any(x in t for x in ["stage", "metastases", "infection", "neuropathy", "diarrhea", "hiv", "malignancy", "tumor"]):
        return "condition"

    return "other"


def extract_entity_text(clause: Dict[str, Any]) -> str:
    anchor = choose_best_anchor(clause)
    if anchor is not None and anchor.get("text"):
        return normalize_text(anchor["text"])

    clause_txt = clause.get("clause_text") or ""
    evid_txt = clause.get("evidence_text") or ""

    # Prefer the clause span the LLM explicitly produced
    if clause_txt:
        return normalize_text(clause_txt)

    if evid_txt:
        return normalize_text(evid_txt)

    return ""


# ----------------------------
# Modifier / provenance helpers
# ----------------------------

def collect_supports_by_label(clause: Dict[str, Any], label: str) -> List[str]:
    supports = (clause.get("bert_candidates") or {}).get("supports", [])
    out = []
    for s in supports:
        if s.get("label") == label and s.get("text"):
            out.append(normalize_text(s["text"]))
    return out


def extract_modifier_text(clause: Dict[str, Any]) -> Optional[str]:
    mods = collect_supports_by_label(clause, "Qualifier")
    if not mods:
        return None
    return " | ".join(dict.fromkeys(mods))


def extract_history_hint(clause_text: str, evidence_text: str) -> Optional[str]:
    text = lower(clause_text + " " + evidence_text)
    if "previously treated" in text or "prior treatment" in text:
        return "previous treatment"
    if "history of" in text:
        return "history"
    if "stable dose" in text:
        return "stable dose"
    if "investigational" in text:
        return "investigational use"
    return None


def extract_exception_context(clause_text: str, evidence_text: str, item_text: str) -> Optional[str]:
    # Prefer local clause/evidence first
    candidates = [evidence_text, clause_text, item_text]
    for text in candidates:
        t = normalize_text(text)
        m = re.search(r"\b(unless|except|with the exception of)\b(.+)$", t, flags=re.IGNORECASE)
        if m:
            return normalize_text(m.group(0))
    return None


def extract_condition_context(clause_text: str, evidence_text: str) -> Optional[str]:
    candidates = [evidence_text, clause_text]
    for text in candidates:
        t = normalize_text(text)
        m = re.search(r"^(if .+?)(?:,|$)", t, flags=re.IGNORECASE)
        if m:
            return normalize_text(m.group(1))
    return None


# ----------------------------
# Operator / value / unit
# ----------------------------

UNIT_PATTERN = r"(?:mg/dl|mg/kg/d|mg/kg|ml/min|cc/min|cells/mm\^3|mmhg|mm hg|meq/l|iu/l|u/l|seconds?|minutes?|hours?|months?|weeks?|years?|yrs?|days?|grade|mg|kg|g|ml|mm|cm|l|%|m\b)"

def extract_between(text: str) -> Optional[Tuple[Any, Any, Optional[str]]]:
    t = lower(text)

    number_pattern = r"\d+(?:\.\d+)?"

    patterns = [
        # between 18 years and 45 years
        rf"\bbetween\s+(?P<min>{number_pattern})\s*(?P<unit1>{UNIT_PATTERN})?\s+and\s+(?P<max>{number_pattern})\s*(?P<unit2>{UNIT_PATTERN})?",

        # minimum of 21 days and maximum of 36 days
        rf"\bminimum\s+of\s+(?P<min>{number_pattern})\s*(?P<unit1>{UNIT_PATTERN})?.*?\bmaximum\s+of\s+(?P<max>{number_pattern})\s*(?P<unit2>{UNIT_PATTERN})?",
    ]

    for pattern in patterns:
        m = re.search(pattern, t, flags=re.IGNORECASE)
        if not m:
            continue

        vmin_raw = m.group("min")
        vmax_raw = m.group("max")

        vmin = float(vmin_raw) if "." in vmin_raw else int(vmin_raw)
        vmax = float(vmax_raw) if "." in vmax_raw else int(vmax_raw)

        unit = m.group("unit2") or m.group("unit1")
        unit = normalize_text(unit) if unit else None

        return vmin, vmax, unit

    return None

def extract_comparison_value_unit(text: str) -> Tuple[Optional[Any], Optional[str]]:
    # normalize spacing but keep case for digits; we match words case‑insensitively
    t = normalize_text(text)

    # Pattern: either digits or a small number word, optional unit
    num_pattern = rf"(\d+(?:\.\d+)?|\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b)\s*({UNIT_PATTERN})?"
    m = re.search(num_pattern, t, flags=re.IGNORECASE)
    if not m:
        return None, None

    raw_num = m.group(1)
    raw_word = m.group(2)
    raw_unit = m.group(3)

    if raw_word:
        value = WORD_NUMBER_MAP.get(raw_word.lower())
    else:
        value = float(raw_num) if "." in raw_num else int(raw_num)

    unit = normalize_text(raw_unit) if raw_unit else None
    return value, unit


def extract_operator(clause: Dict[str, Any], criterion_type: Optional[str]) -> str:
    clause_text = lower(clause.get("clause_text", ""))
    evidence_text = lower(clause.get("evidence_text") or clause.get("clause_text", ""))
    text = clause_text + " " + evidence_text
    is_negated = bool(clause.get("is_negated", False))

    # between X and Y
    if re.search(r"\bbetween\b", text):
        return "between"

    # direct strict comparators and clinical phrases
    if re.search(r"(?<![!])>(?!=)|\babove\b|\bgreater than\b", text):
        return ">"

    if re.search(r"(?<![!])<(?!=)|\bbelow\b|\bless than\b", text):
        return "<"

    # equality
    if re.search(r"(?<![<>=!])=(?![=])|\bequal to\b|\bequals\b", text):
        return "="

    # non-strict comparators
    if re.search(r"(>=|at least|no less than|greater than or equal)", text):
        return ">="

    if re.search(r"(<=|at most|no more than|less than or equal)", text):
        return "<="

    if re.search(r"\bnot in\b", text):
        return "not_in"

    if re.search(r"\b(one of|any of|either)\b", text) and ("," in text or " or " in text):
        return "in"

    if is_negated or re.search(r"\b(no|without|absence of|free of|negative for)\b", text):
        return "not_exists"

    return "exists"


def extract_value_type_value_unit(clause: Dict[str, Any], operator: str) -> Tuple[str, Any, Optional[str]]:
    text = normalize_text(
        (clause.get("clause_text") or "") + " " + (clause.get("evidence_text") or "")
    )
    text_lower = lower(text)

    if operator == "between":
        result = extract_between(text)
        if result is not None:
            vmin, vmax, unit = result
            return "range", {"min": vmin, "max": vmax}, unit
        return "range", {"min": None, "max": None}, None

    if operator in {">", ">=", "<", "<=", "=", "!="}:
        if re.search(r"\b(?:above|greater than)\s+(?:the\s+)?(?:uln|upper limit of normal)\b", text_lower):
            return "scalar", "ULN", None

        if re.search(r"\b(?:below|less than)\s+(?:the\s+)?(?:uln|lower limit of normal)\b", text_lower):
            return "scalar", "ULN", None

        value, unit = extract_comparison_value_unit(text)
        if value is not None:
            return "scalar", value, unit
        return "null", None, None

    if operator in {"in", "not_in"}:
        return "null", None, None

    return "null", None, None


# ----------------------------
# Temporal context
# ----------------------------

def infer_anchor_event(text: str) -> str:
    t = lower(text)
    if "screening" in t:
        return "screening"
    if "randomization" in t:
        return "randomization"
    if "diagnosis" in t:
        return "diagnosis"
    if "surgery" in t:
        return "surgery"
    if "baseline" in t:
        return "baseline"
    if "study start" in t or "treatment start" in t or "first dose" in t:
        return "treatment_start"
    return "baseline"


def extract_temporal_context(clause: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    texts = [
        clause.get("clause_text", ""),
        clause.get("evidence_text", ""),
    ]
    supports = (clause.get("bert_candidates") or {}).get("supports", [])
    texts.extend([s.get("text", "") for s in supports])

    joined = " ".join(normalize_text(x) for x in texts if x)
    t = lower(joined)

    # within X months/weeks/days/years
    m = re.search(r"\bwithin(?: the last)?\s+(\d+(?:\.\d+)?)\s+(day|week|month|year)s?\b", t)
    if m:
        value = float(m.group(1)) if "." in m.group(1) else int(m.group(1))
        return {
            "relation": "within",
            "value": value,
            "unit": m.group(2),
            "anchor_event": infer_anchor_event(t),
        }

    # prior to / before
    if "prior to" in t or "before" in t:
        return {
            "relation": "before",
            "value": None,
            "unit": None,
            "anchor_event": infer_anchor_event(t),
        }

    # after
    if "after" in t:
        return {
            "relation": "after",
            "value": None,
            "unit": None,
            "anchor_event": infer_anchor_event(t),
        }

    # since
    if "since" in t:
        return {
            "relation": "since",
            "value": None,
            "unit": None,
            "anchor_event": infer_anchor_event(t),
        }

    # during screening / during treatment etc.
    if "during" in t:
        return {
            "relation": "during",
            "value": None,
            "unit": None,
            "anchor_event": infer_anchor_event(t),
        }

    return None


# ----------------------------
# History / computability
# ----------------------------

def extract_history_context(clause: Dict[str, Any]) -> Optional[str]:
    text = lower(clause.get("clause_text", "") + " " + clause.get("evidence_text", ""))

    if "previously treated" in text or "prior treatment" in text:
        return "previously_treated"
    if "history of" in text:
        return "prior"
    if "stable dose" in text:
        return "stable_dose"
    if "investigational" in text:
        return "investigational_use"
    if "current" in text:
        return "current"

    return None


NON_COMPUTABLE_MARKERS = [
    "willing",
    "unwilling",
    "able to",
    "unable to",
    "informed consent",
    "noncompliance",
    "preclude study",
    "psychiatric illness that would preclude",
]


def infer_computability(
    clause: Dict[str, Any],
    entity_type: str,
    operator: str,
    evidence_text: str,
) -> Tuple[str, Optional[str]]:
    text = lower(clause.get("clause_text", "") + " " + evidence_text)

    if is_operational_criterion_text(text):
        return "non_computable", "Subjective or operational criterion."

    for marker in NON_COMPUTABLE_MARKERS:
        if marker in text:
            return "non_computable", f"Subjective or operational criterion: {marker}"

    if extract_exception_context(clause.get("clause_text", ""), evidence_text, "") is not None:
        return "partial", "Contains exception or conditional nuance that may need later verification."

    if entity_type == "other" and operator == "exists":
        return "partial", "Entity typing is weak and may require later review."

    return "computable", None


# ----------------------------
# Leaf builder
# ----------------------------

def build_leaf_from_clause(
    trial_id: str,
    item_uid: str,
    criterion_type: Optional[str],
    item_text: str,
    clause: Dict[str, Any],
) -> Dict[str, Any]:
    clause_id = clause["clause_id"]
    criterion_id = f"{item_uid}_{clause_id}"

    anchor = choose_best_anchor(clause)
    entity_type = map_entity_type_from_anchor(anchor)

    clause_text_all = normalize_text(
        (clause.get("clause_text") or "") + " " + (clause.get("evidence_text") or "")
    )

    is_operational = is_operational_criterion_text(clause_text_all)

    # If the whole criterion is operational/subjective, do not let a BERT anchor
    # like "questionnaires" turn it into a clinical procedure.
    if is_operational:
        entity_type = "other"
    elif entity_type is None:
        entity_type = fallback_entity_type_from_text(clause.get("clause_text", ""))

    entity_text = extract_entity_text(clause)

    if is_operational:
        entity_text = normalize_text(clause.get("clause_text") or clause.get("evidence_text") or entity_text)
    operator = extract_operator(clause, criterion_type)
    value_type, value, unit = extract_value_type_value_unit(clause, operator)
    temporal_context = extract_temporal_context(clause)
    history_context = extract_history_context(clause)

    evidence_text = normalize_text(clause.get("evidence_text", ""))
    computability, non_computable_reason = infer_computability(
        clause=clause,
        entity_type=entity_type,
        operator=operator,
        evidence_text=evidence_text,
    )

    provenance = {
        "source_modifier_text": extract_modifier_text(clause),
        "source_condition_text": extract_condition_context(
            clause.get("clause_text", ""),
            evidence_text,
        ),
        "source_exception_context": extract_exception_context(
            clause.get("clause_text", ""),
            evidence_text,
            item_text,
        ),
        "history_context_hint": extract_history_hint(
            clause.get("clause_text", ""),
            evidence_text,
        ),
    }

    # Normalize provenance to null if all fields are null
    if all(v is None for v in provenance.values()):
        provenance = None

    # If comparison operator but no numeric value, downgrade computability
    if operator in {">", ">=", "<", "<=", "=", "!="} and value is None:
        if computability == "computable":
            computability = "partial"
            if not non_computable_reason:
                non_computable_reason = "Comparison operator present but no numeric value extracted."

    criterion = {
        "criterion_id": criterion_id,
        "entity_type": entity_type,
        "entity_text": entity_text,
        "normalized_concept": None,
        "operator": operator,
        "value_type": value_type,
        "value": value,
        "unit": unit,
        "temporal_context": temporal_context,
        "computability": computability,
        "non_computable_reason": non_computable_reason,
        "evidence_text": evidence_text,
        "provenance": provenance,
        "history_context": history_context,
    }

    return {
        "clause_id": clause_id,
        "criterion": criterion,
    }


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ROOT = Path(__file__).resolve().parents[3]

    in_path = ROOT / "outputs" / "extraction" / "pass2_inputs" / "chia_text_only_200_pass2_inputs.jsonl"
    schema_path = ROOT / "schemas" / "rules_v3_pass2_leaf.json"

    out_dir = ROOT / "outputs" / "extraction" / "branch_a" / "pass2_leaves"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "chia_text_only_200_pass2_leaves.jsonl"

    rows = load_jsonl(in_path)
    schema = load_schema(schema_path)
    validator = Draft7Validator(schema)

    output_rows: List[Dict[str, Any]] = []
    n_ok = 0
    n_err = 0

    for row in rows:
        item_uid = row.get("item_uid")
        chia_id = row.get("chia_id")
        doc_id = row.get("document_id")

        if row.get("status") != "ok":
            output_rows.append(
                {
                    "dataset": "CHIA",
                    "stage": "pass2_leaf_extraction",
                    "item_uid": item_uid,
                    "chia_id": chia_id,
                    "document_id": doc_id,
                    "status": "error",
                    "error": f"Pass2 input row status is not ok: {row.get('status')}",
                }
            )
            n_err += 1
            continue

        try:
            payload = row["pass2_input"]
            trial_id = payload["trial_id"]
            criterion_type = payload.get("criterion_type")
            item_text = payload["item_text"]
            clauses = payload["clauses"]

            criteria = [
                build_leaf_from_clause(
                    trial_id=trial_id,
                    item_uid=item_uid,
                    criterion_type=criterion_type,
                    item_text=item_text,
                    clause=clause,
                )
                for clause in clauses
            ]

            pass2_output = {
                "trial_id": trial_id,
                "item_uid": item_uid,
                "chia_id": chia_id,
                "document_id": doc_id,
                "criterion_type": criterion_type,
                "criteria": criteria,
            }

            errors = list(validator.iter_errors(pass2_output))
            if errors:
                msg = "; ".join(e.message for e in errors)
                raise ValidationError(msg)

            rec = {
                "dataset": "CHIA",
                "stage": "pass2_leaf_extraction",
                "item_uid": item_uid,
                "chia_id": chia_id,
                "document_id": doc_id,
                "status": "ok",
                "error": None,
                "pass2_source": "chia_text_only_200_pass2_inputs.jsonl",
                "pass2_output": pass2_output,
            }
            output_rows.append(rec)
            n_ok += 1

        except Exception as e:
            output_rows.append(
                {
                    "dataset": "CHIA",
                    "stage": "pass2_leaf_extraction",
                    "item_uid": item_uid,
                    "chia_id": chia_id,
                    "document_id": doc_id,
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

#python scripts/02_extraction/branch_a/04_extract_pass2_fields_bert_rules.py