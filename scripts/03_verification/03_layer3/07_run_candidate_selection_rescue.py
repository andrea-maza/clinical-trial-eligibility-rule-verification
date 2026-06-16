"""
07_run_candidate_selection_rescue.py

Run the branch-specific Layer 3 candidate-selection rescue.

Branch A:
    - makes no new LLM calls
    - uses the corresponding Branch B leaf as a semantic substitute
    - uses the validated Branch B rescue result when available
    - otherwise routes unresolved cases to human review

Branch B:
    - uses a bounded judge-and-repair loop
    - judges the current Branch B leaf first
    - keeps it unchanged when it meets the judge threshold
    - otherwise generates and judges up to three repair candidates
    - selects only a locally valid candidate that improves the score
    - abstains for human review when no candidate is sufficiently safe

Manual labels are not used. This script writes proposal records but
does not modify the logical rule trees.

Outputs:
    outputs/verification/layer3/candidate_selection_rescue/
        candidate_selection_rescue_results.jsonl
        candidate_selection_rescue_results.csv
        candidate_selection_rescue_raw_prompts.jsonl
        candidate_selection_rescue_summary.json

Run from the repository root:
python scripts/03_verification/03_layer3/07_run_candidate_selection_rescue.py --limit 5 --dry-run
"""


from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# ---------------------------------------------------------------------
# Schema-sensitive allowed values
# ---------------------------------------------------------------------

ALLOWED_ENTITY_TYPES = {
    "condition",
    "drug",
    "procedure",
    "lab",
    "demographic",
    "therapy",
    "biomarker",
    "vital",
    "observation",
    "stage",
    "line_of_therapy",
    "other",
}

ALLOWED_OPERATORS = {
    "exists",
    "not_exists",
    "=",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
    "between",
    "in",
    "not_in",
    "contains",
    "matches",
}

ALLOWED_VALUE_TYPES = {"scalar", "list", "range", "null"}
ALLOWED_COMPUTABILITY = {"computable", "partial", "non_computable"}
ALLOWED_TEMPORAL_RELATIONS = {"before", "after", "during", "within", "since"}
ALLOWED_TEMPORAL_UNITS = {"hour", "day", "week", "month", "year", None}
ALLOWED_ANCHOR_EVENTS = {
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
ALLOWED_HISTORY_CONTEXT = {
    "current",
    "prior",
    "previously_treated",
    "stable_dose",
    "investigational_use",
    "other",
    None,
}

COPYABLE_LEAF_FIELDS = {
    "criterion_id",
    "entity_text",
    "entity_type",
    "normalized_concept",
    "operator",
    "value",
    "value_type",
    "unit",
    "temporal_context",
    "history_context",
    "negated",
    "computability",
    "non_computable_reason",
    "evidence_text",
}

VALID_DECISIONS = {
    "select_candidate",
    "no_change",
    "mark_partial_or_non_computable",
    "human_review",
}

VALID_SELECTED_SOURCES = {
    "A_current",
    "B_current",
    "B_dejure_best",
    "LLM_candidate_1",
    "LLM_candidate_2",
    "LLM_candidate_3",
    "none",
}


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSONL file: {path}")

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


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return

    priority = [
        "plan_id",
        "branch_to_update",
        "criterion_id",
        "execution_group",
        "rescue_strategy",
        "rescue_task_type",
        "decision",
        "selected_source",
        "final_decision",
        "confidence",
        "local_validation_status",
        "local_validation_reasons",
        "selection_reason",
        "allowed_candidate_sources",
        "llm_model",
        "parse_status",
        "A_entity_text",
        "B_entity_text",
        "selected_entity_text",
        "selected_entity_type",
        "selected_operator",
        "selected_value_type",
        "selected_value",
        "selected_unit",
        "selected_computability",
        "selected_temporal_context",
        "selected_history_context",
    ]

    flattened = [flatten_for_csv(row) for row in rows]
    extra = sorted({k for r in flattened for k in r.keys()} - set(priority))
    fieldnames = priority + extra

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in flattened:
            writer.writerow({k: serialize_cell(row.get(k, "")) for k in fieldnames})


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def clean(x: Any) -> str:
    return str(x or "").strip()


def lower(x: Any) -> str:
    return clean(x).lower()


def norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def contains_normalized(container: Any, content: Any) -> bool:
    c = norm(container)
    x = norm(content)
    if not c or not x:
        return False
    return x in c


def to_int(x: Any, default: int = 0) -> int:
    try:
        s = clean(x)
        return int(float(s)) if s else default
    except Exception:
        return default


def boolish(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    return clean(x).lower() in {"1", "true", "yes", "y"}


def strip_internal_fields(leaf: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(leaf, dict):
        return {}
    return {k: deepcopy(v) for k, v in leaf.items() if not str(k).startswith("_")}


def parse_item_clause_from_criterion_id(criterion_id: str) -> Tuple[str, str]:
    criterion_id = clean(criterion_id)
    if "_" not in criterion_id:
        return "", ""
    item_uid, clause_id = criterion_id.rsplit("_", 1)
    return item_uid, clause_id


def selected_leaf_summary(leaf: Any) -> Dict[str, Any]:
    if not isinstance(leaf, dict):
        return {}
    return {
        "selected_entity_text": leaf.get("entity_text"),
        "selected_entity_type": leaf.get("entity_type"),
        "selected_operator": leaf.get("operator"),
        "selected_value_type": leaf.get("value_type"),
        "selected_value": leaf.get("value"),
        "selected_unit": leaf.get("unit"),
        "selected_computability": leaf.get("computability"),
        "selected_temporal_context": leaf.get("temporal_context"),
        "selected_history_context": leaf.get("history_context"),
    }


def flatten_for_csv(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    out.update(selected_leaf_summary(row.get("selected_leaf")))

    a_leaf = row.get("A_current_leaf")
    b_leaf = row.get("B_current_leaf")

    if isinstance(a_leaf, dict):
        out["A_entity_text"] = a_leaf.get("entity_text")
    if isinstance(b_leaf, dict):
        out["B_entity_text"] = b_leaf.get("entity_text")

    if isinstance(out.get("allowed_candidate_sources"), list):
        out["allowed_candidate_sources"] = ";".join(out["allowed_candidate_sources"])

    return out

def sum_trace_token_usage(traces: List[Dict[str, Any]]) -> Dict[str, int]:
    out = {
        "layer3_llm_calls": 0,
        "layer3_prompt_tokens": 0,
        "layer3_completion_tokens": 0,
        "layer3_total_tokens": 0,
    }

    for trace in traces:
        usage = trace.get("token_usage") or {}

        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        total = usage.get("total_tokens")

        if total is None:
            continue

        out["layer3_llm_calls"] += 1
        out["layer3_prompt_tokens"] += int(prompt or 0)
        out["layer3_completion_tokens"] += int(completion or 0)
        out["layer3_total_tokens"] += int(total or 0)

    return out

# ---------------------------------------------------------------------
# Rule-tree leaf indexing
# ---------------------------------------------------------------------

def walk_node_collect_leaves(node: Any, out: Dict[str, Dict[str, Any]], document_id: str) -> None:
    if not isinstance(node, dict):
        return

    if node.get("node_type") == "criterion":
        criterion = node.get("criterion")
        if isinstance(criterion, dict):
            cid = clean(criterion.get("criterion_id"))
            if cid:
                leaf = deepcopy(criterion)
                leaf["_document_id"] = document_id
                out[cid] = leaf
        return

    if node.get("node_type") == "group":
        children = node.get("children", [])
        if isinstance(children, list):
            for child in children:
                walk_node_collect_leaves(child, out, document_id)


def build_leaf_index(ast_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}

    for row in ast_rows:
        if row.get("status", "ok") != "ok":
            continue
        document_id = clean(row.get("document_id"))
        ast = row.get("rules_v3_ast", {})
        if not isinstance(ast, dict):
            continue
        walk_node_collect_leaves(ast.get("inclusion_criteria"), out, document_id)
        walk_node_collect_leaves(ast.get("exclusion_criteria"), out, document_id)

    return out


# ---------------------------------------------------------------------
# Pass2 context indexing
# ---------------------------------------------------------------------

def build_pass2_context_index(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for row in rows:
        if row.get("status") != "ok":
            continue

        payload = row.get("pass2_input", {})
        if not isinstance(payload, dict):
            continue

        item_uid = clean(payload.get("item_uid"))
        item_text = clean(payload.get("item_text"))
        criterion_type = clean(payload.get("criterion_type"))

        for clause in payload.get("clauses", []) or []:
            if not isinstance(clause, dict):
                continue

            clause_id = clean(clause.get("clause_id"))
            if not item_uid or not clause_id:
                continue

            ctx = deepcopy(clause)
            ctx["item_uid"] = item_uid
            ctx["item_text"] = item_text
            ctx["criterion_type"] = criterion_type
            ctx["document_id"] = row.get("document_id")
            ctx["chia_id"] = row.get("chia_id")
            out[(item_uid, clause_id)] = ctx

    return out


def get_context_for_plan(plan: Dict[str, Any], ctx_index: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, Any]:
    item_uid = clean(plan.get("item_uid"))
    clause_id = clean(plan.get("clause_id"))

    if not item_uid or not clause_id:
        item_uid, clause_id = parse_item_clause_from_criterion_id(clean(plan.get("criterion_id")))

    return ctx_index.get((item_uid, clause_id), {})


def source_context_text(context: Dict[str, Any], a_leaf: Dict[str, Any], b_leaf: Dict[str, Any]) -> str:
    parts = [
        context.get("evidence_text"),
        context.get("clause_text"),
        context.get("item_text"),
        a_leaf.get("evidence_text") if isinstance(a_leaf, dict) else "",
        b_leaf.get("evidence_text") if isinstance(b_leaf, dict) else "",
    ]
    return " ".join(clean(x) for x in parts if clean(x))


# ---------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------

def make_client():
    """
    Create the LLM client using the same UMGPT/Azure setup used in extraction.

    Expected .env variables:
        UMGPT_API_BASE
        UMGPT_API_VERSION
        UMGPT_API_KEY
        UMGPT_SHORTCODE

    Optional model variables:
        UMGPT_MODEL_RESCUE
        UMGPT_MODEL_LAYER3_RESCUE
        UMGPT_MODEL_VERIFICATION
        UMGPT_MODEL_PASS2
        UMGPT_MODEL
    """
    from dotenv import load_dotenv
    from openai import AzureOpenAI

    root = Path(__file__).resolve().parents[3]
    load_dotenv(root / ".env")

    api_base = os.getenv("UMGPT_API_BASE")
    api_version = os.getenv("UMGPT_API_VERSION")
    api_key = os.getenv("UMGPT_API_KEY")
    shortcode = os.getenv("UMGPT_SHORTCODE")

    model = (
        os.getenv("UMGPT_MODEL_RESCUE")
        or os.getenv("UMGPT_MODEL_LAYER3_RESCUE")
        or os.getenv("UMGPT_MODEL_VERIFICATION")
        or os.getenv("UMGPT_MODEL_PASS2")
        or os.getenv("UMGPT_MODEL")
        or "gpt-5.2"
    )

    missing = [
        name
        for name, value in {
            "UMGPT_API_BASE": api_base,
            "UMGPT_API_VERSION": api_version,
            "UMGPT_API_KEY": api_key,
            "UMGPT_SHORTCODE": shortcode,
        }.items()
        if not value
    ]

    if missing:
        raise RuntimeError(
            "Missing UMGPT_* env vars. Check your .env file. "
            f"Missing: {missing}"
        )

    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=api_base,
        organization=shortcode,
    )

    return client, model


def extract_usage(response: Any) -> Dict[str, Any]:
    usage = getattr(response, "usage", None)

    if usage is None:
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }

    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def call_llm_json(
    client: Any,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> Tuple[str, Dict[str, Any]]:
    """
    UMGPT/Azure call with retry.

    GPT-5 style models usually need max_completion_tokens.
    Some deployments reject response_format, so this retries without it.
    """
    last_error = None
    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_completion_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            }

            response = client.with_options(timeout=90.0).chat.completions.create(**kwargs)
            return response.choices[0].message.content or "", extract_usage(response)

        except Exception as exc:
            msg = str(exc)
            last_error = exc

            if "response_format" in msg and ("unsupported" in msg.lower() or "invalid" in msg.lower()):
                try:
                    response = client.with_options(timeout=90.0).chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_completion_tokens=max_tokens,
                    )
                    return response.choices[0].message.content or "", extract_usage(response)
                except Exception as exc2:
                    last_error = exc2

            if "max_completion_tokens" in msg and ("unsupported" in msg.lower() or "invalid" in msg.lower()):
                try:
                    response = client.with_options(timeout=90.0).chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return response.choices[0].message.content or "", extract_usage(response)
                except Exception as exc2:
                    last_error = exc2

            wait_sec = min(2 ** attempt, 20)
            print(f"[RETRY] attempt {attempt}/{max_attempts} failed: {type(exc).__name__}: {exc}")
            time.sleep(wait_sec)

    raise RuntimeError(
        f"LLM call failed after {max_attempts} attempts. Last error: {last_error}"
    )

# ---------------------------------------------------------------------
# Parsing and validation
# ---------------------------------------------------------------------

def parse_json_response(text: str) -> Tuple[str, Optional[Dict[str, Any]], str]:
    raw = clean(text)
    if not raw:
        return "empty", None, "empty_response"

    raw = re.sub(r"^```json\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return "parsed", obj, ""
        return "not_object", None, "json_not_object"
    except json.JSONDecodeError as exc:
        # Try extracting first JSON object in case model added text.
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(raw[start:end + 1])
                if isinstance(obj, dict):
                    return "parsed_from_substring", obj, ""
            except Exception:
                pass
        return "parse_error", None, str(exc)


def normalize_temporal_context(tc: Any) -> Any:
    if tc in ["", "null", "None"]:
        return None
    return tc


def normalize_candidate_leaf(leaf: Any, default_leaf: Dict[str, Any], criterion_id: str) -> Optional[Dict[str, Any]]:
    if leaf is None:
        return None
    if not isinstance(leaf, dict):
        return None

    out: Dict[str, Any] = {}
    out["criterion_id"] = criterion_id

    for field in COPYABLE_LEAF_FIELDS:
        if field == "criterion_id":
            continue
        if field in leaf:
            out[field] = deepcopy(leaf.get(field))
        elif field in default_leaf:
            out[field] = deepcopy(default_leaf.get(field))

    # Keep negation schema-safe. Some older outputs used null; the AST apply
    # script expects a boolean when the field is present.
    if "negated" in out:
        out["negated"] = bool(out.get("negated", False))
    elif "negated" in default_leaf:
        out["negated"] = bool(default_leaf.get("negated", False))

    if "non_computable_reason" not in out:
        out["non_computable_reason"] = None

    out["temporal_context"] = normalize_temporal_context(out.get("temporal_context"))

    return out


def value_consistency_reasons(leaf: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []

    operator = leaf.get("operator")
    value_type = leaf.get("value_type")
    value = leaf.get("value")

    if value_type == "null":
        if value is not None:
            reasons.append("null_value_type_with_nonnull_value")
    elif value_type == "scalar":
        if value is None or isinstance(value, (list, dict)):
            reasons.append("scalar_value_type_with_invalid_value")
    elif value_type == "list":
        if not isinstance(value, list) or len(value) == 0:
            reasons.append("list_value_type_with_invalid_value")
    elif value_type == "range":
        if not isinstance(value, dict):
            reasons.append("range_value_type_with_invalid_value")
        elif "min" not in value or "max" not in value:
            reasons.append("range_value_missing_min_or_max_keys")
        elif value.get("min") is None and value.get("max") is None:
            reasons.append("range_with_both_bounds_missing")

    if operator in {"<", "<=", ">", ">=", "=", "!="}:
        if value_type != "scalar" or value is None:
            reasons.append("comparison_operator_without_scalar_value")

    if operator == "between" and value_type != "range":
        reasons.append("between_operator_without_range_value")

    if operator in {"in", "not_in"} and value_type != "list":
        reasons.append("list_operator_without_list_value")

    if operator in {"exists", "not_exists"} and value_type != "null":
        reasons.append("existence_operator_with_non_null_value_type")

    return reasons


def validate_leaf_schema_and_grounding(
    leaf: Optional[Dict[str, Any]],
    context: Dict[str, Any],
    a_leaf: Dict[str, Any],
    b_leaf: Dict[str, Any],
) -> Tuple[str, List[str]]:
    if leaf is None:
        return "fail", ["selected_leaf_is_null"]

    if not isinstance(leaf, dict):
        return "fail", ["selected_leaf_not_object"]

    reasons: List[str] = []

    entity_text = clean(leaf.get("entity_text"))
    evidence_text = clean(leaf.get("evidence_text"))
    entity_type = leaf.get("entity_type")
    operator = leaf.get("operator")
    value_type = leaf.get("value_type")
    computability = leaf.get("computability")
    history_context = leaf.get("history_context")
    temporal_context = normalize_temporal_context(leaf.get("temporal_context"))

    if not entity_text:
        reasons.append("missing_entity_text")

    if not evidence_text:
        reasons.append("missing_evidence_text")

    if entity_type not in ALLOWED_ENTITY_TYPES:
        reasons.append(f"entity_type_not_allowed:{entity_type}")

    if operator not in ALLOWED_OPERATORS:
        reasons.append(f"operator_not_allowed:{operator}")

    if value_type not in ALLOWED_VALUE_TYPES:
        reasons.append(f"value_type_not_allowed:{value_type}")

    if computability not in ALLOWED_COMPUTABILITY:
        reasons.append(f"computability_not_allowed:{computability}")

    if history_context not in ALLOWED_HISTORY_CONTEXT:
        reasons.append(f"history_context_not_allowed:{history_context}")

    if temporal_context is not None:
        if not isinstance(temporal_context, dict):
            reasons.append("temporal_context_must_be_object_or_null")
        else:
            allowed_keys = {"relation", "value", "unit", "anchor_event"}
            bad_keys = set(temporal_context.keys()) - allowed_keys
            if bad_keys:
                reasons.append(f"temporal_context_has_unexpected_keys:{sorted(bad_keys)}")

            relation = temporal_context.get("relation")
            unit = temporal_context.get("unit")
            anchor_event = temporal_context.get("anchor_event")

            if relation not in ALLOWED_TEMPORAL_RELATIONS:
                reasons.append(f"temporal_relation_not_allowed:{relation}")
            if unit not in ALLOWED_TEMPORAL_UNITS:
                reasons.append(f"temporal_unit_not_allowed:{unit}")
            if anchor_event not in ALLOWED_ANCHOR_EVENTS:
                reasons.append(f"temporal_anchor_event_not_allowed:{anchor_event}")

    reasons.extend(value_consistency_reasons(leaf))

    source_text = source_context_text(context, a_leaf, b_leaf)
    if entity_text and evidence_text and not contains_normalized(evidence_text, entity_text):
        # Strict for generated Branch B repair candidates.
        # Flexible for Branch A deterministic substitution if the entity is still grounded
        # somewhere in the original clause/item/source context.
        if not contains_normalized(source_text, entity_text):
            reasons.append("entity_text_not_in_source_context")
    if evidence_text and source_text and not contains_normalized(source_text, evidence_text):
        reasons.append("evidence_text_not_substring_of_source_context")

    if computability == "non_computable" and not clean(leaf.get("non_computable_reason")):
        reasons.append("non_computable_without_reason")

    if reasons:
        return "fail", reasons

    return "pass", []


def error_result(plan: Dict[str, Any], reason: str, a_leaf: Dict[str, Any], b_leaf: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_id": plan.get("plan_id"),
        "branch_to_update": plan.get("branch_to_update"),
        "criterion_id": plan.get("criterion_id"),
        "item_uid": plan.get("item_uid"),
        "clause_id": plan.get("clause_id"),
        "execution_group": plan.get("execution_group"),
        "rescue_strategy": plan.get("rescue_strategy"),
        "rescue_task_type": plan.get("rescue_task_type"),
        "allowed_candidate_sources": [],
        "decision": "human_review",
        "selected_source": "none",
        "selected_leaf": None,
        "generated_candidates": [],
        "candidate_evaluations": [],
        "selection_reason": reason,
        "confidence": "low",
        "final_decision": "human_review",
        "local_validation_status": "fail",
        "local_validation_reasons": [reason],
        "A_current_leaf": strip_internal_fields(a_leaf),
        "B_current_leaf": strip_internal_fields(b_leaf),
        "context": {
            "document_id": context.get("document_id"),
            "chia_id": context.get("chia_id"),
            "item_uid": context.get("item_uid"),
            "clause_id": context.get("clause_id"),
            "item_text": context.get("item_text"),
            "clause_text": context.get("clause_text"),
            "evidence_text": context.get("evidence_text"),
        },
        "parse_status": "error",
    }


# ---------------------------------------------------------------------
# De Jure-style Branch B judge-and-repair
# ---------------------------------------------------------------------

BRANCH_B_DEJURE_CRITERIA = [
    "semantic_completeness",
    "entity_grounding",
    "operator_value_correctness",
    "temporal_history_correctness",
    "polarity_negation_correctness",
    "computability_correctness",
    "fidelity_to_source",
    "non_hallucination",
]


def clamp_score(x: Any, default: float = 0.0) -> float:
    try:
        value = float(x)
    except Exception:
        return default
    return max(0.0, min(5.0, value))


def avg_judge_score(scores: Dict[str, Any]) -> float:
    values = [clamp_score(scores.get(k), 0.0) for k in BRANCH_B_DEJURE_CRITERIA]
    if not values:
        return 0.0
    return sum(values) / len(values)


def critical_errors_from_judgment(judgment: Dict[str, Any]) -> List[str]:
    errors = judgment.get("critical_errors", [])
    if isinstance(errors, list):
        return [clean(x) for x in errors if clean(x)]
    if clean(errors):
        return [clean(errors)]
    return []


def build_branch_b_judge_prompt(
    *,
    plan: Dict[str, Any],
    candidate_id: str,
    candidate_leaf: Dict[str, Any],
    context: Dict[str, Any],
    b_current_leaf: Dict[str, Any],
) -> List[Dict[str, str]]:
    """
    Judge one Branch B candidate with De Jure-style criterion scores.

    This is intentionally separated from generation. The current Branch B leaf is
    judged first. Repair is triggered only if the candidate fails the threshold.
    """
    evidence_text = (
        clean(context.get("evidence_text"))
        or clean(candidate_leaf.get("evidence_text"))
        or clean(b_current_leaf.get("evidence_text"))
    )

    system = (
        "You are a strict clinical-trial eligibility rule judge. "
        "Score the structured criterion leaf against the source text using independent criteria. "
        "Do not reward cosmetic changes. Penalize missing qualifiers, wrong entity, wrong operator/value, "
        "wrong temporal/history context, wrong polarity, unsupported evidence, and hallucination."
    )

    user_payload = {
        "task": "branch_b_candidate_judgment",
        "criterion_id": plan.get("criterion_id"),
        "candidate_id": candidate_id,
        "clinical_text": {
            "item_text": context.get("item_text"),
            "clause_text": context.get("clause_text"),
            "evidence_text": evidence_text,
        },
        "plan_diagnosis": {
            "rescue_strategy": plan.get("rescue_strategy"),
            "rescue_task_type": plan.get("rescue_task_type"),
            "selection_policy": plan.get("selection_policy"),
            "why_selected": plan.get("why_selected"),
            "source_issue_codes": plan.get("source_issue_codes"),
            "source_risk_reasons": plan.get("source_risk_reasons"),
            "source_diagnosis_summary": plan.get("source_diagnosis_summary"),
        },
        "candidate_leaf": strip_internal_fields(candidate_leaf),
        "scoring_criteria_0_to_5": {
            "semantic_completeness": "Does the leaf capture the full atomic clinical meaning of the clause, including important qualifiers?",
            "entity_grounding": "Is entity_text clinically appropriate and supported by evidence_text?",
            "operator_value_correctness": "Are operator, value_type, value, and unit correct for the evidence?",
            "temporal_history_correctness": "Are temporal_context and history_context correct and not missing if required?",
            "polarity_negation_correctness": "Is negated/operator polarity correct?",
            "computability_correctness": "Is computability appropriate: computable, partial, or non_computable?",
            "fidelity_to_source": "Does the leaf preserve source meaning without adding or omitting material content?",
            "non_hallucination": "Is every populated field grounded in the provided source text?",
        },
        "threshold": (
            "For B_current, pass only if average_score >= 4.5/5 and there are no critical_errors. "
            "For generated repair candidates, they may be selected if average_score >= 4.0/5, "
            "they improve over B_current by the required minimum, and there are no critical_errors."
        ),
        "required_output_json": {
            "candidate_id": candidate_id,
            "scores": {k: "integer or float from 0 to 5" for k in BRANCH_B_DEJURE_CRITERIA},
            "average_score": "0 to 5",
            "pass_threshold": "true or false",
            "critical_errors": ["short strings; empty list if none"],
            "critique": "specific field-level critique; say what should be repaired if failing",
            "recommended_action": "keep | repair | human_review",
        },
        "output_rules": [
            "Return valid JSON only.",
            "Do not generate a repaired leaf in this judgment step.",
            "If the candidate is already acceptable, recommended_action must be keep.",
            "If the problem is unsafe condition/exception scope, recommended_action must be human_review.",
        ],
    }

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
    ]


def build_branch_b_repair_prompt(
    *,
    plan: Dict[str, Any],
    attempt_no: int,
    b_current_leaf: Dict[str, Any],
    best_leaf: Dict[str, Any],
    last_judgment: Dict[str, Any],
    context: Dict[str, Any],
) -> List[Dict[str, str]]:
    evidence_text = (
        clean(context.get("evidence_text"))
        or clean(b_current_leaf.get("evidence_text"))
        or clean(best_leaf.get("evidence_text"))
    )

    system = (
        "You are a strict clinical-trial eligibility extraction repair model. "
        "Repair only deficient fields identified by the judge. Do not rewrite a good leaf. "
        "Preserve clinical meaning and exact evidence grounding. If the source does not support a repair, abstain."
    )

    user_payload = {
        "task": "branch_b_dejure_style_targeted_repair",
        "candidate_id": f"LLM_candidate_{attempt_no}",
        "criterion_id": plan.get("criterion_id"),
        "clinical_text": {
            "item_text": context.get("item_text"),
            "clause_text": context.get("clause_text"),
            "evidence_text": evidence_text,
        },
        "plan_diagnosis": {
            "rescue_strategy": plan.get("rescue_strategy"),
            "rescue_task_type": plan.get("rescue_task_type"),
            "selection_policy": plan.get("selection_policy"),
            "why_selected": plan.get("why_selected"),
            "source_issue_codes": plan.get("source_issue_codes"),
            "source_risk_reasons": plan.get("source_risk_reasons"),
            "source_diagnosis_summary": plan.get("source_diagnosis_summary"),
        },
        "current_branch_b_leaf": strip_internal_fields(b_current_leaf),
        "best_leaf_so_far": strip_internal_fields(best_leaf),
        "judge_scores_and_critique": last_judgment,
        "schema_allowed_values": {
            "entity_type": sorted(ALLOWED_ENTITY_TYPES),
            "operator": sorted(ALLOWED_OPERATORS),
            "value_type": sorted(ALLOWED_VALUE_TYPES),
            "computability": sorted(ALLOWED_COMPUTABILITY),
            "temporal_context.relation": sorted(ALLOWED_TEMPORAL_RELATIONS),
            "temporal_context.unit": ["hour", "day", "week", "month", "year", None],
            "temporal_context.anchor_event": sorted(ALLOWED_ANCHOR_EVENTS),
            "history_context": sorted([x for x in ALLOWED_HISTORY_CONTEXT if x is not None]) + [None],
        },
        "repair_rules": [
            "Correct only the deficient fields identified in the critique.",
            "Do not change fields that are already correct.",
            "evidence_text must be an exact substring of the provided item/clause/evidence text.",
            "entity_text must be an exact substring of evidence_text.",
            "Do not flatten condition/exception logic into a simple computable leaf.",
            "If the safest output is to keep the original leaf, return abstain=true and leaf=null.",
        ],
        "required_output_json": {
            "candidate_id": f"LLM_candidate_{attempt_no}",
            "abstain": "true or false",
            "leaf": "repaired leaf object, or null if abstain",
            "repair_reason": "brief field-level explanation",
        },
        "output_rules": [
            "Return valid JSON only.",
            "Do not include markdown.",
        ],
    }

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
    ]


def parse_judgment_or_default(parsed: Optional[Dict[str, Any]], candidate_id: str, parse_status: str, parse_error: str) -> Dict[str, Any]:
    if not isinstance(parsed, dict):
        return {
            "candidate_id": candidate_id,
            "scores": {k: 0 for k in BRANCH_B_DEJURE_CRITERIA},
            "average_score": 0.0,
            "pass_threshold": False,
            "critical_errors": [f"judge_parse_failed:{parse_status}:{parse_error}"],
            "critique": f"Judge parse failed: {parse_error}",
            "recommended_action": "human_review",
        }

    scores = parsed.get("scores", {})
    if not isinstance(scores, dict):
        scores = {}

    avg = parsed.get("average_score")
    avg = clamp_score(avg, avg_judge_score(scores))

    crit = critical_errors_from_judgment(parsed)

    return {
        "candidate_id": candidate_id,
        "scores": {k: clamp_score(scores.get(k), 0.0) for k in BRANCH_B_DEJURE_CRITERIA},
        "average_score": avg,
        "pass_threshold": bool(parsed.get("pass_threshold", avg >= 4.5 and not crit)),
        "critical_errors": crit,
        "critique": clean(parsed.get("critique")),
        "recommended_action": clean(parsed.get("recommended_action")) or ("keep" if avg >= 4.5 and not crit else "repair"),
    }


def judge_branch_b_candidate(
    *,
    client: Any,
    model: str,
    plan: Dict[str, Any],
    candidate_id: str,
    candidate_leaf: Dict[str, Any],
    context: Dict[str, Any],
    b_current_leaf: Dict[str, Any],
    max_tokens: int,
    temperature: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    messages = build_branch_b_judge_prompt(
        plan=plan,
        candidate_id=candidate_id,
        candidate_leaf=candidate_leaf,
        context=context,
        b_current_leaf=b_current_leaf,
    )
    raw_text, token_usage = call_llm_json(
        client=client,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    parse_status, parsed, parse_error = parse_json_response(raw_text)
    judgment = parse_judgment_or_default(parsed, candidate_id, parse_status, parse_error)
    trace = {
        "kind": "branch_b_judge",
        "candidate_id": candidate_id,
        "parse_status": parse_status,
        "parse_error": parse_error,
        "messages": messages,
        "raw_response": raw_text,
        "judgment": judgment,
        "token_usage": token_usage,
        "prompt_tokens": token_usage.get("prompt_tokens"),
        "completion_tokens": token_usage.get("completion_tokens"),
        "total_tokens": token_usage.get("total_tokens"),
    }
    return judgment, trace


def repair_branch_b_candidate(
    *,
    client: Any,
    model: str,
    plan: Dict[str, Any],
    attempt_no: int,
    b_current_leaf: Dict[str, Any],
    best_leaf: Dict[str, Any],
    last_judgment: Dict[str, Any],
    context: Dict[str, Any],
    max_tokens: int,
    temperature: float,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    criterion_id = clean(plan.get("criterion_id"))
    messages = build_branch_b_repair_prompt(
        plan=plan,
        attempt_no=attempt_no,
        b_current_leaf=b_current_leaf,
        best_leaf=best_leaf,
        last_judgment=last_judgment,
        context=context,
    )
    raw_text, token_usage = call_llm_json(
        client=client,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    parse_status, parsed, parse_error = parse_json_response(raw_text)

    candidate_leaf = None
    abstain = False
    repair_reason = ""

    if isinstance(parsed, dict):
        abstain = bool(parsed.get("abstain", False))
        repair_reason = clean(parsed.get("repair_reason"))
        if not abstain:
            candidate_leaf = normalize_candidate_leaf(
                parsed.get("leaf"),
                default_leaf=b_current_leaf,
                criterion_id=criterion_id,
            )

    trace = {
        "kind": "branch_b_repair",
        "candidate_id": f"LLM_candidate_{attempt_no}",
        "parse_status": parse_status,
        "parse_error": parse_error,
        "abstain": abstain,
        "repair_reason": repair_reason,
        "messages": messages,
        "raw_response": raw_text,
        "candidate_leaf": candidate_leaf,
        "token_usage": token_usage,
        "prompt_tokens": token_usage.get("prompt_tokens"),
        "completion_tokens": token_usage.get("completion_tokens"),
        "total_tokens": token_usage.get("total_tokens"),
    }
    return candidate_leaf, trace


def branch_b_dejure_result_from_best(
    *,
    plan: Dict[str, Any],
    a_leaf: Dict[str, Any],
    b_leaf: Dict[str, Any],
    context: Dict[str, Any],
    best_source: str,
    best_leaf: Dict[str, Any],
    best_score: float,
    original_score: float,
    candidate_evaluations: List[Dict[str, Any]],
    generated_candidates: List[Dict[str, Any]],
    traces: List[Dict[str, Any]],
    keep_threshold: float,
    candidate_threshold: float,
    min_improvement: float,
) -> Dict[str, Any]:
    """
    Convert the Branch B De Jure-style loop into one downstream-compatible row.

    Commit policy:
      1. If B_current passes the judge threshold, keep it unchanged.
      2. If B_current fails, use the best generated candidate only when it is
         locally valid and improves over B_current by at least min_improvement.
      3. If no generated candidate improves enough, abstain for human_review.

    This is deliberately more faithful to the judge/repair loop than the older
    single-prompt candidate selector, but it does not blindly apply failed or
    schema-invalid generations.
    """
    criterion_id = clean(plan.get("criterion_id"))

    generated_best = best_source != "B_current"
    original_passed = original_score >= keep_threshold
    generated_passed = best_score >= candidate_threshold
    improved_enough = (best_score - original_score) >= min_improvement

    if best_source == "B_current" and original_passed:
        decision = "no_change"
        selected_source = "B_current"
        selected_leaf = strip_internal_fields(b_leaf)
        confidence = "high" if original_score >= 4.75 else "medium"
        reason = (
            f"B_current passed the De Jure-style judge threshold "
            f"({original_score:.2f}/5 >= {keep_threshold:.2f}/5); kept unchanged."
        )
    elif generated_best and generated_passed and improved_enough:
        decision = "select_candidate"
        selected_source = best_source
        selected_leaf = best_leaf
        confidence = "high" if generated_passed else "medium"
        reason = (
            f"B_current failed or was weaker ({original_score:.2f}/5). "
            f"The best generated candidate was {best_source} with score {best_score:.2f}/5, "
            f"improving by at least {min_improvement:.2f}."
        )
    else:
        decision = "human_review"
        selected_source = "none"
        selected_leaf = None
        confidence = "low"
        reason = (
            f"B_current did not pass and no generated candidate improved enough. "
            f"Best={best_source} score {best_score:.2f}/5; "
            f"B_current score {original_score:.2f}/5."
        )

    local_validation_status = "not_applicable"
    local_validation_reasons: List[str] = []

    if decision in {"select_candidate", "no_change"}:
        local_validation_status, local_validation_reasons = validate_leaf_schema_and_grounding(
            selected_leaf,
            context=context,
            a_leaf=a_leaf,
            b_leaf=b_leaf,
        )
        if local_validation_status != "pass":
            decision = "human_review"
            selected_source = "none"
            selected_leaf = None
            confidence = "low"
            reason = reason + " Local validation failed after selection."

    final_decision = decision if decision == "human_review" or local_validation_status == "pass" else "human_review"

    return {
        "plan_id": plan.get("plan_id"),
        "branch_to_update": "B",
        "criterion_id": criterion_id,
        "item_uid": plan.get("item_uid"),
        "clause_id": plan.get("clause_id"),
        "execution_group": plan.get("execution_group"),
        "rescue_strategy": plan.get("rescue_strategy"),
        "rescue_task_type": plan.get("rescue_task_type"),
        "allowed_candidate_sources": ["B_current", "LLM_candidate_1", "LLM_candidate_2", "LLM_candidate_3", "none"],
        "decision": decision,
        "selected_source": selected_source,
        "selected_leaf": selected_leaf,
        "generated_candidates": generated_candidates,
        "candidate_evaluations": candidate_evaluations,
        "supporting_span": clean(best_leaf.get("evidence_text")) if isinstance(best_leaf, dict) else "",
        "selection_reason": reason,
        "confidence": confidence,
        "final_decision": final_decision,
        "local_validation_status": local_validation_status,
        "local_validation_reasons": local_validation_reasons,
        "branch_b_dejure_keep_threshold": keep_threshold,
        "branch_b_dejure_candidate_threshold": candidate_threshold,
        "branch_b_dejure_min_improvement": min_improvement,
        "branch_b_dejure_original_score": original_score,
        "branch_b_dejure_best_score": best_score,
        "branch_b_dejure_trace": traces,
        "A_current_leaf": strip_internal_fields(a_leaf),
        "B_current_leaf": strip_internal_fields(b_leaf),
        "context": {
            "document_id": context.get("document_id"),
            "chia_id": context.get("chia_id"),
            "item_uid": context.get("item_uid"),
            "clause_id": context.get("clause_id"),
            "item_text": context.get("item_text"),
            "clause_text": context.get("clause_text"),
            "evidence_text": context.get("evidence_text"),
        },
    }


def run_branch_b_dejure_rescue(
    *,
    plan: Dict[str, Any],
    a_leaf: Dict[str, Any],
    b_leaf: Dict[str, Any],
    context: Dict[str, Any],
    client: Any,
    model: str,
    max_tokens: int,
    temperature: float,
    max_attempts: int,
    keep_threshold: float,
    candidate_threshold: float,
    min_improvement: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    traces: List[Dict[str, Any]] = []
    generated_candidates: List[Dict[str, Any]] = []
    candidate_evaluations: List[Dict[str, Any]] = []

    current_judgment, trace = judge_branch_b_candidate(
        client=client,
        model=model,
        plan=plan,
        candidate_id="B_current",
        candidate_leaf=b_leaf,
        context=context,
        b_current_leaf=b_leaf,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    traces.append(trace)

    original_score = clamp_score(current_judgment.get("average_score"), 0.0)
    best_source = "B_current"
    best_leaf = strip_internal_fields(b_leaf)
    best_score = original_score
    best_judgment = current_judgment

    candidate_evaluations.append(
        {
            "candidate_id": "B_current",
            "semantic_score": current_judgment["scores"].get("semantic_completeness", 0),
            "grounding_score": current_judgment["scores"].get("entity_grounding", 0),
            "schema_score": 3,
            "average_score_0_to_5": original_score,
            "critical_errors": current_judgment.get("critical_errors", []),
            "main_problems": current_judgment.get("critical_errors", []) or [current_judgment.get("critique", "")],
        }
    )

    # If current B already passes, do not spend repair calls on a likely-correct leaf.
    if original_score >= keep_threshold and not critical_errors_from_judgment(current_judgment):
        row = branch_b_dejure_result_from_best(
            plan=plan,
            a_leaf=a_leaf,
            b_leaf=b_leaf,
            context=context,
            best_source=best_source,
            best_leaf=best_leaf,
            best_score=best_score,
            original_score=original_score,
            candidate_evaluations=candidate_evaluations,
            generated_candidates=generated_candidates,
            traces=traces,
            keep_threshold=keep_threshold,
            candidate_threshold=candidate_threshold,
            min_improvement=min_improvement,
        )
        return row, traces

    max_attempts = max(0, min(3, max_attempts))

    for attempt_no in range(1, max_attempts + 1):
        candidate_leaf, repair_trace = repair_branch_b_candidate(
            client=client,
            model=model,
            plan=plan,
            attempt_no=attempt_no,
            b_current_leaf=b_leaf,
            best_leaf=best_leaf,
            last_judgment=best_judgment,
            context=context,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        traces.append(repair_trace)

        if candidate_leaf is None:
            continue

        local_status, local_reasons = validate_leaf_schema_and_grounding(
            candidate_leaf,
            context=context,
            a_leaf=a_leaf,
            b_leaf=b_leaf,
        )

        generated_candidates.append(
            {
                "candidate_id": f"LLM_candidate_{attempt_no}",
                "leaf": candidate_leaf,
                "reason": repair_trace.get("repair_reason", ""),
                "local_validation_status": local_status,
                "local_validation_reasons": local_reasons,
            }
        )

        if local_status != "pass":
            candidate_evaluations.append(
                {
                    "candidate_id": f"LLM_candidate_{attempt_no}",
                    "semantic_score": 0,
                    "grounding_score": 0,
                    "schema_score": 0,
                    "average_score_0_to_5": 0,
                    "critical_errors": local_reasons,
                    "main_problems": local_reasons,
                }
            )
            continue

        candidate_judgment, judge_trace = judge_branch_b_candidate(
            client=client,
            model=model,
            plan=plan,
            candidate_id=f"LLM_candidate_{attempt_no}",
            candidate_leaf=candidate_leaf,
            context=context,
            b_current_leaf=b_leaf,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        traces.append(judge_trace)

        candidate_score = clamp_score(candidate_judgment.get("average_score"), 0.0)
        candidate_evaluations.append(
            {
                "candidate_id": f"LLM_candidate_{attempt_no}",
                "semantic_score": candidate_judgment["scores"].get("semantic_completeness", 0),
                "grounding_score": candidate_judgment["scores"].get("entity_grounding", 0),
                "schema_score": 3,
                "average_score_0_to_5": candidate_score,
                "critical_errors": candidate_judgment.get("critical_errors", []),
                "main_problems": candidate_judgment.get("critical_errors", []) or [candidate_judgment.get("critique", "")],
            }
        )

        if candidate_score > best_score and not critical_errors_from_judgment(candidate_judgment):
            best_source = f"LLM_candidate_{attempt_no}"
            best_leaf = candidate_leaf
            best_score = candidate_score
            best_judgment = candidate_judgment

        # Once a generated candidate passes and improves enough, further attempts are not needed.
        if (
            best_source != "B_current"
            and best_score >= candidate_threshold
            and (best_score - original_score) >= min_improvement
        ):
            break

    row = branch_b_dejure_result_from_best(
        plan=plan,
        a_leaf=a_leaf,
        b_leaf=b_leaf,
        context=context,
        best_source=best_source,
        best_leaf=best_leaf,
        best_score=best_score,
        original_score=original_score,
        candidate_evaluations=candidate_evaluations,
        generated_candidates=generated_candidates,
        traces=traces,
        keep_threshold=keep_threshold,
        candidate_threshold=candidate_threshold,
        min_improvement=min_improvement,
    )
    return row, traces



# ---------------------------------------------------------------------
# Branch A deterministic hybrid substitution
# ---------------------------------------------------------------------

APPLICABLE_FOR_REUSE = {"select_candidate", "no_change", "mark_partial_or_non_computable"}


def current_leaf_for_selection(leaf: Dict[str, Any], criterion_id: str) -> Dict[str, Any]:
    """
    Return a schema-safe copy of a current AST leaf for selected_leaf output.
    """
    out = strip_internal_fields(leaf)
    out["criterion_id"] = criterion_id

    # Avoid null negation values in downstream full-AST validation.
    if "negated" in out:
        out["negated"] = bool(out.get("negated", False))

    out["temporal_context"] = normalize_temporal_context(out.get("temporal_context"))

    if out.get("value_type") == "null":
        out["value"] = None

    if "non_computable_reason" not in out:
        out["non_computable_reason"] = None

    return out


def result_has_valid_selected_leaf(result: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(result, dict):
        return False
    if clean(result.get("final_decision")) not in APPLICABLE_FOR_REUSE:
        return False
    if clean(result.get("local_validation_status")) != "pass":
        return False
    return isinstance(result.get("selected_leaf"), dict)


def branch_a_substitution_result(
    *,
    plan: Dict[str, Any],
    a_leaf: Dict[str, Any],
    b_leaf: Dict[str, Any],
    context: Dict[str, Any],
    b_result: Optional[Dict[str, Any]],
    b_requires_dejure: bool,
) -> Dict[str, Any]:
    """
    Branch A rescue is deterministic and cheap:
      - no LLM call;
      - substitute Branch B;
      - if Branch B was in the De Jure subset, use the validated Branch B
        De Jure result instead of raw B_current;
      - if no valid Branch B substitute exists, send to human_review.
    """
    criterion_id = clean(plan.get("criterion_id"))
    validation_reasons: List[str] = []

    selected_leaf: Optional[Dict[str, Any]] = None
    selected_source = "none"
    reason = ""
    confidence = "medium"

    if b_requires_dejure:
        if result_has_valid_selected_leaf(b_result):
            selected_leaf = deepcopy(b_result.get("selected_leaf"))
            selected_leaf["criterion_id"] = criterion_id
            if "negated" in selected_leaf:
                selected_leaf["negated"] = bool(selected_leaf.get("negated", False))
            selected_leaf["temporal_context"] = normalize_temporal_context(selected_leaf.get("temporal_context"))
            selected_source = "B_current" if clean(b_result.get("selected_source")) == "B_current" else "B_dejure_best"
            reason = (
                "Branch A was flagged; corresponding Branch B was also routed to De Jure rescue. "
                f"Using validated Branch B result from source={clean(b_result.get('selected_source'))}."
            )
            confidence = clean(b_result.get("confidence")) or "medium"
        else:
            # Fallback for Branch A:
            # Branch A is the BERT/rules branch, so its rescue is semantic substitution
            # from Branch B. If Branch B De Jure repair did not produce a validated
            # improved candidate, still allow raw B_current as the A substitute,
            # as long as it passes local validation below.
            if isinstance(b_leaf, dict) and b_leaf:
                selected_leaf = current_leaf_for_selection(b_leaf, criterion_id)
                selected_source = "B_current"
                reason = (
                    "Branch A was flagged; corresponding Branch B was routed to De Jure rescue, "
                    "but no validated De Jure replacement was available. Falling back to raw "
                    "B_current as the semantic substitute for Branch A, subject to local validation."
                )
                confidence = "medium"
            else:
                reason = (
                    "Branch A was flagged, but matching Branch B leaf was missing."
                )
    else:
        if isinstance(b_leaf, dict) and b_leaf:
            selected_leaf = current_leaf_for_selection(b_leaf, criterion_id)
            selected_source = "B_current"
            reason = (
                "Branch A was flagged; Branch B was not in the De Jure rescue subset. "
                "Using Branch B current leaf as the semantic substitute without a new Branch A LLM call."
            )
        else:
            reason = "Branch A was flagged, but matching Branch B leaf was missing."

    if selected_leaf is not None:
        status, reasons = validate_leaf_schema_and_grounding(
            selected_leaf,
            context=context,
            a_leaf=a_leaf,
            b_leaf=b_leaf,
        )
        validation_reasons.extend(reasons)
        local_validation_status = "pass" if status == "pass" and not validation_reasons else "fail"
    else:
        local_validation_status = "not_applicable"

    if selected_leaf is not None and local_validation_status == "pass":
        decision = "select_candidate"
        final_decision = "select_candidate"
    else:
        decision = "human_review"
        final_decision = "human_review"
        if selected_leaf is None:
            selected_source = "none"
        confidence = "low"

    return {
        "plan_id": plan.get("plan_id"),
        "branch_to_update": "A",
        "criterion_id": criterion_id,
        "item_uid": plan.get("item_uid"),
        "clause_id": plan.get("clause_id"),
        "execution_group": plan.get("execution_group"),
        "rescue_strategy": plan.get("rescue_strategy"),
        "rescue_task_type": plan.get("rescue_task_type"),
        "allowed_candidate_sources": ["B_current", "B_dejure_best", "none"],
        "decision": decision,
        "selected_source": selected_source,
        "selected_leaf": selected_leaf if final_decision == "select_candidate" else None,
        "generated_candidates": [],
        "candidate_evaluations": [],
        "supporting_span": clean(selected_leaf.get("evidence_text")) if isinstance(selected_leaf, dict) else "",
        "selection_reason": reason,
        "confidence": confidence,
        "final_decision": final_decision,
        "local_validation_status": local_validation_status,
        "local_validation_reasons": validation_reasons,
        "branch_a_rescue_mode": "deterministic_branch_b_substitution",
        "branch_a_b_requires_dejure": int(b_requires_dejure),
        "branch_a_used_b_result_plan_id": clean(b_result.get("plan_id")) if isinstance(b_result, dict) else "",
        "A_current_leaf": strip_internal_fields(a_leaf),
        "B_current_leaf": strip_internal_fields(b_leaf),
        "context": {
            "document_id": context.get("document_id"),
            "chia_id": context.get("chia_id"),
            "item_uid": context.get("item_uid"),
            "clause_id": context.get("clause_id"),
            "item_text": context.get("item_text"),
            "clause_text": context.get("clause_text"),
            "evidence_text": context.get("evidence_text"),
        },
    }


# ---------------------------------------------------------------------
# Existing result handling
# ---------------------------------------------------------------------

def load_existing_results(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for row in load_jsonl(path):
        plan_id = clean(row.get("plan_id"))
        if plan_id:
            out[plan_id] = row
    return out


def select_plan_rows(
    plan_rows: List[Dict[str, Any]],
    existing: Dict[str, Dict[str, Any]],
    branch: Optional[str],
    limit: Optional[int],
    overwrite: bool,
    criterion_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    wanted_ids = {clean(x) for x in (criterion_ids or []) if clean(x)}

    for row in plan_rows:
        plan_id = clean(row.get("plan_id"))
        row_branch = clean(row.get("branch_to_update"))
        row_criterion_id = clean(row.get("criterion_id"))

        if branch and row_branch != branch:
            continue

        if wanted_ids and row_criterion_id not in wanted_ids:
            continue

        if not overwrite and plan_id in existing:
            continue

        rows.append(row)

    if limit is not None:
        rows = rows[:limit]

    return rows


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--branch", choices=["A", "B"], default=None)
    parser.add_argument(
        "--criterion-ids",
        nargs="+",
        default=None,
        help="Optional list of criterion_id values to process, e.g. NCT..._C1 NCT..._C2",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2500)
    parser.add_argument("--sleep", type=float, default=0.2)
    # Kept for backward compatibility. Branch B De Jure is always enabled.
    parser.add_argument("--branch-b-dejure", action="store_true")
    parser.add_argument("--branch-b-max-attempts", type=int, default=3)
    parser.add_argument("--branch-b-keep-threshold", type=float, default=4.5)
    parser.add_argument("--branch-b-candidate-threshold", type=float, default=4.0)
    parser.add_argument("--branch-b-min-improvement", type=float, default=0.25)

    # Backward compatibility. Avoid using this in the final thesis runs.
    parser.add_argument("--branch-b-judge-threshold", type=float, default=None)
    args = parser.parse_args()

    if args.branch_b_judge_threshold is not None:
        args.branch_b_keep_threshold = args.branch_b_judge_threshold

    ROOT = Path(__file__).resolve().parents[3]

    plan_path = (
        ROOT
        / "outputs"
        / "verification"
        / "layer3"
        / "rescue_plan"
        / "main_llm_rescue_plan.jsonl"
    )

    branch_a_ast_path = (
        ROOT
        / "outputs"
        / "verification"
        / "layer3"
        / "safe_structural_repairs"
        / "chia_text_only_200_rules_v3_ast_A_layer3_safe_structural.jsonl"
    )

    branch_b_ast_path = (
        ROOT
        / "outputs"
        / "verification"
        / "layer3"
        / "safe_structural_repairs"
        / "chia_text_only_200_rules_v3_ast_B_layer3_safe_structural.jsonl"
    )

    pass2_inputs_path = (
        ROOT
        / "outputs"
        / "extraction"
        / "pass2_inputs"
        / "chia_text_only_200_pass2_inputs.jsonl"
    )

    out_root = (
        ROOT
        / "outputs"
        / "verification"
        / "layer3"
        / "candidate_selection_rescue"
    )

    out_root.mkdir(parents=True, exist_ok=True)

    out_results_jsonl = out_root / "candidate_selection_rescue_results.jsonl"
    out_results_csv = out_root / "candidate_selection_rescue_results.csv"
    out_prompts_jsonl = out_root / "candidate_selection_rescue_raw_prompts.jsonl"
    out_summary_json = out_root / "candidate_selection_rescue_summary.json"

    print("\nLayer 3 candidate-selection rescue")
    print("Branch A rule tree:", branch_a_ast_path)
    print("Branch B rule tree:", branch_b_ast_path)
    print("Branch B AST:", branch_b_ast_path)
    print("Pass2 inputs:", pass2_inputs_path)
    print("Branch A policy: deterministic Branch-B substitution; no Branch A LLM calls.")
    print("Branch B policy: De Jure-style judge/repair loop; Branch B only.")

    plan_rows = load_jsonl(plan_path)
    a_leaf_index = build_leaf_index(load_jsonl(branch_a_ast_path))
    b_leaf_index = build_leaf_index(load_jsonl(branch_b_ast_path))
    ctx_index = build_pass2_context_index(load_jsonl(pass2_inputs_path))

    existing = {} if args.overwrite else load_existing_results(out_results_jsonl)
    todo = select_plan_rows(
        plan_rows=plan_rows,
        existing=existing,
        branch=args.branch,
        limit=args.limit,
        overwrite=args.overwrite,
        criterion_ids=args.criterion_ids,
    )
    if args.criterion_ids:
        wanted = {clean(x) for x in args.criterion_ids if clean(x)}
        found = {clean(r.get("criterion_id")) for r in todo}

        print("Criterion-id filter requested:", sorted(wanted))
        print("Criterion-id filter selected:", sorted(found))

        missing = wanted - found
        extra = found - wanted

        if missing:
            raise RuntimeError(
                f"Requested criterion_ids not found in selected plan rows: {sorted(missing)}"
            )

        if extra:
            raise RuntimeError(
                f"Unexpected criterion_ids selected: {sorted(extra)}"
            )

        if len(todo) != len(wanted):
            raise RuntimeError(
                f"Expected exactly {len(wanted)} selected rows, but got {len(todo)}."
            )
    # Branch A may depend on Branch B rescue results. Process Branch B first in full runs.
    todo = sorted(
        todo,
        key=lambda r: (
            0 if clean(r.get("branch_to_update")) == "B" else 1,
            to_int(r.get("execution_priority"), 999),
            clean(r.get("criterion_id")),
        ),
    )

    branch_b_rescue_criteria = {
        clean(r.get("criterion_id"))
        for r in plan_rows
        if clean(r.get("branch_to_update")) == "B"
    }

    branch_b_results_by_criterion: Dict[str, Dict[str, Any]] = {
        clean(r.get("criterion_id")): r
        for r in existing.values()
        if clean(r.get("branch_to_update")) == "B" and clean(r.get("criterion_id"))
    }

    needs_llm = (not args.dry_run) and any(clean(r.get("branch_to_update")) == "B" for r in todo)

    if args.dry_run:
        client = None
        model = "DRY_RUN"
    elif needs_llm:
        client, model = make_client()
        print("Layer 3F.2 model:", model)
    else:
        client = None
        model = "NO_LLM_REQUIRED"
        print("Layer 3F.2 model:", model)

    results: List[Dict[str, Any]] = []
    raw_prompts: List[Dict[str, Any]] = []

    for i, plan in enumerate(todo, start=1):
        plan_id = clean(plan.get("plan_id"))
        criterion_id = clean(plan.get("criterion_id"))
        branch = clean(plan.get("branch_to_update"))

        print(f"[{i}/{len(todo)}] Rescue/check: {plan_id} | {branch}")

        a_leaf = a_leaf_index.get(criterion_id, {})
        b_leaf = b_leaf_index.get(criterion_id, {})
        context = get_context_for_plan(plan, ctx_index)

        if not a_leaf or not b_leaf:
            missing = []
            if not a_leaf:
                missing.append("missing_A_leaf")
            if not b_leaf:
                missing.append("missing_B_leaf")
            row = error_result(plan, ";".join(missing), a_leaf, b_leaf, context)
            row["llm_model"] = model
            row["parse_status"] = "missing_leaf"
            results.append(row)
            continue

        if branch == "A":
            raw_prompts.append(
                {
                    "plan_id": plan_id,
                    "branch_to_update": branch,
                    "criterion_id": criterion_id,
                    "mode": "deterministic_branch_b_substitution_no_llm",
                    "b_requires_dejure": int(criterion_id in branch_b_rescue_criteria),
                }
            )

            if args.dry_run:
                continue

            b_result = branch_b_results_by_criterion.get(criterion_id)
            row = branch_a_substitution_result(
                plan=plan,
                a_leaf=a_leaf,
                b_leaf=b_leaf,
                context=context,
                b_result=b_result,
                b_requires_dejure=criterion_id in branch_b_rescue_criteria,
            )
            row["raw_response"] = "NO_LLM_BRANCH_A_DETERMINISTIC_B_SUBSTITUTION"
            row["parse_status"] = "deterministic_branch_a_substitution"
            row["llm_model"] = "NO_LLM_BRANCH_A"
            results.append(row)
            continue

        if branch == "B":
            raw_prompts.append(
                {
                    "plan_id": plan_id,
                    "branch_to_update": branch,
                    "criterion_id": criterion_id,
                    "mode": "branch_b_dejure_judge_repair_loop",
                    "max_attempts": args.branch_b_max_attempts,
                    "keep_threshold": args.branch_b_keep_threshold,
                    "candidate_threshold": args.branch_b_candidate_threshold,
                    "min_improvement": args.branch_b_min_improvement,
                }
            )

            if args.dry_run:
                continue

            raw_text = ""
            try:
                row, dejure_traces = run_branch_b_dejure_rescue(
                    plan=plan,
                    a_leaf=a_leaf,
                    b_leaf=b_leaf,
                    context=context,
                    client=client,
                    model=model,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    max_attempts=args.branch_b_max_attempts,
                    keep_threshold=args.branch_b_keep_threshold,
                    candidate_threshold=args.branch_b_candidate_threshold,
                    min_improvement=args.branch_b_min_improvement,
                )
                row["raw_response"] = json.dumps(dejure_traces, ensure_ascii=False)
                row["parse_status"] = "dejure_loop"
                row["llm_model"] = model
                row.update(sum_trace_token_usage(dejure_traces))
                results.append(row)
                branch_b_results_by_criterion[criterion_id] = row
            except Exception as exc:
                row = error_result(plan, f"llm_call_failed:{type(exc).__name__}:{exc}", a_leaf, b_leaf, context)
                row["raw_response"] = raw_text
                row["llm_model"] = model
                row["parse_status"] = "dejure_error"
                results.append(row)
                branch_b_results_by_criterion[criterion_id] = row

            time.sleep(args.sleep)
            continue

        row = error_result(plan, f"unknown_branch:{branch}", a_leaf, b_leaf, context)
        row["llm_model"] = model
        row["parse_status"] = "unknown_branch"
        results.append(row)

    if args.overwrite:
        merged = results
    else:
        merged_by_id = dict(existing)
        for row in results:
            pid = clean(row.get("plan_id"))
            if pid:
                merged_by_id[pid] = row
        merged = list(merged_by_id.values())

    write_jsonl(out_results_jsonl, merged)
    write_csv(out_results_csv, merged)
    write_jsonl(out_prompts_jsonl, raw_prompts)

    validation_reasons = Counter()
    for row in merged:
        for reason in row.get("local_validation_reasons", []) or []:
            validation_reasons[reason] += 1

    token_summary = {
        "layer3_llm_calls": sum(int(r.get("layer3_llm_calls", 0) or 0) for r in merged),
        "layer3_prompt_tokens": sum(int(r.get("layer3_prompt_tokens", 0) or 0) for r in merged),
        "layer3_completion_tokens": sum(int(r.get("layer3_completion_tokens", 0) or 0) for r in merged),
        "layer3_total_tokens": sum(int(r.get("layer3_total_tokens", 0) or 0) for r in merged),
    }

    summary = {
        "stage": "06f2_branch_specific_hybrid_and_dejure_rescue",
        "token_usage": token_summary,
        "description": (
            "Branch-specific Layer 3 rescue. Manual labels are not used. "
            "Branch A makes no LLM calls and uses Branch B as semantic substitute. "
            "Branch B uses a De Jure-style judge/repair loop: judge B_current, repair up to three attempts if needed, keep the best validated candidate."
        ),
        "inputs": {
            "plan": str(plan_path),
            "branch_a_ast": str(branch_a_ast_path),
            "branch_b_ast": str(branch_b_ast_path),
            "pass2_inputs": str(pass2_inputs_path),
        },
        "outputs": {
            "results_jsonl": str(out_results_jsonl),
            "results_csv": str(out_results_csv),
            "raw_prompts_jsonl": str(out_prompts_jsonl),
            "summary_json": str(out_summary_json),
        },
        "run_settings": {
            "limit": args.limit,
            "branch": args.branch,
            "dry_run": args.dry_run,
            "overwrite": args.overwrite,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "sleep": args.sleep,
            "branch_b_dejure_always_enabled": True,
            "branch_b_max_attempts": args.branch_b_max_attempts,
            "branch_b_keep_threshold": args.branch_b_keep_threshold,
            "branch_b_candidate_threshold": args.branch_b_candidate_threshold,
            "branch_b_min_improvement": args.branch_b_min_improvement,
        },
        "counts": {
            "plan_rows_total": len(plan_rows),
            "selected_todo_rows": len(todo),
            "processed_this_run": len(results),
            "total_results": len(merged),
            "branch_b_rescue_criteria_in_plan": len(branch_b_rescue_criteria),
            "counts_by_branch_to_update": dict(Counter(clean(r.get("branch_to_update")) for r in merged)),
            "counts_by_final_decision": dict(Counter(clean(r.get("final_decision")) for r in merged)),
            "counts_by_selected_source": dict(Counter(clean(r.get("selected_source")) for r in merged)),
            "counts_by_validation_status": dict(Counter(clean(r.get("local_validation_status")) for r in merged)),
            "counts_by_parse_status": dict(Counter(clean(r.get("parse_status")) for r in merged)),
            "validation_reasons": dict(validation_reasons.most_common()),
        },
    }

    write_json(out_summary_json, summary)

    print("\nDONE")
    print("Results JSONL:", out_results_jsonl)
    print("Results CSV:", out_results_csv)
    print("Raw prompts JSONL:", out_prompts_jsonl)
    print("Summary JSON:", out_summary_json)

    print("\nProcessed this run:", len(results))
    print("Total results:", len(merged))
    print("Counts by branch:", summary["counts"]["counts_by_branch_to_update"])
    print("Final decision counts:", summary["counts"]["counts_by_final_decision"])
    print("Selected source counts:", summary["counts"]["counts_by_selected_source"])
    print("Validation status counts:", summary["counts"]["counts_by_validation_status"])
    print("Parse status counts:", summary["counts"]["counts_by_parse_status"])
    if validation_reasons:
        print("Validation reasons:", dict(validation_reasons.most_common()))

    if args.dry_run:
        print("Dry run completed. No LLM calls were made and no new results were processed.")


if __name__ == "__main__":
    main()

# Run from the repository root:
# python scripts/03_verification/03_layer3/07_run_candidate_selection_rescue.py --limit 5 --dry-run
# python scripts/03_verification/03_layer3/07_run_candidate_selection_rescue.py --branch B --limit 5 --overwrite
# python scripts/03_verification/03_layer3/07_run_candidate_selection_rescue.py --overwrite