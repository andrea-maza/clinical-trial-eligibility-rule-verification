import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from jsonschema import Draft7Validator, ValidationError
from openai import AzureOpenAI, APIError, APITimeoutError, APIConnectionError, RateLimitError, BadRequestError


# --------------------------------------------------
# Config
# --------------------------------------------------

SMOKE_TEST_N = None  # set to None after checking outputs

# For clean Branch B comparison, keep this False.
# If True, the LLM also sees BERT candidates as hints.
INCLUDE_BERT_HINTS = False


# --------------------------------------------------
# IO helpers
# --------------------------------------------------

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


def load_existing_done_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()

    done = set()
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("status") == "ok" and rec.get("item_uid"):
                    done.add(rec["item_uid"])
            except Exception:
                continue
    return done

def coerce_string_null(value):
    if isinstance(value, str) and value.strip().lower() in {"null", "none", ""}:
        return None
    return value
# --------------------------------------------------
# JSON cleaning
# --------------------------------------------------

def clean_json_text(raw: str) -> str:
    raw = (raw or "").strip()

    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    return raw


# --------------------------------------------------
# Prompt input builder
# --------------------------------------------------

def simplify_clause_for_prompt(clause: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "clause_id": clause.get("clause_id"),
        "clause_text": clause.get("clause_text"),
        "evidence_text": clause.get("evidence_text"),
        "is_negated": clause.get("is_negated"),
        "connector_to_next": clause.get("connector_to_next"),
        "quantifier": clause.get("quantifier"),
    }

    if INCLUDE_BERT_HINTS:
        out["bert_candidates"] = clause.get("bert_candidates", {"anchors": [], "supports": []})

    return out


def build_user_content(payload: Dict[str, Any]) -> str:
    clauses = [simplify_clause_for_prompt(c) for c in payload.get("clauses", [])]

    request_obj = {
        "trial_id": payload.get("trial_id"),
        "item_uid": payload.get("item_uid"),
        "chia_id": payload.get("chia_id"),
        "document_id": payload.get("document_id"),
        "criterion_type": payload.get("criterion_type"),
        "item_text": payload.get("item_text"),
        "clauses": clauses,
    }

    return json.dumps(request_obj, ensure_ascii=False, indent=2)

ALLOWED_HISTORY_CONTEXT = {
    "current",
    "prior",
    "previously_treated",
    "stable_dose",
    "investigational_use",
    "other",
    None,
}

TEMPORAL_RELATIONS = {"before", "after", "during", "within", "since"}

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


def normalize_temporal_operator_misuse(criterion: Dict[str, Any]) -> None:
    """
    Fix cases where the LLM incorrectly puts temporal relations
    such as 'within' or 'before' in operator.
    """
    op = criterion.get("operator")

    if op not in TEMPORAL_RELATIONS:
        return

    # Move temporal meaning into temporal_context if needed.
    if criterion.get("temporal_context") is None:
        criterion["temporal_context"] = {
            "relation": op,
            "value": criterion.get("value"),
            "unit": criterion.get("unit"),
            "anchor_event": "other",
        }

    # Keep clinical criterion as existence-based.
    criterion["operator"] = "exists"
    criterion["value_type"] = "null"
    criterion["value"] = None
    criterion["unit"] = None


def normalize_temporal_context(criterion: Dict[str, Any]) -> None:
    tc = coerce_string_null(criterion.get("temporal_context"))

    # temporal_context must be either a dict or None.
    # If the LLM returns an invalid string like "partial",
    # keep the uncertainty in computability/non_computable_reason,
    # but make temporal_context schema-valid.
    if not isinstance(tc, dict):
        criterion["temporal_context"] = None
        return

    criterion["temporal_context"] = tc

    tc["value"] = coerce_string_null(tc.get("value"))
    tc["unit"] = coerce_string_null(tc.get("unit"))
    tc["anchor_event"] = coerce_string_null(tc.get("anchor_event"))

    if tc.get("unit") not in ALLOWED_TEMPORAL_UNITS:
        tc["unit"] = None

    if tc.get("anchor_event") not in ALLOWED_ANCHOR_EVENTS:
        tc["anchor_event"] = "other"

    if tc.get("relation") not in TEMPORAL_RELATIONS:
        tc["relation"] = "during"


def normalize_history_context(criterion: Dict[str, Any]) -> None:
    hc = coerce_string_null(criterion.get("history_context"))

    if hc not in ALLOWED_HISTORY_CONTEXT:
        # Example: LLM returns "during", which is temporal, not history.
        hc = "other"

    criterion["history_context"] = hc
# --------------------------------------------------
# Post-processing
# --------------------------------------------------

def normalize_pass2_output(
    parsed: Dict[str, Any],
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Force metadata and criterion_id/evidence_text to stay aligned with Pass 1.
    This prevents the LLM from drifting.
    """
    item_uid = payload["item_uid"]
    clauses = payload.get("clauses", [])
    clauses_by_id = {c["clause_id"]: c for c in clauses}
    expected_clause_ids = [c["clause_id"] for c in clauses]

    parsed["trial_id"] = payload.get("trial_id")
    parsed["item_uid"] = item_uid
    parsed["chia_id"] = payload.get("chia_id")
    parsed["document_id"] = payload.get("document_id")
    parsed["criterion_type"] = payload.get("criterion_type")

    criteria = parsed.get("criteria")
    if not isinstance(criteria, list):
        raise ValueError("Missing or invalid criteria list.")

    seen = set()
    normalized_criteria = []

    for entry in criteria:
        clause_id = entry.get("clause_id")
        if clause_id not in clauses_by_id:
            raise ValueError(f"Unexpected clause_id from model: {clause_id}")

        if clause_id in seen:
            raise ValueError(f"Duplicate clause_id from model: {clause_id}")
        seen.add(clause_id)

        criterion = entry.get("criterion")
        if not isinstance(criterion, dict):
            raise ValueError(f"Missing criterion object for clause_id={clause_id}")

        clause = clauses_by_id[clause_id]

        # Force stable identifiers and evidence grounding.
        criterion["criterion_id"] = f"{item_uid}_{clause_id}"
        criterion["evidence_text"] = clause.get("evidence_text", "")

        # Fill optional fields so downstream scripts see a stable shape.
        criterion.setdefault("normalized_concept", None)
        criterion.setdefault("value", None)
        criterion.setdefault("unit", None)
        criterion.setdefault("temporal_context", None)
        criterion.setdefault("non_computable_reason", None)
        criterion.setdefault("provenance", None)
        criterion.setdefault("history_context", None)

        # Convert accidental string "null" to real JSON null.
        for key in [
            "normalized_concept",
            "value",
            "unit",
            "temporal_context",
            "non_computable_reason",
            "provenance",
            "history_context",
        ]:
            criterion[key] = coerce_string_null(criterion.get(key))
        
        # Fix common LLM schema slips before validation.
        normalize_temporal_operator_misuse(criterion)
        normalize_temporal_context(criterion)
        normalize_history_context(criterion)

        if isinstance(criterion.get("temporal_context"), dict):
            tc = criterion["temporal_context"]
            tc["value"] = coerce_string_null(tc.get("value"))
            tc["unit"] = coerce_string_null(tc.get("unit"))
            tc["anchor_event"] = coerce_string_null(tc.get("anchor_event"))

        # Normalize common consistency issues.
        if criterion.get("value_type") == "null":
            criterion["value"] = None

        if criterion.get("computability") == "non_computable":
            if not criterion.get("non_computable_reason"):
                criterion["non_computable_reason"] = "Marked as non-computable by LLM Pass 2."

        normalized_criteria.append(
            {
                "clause_id": clause_id,
                "criterion": criterion,
            }
        )

    missing = set(expected_clause_ids) - seen
    if missing:
        raise ValueError(f"LLM omitted clause_ids: {sorted(missing)}")

    # Keep the original Pass 1 order.
    normalized_criteria = sorted(
        normalized_criteria,
        key=lambda x: expected_clause_ids.index(x["clause_id"]),
    )

    parsed["criteria"] = normalized_criteria
    return parsed

def is_content_filter_error(e) -> bool:
    msg = str(e).lower()
    return (
        "content_filter" in msg
        or "responsibleaipolicyviolation" in msg
        or "content management policy" in msg
    )


def call_llm_with_retry(client, model: str, system_prompt: str, user_content: str, max_attempts: int = 3):
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            return client.with_options(timeout=90.0).chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0,
            )

        except BadRequestError as e:
            # Content filter errors are deterministic for the same prompt.
            # Retrying wastes time, so mark this item as error and continue.
            if is_content_filter_error(e):
                raise RuntimeError(f"CONTENT_FILTER_BLOCKED: {e}")
            raise

        except (APITimeoutError, APIConnectionError, RateLimitError, APIError) as e:
            last_error = e
            wait_sec = min(2 ** attempt, 20)
            print(f"[RETRY] attempt {attempt}/{max_attempts} failed: {type(e).__name__}: {e}")
            time.sleep(wait_sec)

    raise RuntimeError(f"LLM call failed after {max_attempts} attempts. Last error: {last_error}")

# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    ROOT = Path(__file__).resolve().parents[3]

    in_path = ROOT / "outputs" / "extraction" / "pass2_inputs" / "chia_text_only_200_pass2_inputs.jsonl"
    schema_path = ROOT / "schemas" / "rules_v3_pass2_leaf.json"
    prompt_path = ROOT / "prompts" / "pass2_llm_leaf_prompt.txt"

    out_dir = ROOT / "outputs" / "extraction" / "branch_b" / "pass2_leaves_llm"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "chia_text_only_200_pass2_leaves_llm.jsonl"

    load_dotenv(ROOT / ".env")

    api_base = os.getenv("UMGPT_API_BASE")
    api_version = os.getenv("UMGPT_API_VERSION")
    api_key = os.getenv("UMGPT_API_KEY")
    shortcode = os.getenv("UMGPT_SHORTCODE")
    default_model = os.getenv("UMGPT_MODEL", "gpt-5.2")
    pass2_model = os.getenv("UMGPT_MODEL_PASS2", default_model)
    print("Pass 2 model:", pass2_model)

    if not all([api_base, api_version, api_key, shortcode]):
        raise RuntimeError("Missing UMGPT_* env vars. Check your .env file.")

    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=api_base,
        organization=shortcode,
    )

    system_prompt = prompt_path.read_text(encoding="utf-8")

    schema = load_schema(schema_path)
    validator = Draft7Validator(schema)

    rows = load_jsonl(in_path)
    rows = [r for r in rows if r.get("status") == "ok"]

    if SMOKE_TEST_N is not None:
        rows = rows[:SMOKE_TEST_N]

    done_ids = load_existing_done_ids(out_path)

    print("Input rows:", len(rows))
    print("Already done:", len(done_ids))
    print("Writing to:", out_path)
    print("Include BERT hints:", INCLUDE_BERT_HINTS)

    n_ok = 0
    n_err = 0

    with open(out_path, "a", encoding="utf-8") as f_out:
        for idx, row in enumerate(rows, start=1):
            payload = row["pass2_input"]
            item_uid = payload["item_uid"]

            if item_uid in done_ids:
                continue

            t0 = time.time()
            raw_output = None
            parsed = None
            status = "ok"
            error = None
            prompt_tokens = None
            completion_tokens = None
            total_tokens = None

            try:
                user_content = build_user_content(payload)

                print("Input chars:", len(user_content))
                print(f"[{idx}/{len(rows)}] Calling LLM for {item_uid}...")
                resp = call_llm_with_retry(
                    client=client,
                    model=pass2_model,
                    system_prompt=system_prompt,
                    user_content=user_content,
                    max_attempts=3,
                )
                print(f"[{idx}/{len(rows)}] Done in {time.time()-t0:.1f}s")
                print("Pass 2 model:", pass2_model)

                raw_output = resp.choices[0].message.content or ""
                cleaned = clean_json_text(raw_output)
                parsed = json.loads(cleaned)
                parsed = normalize_pass2_output(parsed, payload)

                errors = list(validator.iter_errors(parsed))
                if errors:
                    msg = "; ".join(e.message for e in errors)
                    raise ValidationError(msg)

                if resp.usage is not None:
                    prompt_tokens = getattr(resp.usage, "prompt_tokens", None)
                    completion_tokens = getattr(resp.usage, "completion_tokens", None)
                    total_tokens = getattr(resp.usage, "total_tokens", None)

                n_ok += 1

            except Exception as e:
                status = "error"
                error = str(e)
                n_err += 1

                if n_err <= 5:
                    print(f"[ERROR] item_uid={item_uid}: {error}")

            rec = {
                "dataset": "CHIA",
                "stage": "pass2_llm_leaf_extraction",
                "branch": "B_llm_pass2",
                "item_uid": item_uid,
                "chia_id": payload.get("chia_id"),
                "document_id": payload.get("document_id"),
                "status": status,
                "error": error,
                "model": pass2_model,
                "temperature": 0,
                "include_bert_hints": INCLUDE_BERT_HINTS,
                "prompt_id": "pass2_llm_leaf_prompt",
                "schema_id": "rules_v3_pass2_leaf",
                "pass2_source": "chia_text_only_200_pass2_inputs.jsonl",
                "raw_output": raw_output,
                "pass2_output": parsed,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "timestamp": time.time(),
                "latency_sec": round(time.time() - t0, 3),
            }

            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f_out.flush()

            if idx % 10 == 0:
                print(f"Processed {idx}/{len(rows)} | ok={n_ok} err={n_err}")

    print("DONE | ok=", n_ok, "err=", n_err)


if __name__ == "__main__":
    main()


# Run from the repository root: #
#  python scripts/02_extraction/branch_b/04_extract_pass2_llm_leaves.py