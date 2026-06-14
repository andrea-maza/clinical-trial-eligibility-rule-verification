import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from jsonschema import Draft7Validator, ValidationError
from openai import AzureOpenAI

s
def split_top_level_items_with_offsets(text: str) -> list[dict]:
    """
    Split one CHIA row text into top-level items and keep char offsets
    relative to the original full_text.
    """
    items = []

    for match in re.finditer(r"[^\n]+", text):
        raw_line = match.group(0)
        raw_start = match.start()

        if not raw_line.strip():
            continue

        # remove leading whitespace
        leading_ws = len(raw_line) - len(raw_line.lstrip())
        start_char = raw_start + leading_ws
        content = raw_line.lstrip()

        # remove simple bullet/number prefixes if present
        bullet_match = re.match(r"(?:[-•*]|\d+[\.\)])\s*", content)
        if bullet_match:
            start_char += bullet_match.end()
            content = content[bullet_match.end():]

        # trim trailing whitespace only
        content = content.rstrip()

        if not content:
            continue

        end_char = start_char + len(content)

        items.append(
            {
                "text": content,
                "start_char": start_char,
                "end_char": end_char,
            }
        )

    return items

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_existing_done_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()

    done = set()
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("status") == "ok":
                    item_uid = rec.get("item_uid")
                    if item_uid:
                        done.add(item_uid)
            except Exception:
                continue
    return done


def load_schema(schema_path: Path) -> dict:
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def strip_markdown_fences(text: str) -> str:
    """
    Remove optional ```json ... ``` fences if the model adds them.
    """
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    if text.lower().startswith("json"):
        text = text[4:].lstrip()

    return text


def assign_clause_ids(parsed_obj: dict) -> dict:
    """
    Add local clause IDs after schema validation.
    The LLM does not need to generate them.
    """
    enriched = json.loads(json.dumps(parsed_obj))  # deep copy

    clauses = enriched.get("clauses", [])
    for i, clause in enumerate(clauses, start=1):
        clause["clause_id"] = f"C{i}"

    return enriched


def run_extra_sanity_checks(parsed_obj: dict, item_text: str) -> None:
    """
    Checks beyond schema validation.
    """
    clauses = parsed_obj.get("clauses", [])
    if not clauses:
        raise ValueError("No clauses returned.")

    if clauses[-1].get("connector_to_next") is not None:
        raise ValueError("Last clause must have connector_to_next = null.")

    for i, clause in enumerate(clauses, start=1):
        evidence = str(clause.get("evidence_text", "")).strip()

        if not evidence:
            raise ValueError(f"Clause {i} has empty evidence_text.")

        if evidence not in item_text:
            raise ValueError(
                f"Clause {i} evidence_text is not an exact substring of item_text."
            )


def main():
    ROOT = Path(__file__).resolve().parents[2]

    in_path = ROOT / "data" / "processed" / "chia_text_only_200.jsonl"
    prompt_path = ROOT / "prompts" / "pass1_flat_prompt.txt"
    schema_path = ROOT / "schemas" / "rules_v3_pass1_flat.json"

    out_dir = ROOT / "outputs" / "extraction" / "pass1_flat"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "chia_text_only_200_pass1_flat.jsonl"

    smoke_test_n = None  # set to None to run all rows

    load_dotenv(ROOT / ".env")
    api_base = os.getenv("UMGPT_API_BASE")
    api_version = os.getenv("UMGPT_API_VERSION")
    api_key = os.getenv("UMGPT_API_KEY")
    shortcode = os.getenv("UMGPT_SHORTCODE")
    model = os.getenv("UMGPT_MODEL", "gpt-5.2")

    if not all([api_base, api_version, api_key, shortcode]):
        raise RuntimeError("Missing UMGPT_* env vars. Check your .env file.")

    system_prompt = prompt_path.read_text(encoding="utf-8")
    schema = load_schema(schema_path)
    validator = Draft7Validator(schema)

    print("Prompt path:", prompt_path)
    print("Schema path:", schema_path)
    print("Output path:", out_path)

    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=api_base,
        organization=shortcode,
    )

    rows = load_jsonl(in_path)
    if smoke_test_n is not None:
        rows = rows[:smoke_test_n]

    done_ids = load_existing_done_ids(out_path)

    print("Input rows:", len(rows))
    print("Already done:", len(done_ids))

    n_ok = 0
    n_err = 0

    with open(out_path, "a", encoding="utf-8") as f_out:
        for idx, row in enumerate(rows, start=1):
            chia_id = row.get("chia_id")
            doc_id = row.get("document_id")
            txt = (row.get("text") or "").strip()

            if not txt:
                continue

            criterion_type_hint = "exclusion" if str(chia_id).endswith("_exc") else "inclusion"

            top_level_items = split_top_level_items_with_offsets(txt)

            for item_index, item in enumerate(top_level_items, start=1):
                item_uid = f"{chia_id}__item{item_index}"
                item_text = item["text"]
                item_start_char = item["start_char"]
                item_end_char = item["end_char"]

                if item_uid in done_ids:
                    continue

                t0 = time.time()
                status = "ok"
                error = None
                raw_output = None
                parsed_pass1 = None
                parsed_pass1_with_ids = None

                prompt_tokens = None
                completion_tokens = None
                total_tokens = None

                try:
                    user_content = (
                        f"Trial ID: {doc_id}\n"
                        f"Criterion Type Hint: {criterion_type_hint}\n\n"
                        f"Criteria Text:\n{item_text}"
                    )

                    resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content},
                        ],
                        temperature=0,
                    )

                    if hasattr(resp, "usage") and resp.usage is not None:
                        prompt_tokens = getattr(resp.usage, "prompt_tokens", None)
                        completion_tokens = getattr(resp.usage, "completion_tokens", None)
                        total_tokens = getattr(resp.usage, "total_tokens", None)

                    raw_output = resp.choices[0].message.content or ""
                    raw_output = strip_markdown_fences(raw_output)

                    parsed_pass1 = json.loads(raw_output)

                    if not isinstance(parsed_pass1, dict):
                        raise ValueError("Model output is not a JSON object.")

                    # force known metadata from pipeline
                    parsed_pass1["trial_id"] = doc_id
                    parsed_pass1["criterion_type"] = criterion_type_hint

                    #deterministic correction:
                    # the last clause cannot connect to a next clause
                    clauses = parsed_pass1.get("clauses", [])
                    if clauses:
                        clauses[-1]["connector_to_next"] = None

                    errors = list(validator.iter_errors(parsed_pass1))
                    if errors:
                        msg = "; ".join(e.message for e in errors)
                        raise ValidationError(msg)

                    run_extra_sanity_checks(parsed_pass1, item_text)

                    parsed_pass1_with_ids = assign_clause_ids(parsed_pass1)

                    n_ok += 1

                except (json.JSONDecodeError, ValidationError, ValueError) as e:
                    status = "error"
                    error = str(e)
                    n_err += 1
                except Exception as e:
                    status = "error"
                    error = f"Unexpected error: {str(e)}"
                    n_err += 1

                if status == "error" and n_err <= 5:
                    print(f"[ERROR] chia_id={chia_id} item={item_index}: {error}")

                rec = {
                    "dataset": "CHIA",
                    "prompt_id": "pass1_flat_prompt",
                    "schema_id": "rules_v3_pass1_flat",
                    "schema_version": "v3_pass1_flat",
                    "model": model,
                    "temperature": 0,
                    "chia_id": chia_id,
                    "document_id": doc_id,
                    "criterion_type_hint": criterion_type_hint,
                    "item_index": item_index,
                    "item_uid": item_uid,
                    "item_text": item_text,
                    "item_start_char": item_start_char,
                    "item_end_char": item_end_char,
                    "full_text": txt,
                    "raw_output": raw_output,
                    "parsed_pass1": parsed_pass1,
                    "parsed_pass1_with_ids": parsed_pass1_with_ids,
                    "status": status,
                    "error": error,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "timestamp": time.time(),
                    "latency_sec": round(time.time() - t0, 3),
                }

                f_out.write(json.dumps(rec) + "\n")
                f_out.flush()
                print(f"Processed item {item_uid} | status={status} | ok={n_ok} err={n_err}", flush=True)
                if status == "ok":
                    done_ids.add(item_uid)

    print("DONE | ok =", n_ok, "err =", n_err)


if __name__ == "__main__":
    main()


# Run from the repository root: 
# python scripts/02_extraction/01_extract_pass1_flat_chia.py