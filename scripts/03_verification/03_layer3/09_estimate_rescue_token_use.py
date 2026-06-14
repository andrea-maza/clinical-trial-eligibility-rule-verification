"""
09_estimate_rescue_token_use.py

Estimate the Layer 3 LLM token use retrospectively from the previously
saved prompts and raw responses.

This script does not call the LLM and does not modify any rule trees.
The estimates are used only to describe the computational burden of the
previously completed Branch B judge-and-repair process.

Run from the repository root:
python scripts/03_verification/03_layer3/09_estimate_rescue_token_use.py
"""

import json
import statistics
from pathlib import Path
from collections import Counter

try:
    import tiktoken
except ImportError:
    tiktoken = None


def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_encoding():
    if tiktoken is None:
        return None

    # GPT-5.2 may not be recognized by older tiktoken versions.
    # o200k_base is a reasonable modern fallback.
    try:
        return tiktoken.encoding_for_model("gpt-5.2")
    except Exception:
        try:
            return tiktoken.get_encoding("o200k_base")
        except Exception:
            return tiktoken.get_encoding("cl100k_base")


def count_text_tokens(text, encoding):
    text = "" if text is None else str(text)

    if encoding is None:
        # Rough fallback only. Prefer installing tiktoken.
        return int(len(text) / 4)

    return len(encoding.encode(text))


def count_messages_tokens(messages, encoding):
    """
    Approximate chat-message token count.
    This is not exact API accounting, but it is good enough
    for retrospective burden comparison when API usage was not saved.
    """
    if not isinstance(messages, list):
        return 0

    total = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        total += count_text_tokens(msg.get("role", ""), encoding)
        total += count_text_tokens(msg.get("content", ""), encoding)

    return total


def find_traces(obj):
    """
    Recursively collect trace dictionaries that contain messages and raw_response.
    This handles both candidate_selection_rescue_results.jsonl and raw_prompts files.
    """
    traces = []

    if isinstance(obj, dict):
        if "messages" in obj and "raw_response" in obj:
            traces.append(obj)

        for value in obj.values():
            traces.extend(find_traces(value))

    elif isinstance(obj, list):
        for value in obj:
            traces.extend(find_traces(value))

    return traces


def mean_or_none(values):
    return round(statistics.mean(values), 2) if values else None


def median_or_none(values):
    return round(statistics.median(values), 2) if values else None

def main():
    ROOT = Path(__file__).resolve().parents[3]

    results_path = (
        ROOT
        / "outputs"
        / "verification"
        / "layer3"
        / "candidate_selection_rescue"
        / "candidate_selection_rescue_results.jsonl"
    )

    rows = load_jsonl(results_path)
    encoding = get_encoding()

    call_rows = []
    seen_call_keys = set()

    for row_i, row in enumerate(rows):
        branch = row.get("branch_to_update", "")
        plan_id = row.get("plan_id", "")
        traces = find_traces(row)

        for trace_i, trace in enumerate(traces):
            kind = trace.get("kind", "unknown")
            candidate_id = trace.get("candidate_id", "")

            # avoid double counting if the same trace appears twice
            call_key = (plan_id, kind, candidate_id, trace_i)
            if call_key in seen_call_keys:
                continue
            seen_call_keys.add(call_key)

            input_tokens = count_messages_tokens(trace.get("messages", []), encoding)
            output_tokens = count_text_tokens(trace.get("raw_response", ""), encoding)
            total_tokens = input_tokens + output_tokens
            weighted_burden = input_tokens + 8 * output_tokens

            call_rows.append({
                "branch": branch,
                "plan_id": plan_id,
                "kind": kind,
                "candidate_id": candidate_id,
                "input_tokens_est": input_tokens,
                "output_tokens_est": output_tokens,
                "total_tokens_est": total_tokens,
                "weighted_token_burden_est": weighted_burden,
            })

    by_branch = {}
    for branch in sorted(set(r["branch"] for r in call_rows)):
        bucket = [r for r in call_rows if r["branch"] == branch]
        by_branch[branch] = {
            "llm_calls_est": len(bucket),
            "input_tokens_est": sum(r["input_tokens_est"] for r in bucket),
            "output_tokens_est": sum(r["output_tokens_est"] for r in bucket),
            "total_tokens_est": sum(r["total_tokens_est"] for r in bucket),
            "weighted_token_burden_est": sum(r["weighted_token_burden_est"] for r in bucket),
        }

    by_kind = {}
    for kind in sorted(set(r["kind"] for r in call_rows)):
        bucket = [r for r in call_rows if r["kind"] == kind]
        by_kind[kind] = {
            "llm_calls_est": len(bucket),
            "input_tokens_est": sum(r["input_tokens_est"] for r in bucket),
            "output_tokens_est": sum(r["output_tokens_est"] for r in bucket),
            "total_tokens_est": sum(r["total_tokens_est"] for r in bucket),
            "weighted_token_burden_est": sum(r["weighted_token_burden_est"] for r in bucket),
        }

    summary = {
        "stage": "layer3_candidate_selection_rescue_estimated_token_use",
        "important_note": (
            "This is a retrospective estimate from saved prompts and raw responses. "
            "It does not make new LLM calls and may differ slightly from exact API accounting."
        ),
        "source_file": str(results_path),
        "n_result_rows": len(rows),
        "n_llm_calls_est": len(call_rows),
        "by_branch": by_branch,
        "by_call_kind": by_kind,
        "overall": {
            "llm_calls_est": len(call_rows),
            "input_tokens_est": sum(r["input_tokens_est"] for r in call_rows),
            "output_tokens_est": sum(r["output_tokens_est"] for r in call_rows),
            "total_tokens_est": sum(r["total_tokens_est"] for r in call_rows),
            "weighted_token_burden_est": sum(r["weighted_token_burden_est"] for r in call_rows),
            "input_tokens_mean_est": mean_or_none([r["input_tokens_est"] for r in call_rows]),
            "input_tokens_median_est": median_or_none([r["input_tokens_est"] for r in call_rows]),
            "output_tokens_mean_est": mean_or_none([r["output_tokens_est"] for r in call_rows]),
            "output_tokens_median_est": median_or_none([r["output_tokens_est"] for r in call_rows]),
        },
    }

    out_dir = (
        ROOT
        / "outputs"
        / "verification"
        / "layer3"
        / "candidate_selection_rescue"
    )

    out_summary = out_dir / "layer3_rescue_token_use_estimate_summary.json"
    out_calls = out_dir / "layer3_rescue_token_use_estimate_calls.jsonl"

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(out_calls, "w", encoding="utf-8") as f:
        for r in call_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n===== LAYER 3 RESCUE TOKEN USE ESTIMATE =====")
    print(json.dumps(summary, indent=2))
    print("\nSaved:")
    print(out_summary)
    print(out_calls)


if __name__ == "__main__":
    main()


#python scripts/03_verification/03_layer3/09_estimate_rescue_token_use.py