# scripts/02_extraction/branch_b/04b_summarize_pass2_token_use.py

import json
import statistics
from pathlib import Path


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean_or_none(values):
    return round(statistics.mean(values), 2) if values else None


def median_or_none(values):
    return round(statistics.median(values), 2) if values else None


def main():
    root = Path(__file__).resolve().parents[3]

    pass2_dir = (
        root
        / "outputs"
        / "extraction"
        / "branch_b"
        / "pass2_leaves_llm"
    )

    pass2_path = pass2_dir / "chia_text_only_200_pass2_leaves_llm.jsonl"

    rows = load_jsonl(pass2_path)

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    error_rows = [row for row in rows if row.get("status") != "ok"]

    prompt_tokens = [row.get("prompt_tokens", 0) or 0 for row in ok_rows]
    completion_tokens = [
        row.get("completion_tokens", 0) or 0 for row in ok_rows
    ]
    total_tokens = [row.get("total_tokens", 0) or 0 for row in ok_rows]

    input_total = sum(prompt_tokens)
    output_total = sum(completion_tokens)
    total_token_sum = sum(total_tokens)

    # Relative token weighting used for the cost comparison:
    # input tokens = 1 unit; output tokens = 8 units.
    weighted_burden = input_total + 8 * output_total

    summary = {
        "stage": "pass2_llm_leaf_extraction",
        "branch": "B_llm_pass2",
        "file": pass2_path.relative_to(root).as_posix(),
        "n_rows_total": len(rows),
        "n_ok_rows": len(ok_rows),
        "n_error_rows": len(error_rows),
        "input_tokens_total": input_total,
        "output_tokens_total": output_total,
        "total_tokens_sum": total_token_sum,
        "weighted_token_burden": round(weighted_burden, 2),
        "input_tokens_mean": mean_or_none(prompt_tokens),
        "input_tokens_median": median_or_none(prompt_tokens),
        "output_tokens_mean": mean_or_none(completion_tokens),
        "output_tokens_median": median_or_none(completion_tokens),
        "total_tokens_mean": mean_or_none(total_tokens),
        "total_tokens_median": median_or_none(total_tokens),
    }

    print("\n===== PASS 2 BRANCH B TOKEN USE SUMMARY =====")
    for key, value in summary.items():
        print(f"{key}: {value}")

    out_path = pass2_dir / "pass2_token_use_summary.json"

    with out_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nSaved summary to:")
    print(out_path)


if __name__ == "__main__":
    main()


# Run from the repository root:
# python scripts/02_extraction/branch_b/04b_summarize_pass2_token_use.py
