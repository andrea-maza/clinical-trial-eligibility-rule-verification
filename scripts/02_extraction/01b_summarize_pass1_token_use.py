import json
import statistics
from pathlib import Path


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean_or_none(values):
    return round(statistics.mean(values), 2) if values else None


def median_or_none(values):
    return round(statistics.median(values), 2) if values else None


def main():
    ROOT = Path(__file__).resolve().parents[2]

    pass1_path = (
        ROOT
        / "outputs"
        / "extraction"
        / "pass1_flat"
        / "chia_text_only_200_pass1_flat.jsonl"
    )

    rows = load_jsonl(pass1_path)

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    err_rows = [r for r in rows if r.get("status") != "ok"]

    prompt_tokens = [r.get("prompt_tokens", 0) or 0 for r in ok_rows]
    completion_tokens = [r.get("completion_tokens", 0) or 0 for r in ok_rows]
    total_tokens = [r.get("total_tokens", 0) or 0 for r in ok_rows]

    # Existing Pass 1 output did not store cached token details.
    # Therefore cached input is set to 0
    cached_input_tokens = 0

    input_total = sum(prompt_tokens)
    output_total = sum(completion_tokens)
    total_token_sum = sum(total_tokens)

    # Relative burden based on the model prices:
    # input = 1 unit, cached input = 0.1 units, output = 8 units
    weighted_burden = input_total + 0.1 * cached_input_tokens + 8 * output_total

    summary = {
        "stage": "pass1_logical_decomposition",
        "file": str(pass1_path),
        "n_rows_total": len(rows),
        "n_ok_rows": len(ok_rows),
        "n_error_rows": len(err_rows),
        "input_tokens_total": input_total,
        "cached_input_tokens_total": cached_input_tokens,
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

    print("\n===== PASS 1 TOKEN USE SUMMARY =====")
    for key, value in summary.items():
        print(f"{key}: {value}")

    out_path = (
        ROOT
        / "outputs"
        / "extraction"
        / "pass1_flat"
        / "pass1_token_use_summary.json"
    )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved summary to:")
    print(out_path)


if __name__ == "__main__":
    main()

# Run from the repository root: 
# # python scripts/02_extraction/01b_summarize_pass1_token_use.py