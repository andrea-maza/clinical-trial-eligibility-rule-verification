"""
10_calculate_branch_a_hybrid_tokens.py

Estimate the attributable Branch B Pass 2 token burden for Branch A
leaves that used a Branch B leaf as their Layer 3 semantic substitute.

Branch A made no new LLM calls during Layer 3. This script allocates the
existing item-level Branch B Pass 2 token use equally across the leaves
returned for each item.

This script does not call the LLM and does not modify rule trees.

Run from the repository root:
python scripts/03_verification/03_layer3/10_calculate_branch_a_hybrid_tokens.py
"""

import json
from pathlib import Path
from collections import Counter


def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def clean(x):
    return str(x or "").strip()


def get_criteria_count(pass2_row):
    out = pass2_row.get("pass2_output") or {}
    criteria = out.get("criteria") or []
    return len(criteria) if isinstance(criteria, list) else 0


def criterion_ids_from_pass2_row(pass2_row):
    out = pass2_row.get("pass2_output") or {}
    criteria = out.get("criteria") or []

    ids = []
    for entry in criteria:
        criterion = entry.get("criterion") or {}
        cid = clean(criterion.get("criterion_id"))
        if cid:
            ids.append(cid)

    return ids


def build_allocated_pass2_token_index(pass2_rows):
    """
    Allocate Branch B Pass 2 item-level token use to each criterion leaf
    returned by that item.
    """
    token_index = {}

    for row in pass2_rows:
        if row.get("status") != "ok":
            continue

        criterion_ids = criterion_ids_from_pass2_row(row)
        n = len(criterion_ids)

        if n == 0:
            continue

        input_tokens = row.get("prompt_tokens") or 0
        output_tokens = row.get("completion_tokens") or 0
        total_tokens = row.get("total_tokens") or (input_tokens + output_tokens)

        allocated_input = input_tokens / n
        allocated_output = output_tokens / n
        allocated_total = total_tokens / n
        allocated_weighted = allocated_input + 8 * allocated_output

        for cid in criterion_ids:
            token_index[cid] = {
                "allocated_input_tokens": allocated_input,
                "allocated_output_tokens": allocated_output,
                "allocated_total_tokens": allocated_total,
                "allocated_weighted_burden": allocated_weighted,
                "source_item_uid": row.get("item_uid"),
                "criteria_in_item": n,
            }

    return token_index

def main():
    ROOT = Path(__file__).resolve().parents[3]

    pass2_b_path = (
        ROOT
        / "outputs"
        / "extraction"
        / "branch_b"
        / "pass2_leaves_llm"
        / "chia_text_only_200_pass2_leaves_llm.jsonl"
    )

    rescue_results_path = (
        ROOT
        / "outputs"
        / "verification"
        / "layer3"
        / "candidate_selection_rescue"
        / "candidate_selection_rescue_results.jsonl"
    )

    out_dir = (
        ROOT
        / "outputs"
        / "verification"
        / "layer3"
        / "candidate_selection_rescue"
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    out_summary = (
        out_dir / "branch_a_hybrid_attributable_pass2_token_summary.json"
    )
    out_rows = (
        out_dir / "branch_a_hybrid_attributable_pass2_token_rows.jsonl"
    )

    pass2_rows = load_jsonl(pass2_b_path)
    rescue_rows = load_jsonl(rescue_results_path)

    token_index = build_allocated_pass2_token_index(pass2_rows)

    branch_a_hybrid_rows = []

    for row in rescue_rows:
        if clean(row.get("branch_to_update")) != "A":
            continue

        final_decision = clean(row.get("final_decision"))
        selected_source = clean(row.get("selected_source"))

        # Branch A hybrid substitution uses Branch B output.
        uses_branch_b = selected_source in {
            "B_current",
            "B_dejure_best",
        }

        if final_decision != "select_candidate":
            continue

        if not uses_branch_b:
            continue

        criterion_id = clean(row.get("criterion_id"))
        allocated = token_index.get(criterion_id)

        if not allocated:
            branch_a_hybrid_rows.append({
                "criterion_id": criterion_id,
                "selected_source": selected_source,
                "token_allocation_status": "missing_branch_b_pass2_token_allocation",
                "allocated_input_tokens": 0,
                "allocated_output_tokens": 0,
                "allocated_total_tokens": 0,
                "allocated_weighted_burden": 0,
            })
            continue

        branch_a_hybrid_rows.append({
            "criterion_id": criterion_id,
            "selected_source": selected_source,
            "token_allocation_status": "allocated_from_branch_b_pass2_item_call",
            **allocated,
        })

    summary = {
        "stage": "branch_a_hybrid_attributable_pass2_token_use",
        "important_note": (
            "Branch A made no new Layer 3 LLM calls. This retrospective "
            "calculation estimates the attributable Branch B Pass 2 token "
            "burden for Branch A leaves that reused Branch B as a semantic "
            "substitute. Because Branch B Pass 2 calls were made per item, "
            "tokens are allocated equally across the leaves returned for "
            "that item."
        ),
        "n_branch_a_hybrid_substituted_leaves": len(branch_a_hybrid_rows),
        "selected_source_counts": dict(Counter(r["selected_source"] for r in branch_a_hybrid_rows)),
        "allocation_status_counts": dict(Counter(r["token_allocation_status"] for r in branch_a_hybrid_rows)),
        "allocated_input_tokens": round(sum(r["allocated_input_tokens"] for r in branch_a_hybrid_rows), 2),
        "allocated_output_tokens": round(sum(r["allocated_output_tokens"] for r in branch_a_hybrid_rows), 2),
        "allocated_total_tokens": round(sum(r["allocated_total_tokens"] for r in branch_a_hybrid_rows), 2),
        "allocated_weighted_burden": round(sum(r["allocated_weighted_burden"] for r in branch_a_hybrid_rows), 2),
    }

    with open(out_rows, "w", encoding="utf-8") as f:
        for r in branch_a_hybrid_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n===== BRANCH A HYBRID ATTRIBUTABLE PASS 2 TOKEN USE =====")
    print(json.dumps(summary, indent=2))
    print("\nSaved:")
    print(out_summary)
    print(out_rows)


if __name__ == "__main__":
    main()

#python scripts/03_verification/03_layer3/10_calculate_branch_a_hybrid_tokens.py