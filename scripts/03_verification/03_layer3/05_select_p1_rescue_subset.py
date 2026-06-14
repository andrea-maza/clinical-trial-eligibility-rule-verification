"""
05_select_p1_rescue_subset.py

Select the first runnable subset from the broad Layer 3 targeted-rescue
candidate queue.

The P1 subset contains:
    - conservative structural recovery candidates
    - mandatory Branch B LLM-judge candidates

The script also:
    - writes separate files for both P1 groups
    - records candidates not selected for P1
    - separates duplicate-pruning cases for review
    - validates candidate identifiers, branches, stages, and tasks

This script does not call the LLM, modify logical rule trees, apply
repairs, or accept any candidate output.

Outputs:
    outputs/verification/layer3/p1_rescue_subset/

Run from the repository root:
python scripts/03_verification/03_layer3/05_select_p1_rescue_subset.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[3]

INPUT_JSONL = (
    ROOT
    / "outputs"
    / "verification"
    / "layer3"
    / "targeted_rescue_candidates"
    / "layer3_targeted_llm_rescue_candidates.jsonl"
)

INPUT_CSV = (
    ROOT
    / "outputs"
    / "verification"
    / "layer3"
    / "targeted_rescue_candidates"
    / "layer3_targeted_llm_rescue_candidates.csv"
)

OUT_DIR = (
    ROOT
    / "outputs"
    / "verification"
    / "layer3"
    / "p1_rescue_subset"
)

OUT_P1_JSONL = OUT_DIR / "p1_rescue_execution_subset.jsonl"
OUT_P1_CSV = OUT_DIR / "p1_rescue_execution_subset.csv"

OUT_P1_CONSERVATIVE_JSONL = (
    OUT_DIR / "p1_conservative_structural_recovery.jsonl"
)
OUT_P1_CONSERVATIVE_CSV = (
    OUT_DIR / "p1_conservative_structural_recovery.csv"
)

OUT_P1_BRANCH_B_JUDGE_JSONL = (
    OUT_DIR / "p1_branch_b_mandatory_llm_judge.jsonl"
)
OUT_P1_BRANCH_B_JUDGE_CSV = (
    OUT_DIR / "p1_branch_b_mandatory_llm_judge.csv"
)

OUT_DUPLICATE_REVIEW_CSV = (
    OUT_DIR / "p1_duplicate_prune_candidates_review_only.csv"
)
OUT_NOT_SELECTED_CSV = OUT_DIR / "not_selected_for_p1_execution.csv"

OUT_SUMMARY_JSON = OUT_DIR / "p1_rescue_execution_subset_summary.json"


P1_RUN_STAGES = {
    "P1_run_now_conservative_structural_recovery",
    "P1_run_now_branch_b_mandatory_judge",
}

P1_CONSERVATIVE_STAGE = "P1_run_now_conservative_structural_recovery"
P1_BRANCH_B_STAGE = "P1_run_now_branch_b_mandatory_judge"

DUPLICATE_TASK = "duplicate_prune_or_merge_judge"


# ---------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL not found: {path}")

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
        path.write_text("", encoding="utf-8")
        return

    priority_cols = [
        "candidate_id",
        "selected_for_p1",
        "p1_group",
        "not_selected_reason",
        "review_only_reason",

        "branch",
        "criterion_id",
        "document_id",
        "run_stage",
        "candidate_kind",
        "rescue_task_type",
        "requires_judge_first",
        "expected_llm_action",
        "max_attempts",
        "requires_reverification_after_llm",

        "entity_text",
        "entity_type",
        "operator",
        "value_type",
        "value",
        "unit",
        "computability",
        "evidence_text",

        "layer3_primary_action",
        "layer3_action_family",
        "branch_b_final_routing_decision",
        "branch_b_semantic_grounding_support",
        "branch_b_semantic_grounding_risk_label",
        "branch_a_leaf_support",
        "branch_a_risk_label",

        "conservative_issues",
        "all_layer1_codes",
        "diagnosis_summary",
        "prompt_input_json",
    ]

    cols: List[str] = []
    seen = set()

    for c in priority_cols:
        if any(c in r for r in rows):
            cols.append(c)
            seen.add(c)

    for r in rows:
        for c in r:
            if c not in seen:
                cols.append(c)
                seen.add(c)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()

        for r in rows:
            writer.writerow({c: serialize_cell(r.get(c, "")) for c in cols})


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def clean(x: Any) -> str:
    return str(x or "").strip()


def to_int(x: Any, default: int = 0) -> int:
    s = clean(x)

    if not s:
        return default

    try:
        return int(float(s))
    except Exception:
        return default


def p1_group(row: Dict[str, Any]) -> str:
    run_stage = clean(row.get("run_stage"))

    if run_stage == P1_CONSERVATIVE_STAGE:
        return "p1_conservative_structural_recovery"

    if run_stage == P1_BRANCH_B_STAGE:
        return "p1_branch_b_mandatory_llm_judge"

    return ""


def is_duplicate_review_only(row: Dict[str, Any]) -> bool:
    return clean(row.get("rescue_task_type")) == DUPLICATE_TASK


def should_select_for_p1(row: Dict[str, Any]) -> bool:
    run_stage = clean(row.get("run_stage"))
    return run_stage in P1_RUN_STAGES


def add_selection_metadata(row: Dict[str, Any], selected: bool, reason: str = "") -> Dict[str, Any]:
    out = dict(row)

    out["selected_for_p1"] = 1 if selected else 0
    out["p1_group"] = p1_group(row) if selected else ""
    out["not_selected_reason"] = "" if selected else reason
    out["review_only_reason"] = ""

    return out


def sort_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda r: (
            to_int(r.get("run_stage_order"), default=99),
            clean(r.get("p1_group")),
            clean(r.get("branch")),
            clean(r.get("criterion_id")),
            clean(r.get("rescue_task_type")),
        ),
    )


def find_repeated_criterion_ids(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = Counter(clean(r.get("branch")) + "::" + clean(r.get("criterion_id")) for r in rows)

    return {
        key: count
        for key, count in counts.items()
        if count > 1
    }


def validate_p1_rows(rows: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []

    for r in rows:
        candidate_id = clean(r.get("candidate_id"))
        run_stage = clean(r.get("run_stage"))
        branch = clean(r.get("branch"))
        task = clean(r.get("rescue_task_type"))

        if not candidate_id:
            errors.append("missing_candidate_id")

        if run_stage not in P1_RUN_STAGES:
            errors.append(f"{candidate_id}:non_p1_run_stage:{run_stage}")

        if run_stage == P1_BRANCH_B_STAGE and branch != "B":
            errors.append(f"{candidate_id}:branch_b_stage_but_branch_is_{branch}")

        if run_stage == P1_CONSERVATIVE_STAGE:
            conservative_issues = clean(r.get("conservative_issues"))
            if not conservative_issues:
                errors.append(f"{candidate_id}:conservative_stage_without_conservative_issues")

        if task == DUPLICATE_TASK:
            errors.append(f"{candidate_id}:duplicate_task_should_be_review_only_not_executed")

    return sorted(set(errors))


def candidate_preview(rows: List[Dict[str, Any]], n: int = 5) -> List[Dict[str, Any]]:
    preview = []

    for r in rows[:n]:
        preview.append({
            "candidate_id": clean(r.get("candidate_id")),
            "branch": clean(r.get("branch")),
            "criterion_id": clean(r.get("criterion_id")),
            "run_stage": clean(r.get("run_stage")),
            "rescue_task_type": clean(r.get("rescue_task_type")),
            "entity_text": clean(r.get("entity_text")),
            "operator": clean(r.get("operator")),
            "value_type": clean(r.get("value_type")),
            "value": clean(r.get("value")),
            "evidence_text": clean(r.get("evidence_text"))[:250],
        })

    return preview


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLayer 3 P1 rescue subset selector")
    print("Input JSONL:", INPUT_JSONL)

    all_rows = read_jsonl(INPUT_JSONL)

    selected_rows: List[Dict[str, Any]] = []
    not_selected_rows: List[Dict[str, Any]] = []
    duplicate_review_rows: List[Dict[str, Any]] = []

    seen_candidate_ids = set()
    duplicate_candidate_ids = []

    for row in all_rows:
        candidate_id = clean(row.get("candidate_id"))

        if candidate_id in seen_candidate_ids:
            duplicate_candidate_ids.append(candidate_id)
            continue

        seen_candidate_ids.add(candidate_id)

        if not should_select_for_p1(row):
            reason = f"run_stage_not_in_p1:{clean(row.get('run_stage'))}"
            not_selected_rows.append(add_selection_metadata(row, selected=False, reason=reason))
            continue

        # Duplicate pruning is not an LLM rescue execution call yet.
        # It should be handled by deterministic duplicate analysis or review.
        if is_duplicate_review_only(row):
            out = add_selection_metadata(row, selected=False, reason="")
            out["review_only_reason"] = "duplicate_prune_or_merge_should_not_be_executed_as_p1_llm_rescue"
            duplicate_review_rows.append(out)
            continue

        selected_rows.append(add_selection_metadata(row, selected=True))

    selected_rows = sort_rows(selected_rows)
    not_selected_rows = sort_rows(not_selected_rows)
    duplicate_review_rows = sort_rows(duplicate_review_rows)

    conservative_rows = [
        r for r in selected_rows
        if clean(r.get("run_stage")) == P1_CONSERVATIVE_STAGE
    ]

    branch_b_judge_rows = [
        r for r in selected_rows
        if clean(r.get("run_stage")) == P1_BRANCH_B_STAGE
    ]

    validation_errors = validate_p1_rows(selected_rows)
    repeated_selected_criteria = find_repeated_criterion_ids(selected_rows)

    # Write outputs
    write_jsonl(OUT_P1_JSONL, selected_rows)
    write_csv(OUT_P1_CSV, selected_rows)

    write_jsonl(OUT_P1_CONSERVATIVE_JSONL, conservative_rows)
    write_csv(OUT_P1_CONSERVATIVE_CSV, conservative_rows)

    write_jsonl(OUT_P1_BRANCH_B_JUDGE_JSONL, branch_b_judge_rows)
    write_csv(OUT_P1_BRANCH_B_JUDGE_CSV, branch_b_judge_rows)

    write_csv(OUT_DUPLICATE_REVIEW_CSV, duplicate_review_rows)
    write_csv(OUT_NOT_SELECTED_CSV, not_selected_rows)

    counts_by_run_stage = Counter(clean(r.get("run_stage")) for r in selected_rows)
    counts_by_branch = Counter(clean(r.get("branch")) for r in selected_rows)
    counts_by_task = Counter(clean(r.get("rescue_task_type")) for r in selected_rows)
    counts_by_group = Counter(clean(r.get("p1_group")) for r in selected_rows)
    counts_requires_judge = Counter(str(r.get("requires_judge_first")) for r in selected_rows)

    summary = {
        "description": (
            "Selection of the first P1 rescue execution subset. "
            "This script only selects candidates and does not call the LLM."
        ),
        "n_total_candidates_from_06d": len(all_rows),
        "n_selected_for_p1_execution": len(selected_rows),
        "n_p1_conservative_structural_recovery": len(conservative_rows),
        "n_p1_branch_b_mandatory_llm_judge": len(branch_b_judge_rows),
        "n_duplicate_review_only": len(duplicate_review_rows),
        "n_not_selected_for_p1": len(not_selected_rows),
        "duplicate_candidate_ids_in_input_count": len(duplicate_candidate_ids),
        "duplicate_candidate_ids_in_input_examples": duplicate_candidate_ids[:20],
        "repeated_branch_criterion_ids_in_selected_count": len(repeated_selected_criteria),
        "repeated_branch_criterion_ids_in_selected_examples": dict(list(repeated_selected_criteria.items())[:20]),
        "validation_errors_count": len(validation_errors),
        "validation_errors_examples": validation_errors[:50],
        "counts_by_p1_group": dict(counts_by_group.most_common()),
        "counts_by_run_stage": dict(counts_by_run_stage.most_common()),
        "counts_by_branch": dict(counts_by_branch.most_common()),
        "counts_by_rescue_task_type": dict(counts_by_task.most_common()),
        "counts_by_requires_judge_first": dict(counts_requires_judge.most_common()),
        "outputs": {
            "p1_jsonl": str(OUT_P1_JSONL),
            "p1_csv": str(OUT_P1_CSV),
            "p1_conservative_jsonl": str(OUT_P1_CONSERVATIVE_JSONL),
            "p1_conservative_csv": str(OUT_P1_CONSERVATIVE_CSV),
            "p1_branch_b_judge_jsonl": str(OUT_P1_BRANCH_B_JUDGE_JSONL),
            "p1_branch_b_judge_csv": str(OUT_P1_BRANCH_B_JUDGE_CSV),
            "duplicate_review_csv": str(OUT_DUPLICATE_REVIEW_CSV),
            "not_selected_csv": str(OUT_NOT_SELECTED_CSV),
            "summary_json": str(OUT_SUMMARY_JSON),
        },
        "execution_notes": [
            "Run the conservative structural recovery queue first or together with Branch B mandatory judge, but keep outputs separate.",
            "Branch B mandatory candidates should go to LLM-as-judge before repair.",
            "Conservative structural candidates can go directly to targeted field recovery.",
            "Do not run P2, P3, or P4 yet.",
            "Do not execute duplicate-prune candidates as blind LLM rescue; handle duplicates separately.",
            "Any LLM output must be re-run through Layer 1 and Layer 2 before final acceptance.",
        ],
        "preview_selected_first_5": candidate_preview(selected_rows, n=5),
        "preview_conservative_first_5": candidate_preview(conservative_rows, n=5),
        "preview_branch_b_judge_first_5": candidate_preview(branch_b_judge_rows, n=5),
    }

    write_json(OUT_SUMMARY_JSON, summary)

    print("\nDONE")
    print("P1 selected JSONL:", OUT_P1_JSONL)
    print("P1 selected CSV:", OUT_P1_CSV)
    print("Branch B judge JSONL:", OUT_P1_BRANCH_B_JUDGE_JSONL)
    print("Conservative recovery JSONL:", OUT_P1_CONSERVATIVE_JSONL)
    print("Summary JSON:", OUT_SUMMARY_JSON)

    print("\nCounts:")
    print("Total queue candidates:", len(all_rows))
    print("Selected for P1:", len(selected_rows))
    print("P1 conservative:", len(conservative_rows))
    print("P1 Branch B mandatory judge:", len(branch_b_judge_rows))
    print("Duplicate review only:", len(duplicate_review_rows))
    print("Not selected:", len(not_selected_rows))

    print("\nCounts by task:")
    print(dict(counts_by_task.most_common()))

    print("\nRequires judge first:")
    print(dict(counts_requires_judge.most_common()))

    if validation_errors:
        print("\nVALIDATION ERRORS FOUND:")
        for err in validation_errors[:20]:
            print(" -", err)
    else:
        print("\nValidation: OK")


if __name__ == "__main__":
    main()

# Run from the repository root:
# python scripts/03_verification/03_layer3/05_select_p1_rescue_subset.py