"""
04_evaluate_entity_grounding.py

Evaluate post-verification entity grounding for Branch A and Branch B.

The script imports the exact pre-verification grounding functions so
that the same diagnostic logic is used before and after verification.

The evaluation checks:
    - whether entity_text appears in evidence_text
    - whether PubMedBERT candidate spans are available
    - whether entity_text overlaps those candidate spans
    - whether the entity type agrees with the best candidate span

Inputs:
    outputs/evaluation/post_verification/
        post_verification_pass2_leaves/
    outputs/extraction/pass2_inputs/

Outputs:
    outputs/evaluation/post_verification/
        entity_grounding_post_verification_A_B/

This script does not call the LLM, use manual labels, or modify predictions.

Run from the repository root:
python scripts/04_evaluation/02_post_verification/04_evaluate_entity_grounding.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[3]

PRE_EVAL_SCRIPT = (
    ROOT
    / "scripts"
    / "04_evaluation"
    / "01_pre_verification"
    / "03_evaluate_entity_grounding.py"
)

PASS2_INPUTS_PATH = (
    ROOT
    / "outputs"
    / "extraction"
    / "pass2_inputs"
    / "chia_text_only_200_pass2_inputs.jsonl"
)

A_POST_PASS2_LEAVES = (
    ROOT
    / "outputs"
    / "evaluation"
    / "post_verification"
    / "post_verification_pass2_leaves"
    / "chia_text_only_200_post_verification_pass2_leaves_A.jsonl"
)

B_POST_PASS2_LEAVES = (
    ROOT
    / "outputs"
    / "evaluation"
    / "post_verification"
    / "post_verification_pass2_leaves"
    / "chia_text_only_200_post_verification_pass2_leaves_B.jsonl"
)

OUT_ROOT = (
    ROOT
    / "outputs"
    / "evaluation"
    / "post_verification"
    / "entity_grounding_post_verification_A_B"
)

PRE_SUMMARY_PATH = (
    ROOT
    / "outputs"
    / "evaluation"
    / "pre_verification"
    / "entity_grounding_pre_verification_A_B"
    / "all_branch_summary.json"
)

OUT_ALL_SUMMARY = OUT_ROOT / "all_branch_summary.json"
OUT_PRE_POST_COMPARISON = (
    OUT_ROOT / "pre_post_entity_grounding_comparison.json"
)


def import_pre_eval_module():
    if not PRE_EVAL_SCRIPT.exists():
        raise FileNotFoundError(f"Pre evaluation script not found: {PRE_EVAL_SCRIPT}")

    spec = importlib.util.spec_from_file_location(
        "pre_entity_grounding_eval",
        PRE_EVAL_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import: {PRE_EVAL_SCRIPT}")

    module = importlib.util.module_from_spec(spec)

    sys.path.insert(0, str(PRE_EVAL_SCRIPT.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(PRE_EVAL_SCRIPT.parent))
        except ValueError:
            pass

    return module


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    return json.loads(path.read_text(encoding="utf-8"))


def delta_float(post: Any, pre: Any) -> Optional[float]:
    if post is None or pre is None:
        return None

    try:
        return round(float(post) - float(pre), 4)
    except Exception:
        return None


def compact_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "n_items_evaluated": summary["n_items_evaluated"],
        "n_total_clauses": summary["n_total_clauses"],
        "entity_nonempty_rate": summary["entity_nonempty_rate"],
        "entity_in_evidence_rate": summary["entity_in_evidence_rate"],
        "anchor_present_rate": summary["anchor_present_rate"],
        "no_anchor_rate": summary["no_anchor_rate"],
        "any_bert_candidate_present_rate": summary["any_bert_candidate_present_rate"],
        "no_bert_candidate_rate": summary["no_bert_candidate_rate"],
        "entity_in_evidence_rate_when_anchor_present": summary[
            "entity_in_evidence_rate_when_anchor_present"
        ],
        "entity_in_evidence_rate_when_no_anchor": summary[
            "entity_in_evidence_rate_when_no_anchor"
        ],
        "entity_overlaps_any_anchor_rate_when_anchor_present": summary[
            "entity_overlaps_any_anchor_rate_when_anchor_present"
        ],
        "entity_overlaps_best_anchor_rate_when_anchor_present": summary[
            "entity_overlaps_best_anchor_rate_when_anchor_present"
        ],
        "entity_overlaps_any_anchor_rate": summary["entity_overlaps_any_anchor_rate"],
        "entity_overlaps_best_anchor_rate": summary["entity_overlaps_best_anchor_rate"],
        "best_anchor_type_match_rate": summary["best_anchor_type_match_rate"],
        "generic_entity_rate": summary["generic_entity_rate"],
        "issue_counts": summary["issue_counts"],
    }


def build_pre_post_comparison(post_summaries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    pre = read_json(PRE_SUMMARY_PATH)

    mapping = {
        "A_post_verification": "A_bert_rules",
        "B_post_verification": "B_llm_pass2",
    }

    rate_keys = [
        "entity_nonempty_rate",
        "entity_in_evidence_rate",
        "anchor_present_rate",
        "no_anchor_rate",
        "any_bert_candidate_present_rate",
        "no_bert_candidate_rate",
        "entity_in_evidence_rate_when_anchor_present",
        "entity_in_evidence_rate_when_no_anchor",
        "entity_overlaps_any_anchor_rate_when_anchor_present",
        "entity_overlaps_best_anchor_rate_when_anchor_present",
        "entity_overlaps_any_anchor_rate",
        "entity_overlaps_best_anchor_rate",
        "best_anchor_type_match_rate",
        "generic_entity_rate",
    ]

    comparison: Dict[str, Any] = {}

    for post_branch, pre_branch in mapping.items():
        pre_summary = pre.get(pre_branch, {})
        post_summary = post_summaries.get(post_branch, {})

        deltas = {
            key: delta_float(post_summary.get(key), pre_summary.get(key))
            for key in rate_keys
        }

        pre_issues = pre_summary.get("issue_counts", {})
        post_issues = post_summary.get("issue_counts", {})

        issue_keys = sorted(set(pre_issues) | set(post_issues))

        issue_delta = {
            key: int(post_issues.get(key, 0)) - int(pre_issues.get(key, 0))
            for key in issue_keys
        }

        comparison[post_branch] = {
            "pre_branch": pre_branch,
            "post_branch": post_branch,
            "pre": pre_summary,
            "post": post_summary,
            "delta_rates": deltas,
            "delta_issue_counts": issue_delta,
            "interpretation_note": (
                "This is an entity-grounding diagnostic against Pass 2 candidate spans. "
                "It is not full semantic correctness."
            ),
        }

    return comparison


def check_inputs() -> None:
    required = [
        PRE_EVAL_SCRIPT,
        PASS2_INPUTS_PATH,
        A_POST_PASS2_LEAVES,
        B_POST_PASS2_LEAVES,
        PRE_SUMMARY_PATH,
    ]

    missing = [str(p) for p in required if not p.exists()]

    if missing:
        raise FileNotFoundError("Missing required input(s):\n" + "\n".join(missing))

    bad_paths = []

    for p in [PASS2_INPUTS_PATH, A_POST_PASS2_LEAVES, B_POST_PASS2_LEAVES]:
        if "chia_text_only_100" in str(p):
            bad_paths.append(str(p))

    if "b_fusion" in str(PRE_SUMMARY_PATH):
        bad_paths.append(str(PRE_SUMMARY_PATH))

    if bad_paths:
        raise RuntimeError("Old/inconsistent paths detected:\n" + "\n".join(bad_paths))


def main() -> None:
    check_inputs()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    pre_eval = import_pre_eval_module()

    print("\nPost-verification entity grounding diagnostic")
    print("Pre-verification evaluation logic:", PRE_EVAL_SCRIPT)
    print("Pass2 inputs:", PASS2_INPUTS_PATH)
    print("Branch A post leaves:", A_POST_PASS2_LEAVES)
    print("Branch B post leaves:", B_POST_PASS2_LEAVES)

    branch_paths = {
        "A_post_verification": A_POST_PASS2_LEAVES,
        "B_post_verification": B_POST_PASS2_LEAVES,
    }

    all_summaries: Dict[str, Dict[str, Any]] = {}

    for branch_name, pass2_leaves_path in branch_paths.items():
        summary = pre_eval.evaluate_branch(
            branch_name=branch_name,
            pass2_inputs_path=PASS2_INPUTS_PATH,
            pass2_leaves_path=pass2_leaves_path,
            out_root=OUT_ROOT,
        )

        if summary is not None:
            summary["evaluation_stage"] = "entity_grounding_post_verification_A_B"
            write_json(OUT_ROOT / branch_name / "summary.json", summary)

            all_summaries[branch_name] = compact_summary(summary)

    write_json(OUT_ALL_SUMMARY, all_summaries)

    pre_post = build_pre_post_comparison(all_summaries)
    write_json(OUT_PRE_POST_COMPARISON, pre_post)

    print("\n===== WROTE ENTITY GROUNDING POST-VERIFICATION SUMMARY =====")
    print(OUT_ALL_SUMMARY)
    print(OUT_PRE_POST_COMPARISON)

    print("\nCompact post summaries:")
    for branch, summary in all_summaries.items():
        print(f"\n--- {branch} ---")
        print("Total clauses:", summary["n_total_clauses"])
        print("Entity in evidence:", summary["entity_in_evidence_rate"])
        print("Anchor present:", summary["anchor_present_rate"])
        print("Entity overlaps best anchor:", summary["entity_overlaps_best_anchor_rate"])
        print("Best anchor type match:", summary["best_anchor_type_match_rate"])
        print("Generic entity rate:", summary["generic_entity_rate"])


if __name__ == "__main__":
    main()

# Run from the repository root:
# python scripts/04_evaluation/02_post_verification/04_evaluate_entity_grounding.py