"""
03_evaluate_gold_node_proxy.py

Run the CHIA gold-node proxy evaluation on the post-verification
Branch A and Branch B leaves.

The script imports the exact pre-verification evaluation functions so
that the same matching logic is used before and after verification.

This proxy measures entity span and type alignment. It is not a
complete semantic evaluation and does not use manual labels.

Inputs:
    outputs/evaluation/post_verification/
        post_verification_pass2_leaves/
    outputs/extraction/pass2_inputs/
    data/processed/chia_struct_eval_200_gold_graph.jsonl

Outputs:
    outputs/evaluation/post_verification/
        gold_nodes_fair_leaf_proxy_post_verification_A_B/

This script does not call the LLM and does not modify predictions.

Run from the repository root:
python scripts/04_evaluation/02_post_verification/03_evaluate_gold_node_proxy.py
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
    / "02_evaluate_gold_node_proxy.py"
)

GOLD_PATH = (
    ROOT
    / "data"
    / "processed"
    / "chia_struct_eval_200_gold_graph.jsonl"
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
    / "gold_nodes_fair_leaf_proxy_post_verification_A_B"
)

PRE_SUMMARY_PATH = (
    ROOT
    / "outputs"
    / "evaluation"
    / "pre_verification"
    / "gold_nodes_fair_leaf_proxy_pre_verification_A_B"
    / "all_branch_summary.json"
)

OUT_ALL_SUMMARY = OUT_ROOT / "all_branch_summary.json"
OUT_PRE_POST_COMPARISON = (
    OUT_ROOT / "pre_post_gold_nodes_comparison.json"
)



def import_pre_eval_module():
    if not PRE_EVAL_SCRIPT.exists():
        raise FileNotFoundError(
            f"Pre-verification evaluation script not found: {PRE_EVAL_SCRIPT}"
        )

    spec = importlib.util.spec_from_file_location(
        "pre_verification_gold_node_proxy",
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


def compact_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "total_leaves": summary["coverage"]["stats"].get("total_leaves", 0),
        "comparable_leaves": summary["coverage"]["stats"].get("comparable_leaves", 0),
        "non_comparable_leaves": summary["coverage"]["stats"].get("non_comparable_leaves", 0),
        "span_recoverable_comparable_leaves": summary["coverage"]["stats"].get(
            "span_recoverable_comparable_leaves", 0
        ),
        "span_not_recoverable_comparable_leaves": summary["coverage"]["stats"].get(
            "span_not_recoverable_comparable_leaves", 0
        ),
        "n_total_pred_comparable_nodes": summary["n_total_pred_comparable_nodes"],
        "n_total_gold_nodes": summary["n_total_gold_nodes"],
        "micro_exact_mention": summary["micro_exact_mention"],
        "micro_soft_mention": summary["micro_soft_mention"],
        "micro_exact_typed": summary["micro_exact_typed"],
        "micro_soft_typed": summary["micro_soft_typed"],
        "pred_entity_type_counts": summary["coverage"].get("pred_entity_type_counts", {}),
        "non_comparable_entity_type_counts": summary["coverage"].get(
            "non_comparable_entity_type_counts", {}
        ),
        "span_source_counts": summary["coverage"].get("span_source_counts", {}),
    }


def get_f1(block: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(block, dict):
        return None
    return block.get("f1")


def build_pre_post_comparison(post_summaries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    pre = read_json(PRE_SUMMARY_PATH)

    mapping = {
        "A_post_verification": "A_bert_rules",
        "B_post_verification": "B_llm_pass2",
    }

    comparison: Dict[str, Any] = {}

    for post_branch, pre_branch in mapping.items():
        pre_summary = pre.get(pre_branch, {})
        post_summary = post_summaries.get(post_branch, {})

        comparison[post_branch] = {
            "pre_branch": pre_branch,
            "post_branch": post_branch,
            "pre": pre_summary,
            "post": post_summary,
            "delta": {
                "total_leaves": post_summary.get("total_leaves", 0)
                - pre_summary.get("total_leaves", 0),
                "comparable_leaves": post_summary.get("comparable_leaves", 0)
                - pre_summary.get("comparable_leaves", 0),
                "span_recoverable_comparable_leaves": post_summary.get(
                    "span_recoverable_comparable_leaves", 0
                )
                - pre_summary.get("span_recoverable_comparable_leaves", 0),
                "span_not_recoverable_comparable_leaves": post_summary.get(
                    "span_not_recoverable_comparable_leaves", 0
                )
                - pre_summary.get("span_not_recoverable_comparable_leaves", 0),
                "exact_mention_f1": (
                    get_f1(post_summary.get("micro_exact_mention"))
                    - get_f1(pre_summary.get("micro_exact_mention"))
                    if get_f1(post_summary.get("micro_exact_mention")) is not None
                    and get_f1(pre_summary.get("micro_exact_mention")) is not None
                    else None
                ),
                "soft_mention_f1": (
                    get_f1(post_summary.get("micro_soft_mention"))
                    - get_f1(pre_summary.get("micro_soft_mention"))
                    if get_f1(post_summary.get("micro_soft_mention")) is not None
                    and get_f1(pre_summary.get("micro_soft_mention")) is not None
                    else None
                ),
                "exact_typed_f1": (
                    get_f1(post_summary.get("micro_exact_typed"))
                    - get_f1(pre_summary.get("micro_exact_typed"))
                    if get_f1(post_summary.get("micro_exact_typed")) is not None
                    and get_f1(pre_summary.get("micro_exact_typed")) is not None
                    else None
                ),
                "soft_typed_f1": (
                    get_f1(post_summary.get("micro_soft_typed"))
                    - get_f1(pre_summary.get("micro_soft_typed"))
                    if get_f1(post_summary.get("micro_soft_typed")) is not None
                    and get_f1(pre_summary.get("micro_soft_typed")) is not None
                    else None
                ),
            },
            "interpretation_note": (
                "This is a Level A CHIA-style proxy. It measures entity span/type "
                "alignment, not full semantic correctness."
            ),
        }

    return comparison


def check_inputs() -> None:
    required = [
        GOLD_PATH,
        PASS2_INPUTS_PATH,
        A_POST_PASS2_LEAVES,
        B_POST_PASS2_LEAVES,
        PRE_EVAL_SCRIPT,
        PRE_SUMMARY_PATH,
    ]

    missing = [str(p) for p in required if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing required input(s):\n" + "\n".join(missing)
        )

    bad_paths = []

    if "chia_text_only_100" in str(PASS2_INPUTS_PATH):
        bad_paths.append(str(PASS2_INPUTS_PATH))

    if "chia_text_only_100" in str(A_POST_PASS2_LEAVES):
        bad_paths.append(str(A_POST_PASS2_LEAVES))

    if "chia_text_only_100" in str(B_POST_PASS2_LEAVES):
        bad_paths.append(str(B_POST_PASS2_LEAVES))

    if "b_fusion" in str(PRE_SUMMARY_PATH):
        bad_paths.append(str(PRE_SUMMARY_PATH))

    if bad_paths:
        raise RuntimeError(
            "Old/inconsistent paths detected:\n" + "\n".join(bad_paths)
        )


def main() -> None:
    check_inputs()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    pre_eval = import_pre_eval_module()

    print("\nPost-verification Level A CHIA gold-node fair leaf proxy")
    print("Pre-verification evaluation logic:", PRE_EVAL_SCRIPT)
    print("Gold:", GOLD_PATH)
    print("Pass2 inputs:", PASS2_INPUTS_PATH)
    print("Branch A post leaves:", A_POST_PASS2_LEAVES)
    print("Branch B post leaves:", B_POST_PASS2_LEAVES)

    gold_rows = pre_eval.load_jsonl(GOLD_PATH)
    pass2_input_rows = pre_eval.load_jsonl(PASS2_INPUTS_PATH)

    gold_by_doc = pre_eval.build_gold_index(gold_rows)
    pass2_input_index = pre_eval.build_pass2_input_index(pass2_input_rows)

    branch_paths = {
        "A_post_verification": A_POST_PASS2_LEAVES,
        "B_post_verification": B_POST_PASS2_LEAVES,
    }

    all_summaries: Dict[str, Dict[str, Any]] = {}

    for branch_name, path in branch_paths.items():
        summary = pre_eval.run_branch(
            root=ROOT,
            branch_name=branch_name,
            pass2_leaves_path=path,
            gold_by_doc=gold_by_doc,
            pass2_input_index=pass2_input_index,
            out_root=OUT_ROOT,
        )

        if summary is not None:
            summary["evaluation_stage"] = (
                "level_a_gold_nodes_fair_leaf_proxy_post_verification_A_B"
            )
            pre_eval.write_json(
                OUT_ROOT / branch_name / "summary.json",
                summary,
            )

            compact = compact_summary(summary)
            compact["pass2_leaves_file"] = str(path)
            all_summaries[branch_name] = compact


    write_json(OUT_ALL_SUMMARY, all_summaries)

    pre_post = build_pre_post_comparison(all_summaries)
    write_json(OUT_PRE_POST_COMPARISON, pre_post)

    print("\n===== WROTE POST-VERIFICATION GOLD-NODE SUMMARY =====")
    print(OUT_ALL_SUMMARY)
    print(OUT_PRE_POST_COMPARISON)

    print("\nCompact post summaries:")
    for branch, summary in all_summaries.items():
        print(f"\n--- {branch} ---")
        print("Total leaves:", summary["total_leaves"])
        print("Comparable leaves:", summary["comparable_leaves"])
        print("Span recoverable comparable leaves:", summary["span_recoverable_comparable_leaves"])
        print("Span NOT recoverable comparable leaves:", summary["span_not_recoverable_comparable_leaves"])
        print("Gold nodes:", summary["n_total_gold_nodes"])
        print("Pred comparable nodes:", summary["n_total_pred_comparable_nodes"])
        print("Exact mention:", summary["micro_exact_mention"])
        print("Soft mention:", summary["micro_soft_mention"])
        print("Exact typed:", summary["micro_exact_typed"])
        print("Soft typed:", summary["micro_soft_typed"])


if __name__ == "__main__":
    main()

# Run from the repository root:
# python scripts/04_evaluation/02_post_verification/03_evaluate_gold_node_proxy.py