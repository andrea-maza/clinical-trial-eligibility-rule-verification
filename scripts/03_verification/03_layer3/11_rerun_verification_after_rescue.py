"""
11_rerun_verification_after_rescue.py

Rerun Layer 1 and Layer 2 verification after the validated Layer 3
rescue proposals have been applied.

The script:
    - rebuilds the shared deterministic Layer 1 inventory
    - reruns Pass 1--Pass 2 consistency checks
    - reruns the Branch A and Branch B Layer 1 policies
    - recomputes Branch A Layer 2 support and risk
    - recomputes Branch B Layer 2 grounding and execution support
    - compares pre-rescue and post-rescue verification counts

This script does not call the LLM, modify the rule trees, use manual
labels, or assign final decisions.

Run from the repository root:
python scripts/03_verification/03_layer3/11_rerun_verification_after_rescue.py
"""


from __future__ import annotations

import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[3]

LAYER1_SCRIPT_DIR = (
    ROOT / "scripts" / "03_verification" / "01_layer1"
)
LAYER2_SCRIPT_DIR = (
    ROOT / "scripts" / "03_verification" / "02_layer2"
)

L1_COMMON_SCRIPT = (
    LAYER1_SCRIPT_DIR / "01_build_deterministic_inventory.py"
)
L1D_SCRIPT = (
    LAYER1_SCRIPT_DIR / "02_check_pass1_pass2_consistency.py"
)
L1_POLICY_A_SCRIPT = (
    LAYER1_SCRIPT_DIR / "03_apply_policy_branch_a.py"
)
L1_POLICY_B_SCRIPT = (
    LAYER1_SCRIPT_DIR / "04_apply_policy_branch_b.py"
)

L2_A_INVENTORY_SCRIPT = (
    LAYER2_SCRIPT_DIR / "01_inventory_branch_a_support_signals.py"
)
L2_A_SCORE_SCRIPT = (
    LAYER2_SCRIPT_DIR / "02_score_branch_a_leaf_risk.py"
)
L2_B_INVENTORY_SCRIPT = (
    LAYER2_SCRIPT_DIR / "05_inventory_branch_b_support_signals.py"
)
L2_B_SCREEN_SCRIPT = (
    LAYER2_SCRIPT_DIR / "06_screen_branch_b_grounding.py"
)

A_RESCUED_AST = (
    ROOT
    / "outputs"
    / "verification"
    / "layer3"
    / "applied_candidate_selection_rescue"
    / "chia_text_only_200_rules_v3_ast_A_layer3_candidate_selection_rescue.jsonl"
)

B_RESCUED_AST = (
    ROOT
    / "outputs"
    / "verification"
    / "layer3"
    / "applied_candidate_selection_rescue"
    / "chia_text_only_200_rules_v3_ast_B_layer3_candidate_selection_rescue.jsonl"
)

PASS2_INPUTS = (
    ROOT
    / "outputs"
    / "extraction"
    / "pass2_inputs"
    / "chia_text_only_200_pass2_inputs.jsonl"
)

PASS1_OUTPUTS = (
    ROOT
    / "outputs"
    / "extraction"
    / "pass1_flat"
    / "chia_text_only_200_pass1_flat.jsonl"
)

OUT_ROOT = (
    ROOT
    / "outputs"
    / "verification"
    / "layer3"
    / "post_rescue_verification"
)

L1_COMMON_DIR = (
    OUT_ROOT / "layer1" / "deterministic_inventory"
)
L1D_DIR = (
    OUT_ROOT / "layer1" / "pass1_pass2_consistency"
)
L1_POLICY_A_DIR = (
    OUT_ROOT / "layer1" / "policy_branch_a"
)
L1_POLICY_B_DIR = (
    OUT_ROOT / "layer1" / "policy_branch_b"
)
L2_A_DIR = OUT_ROOT / "layer2" / "branch_a"
L2_B_DIR = OUT_ROOT / "layer2" / "branch_b"

SUMMARY_JSON = OUT_ROOT / "post_rescue_verification_summary.json"

# ---------------------------------------------------------------------
# Basic IO
# ---------------------------------------------------------------------

def import_module_from_path(module_name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required script not found: {path}")

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {path}")

    module = importlib.util.module_from_spec(spec)

    # Needed for modules that import sibling files.
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(path.parent))
        except ValueError:
            pass

    return module


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def count_csv_rows(path: Path) -> int:
    return len(read_csv(path))


def clean(x: Any) -> str:
    return str(x or "").strip()


# ---------------------------------------------------------------------
# Layer 1 common deterministic inventory
# ---------------------------------------------------------------------

def run_layer1_common() -> Dict[str, Any]:
    mod00 = import_module_from_path(
        "post_rescue_layer1_common",
        L1_COMMON_SCRIPT,
    )

    L1_COMMON_DIR.mkdir(parents=True, exist_ok=True)

    schema_candidates = sorted(
        (ROOT / "schemas").rglob("rules_v3.json")
    )

    if len(schema_candidates) != 1:
        raise RuntimeError(
            "Expected exactly one rules_v3.json schema, found: "
            + ", ".join(str(path) for path in schema_candidates)
        )

    schema_path = schema_candidates[0]
    schema = mod00.load_schema(schema_path)
    validator = mod00.Draft7Validator(schema)

    branch_paths = {
        "A_post_rescue": A_RESCUED_AST,
        "B_post_rescue": B_RESCUED_AST,
    }

    all_leaf_rows = []
    all_ast_rows = []
    summaries = {}

    for branch_name, ast_path in branch_paths.items():
        leaf_rows, ast_rows, summary = mod00.scan_branch(
            branch_name=branch_name,
            ast_path=ast_path,
            schema_validator=validator,
        )
        all_leaf_rows.extend(leaf_rows)
        all_ast_rows.extend(ast_rows)
        summaries[branch_name] = summary

    leaf_csv = L1_COMMON_DIR / "deterministic_verification_inventory_leaf_level.csv"
    ast_csv = L1_COMMON_DIR / "deterministic_verification_inventory_ast_level.csv"
    summary_json = L1_COMMON_DIR / "deterministic_verification_inventory_summary.json"

    fieldnames = [
        "branch",
        "document_id",
        "path",
        "criterion_id",
        "item_uid",
        "clause_id",
        "entity_type",
        "entity_text",
        "operator",
        "value_type",
        "value",
        "unit",
        "temporal_context",
        "history_context",
        "computability",
        "non_computable_reason",
        "evidence_text",
        "deterministic_issues",
        "layer1a_action_category",
        "layer1a_action_hint",
        "layer1c_source_text_warnings",
        "layer1c_action",
    ]

    ast_fieldnames = [
        "branch",
        "document_id",
        "issue_type",
        "criterion_id",
        "paths",
        "layer1b_action_category",
        "detail",
    ]

    mod00.write_csv(leaf_csv, all_leaf_rows, fieldnames)
    mod00.write_csv(ast_csv, all_ast_rows, ast_fieldnames)

    combined_leaf_issue_counter = Counter()
    combined_ast_issue_counter = Counter()
    combined_action_counter = Counter()
    combined_source_warning_counter = Counter()
    combined_source_action_counter = Counter()

    for s in summaries.values():
        combined_leaf_issue_counter.update(s["layer1a_hard_leaf_issue_counts"])
        combined_ast_issue_counter.update(s["layer1b_ast_integrity_issue_counts"])
        combined_action_counter.update(s["action_category_counts"])
        combined_source_warning_counter.update(s["layer1c_source_text_warning_counts"])
        combined_source_action_counter.update(s["layer1c_action_counts"])

    summary = {
        "stage": "06g_post_rescue_layer1_common",
        "description": "Post-rescue Layer 1 common deterministic inventory.",
        "inputs": {
            "schema": str(schema_path),
            "A_post_rescue": str(A_RESCUED_AST),
            "B_post_rescue": str(B_RESCUED_AST),
        },
        "outputs": {
            "leaf_csv": str(leaf_csv),
            "ast_csv": str(ast_csv),
            "summary_json": str(summary_json),
        },
        "branches": summaries,
        "combined": {
            "layer1a_hard_leaf_issue_counts": dict(combined_leaf_issue_counter.most_common()),
            "layer1b_ast_integrity_issue_counts": dict(combined_ast_issue_counter.most_common()),
            "layer1c_source_text_warning_counts": dict(combined_source_warning_counter.most_common()),
            "action_category_counts": dict(combined_action_counter.most_common()),
            "layer1c_action_counts": dict(combined_source_action_counter.most_common()),
        },
    }

    mod00.write_json(summary_json, summary)

    return summary


# ---------------------------------------------------------------------
# Layer 1D Pass1/Pass2 consistency
# ---------------------------------------------------------------------

def run_layer1d() -> Dict[str, Any]:
    mod01 = import_module_from_path(
        "post_rescue_layer1d",
        L1D_SCRIPT,
    )

    L1D_DIR.mkdir(parents=True, exist_ok=True)

    mod01.PASS1_PATH_OVERRIDE = PASS1_OUTPUTS

    mod01.BRANCH_AST_PATHS = {
        "A_post_rescue": A_RESCUED_AST,
        "B_post_rescue": B_RESCUED_AST,
    }
    mod01.OUT_DIR = L1D_DIR
    mod01.AUDIT_CSV_PATH = L1D_DIR / "layer1d_pass1_pass2_consistency_audit.csv"
    mod01.SUMMARY_JSON_PATH = L1D_DIR / "layer1d_pass1_pass2_consistency_summary.json"

    mod01.main()

    return read_json(mod01.SUMMARY_JSON_PATH)


# ---------------------------------------------------------------------
# Layer 1 policies
# ---------------------------------------------------------------------

def run_layer1_policy_a() -> Dict[str, Any]:
    mod02a = import_module_from_path(
        "post_rescue_layer1_policy_a",
        L1_POLICY_A_SCRIPT,
    )

    L1_POLICY_A_DIR.mkdir(parents=True, exist_ok=True)

    mod02a.LAYER1_INV_DIR = L1_COMMON_DIR
    mod02a.LAYER1D_DIR = L1D_DIR
    mod02a.LEAF_INVENTORY_CSV = L1_COMMON_DIR / "deterministic_verification_inventory_leaf_level.csv"
    mod02a.AST_INVENTORY_CSV = L1_COMMON_DIR / "deterministic_verification_inventory_ast_level.csv"
    mod02a.LAYER1D_AUDIT_CSV = L1D_DIR / "layer1d_pass1_pass2_consistency_audit.csv"

    mod02a.OUT_DIR = L1_POLICY_A_DIR
    mod02a.OUT_CSV = L1_POLICY_A_DIR / "layer1_policy_branch_a_leaf_level.csv"
    mod02a.OUT_JSON = L1_POLICY_A_DIR / "layer1_policy_branch_a_summary.json"

    mod02a.DETERMINISTIC_BRANCH_NAME = "A_post_rescue"
    mod02a.LAYER1D_BRANCH_NAME = "A_post_rescue"
    mod02a.POLICY_BRANCH_NAME = "A"

    mod02a.main()

    return read_json(mod02a.OUT_JSON)


def run_layer1_policy_b() -> Dict[str, Any]:
    mod02b = import_module_from_path(
        "post_rescue_layer1_policy_b",
        L1_POLICY_B_SCRIPT,
    )

    L1_POLICY_B_DIR.mkdir(parents=True, exist_ok=True)

    mod02b.LAYER1_INV_DIR = L1_COMMON_DIR
    mod02b.LAYER1D_DIR = L1D_DIR
    mod02b.LEAF_INVENTORY_CSV = L1_COMMON_DIR / "deterministic_verification_inventory_leaf_level.csv"
    mod02b.AST_INVENTORY_CSV = L1_COMMON_DIR / "deterministic_verification_inventory_ast_level.csv"
    mod02b.LAYER1D_AUDIT_CSV = L1D_DIR / "layer1d_pass1_pass2_consistency_audit.csv"

    mod02b.OUT_DIR = L1_POLICY_B_DIR
    mod02b.OUT_CSV = L1_POLICY_B_DIR / "layer1_policy_branch_b_leaf_level.csv"
    mod02b.OUT_JSON = L1_POLICY_B_DIR / "layer1_policy_branch_b_summary.json"

    mod02b.DETERMINISTIC_BRANCH_NAME = "B_post_rescue"
    mod02b.LAYER1D_BRANCH_NAME = "B_post_rescue"
    mod02b.POLICY_BRANCH_NAME = "B"

    mod02b.main()

    return read_json(mod02b.OUT_JSON)


# ---------------------------------------------------------------------
# Layer 2 Branch A
# ---------------------------------------------------------------------

def run_layer2_branch_a_inventory() -> Dict[str, Any]:
    mod03 = import_module_from_path(
        "post_rescue_layer2_a_inventory",
        L2_A_INVENTORY_SCRIPT,
    )

    L2_A_DIR.mkdir(parents=True, exist_ok=True)

    out_csv = L2_A_DIR / "layer2_branch_a_support_inventory_leaf_level.csv"
    out_json = L2_A_DIR / "layer2_branch_a_support_inventory_summary.json"

    # Force Layer 1 issue lookup to use the post-rescue Layer 1 outputs.
    mod03.LAYER1_OPTIONAL_FILES = [
        ("layer1_inventory", L1_COMMON_DIR / "deterministic_verification_inventory_leaf_level.csv"),
        ("layer1d_pass1_consistency", L1D_DIR / "layer1d_pass1_pass2_consistency_audit.csv"),
        ("layer1_policy_branch_a", L1_POLICY_A_DIR / "layer1_policy_branch_a_leaf_level.csv"),
    ]

    leaves = mod03.load_branch_a_leaves_from_ast(A_RESCUED_AST, A_RESCUED_AST.name)
    pass2_rows = mod03.load_jsonl(PASS2_INPUTS)
    pass2_clause_index = mod03.build_pass2_clause_index(pass2_rows)

    leaf_ids = {leaf["criterion"].get("criterion_id") for leaf in leaves}
    pass2_ids = set(pass2_clause_index.keys())

    missing_in_pass2 = sorted(x for x in leaf_ids if x not in pass2_ids)

    if missing_in_pass2:
        raise RuntimeError(
            "Post-rescue Branch A AST leaves are not compatible with Pass2 inputs. "
            f"Missing in Pass2 index: {len(missing_in_pass2)}. "
            f"Examples: {missing_in_pass2[:5]}"
        )

    layer1_issue_index = mod03.read_layer1_issue_index(ROOT)

    inventory_rows = []
    missing_pass2_context = 0

    for leaf in leaves:
        criterion_id = leaf["criterion"].get("criterion_id")
        if criterion_id not in pass2_clause_index:
            missing_pass2_context += 1

        inventory_rows.append(
            mod03.compute_leaf_inventory_row(
                leaf=leaf,
                pass2_clause_index=pass2_clause_index,
                layer1_issue_index=layer1_issue_index,
            )
        )

    summary = mod03.make_summary(
        rows=inventory_rows,
        source_ast_path=A_RESCUED_AST,
        pass2_input_path=PASS2_INPUTS,
        layer1_issue_index=layer1_issue_index,
    )
    summary["missing_pass2_clause_context"] = missing_pass2_context
    summary["stage"] = "06g_post_rescue_layer2_branch_a_inventory"

    mod03.write_csv(out_csv, inventory_rows)
    mod03.write_json(out_json, summary)

    return summary


def run_layer2_branch_a_score() -> Dict[str, Any]:
    mod03a = import_module_from_path(
        "post_rescue_layer2_a_score",
        L2_A_SCORE_SCRIPT,
    )

    in_csv = L2_A_DIR / "layer2_branch_a_support_inventory_leaf_level.csv"
    out_csv = L2_A_DIR / "layer2_branch_a_leaf_risk_scores.csv"
    out_json = L2_A_DIR / "layer2_branch_a_risk_summary.json"

    inventory_rows = mod03a.read_csv(in_csv)
    scored_rows = [mod03a.compute_leaf_score(row) for row in inventory_rows]
    summary = mod03a.make_summary(scored_rows, in_csv, out_csv)
    summary["stage"] = "06g_post_rescue_layer2_branch_a_score"

    mod03a.write_csv(out_csv, scored_rows)
    mod03a.write_json(out_json, summary)

    return summary


# ---------------------------------------------------------------------
# Layer 2 Branch B
# ---------------------------------------------------------------------

def run_layer2_branch_b_inventory() -> Dict[str, Any]:
    mod04 = import_module_from_path(
        "post_rescue_layer2_b_inventory",
        L2_B_INVENTORY_SCRIPT,
    )

    L2_B_DIR.mkdir(parents=True, exist_ok=True)

    out_csv = L2_B_DIR / "layer2_branch_b_support_inventory_leaf_level.csv"
    out_json = L2_B_DIR / "layer2_branch_b_support_inventory_summary.json"

    ast_docs = mod04.read_jsonl(B_RESCUED_AST)
    pass2_rows = mod04.read_jsonl(PASS2_INPUTS)
    pass2_index = mod04.build_pass2_context_index(pass2_rows)

    branch_a_score_csv = L2_A_DIR / "layer2_branch_a_leaf_risk_scores.csv"
    branch_a_rows = mod04.read_csv(branch_a_score_csv)
    branch_a_index = mod04.build_branch_a_index(branch_a_rows)

    policy_rows = mod04.read_csv(L1_POLICY_B_DIR / "layer1_policy_branch_b_leaf_level.csv")
    policy_index = mod04.build_branch_b_layer1_policy_index(policy_rows)

    inventory_rows = []

    for doc in ast_docs:
        leaves = mod04.extract_leaves_from_doc(doc)

        for leaf, leaf_path in leaves:
            criterion_id = mod04.get_leaf_value(leaf, "criterion_id", "")

            layer1_meta = mod04.get_layer1_metadata(
                doc=doc,
                leaf_path=leaf_path,
                criterion_id=criterion_id,
            )

            pass2_context = mod04.find_pass2_context(leaf, doc, pass2_index)
            branch_a_row = mod04.find_branch_a_row(leaf, doc, branch_a_index)

            policy_row = mod04.find_branch_b_layer1_policy_row(
                leaf=leaf,
                doc=doc,
                policy_index=policy_index,
            )

            row = mod04.compute_leaf_inventory_row(
                doc=doc,
                leaf=leaf,
                leaf_path=leaf_path,
                pass2_context=pass2_context,
                branch_a_row=branch_a_row,
                layer1_policy_row=policy_row,
                layer1_meta=layer1_meta,
            )
            inventory_rows.append(row)

    mod04.write_csv(out_csv, inventory_rows)

    summary = mod04.make_summary(
        rows=inventory_rows,
        ast_path=B_RESCUED_AST,
        pass2_path=PASS2_INPUTS,
        branch_a_path=branch_a_score_csv,
    )
    summary["stage"] = "06g_post_rescue_layer2_branch_b_inventory"

    mod04.write_json(out_json, summary)

    return summary


def run_layer2_branch_b_screen() -> Dict[str, Any]:
    mod04a = import_module_from_path(
        "post_rescue_layer2_b_screen",
        L2_B_SCREEN_SCRIPT,
    )

    in_csv = L2_B_DIR / "layer2_branch_b_support_inventory_leaf_level.csv"
    out_csv = L2_B_DIR / "layer2_branch_b_grounding_screen_leaf_level.csv"
    out_json = L2_B_DIR / "layer2_branch_b_grounding_screen_summary.json"

    inventory_rows = mod04a.read_csv(in_csv)

    if not inventory_rows:
        raise RuntimeError("Post-rescue Branch B Layer 2 inventory is empty.")

    scored_rows = [mod04a.score_row(row) for row in inventory_rows]

    mod04a.write_csv(out_csv, scored_rows)

    summary = mod04a.summarize_scores(scored_rows)
    summary["stage"] = "06g_post_rescue_layer2_branch_b_grounding_screen"

    mod04a.write_json(out_json, summary)

    return summary


# ---------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------

def compare_pre_post_counts() -> Dict[str, Any]:
    pre_a_score = (
        ROOT
        / "outputs"
        / "verification"
        / "layer2"
        / "branch_a"
        / "layer2_branch_a_leaf_risk_scores.csv"
    )

    post_a_score = (
        L2_A_DIR
        / "layer2_branch_a_leaf_risk_scores.csv"
    )

    pre_b_screen = (
        ROOT
        / "outputs"
        / "verification"
        / "layer2"
        / "branch_b"
        / "layer2_branch_b_grounding_screen_leaf_level.csv"
    )

    post_b_screen = (
        L2_B_DIR
        / "layer2_branch_b_grounding_screen_leaf_level.csv"
    )

    def counter_from_csv(path: Path, col: str) -> Dict[str, int]:
        rows = read_csv(path)
        return dict(
            Counter(clean(r.get(col)) for r in rows).most_common()
        )

    return {
        "branch_a_risk_label_counts": {
            "pre": (
                counter_from_csv(pre_a_score, "risk_label")
                if pre_a_score.exists()
                else {}
            ),
            "post": (
                counter_from_csv(post_a_score, "risk_label")
                if post_a_score.exists()
                else {}
            ),
        },
        "branch_b_semantic_grounding_risk_label_counts": {
            "pre": (
                counter_from_csv(
                    pre_b_screen,
                    "semantic_grounding_risk_label",
                )
                if pre_b_screen.exists()
                else {}
            ),
            "post": (
                counter_from_csv(
                    post_b_screen,
                    "semantic_grounding_risk_label",
                )
                if post_b_screen.exists()
                else {}
            ),
        },
        "branch_b_final_routing_decision_counts": {
            "pre": (
                counter_from_csv(
                    pre_b_screen,
                    "final_routing_decision",
                )
                if pre_b_screen.exists()
                else {}
            ),
            "post": (
                counter_from_csv(
                    post_b_screen,
                    "final_routing_decision",
                )
                if post_b_screen.exists()
                else {}
            ),
        },
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("\nLayer 3: rerun verification after rescue")
    print("Branch A post-rescue rule tree:", A_RESCUED_AST)
    print("Branch B post-rescue rule tree:", B_RESCUED_AST)
    print("Output root:", OUT_ROOT)

    if not A_RESCUED_AST.exists():
        raise FileNotFoundError(f"Missing Branch A post-rescue AST: {A_RESCUED_AST}")

    if not B_RESCUED_AST.exists():
        raise FileNotFoundError(f"Missing Branch B post-rescue AST: {B_RESCUED_AST}")

    if not PASS2_INPUTS.exists():
        raise FileNotFoundError(f"Missing Pass2 inputs: {PASS2_INPUTS}")
    
    if "100" in str(A_RESCUED_AST) or "100" in str(B_RESCUED_AST) or "100" in str(PASS2_INPUTS):
        raise RuntimeError("06G is still pointing to old 100-run files. Change paths to 200.")

    if "200" not in str(A_RESCUED_AST) or "200" not in str(B_RESCUED_AST) or "200" not in str(PASS2_INPUTS):
        raise RuntimeError("06G expected CHIA-200 inputs but paths do not contain 200.")

    print("\n[1/7] Re-running Layer 1 common deterministic inventory...")
    l1_common = run_layer1_common()

    print("\n[2/7] Re-running Layer 1D Pass1/Pass2 consistency...")
    l1d = run_layer1d()

    print("\n[3/7] Re-running Branch A Layer 1 policy...")
    l1_policy_a = run_layer1_policy_a()

    print("\n[4/7] Re-running Branch B Layer 1 policy...")
    l1_policy_b = run_layer1_policy_b()

    print("\n[5/7] Re-running Branch A Layer 2 inventory and score...")
    l2_a_inventory = run_layer2_branch_a_inventory()
    l2_a_score = run_layer2_branch_a_score()

    print("\n[6/7] Re-running Branch B Layer 2 inventory and grounding screen...")
    l2_b_inventory = run_layer2_branch_b_inventory()
    l2_b_screen = run_layer2_branch_b_screen()

    print("\n[7/7] Writing combined post-rescue summary...")
    comparison = compare_pre_post_counts()

    summary = {
        "stage": "11_rerun_verification_after_rescue",
        "description": (
            "Reruns Layer 1 and Layer 2 verification after validated "
            "candidate-selection rescue was applied. This step does not call "
            "the LLM, modify rule trees, or use manual labels."
        ),
        "inputs": {
            "branch_a_post_rescue_ast": str(A_RESCUED_AST),
            "branch_b_post_rescue_ast": str(B_RESCUED_AST),
            "pass2_inputs": str(PASS2_INPUTS),
        },
        "output_root": str(OUT_ROOT),
        "outputs": {
            "layer1_common_leaf_csv": str(L1_COMMON_DIR / "deterministic_verification_inventory_leaf_level.csv"),
            "layer1_common_ast_csv": str(L1_COMMON_DIR / "deterministic_verification_inventory_ast_level.csv"),
            "layer1d_audit_csv": str(L1D_DIR / "layer1d_pass1_pass2_consistency_audit.csv"),
            "layer1_policy_a_csv": str(L1_POLICY_A_DIR / "layer1_policy_branch_a_leaf_level.csv"),
            "layer1_policy_b_csv": str(L1_POLICY_B_DIR / "layer1_policy_branch_b_leaf_level.csv"),
            "layer2_a_inventory_csv": str(L2_A_DIR / "layer2_branch_a_support_inventory_leaf_level.csv"),
            "layer2_a_score_csv": str(L2_A_DIR / "layer2_branch_a_leaf_risk_scores.csv"),
            "layer2_b_inventory_csv": str(L2_B_DIR / "layer2_branch_b_support_inventory_leaf_level.csv"),
            "layer2_b_screen_csv": str(L2_B_DIR / "layer2_branch_b_grounding_screen_leaf_level.csv"),
            "summary_json": str(SUMMARY_JSON),
        },
        "row_counts": {
            "layer1_common_leaf_rows": count_csv_rows(L1_COMMON_DIR / "deterministic_verification_inventory_leaf_level.csv"),
            "layer1d_audit_rows": count_csv_rows(L1D_DIR / "layer1d_pass1_pass2_consistency_audit.csv"),
            "layer1_policy_a_rows": count_csv_rows(L1_POLICY_A_DIR / "layer1_policy_branch_a_leaf_level.csv"),
            "layer1_policy_b_rows": count_csv_rows(L1_POLICY_B_DIR / "layer1_policy_branch_b_leaf_level.csv"),
            "layer2_a_score_rows": count_csv_rows(L2_A_DIR / "layer2_branch_a_leaf_risk_scores.csv"),
            "layer2_b_screen_rows": count_csv_rows(L2_B_DIR / "layer2_branch_b_grounding_screen_leaf_level.csv"),
        },
        "key_post_rescue_summaries": {
            "layer1_common": l1_common.get("combined", {}),
            "layer1d": l1d.get("combined", {}),
            "branch_a_layer2_risk_counts": l2_a_score.get("risk_label_counts", {}),
            "branch_b_semantic_grounding_risk_counts": l2_b_screen.get("semantic_grounding_risk_label_counts", {}),
            "branch_b_final_routing_counts": l2_b_screen.get("final_routing_decision_counts", {}),
        },
        "pre_post_comparison": comparison,
        "method_notes": [
            "No LLM calls are made in this script.",
            "No rule-tree files are modified in this script.",
            "These outputs are used by the final-decision script.",
            "Manual evaluation is performed separately on post-rescue outputs.",
        ],
    }

    write_json(SUMMARY_JSON, summary)

    print("\nDONE")
    print("Post-rescue verification summary:", SUMMARY_JSON)

    print("\nRow counts:")
    print(summary["row_counts"])

    print("\nBranch A risk counts, pre vs post:")
    print(comparison["branch_a_risk_label_counts"])

    print("\nBranch B semantic grounding risk counts, pre vs post:")
    print(comparison["branch_b_semantic_grounding_risk_label_counts"])

    print("\nBranch B final routing counts, pre vs post:")
    print(comparison["branch_b_final_routing_decision_counts"])


if __name__ == "__main__":
    main()

# Run from the repository root:
# python scripts/03_verification/03_layer3/11_rerun_verification_after_rescue.py