""" 
01_inspect_layer1_action_hints.py
Inspect examples of deterministic Layer 1 action hints and source-text
warnings. This diagnostic script does not modify any logical rule tree. 

Run from the repository root: 
python scripts/03_verification/diagnostics/01_inspect_layer1_action_hints.py 
"""
import csv
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]

CSV_PATH = (
    ROOT
    / "outputs"
    / "verification"
    / "layer1"
    / "deterministic_inventory"
    / "deterministic_verification_inventory_leaf_level.csv"
)

# These are action hints from Layer 1A.
# This script does NOT repair anything.
# It only prints examples for inspection.
ACTION_HINT_CANDIDATES = {
    "safe_normalization_candidate",
    "conservative_rewrite_candidate",
}

MAX_EXAMPLES_PER_ISSUE = 5


REQUIRED_COLUMNS = {
    "branch",
    "document_id",
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
}


def split_issues(x: str):
    if not x:
        return []
    return [i.strip() for i in str(x).split(";") if i.strip()]


def short(x, n=220):
    x = str(x or "").replace("\n", " ").strip()
    return x if len(x) <= n else x[:n] + "..."


def require_columns(rows):
    if not rows:
        raise RuntimeError(f"No rows found in {CSV_PATH}")

    missing = REQUIRED_COLUMNS - set(rows[0].keys())

    if missing:
        raise RuntimeError(
            "The input CSV does not match the current Layer 1 inventory format. "
            f"Missing columns: {sorted(missing)}"
        )


def print_row(r):
    print(f"  BRANCH: {r['branch']}")
    print(f"  DOC:    {r['document_id']}")
    print(f"  ITEM:   {r['item_uid']}")
    print(f"  CLAUSE: {r['clause_id']}")
    print(f"  ENTITY: [{r['entity_type']}] {r['entity_text']}")
    print(
        f"  OP:     {r['operator']} | "
        f"value_type={r['value_type']} | "
        f"value={r['value']} | "
        f"unit={r['unit']}"
    )
    print(f"  TEMP:   {r['temporal_context']}")
    print(f"  HIST:   {r['history_context']}")
    print(f"  COMP:   {r['computability']} | reason={short(r['non_computable_reason'], 140)}")
    print(f"  L1A ISSUES: {r['deterministic_issues']}")
    print(f"  L1A ACTION: {r['layer1a_action_category']}")
    print(f"  L1A HINT:   {short(r['layer1a_action_hint'], 180)}")
    print(f"  L1C WARN:   {r['layer1c_source_text_warnings']}")
    print(f"  L1C ACTION: {r['layer1c_action']}")
    print(f"  EVID:       {short(r['evidence_text'], 260)}")


def print_issue_examples(title, issue_to_rows, max_examples):
    for issue, issue_rows in issue_to_rows.items():
        print("\n" + "=" * 100)
        print(title, issue)
        print(f"N rows: {len(issue_rows)}")
        print("=" * 100)

        for idx, r in enumerate(issue_rows[:max_examples], start=1):
            print(f"\nExample {idx}")
            print_row(r)


def main():
    with open(CSV_PATH, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    require_columns(rows)

    action_hint_rows = [
        r for r in rows
        if r["layer1a_action_category"] in ACTION_HINT_CANDIDATES
    ]

    layer1a_flag_only_rows = [
        r for r in rows
        if r["layer1a_action_category"] == "flag_only"
        and r["deterministic_issues"]
    ]

    layer1c_flag_rows = [
        r for r in rows
        if r["layer1c_action"] == "flag_only"
        and r["layer1c_source_text_warnings"]
    ]

    print("\n===== LAYER 1A / 1C ACTION-HINT INSPECTION =====")
    print("Input CSV:", CSV_PATH)
    print("Total leaf rows:", len(rows))

    print("\n--- Rows by branch ---")
    print(Counter(r["branch"] for r in rows))

    print("\n--- Layer 1A action categories ---")
    print(Counter(r["layer1a_action_category"] for r in rows))

    print("\n--- Layer 1C actions ---")
    print(Counter(r["layer1c_action"] for r in rows))

    print("\n--- Main row groups ---")
    print("Layer 1A safe/conservative action-hint rows:", len(action_hint_rows))
    print("Layer 1A flag-only rows:", len(layer1a_flag_only_rows))
    print("Layer 1C source-warning rows:", len(layer1c_flag_rows))

    print("\n--- Layer 1A action-hint rows by branch ---")
    print(Counter(r["branch"] for r in action_hint_rows))

    print("\n--- Layer 1A flag-only rows by branch ---")
    print(Counter(r["branch"] for r in layer1a_flag_only_rows))

    print("\n--- Layer 1C source-warning rows by branch ---")
    print(Counter(r["branch"] for r in layer1c_flag_rows))

    # --------------------------------------------------------
    # Layer 1A safe/conservative candidates
    # --------------------------------------------------------
    action_issue_counter = Counter()
    action_issue_to_rows = defaultdict(list)

    for r in action_hint_rows:
        for issue in split_issues(r["deterministic_issues"]):
            action_issue_counter[issue] += 1
            action_issue_to_rows[issue].append(r)

    print("\n===== LAYER 1A SAFE / CONSERVATIVE ACTION-HINT ISSUES =====")
    for issue, count in action_issue_counter.most_common():
        print(f"{issue}: {count}")

    print_issue_examples(
        title="LAYER 1A ACTION-HINT ISSUE:",
        issue_to_rows=action_issue_to_rows,
        max_examples=MAX_EXAMPLES_PER_ISSUE,
    )

    # --------------------------------------------------------
    # Layer 1A flag-only issues
    # --------------------------------------------------------
    flag_issue_counter = Counter()
    flag_issue_to_rows = defaultdict(list)

    for r in layer1a_flag_only_rows:
        for issue in split_issues(r["deterministic_issues"]):
            flag_issue_counter[issue] += 1
            flag_issue_to_rows[issue].append(r)

    print("\n===== LAYER 1A FLAG-ONLY ISSUES: NOT TO AUTO-REPAIR =====")
    for issue, count in flag_issue_counter.most_common():
        print(f"{issue}: {count}")

    print_issue_examples(
        title="LAYER 1A FLAG-ONLY ISSUE:",
        issue_to_rows=flag_issue_to_rows,
        max_examples=3,
    )

    # --------------------------------------------------------
    # Layer 1C source-text warnings
    # --------------------------------------------------------
    l1c_counter = Counter()
    l1c_to_rows = defaultdict(list)

    for r in layer1c_flag_rows:
        for issue in split_issues(r["layer1c_source_text_warnings"]):
            l1c_counter[issue] += 1
            l1c_to_rows[issue].append(r)

    print("\n===== LAYER 1C SOURCE-TEXT WARNING ISSUES =====")
    for issue, count in l1c_counter.most_common():
        print(f"{issue}: {count}")

    print_issue_examples(
        title="LAYER 1C WARNING:",
        issue_to_rows=l1c_to_rows,
        max_examples=3,
    )


if __name__ == "__main__":
    main()

# Run from the repository root:
# python .\scripts\03_verification\01_layer1\diagnostics\01_inspect_layer1_action_hints.py