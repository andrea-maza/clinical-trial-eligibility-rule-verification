"""
06_summarize_manual_semantic_labels.py

Summarize the completed post-verification manual semantic labels for
Branch A and Branch B.

Unchanged branch-specific leaves may reuse their pre-verification
labels. Leaves changed by verification or rescue must have independent
post-verification labels.

Inputs:
    outputs/evaluation/post_verification/
        semantic_manual_post_verification_A_B/
            semantic_clause_sheet_A_B_post_verification.csv

    outputs/evaluation/pre_verification/
        semantic_manual_pre_verification_A_B_summary/
            reviewed_semantic_clause_labels_A_B.csv

Outputs:
    outputs/evaluation/post_verification/
        semantic_manual_post_verification_A_B_summary/
            merged_semantic_post_labels_A_B.csv
            semantic_manual_post_summary_A_B.json

This script summarizes existing labels. It does not create manual
labels, call the LLM, or modify predictions.

Run from the repository root:
python scripts/04_evaluation/02_post_verification/06_summarize_manual_semantic_labels.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


VALID_LABELS = {"correct", "partial", "incorrect"}

LABEL_SCORE = {
    "incorrect": 0,
    "partial": 1,
    "correct": 2,
}


# --------------------------------------------------
# IO helpers
# --------------------------------------------------

def load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV file: {path}")

    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
    last_error = None

    for enc in encodings:
        try:
            rows = []

            with path.open("r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    rows.append(
                        {
                            k: (v.strip() if isinstance(v, str) else v)
                            for k, v in row.items()
                        }
                    )

            print(f"Loaded {path.name} with encoding: {enc}")
            return rows

        except UnicodeDecodeError as exc:
            last_error = exc

    raise RuntimeError(f"Could not decode {path}. Last error: {last_error}")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({c: row.get(c, "") for c in fieldnames})


# --------------------------------------------------
# Basic helpers
# --------------------------------------------------

def normalize_label(x: Any) -> str:
    return str(x or "").strip().lower()


def is_valid_label(x: Any) -> bool:
    return normalize_label(x) in VALID_LABELS


def safe_pct(num: int, den: int) -> Optional[float]:
    if den == 0:
        return None

    return round(100.0 * num / den, 2)


def split_issue_types(x: Any) -> List[str]:
    if not x:
        return []

    out = []

    for part in str(x).replace(",", ";").split(";"):
        issue = part.strip().lower()

        if issue:
            out.append(issue)

    return out


def label_score(label: Any) -> Optional[int]:
    label = normalize_label(label)

    if label not in LABEL_SCORE:
        return None

    return LABEL_SCORE[label]


def label_change(pre_label: Any, post_label: Any) -> str:
    pre = normalize_label(pre_label)
    post = normalize_label(post_label)

    if pre not in VALID_LABELS or post not in VALID_LABELS:
        return ""

    pre_score = LABEL_SCORE[pre]
    post_score = LABEL_SCORE[post]

    if post_score > pre_score:
        return "improved"

    if post_score < pre_score:
        return "worsened"

    return "same"

def normalize_cell(x: Any) -> str:
    return str(x or "").strip()


def make_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return (
        normalize_cell(row.get("item_uid")),
        normalize_cell(row.get("clause_id")),
    )


def build_pre_manual_index(pre_rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, str]]:
    out = {}

    for row in pre_rows:
        key = make_key(row)

        if key[0] and key[1]:
            out[key] = row

    return out


def attach_pre_reference_columns(
    post_rows: List[Dict[str, str]],
    pre_index: Dict[Tuple[str, str], Dict[str, str]],
) -> List[Dict[str, Any]]:
    """
    Adds pre-verification labels internally for summary calculations only.
    This does not modify the original manual labeling CSV.
    """
    out = []

    for row in post_rows:
        row = dict(row)
        pre = pre_index.get(make_key(row), {})

        for branch in ["A", "B"]:
            pre_leaf_col = f"{branch}_leaf"
            post_leaf_col = f"{branch}_post_leaf"

            pre_label_col = f"manual_{branch}_leaf_label"
            pre_issue_col = f"manual_{branch}_issue_type"

            pre_leaf = normalize_cell(pre.get(pre_leaf_col))
            post_leaf = normalize_cell(row.get(post_leaf_col))

            row[f"manual_{branch}_pre_leaf_label_reference_only"] = normalize_label(
                pre.get(pre_label_col)
            )
            row[f"manual_{branch}_pre_issue_type_reference_only"] = normalize_cell(
                pre.get(pre_issue_col)
            )

            if pre:
                row[f"{branch}_leaf_changed_post_vs_pre"] = (
                    "1" if pre_leaf != post_leaf else "0"
                )
            else:
                row[f"{branch}_leaf_changed_post_vs_pre"] = ""

        out.append(row)

    return out

def compare_b_to_a(row: Dict[str, Any]) -> str:
    a = normalize_label(row.get("manual_A_post_leaf_label"))
    b = normalize_label(row.get("manual_B_post_leaf_label"))

    if a not in VALID_LABELS or b not in VALID_LABELS:
        return "unlabeled"

    if LABEL_SCORE[b] > LABEL_SCORE[a]:
        return "B_better_than_A"

    if LABEL_SCORE[b] < LABEL_SCORE[a]:
        return "B_worse_than_A"

    return "B_same_as_A"


def infer_best_branch_ab(row: Dict[str, Any]) -> str:
    a = normalize_label(row.get("manual_A_post_leaf_label"))
    b = normalize_label(row.get("manual_B_post_leaf_label"))

    if a not in VALID_LABELS or b not in VALID_LABELS:
        return ""

    a_score = LABEL_SCORE[a]
    b_score = LABEL_SCORE[b]

    if a_score == 0 and b_score == 0:
        return "none"

    if a_score > b_score:
        return "A"

    if b_score > a_score:
        return "B"

    return "tie"


def chosen_branch_manual_label(row: Dict[str, Any]) -> str:
    chosen = str(row.get("cross_chosen_branch") or "").strip()

    if chosen == "A":
        return normalize_label(row.get("manual_A_post_leaf_label"))

    if chosen == "B":
        return normalize_label(row.get("manual_B_post_leaf_label"))

    return ""


# --------------------------------------------------
# Branch summaries
# --------------------------------------------------

def summarize_branch(rows: List[Dict[str, Any]], branch: str) -> Dict[str, Any]:
    label_field = f"manual_{branch}_post_leaf_label"
    issue_field = f"manual_{branch}_post_issue_type"

    total_rows = len(rows)

    labeled = [
        row for row in rows
        if is_valid_label(row.get(label_field))
    ]

    label_counts = Counter(
        normalize_label(row.get(label_field))
        for row in labeled
    )

    issue_counts = Counter()

    for row in labeled:
        label = normalize_label(row.get(label_field))

        if label in {"partial", "incorrect"}:
            issue_counts.update(split_issue_types(row.get(issue_field)))

    missing_issue_count = 0

    for row in labeled:
        label = normalize_label(row.get(label_field))
        issue = str(row.get(issue_field) or "").strip()

        if label in {"partial", "incorrect"} and not issue:
            missing_issue_count += 1

    return {
        "branch": branch,
        "n_total_rows": total_rows,
        "n_labeled_rows": len(labeled),
        "n_unlabeled_rows": total_rows - len(labeled),
        "label_coverage_pct": safe_pct(len(labeled), total_rows),
        "label_counts": dict(label_counts),
        "label_percentages_among_labeled": {
            label: safe_pct(label_counts.get(label, 0), len(labeled))
            for label in ["correct", "partial", "incorrect"]
        },
        "issue_counts_among_partial_or_incorrect": dict(issue_counts.most_common()),
        "partial_or_incorrect_rows_missing_issue_type": missing_issue_count,
    }


def summarize_branch_by_stratum(rows: List[Dict[str, Any]], branch: str) -> Dict[str, Any]:
    label_field = f"manual_{branch}_post_leaf_label"

    buckets = defaultdict(list)

    for row in rows:
        stratum = str(row.get("stratum") or "unknown").strip().lower()
        buckets[stratum].append(row)

    out = {}

    for stratum, bucket in sorted(buckets.items()):
        labeled = [
            row for row in bucket
            if is_valid_label(row.get(label_field))
        ]

        counts = Counter(
            normalize_label(row.get(label_field))
            for row in labeled
        )

        out[stratum] = {
            "n_total_rows": len(bucket),
            "n_labeled_rows": len(labeled),
            "coverage_pct": safe_pct(len(labeled), len(bucket)),
            "label_counts": dict(counts),
            "label_percentages_among_labeled": {
                label: safe_pct(counts.get(label, 0), len(labeled))
                for label in ["correct", "partial", "incorrect"]
            },
        }

    return out


def summarize_branch_by_final_decision(rows: List[Dict[str, Any]], branch: str) -> Dict[str, Any]:
    label_field = f"manual_{branch}_post_leaf_label"
    decision_field = f"{branch}_final_decision"

    buckets = defaultdict(list)

    for row in rows:
        decision = str(row.get(decision_field) or "missing").strip()
        buckets[decision].append(row)

    out = {}

    for decision, bucket in sorted(buckets.items()):
        labeled = [
            row for row in bucket
            if is_valid_label(row.get(label_field))
        ]

        counts = Counter(
            normalize_label(row.get(label_field))
            for row in labeled
        )

        out[decision] = {
            "n_total_rows": len(bucket),
            "n_labeled_rows": len(labeled),
            "coverage_pct": safe_pct(len(labeled), len(bucket)),
            "label_counts": dict(counts),
            "label_percentages_among_labeled": {
                label: safe_pct(counts.get(label, 0), len(labeled))
                for label in ["correct", "partial", "incorrect"]
            },
        }

    return out


def summarize_branch_by_rescue_status(rows: List[Dict[str, Any]], branch: str) -> Dict[str, Any]:
    label_field = f"manual_{branch}_post_leaf_label"
    rescue_field = f"{branch}_rescue_apply_status"

    buckets = defaultdict(list)

    for row in rows:
        status = str(row.get(rescue_field) or "not_rescued").strip()
        buckets[status].append(row)

    out = {}

    for status, bucket in sorted(buckets.items()):
        labeled = [
            row for row in bucket
            if is_valid_label(row.get(label_field))
        ]

        counts = Counter(
            normalize_label(row.get(label_field))
            for row in labeled
        )

        out[status] = {
            "n_total_rows": len(bucket),
            "n_labeled_rows": len(labeled),
            "coverage_pct": safe_pct(len(labeled), len(bucket)),
            "label_counts": dict(counts),
            "label_percentages_among_labeled": {
                label: safe_pct(counts.get(label, 0), len(labeled))
                for label in ["correct", "partial", "incorrect"]
            },
        }

    return out


# --------------------------------------------------
# Pre/post transition summaries
# --------------------------------------------------

def summarize_pre_post_transition(
    rows: List[Dict[str, Any]],
    branch: str,
    changed_only: bool = False,
) -> Dict[str, Any]:
    pre_field = f"manual_{branch}_pre_leaf_label_reference_only"
    post_field = f"manual_{branch}_post_leaf_label"
    changed_field = f"{branch}_leaf_changed_post_vs_pre"

    comparable = [
        row for row in rows
        if is_valid_label(row.get(pre_field))
        and is_valid_label(row.get(post_field))
    ]

    if changed_only:
        comparable = [
            row for row in comparable
            if str(row.get(changed_field) or "").strip() == "1"
        ]

    transition_counts = Counter()
    direction_counts = Counter()

    for row in comparable:
        pre = normalize_label(row.get(pre_field))
        post = normalize_label(row.get(post_field))

        transition_counts[f"{pre}_to_{post}"] += 1
        direction_counts[label_change(pre, post)] += 1

    return {
        "branch": branch,
        "changed_only": changed_only,
        "n_rows_with_pre_and_post_labels": len(comparable),
        "transition_counts": dict(transition_counts.most_common()),
        "direction_counts": dict(direction_counts.most_common()),
        "direction_percentages": {
            k: safe_pct(v, len(comparable))
            for k, v in direction_counts.items()
        },
        "important_note": (
            "Pre labels are used as reference labels. "
            "The changed_only version is the main one for evaluating Layer 3 effect, "
            "because unchanged rows may have copied labels."
        ),
    }


# --------------------------------------------------
# Branch comparison summaries
# --------------------------------------------------

def summarize_b_vs_a(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    comparisons = [
        compare_b_to_a(row)
        for row in rows
    ]

    comparisons = [
        x for x in comparisons
        if x != "unlabeled"
    ]

    counts = Counter(comparisons)

    return {
        "n_rows_with_A_and_B_post_labels": len(comparisons),
        "counts": dict(counts),
        "percentages": {
            k: safe_pct(v, len(comparisons))
            for k, v in counts.items()
        },
    }


def summarize_inferred_best_ab(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    usable = [
        row for row in rows
        if row.get("manual_inferred_best_branch_A_B_post")
    ]

    counts = Counter(
        row["manual_inferred_best_branch_A_B_post"]
        for row in usable
    )

    return {
        "n_rows_with_A_and_B_post_labels": len(usable),
        "counts": dict(counts),
        "percentages": {
            k: safe_pct(v, len(usable))
            for k, v in counts.items()
        },
    }


def summarize_cross_branch_choice(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    labeled = [
        row for row in rows
        if is_valid_label(row.get("manual_chosen_branch_post_label"))
    ]

    chosen_counts = Counter(
        str(row.get("cross_chosen_branch") or "").strip()
        for row in rows
        if str(row.get("cross_chosen_branch") or "").strip()
    )

    decision_counts = Counter(
        str(row.get("cross_chosen_decision") or "").strip()
        for row in rows
        if str(row.get("cross_chosen_decision") or "").strip()
    )

    label_counts = Counter(
        normalize_label(row.get("manual_chosen_branch_post_label"))
        for row in labeled
    )

    return {
        "chosen_branch_counts": dict(chosen_counts.most_common()),
        "chosen_decision_counts": dict(decision_counts.most_common()),
        "n_rows_with_manual_label_for_chosen_branch": len(labeled),
        "manual_label_counts_for_chosen_branch": dict(label_counts.most_common()),
        "manual_label_percentages_for_chosen_branch": {
            label: safe_pct(label_counts.get(label, 0), len(labeled))
            for label in ["correct", "partial", "incorrect"]
        },
    }


# --------------------------------------------------
# Enrich rows
# --------------------------------------------------

def enrich_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched = []

    for row in rows:
        row = dict(row)

        a_pre = normalize_label(row.get("manual_A_pre_leaf_label_reference_only"))
        b_pre = normalize_label(row.get("manual_B_pre_leaf_label_reference_only"))
        a_post = normalize_label(row.get("manual_A_post_leaf_label"))
        b_post = normalize_label(row.get("manual_B_post_leaf_label"))

        row["A_pre_to_post_change"] = label_change(a_pre, a_post)
        row["B_pre_to_post_change"] = label_change(b_pre, b_post)

        row["B_vs_A_post"] = compare_b_to_a(row)
        row["manual_inferred_best_branch_A_B_post"] = infer_best_branch_ab(row)
        row["manual_chosen_branch_post_label"] = chosen_branch_manual_label(row)

        enriched.append(row)

    return enriched


# --------------------------------------------------
# Summary
# --------------------------------------------------

def build_summary(
    rows: List[Dict[str, Any]],
    input_sheet: Path,
    pre_reviewed_sheet: Path,
    output_csv: Path,
    output_json: Path,
) -> Dict[str, Any]:
    return {
        "stage": "semantic_manual_post_verification_A_B_summary",
        "inputs": {
            "post_manual_sheet": str(input_sheet),
            "pre_reviewed_manual_labels": str(pre_reviewed_sheet),
        },
        "outputs": {
            "merged_post_labels_csv": str(output_csv),
            "summary_json": str(output_json),
        },
        "interpretation_note": (
            "Manual labels evaluate post-verification Pass 2 leaf semantics. "
            "Pre-verification labels may be reused when a branch-specific leaf "
            "is unchanged. Changed leaves require independent post-verification "
            "review."
        ),
        "valid_manual_labels": ["correct", "partial", "incorrect"],
        "row_counts": {
            "total_clause_rows": len(rows),
        },
        "post_leaf_semantic_summary": {
            "A": summarize_branch(rows, "A"),
            "B": summarize_branch(rows, "B"),
        },
        "post_leaf_semantic_by_stratum": {
            "A": summarize_branch_by_stratum(rows, "A"),
            "B": summarize_branch_by_stratum(rows, "B"),
        },
        "post_leaf_semantic_by_final_decision": {
            "A": summarize_branch_by_final_decision(rows, "A"),
            "B": summarize_branch_by_final_decision(rows, "B"),
        },
        "post_leaf_semantic_by_rescue_status": {
            "A": summarize_branch_by_rescue_status(rows, "A"),
            "B": summarize_branch_by_rescue_status(rows, "B"),
        },
        "pre_post_transition_reference_only": {
            "A_all_labeled": summarize_pre_post_transition(rows, "A", changed_only=False),
            "B_all_labeled": summarize_pre_post_transition(rows, "B", changed_only=False),
            "A_changed_only": summarize_pre_post_transition(rows, "A", changed_only=True),
            "B_changed_only": summarize_pre_post_transition(rows, "B", changed_only=True),
        },
        "branch_comparison_A_vs_B_post": {
            "B_vs_A": summarize_b_vs_a(rows),
            "inferred_best_A_B": summarize_inferred_best_ab(rows),
        },
        "cross_branch_choice_summary": summarize_cross_branch_choice(rows),
        "quality_checks": {
            "A_post_partial_or_incorrect_missing_issue_type": summarize_branch(rows, "A")[
                "partial_or_incorrect_rows_missing_issue_type"
            ],
            "B_post_partial_or_incorrect_missing_issue_type": summarize_branch(rows, "B")[
                "partial_or_incorrect_rows_missing_issue_type"
            ],
        },
        "method_notes": [
            "Pass 1 logic is not relabeled because Pass 1 did not change.",
            "Unchanged branch-specific leaves may reuse pre-verification labels.",
            "Changed leaves require independent post-verification labels.",
            "Changed-only transitions provide the main estimate of the Layer 3 effect.",
        ],
    }


# --------------------------------------------------
# Main
# --------------------------------------------------
def main() -> None:
    ROOT = Path(__file__).resolve().parents[3]

    input_sheet = (
        ROOT
        / "outputs"
        / "evaluation"
        / "post_verification"
        / "semantic_manual_post_verification_A_B"
        / "semantic_clause_sheet_A_B_post_verification.csv"
    )

    pre_reviewed_sheet = (
        ROOT
        / "outputs"
        / "evaluation"
        / "pre_verification"
        / "semantic_manual_pre_verification_A_B_summary"
        / "reviewed_semantic_clause_labels_A_B.csv"
    )

    out_dir = (
        ROOT
        / "outputs"
        / "evaluation"
        / "post_verification"
        / "semantic_manual_post_verification_A_B_summary"
    )

    merged_csv_path = (
        out_dir / "merged_semantic_post_labels_A_B.csv"
    )
    summary_json_path = (
        out_dir / "semantic_manual_post_summary_A_B.json"
    )

    required_inputs = {
        "completed post-verification manual sheet": input_sheet,
        "completed pre-verification reviewed labels": pre_reviewed_sheet,
    }

    for name, path in required_inputs.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_csv(input_sheet)
    pre_rows = load_csv(pre_reviewed_sheet)

    pre_index = build_pre_manual_index(pre_rows)
    rows_with_pre_reference = attach_pre_reference_columns(rows, pre_index)

    enriched_rows = enrich_rows(rows_with_pre_reference)

    fieldnames = list(enriched_rows[0].keys()) if enriched_rows else []

    write_csv(merged_csv_path, enriched_rows, fieldnames)

    summary = build_summary(
        rows=enriched_rows,
        input_sheet=input_sheet,
        pre_reviewed_sheet=pre_reviewed_sheet,
        output_csv=merged_csv_path,
        output_json=summary_json_path,
    )

    write_json(summary_json_path, summary)

    print("\n===== POST-VERIFICATION SEMANTIC MANUAL SUMMARY: A/B =====")
    print("Merged CSV:", merged_csv_path)
    print("Summary JSON:", summary_json_path)

    print("\n--- Branch A post semantic summary ---")
    a = summary["post_leaf_semantic_summary"]["A"]
    print(f"Labeled rows: {a['n_labeled_rows']}/{a['n_total_rows']} ({a['label_coverage_pct']}%)")
    print("Label counts:", a["label_counts"])
    print("Label %:", a["label_percentages_among_labeled"])
    print("Issue counts:", a["issue_counts_among_partial_or_incorrect"])
    print("Missing issue types:", a["partial_or_incorrect_rows_missing_issue_type"])

    print("\n--- Branch B post semantic summary ---")
    b = summary["post_leaf_semantic_summary"]["B"]
    print(f"Labeled rows: {b['n_labeled_rows']}/{b['n_total_rows']} ({b['label_coverage_pct']}%)")
    print("Label counts:", b["label_counts"])
    print("Label %:", b["label_percentages_among_labeled"])
    print("Issue counts:", b["issue_counts_among_partial_or_incorrect"])
    print("Missing issue types:", b["partial_or_incorrect_rows_missing_issue_type"])

    print("\n--- Pre/post transition reference only ---")
    print("A all labeled:", summary["pre_post_transition_reference_only"]["A_all_labeled"])
    print("B all labeled:", summary["pre_post_transition_reference_only"]["B_all_labeled"])
    print("A changed only:", summary["pre_post_transition_reference_only"]["A_changed_only"])
    print("B changed only:", summary["pre_post_transition_reference_only"]["B_changed_only"])

    print("\n--- Branch comparison A vs B post ---")
    print(summary["branch_comparison_A_vs_B_post"])

    print("\n--- Cross-branch choice summary ---")
    print(summary["cross_branch_choice_summary"])


if __name__ == "__main__":
    main()

# python scripts/04_evaluation/02_post_verification/06_summarize_manual_semantic_labels.py