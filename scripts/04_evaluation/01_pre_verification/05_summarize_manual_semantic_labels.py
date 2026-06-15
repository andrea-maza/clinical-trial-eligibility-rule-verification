"""
05_summarize_manual_semantic_labels.py

Summarize the completed pre-verification manual semantic labels for
Branch A and Branch B.

Inputs:
    outputs/evaluation/pre_verification/
        semantic_manual_pre_verification_A_B/
            semantic_pass1_logic_sheet.csv
            semantic_clause_sheet_A_B.csv

Outputs:
    outputs/evaluation/pre_verification/
        semantic_manual_pre_verification_A_B_summary/

Important:
    This script must be run only on the completed manually reviewed
    CSV files. It does not create manual labels.

This script does not call the LLM and does not modify extraction outputs.

Run from the repository root only when intentionally rebuilding the
manual-label summaries:
python scripts/04_evaluation/01_pre_verification/05_summarize_manual_semantic_labels.py
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


VALID_LEAF_LABELS = {"correct", "partial", "incorrect", "unsupported"}
VALID_LOGIC_LABELS = {"correct", "partial", "incorrect"}

VALID_BEST_VALUES = {"A", "B", "tie", "none"}

LABEL_SCORE = {
    "unsupported": 0,
    "incorrect": 0,
    "partial": 1,
    "correct": 2,
}


# --------------------------------------------------
# IO helpers
# --------------------------------------------------

def load_csv(path: Path) -> List[Dict[str, str]]:
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
    last_error = None

    for enc in encodings:
        try:
            rows = []
            with open(path, "r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append({
                        k: (v.strip() if isinstance(v, str) else v)
                        for k, v in row.items()
                    })

            print(f"Loaded {path.name} with encoding: {enc}")
            return rows

        except UnicodeDecodeError as e:
            last_error = e

    raise RuntimeError(f"Could not decode {path}. Last error: {last_error}")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------
# Basic helpers
# --------------------------------------------------

def normalize_label(x: Any) -> str:
    return str(x or "").strip().lower()


def normalize_best(x: Any) -> str:
    value = str(x or "").strip()

    if value.lower() == "tie":
        return "tie"
    if value.lower() == "none":
        return "none"
    if value.upper() in {"A", "B"}:
        return value.upper()

    return ""


def is_valid_leaf_label(x: Any) -> bool:
    return normalize_label(x) in VALID_LEAF_LABELS


def is_valid_logic_label(x: Any) -> bool:
    return normalize_label(x) in VALID_LOGIC_LABELS


def split_issue_types(x: Any) -> List[str]:
    if not x:
        return []

    out = []

    for part in str(x).replace(",", ";").split(";"):
        issue = part.strip().lower()
        if issue:
            out.append(issue)

    return out


def safe_pct(num: int, den: int) -> Optional[float]:
    if den == 0:
        return None
    return round(100.0 * num / den, 2)


def get_branch_label(row: Dict[str, Any], branch: str) -> str:
    return normalize_label(row.get(f"manual_{branch}_leaf_label"))


# --------------------------------------------------
# Pass 1 logic summaries
# --------------------------------------------------

def summarize_pass1_logic(logic_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    labeled = [
        r for r in logic_rows
        if is_valid_logic_label(r.get("manual_logic_label"))
    ]

    label_counts = Counter(
        normalize_label(r.get("manual_logic_label"))
        for r in labeled
    )

    issue_counts = Counter()
    missing_issue_count = 0

    for r in labeled:
        label = normalize_label(r.get("manual_logic_label"))
        issue = str(r.get("manual_logic_issue_type") or "").strip()

        if label in {"partial", "incorrect"}:
            issue_counts.update(split_issue_types(issue))

            if not issue:
                missing_issue_count += 1

    return {
        "n_total_items": len(logic_rows),
        "n_labeled_items": len(labeled),
        "n_unlabeled_items": len(logic_rows) - len(labeled),
        "label_coverage_pct": safe_pct(len(labeled), len(logic_rows)),
        "label_counts": {
            label: label_counts.get(label, 0)
            for label in ["correct", "partial", "incorrect"]
        },
        "label_percentages_among_labeled": {
            label: safe_pct(label_counts.get(label, 0), len(labeled))
            for label in ["correct", "partial", "incorrect"]
        },
        "issue_counts_among_partial_or_incorrect": dict(issue_counts.most_common()),
        "partial_or_incorrect_rows_missing_issue_type": missing_issue_count,
    }


def summarize_pass1_logic_by_stratum(logic_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets = defaultdict(list)

    for r in logic_rows:
        stratum = str(r.get("stratum") or "unknown").strip().lower()
        buckets[stratum].append(r)

    out = {}

    for stratum, bucket in sorted(buckets.items()):
        labeled = [
            r for r in bucket
            if is_valid_logic_label(r.get("manual_logic_label"))
        ]

        counts = Counter(
            normalize_label(r.get("manual_logic_label"))
            for r in labeled
        )

        out[stratum] = {
            "n_total_items": len(bucket),
            "n_labeled_items": len(labeled),
            "coverage_pct": safe_pct(len(labeled), len(bucket)),
            "label_counts": {
                label: counts.get(label, 0)
                for label in ["correct", "partial", "incorrect"]
            },
            "label_percentages_among_labeled": {
                label: safe_pct(counts.get(label, 0), len(labeled))
                for label in ["correct", "partial", "incorrect"]
            },
        }

    return out


# --------------------------------------------------
# Branch summaries
# --------------------------------------------------

def summarize_branch(rows: List[Dict[str, Any]], branch: str) -> Dict[str, Any]:
    label_field = f"manual_{branch}_leaf_label"
    issue_field = f"manual_{branch}_issue_type"

    total_rows = len(rows)

    labeled = [
        r for r in rows
        if is_valid_leaf_label(r.get(label_field))
    ]

    label_counts = Counter(
        normalize_label(r.get(label_field))
        for r in labeled
    )

    issue_counts = Counter()
    missing_issue_count = 0

    for r in labeled:
        label = normalize_label(r.get(label_field))
        issue = str(r.get(issue_field) or "").strip()

        if label != "correct":
            issue_counts.update(split_issue_types(issue))

            if not issue:
                missing_issue_count += 1

    return {
        "branch": branch,
        "n_total_rows": total_rows,
        "n_labeled_rows": len(labeled),
        "n_unlabeled_rows": total_rows - len(labeled),
        "label_coverage_pct": safe_pct(len(labeled), total_rows),
        "label_counts": {
            label: label_counts.get(label, 0)
            for label in ["correct", "partial", "incorrect", "unsupported"]
        },
        "label_percentages_among_labeled": {
            label: safe_pct(label_counts.get(label, 0), len(labeled))
            for label in ["correct", "partial", "incorrect", "unsupported"]
        },
        "issue_counts_among_non_correct": dict(issue_counts.most_common()),
        "non_correct_rows_missing_issue_type": missing_issue_count,
    }


def summarize_branch_by_stratum(rows: List[Dict[str, Any]], branch: str) -> Dict[str, Any]:
    label_field = f"manual_{branch}_leaf_label"

    buckets = defaultdict(list)

    for r in rows:
        stratum = str(r.get("stratum") or "unknown").strip().lower()
        buckets[stratum].append(r)

    out = {}

    for stratum, bucket in sorted(buckets.items()):
        labeled = [
            r for r in bucket
            if is_valid_leaf_label(r.get(label_field))
        ]

        counts = Counter(
            normalize_label(r.get(label_field))
            for r in labeled
        )

        out[stratum] = {
            "n_total_rows": len(bucket),
            "n_labeled_rows": len(labeled),
            "coverage_pct": safe_pct(len(labeled), len(bucket)),
            "label_counts": {
                label: counts.get(label, 0)
                for label in ["correct", "partial", "incorrect", "unsupported"]
            },
            "label_percentages_among_labeled": {
                label: safe_pct(counts.get(label, 0), len(labeled))
                for label in ["correct", "partial", "incorrect", "unsupported"]
            },
        }

    return out


# --------------------------------------------------
# A/B comparison
# --------------------------------------------------

def infer_best_branch_from_labels(row: Dict[str, Any]) -> str:
    a_label = get_branch_label(row, "A")
    b_label = get_branch_label(row, "B")

    if a_label not in VALID_LEAF_LABELS or b_label not in VALID_LEAF_LABELS:
        return ""

    a_score = LABEL_SCORE[a_label]
    b_score = LABEL_SCORE[b_label]

    if a_score == 0 and b_score == 0:
        return "none"

    if a_score > b_score:
        return "A"

    if b_score > a_score:
        return "B"

    return "tie"


def compare_b_to_a(row: Dict[str, Any]) -> str:
    a_label = get_branch_label(row, "A")
    b_label = get_branch_label(row, "B")

    if a_label not in VALID_LEAF_LABELS or b_label not in VALID_LEAF_LABELS:
        return "unlabeled"

    if LABEL_SCORE[b_label] > LABEL_SCORE[a_label]:
        return "B_better_than_A"

    if LABEL_SCORE[b_label] < LABEL_SCORE[a_label]:
        return "B_worse_than_A"

    return "B_same_as_A"


def summarize_b_vs_a(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    comparisons = [
        r.get("B_vs_A")
        for r in rows
        if r.get("B_vs_A") and r.get("B_vs_A") != "unlabeled"
    ]

    counts = Counter(comparisons)

    return {
        "n_comparable_rows": len(comparisons),
        "counts": dict(counts),
        "percentages": {
            k: safe_pct(v, len(comparisons))
            for k, v in counts.items()
        },
    }


def summarize_manual_best(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    usable = [
        normalize_best(r.get("manual_best_branch_A_B"))
        for r in rows
        if normalize_best(r.get("manual_best_branch_A_B")) in VALID_BEST_VALUES
    ]

    counts = Counter(usable)

    return {
        "n_rows_with_manual_best": len(usable),
        "counts": dict(counts),
        "percentages": {
            k: safe_pct(v, len(usable))
            for k, v in counts.items()
        },
    }


def summarize_inferred_best(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    usable = [
        r.get("manual_inferred_best_branch")
        for r in rows
        if r.get("manual_inferred_best_branch")
    ]

    counts = Counter(usable)

    return {
        "n_rows_with_both_branch_labels": len(usable),
        "counts": dict(counts),
        "percentages": {
            k: safe_pct(v, len(usable))
            for k, v in counts.items()
        },
    }


def add_comparison_columns(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []

    for r in rows:
        row = dict(r)

        row["manual_A_leaf_label"] = normalize_label(row.get("manual_A_leaf_label"))
        row["manual_B_leaf_label"] = normalize_label(row.get("manual_B_leaf_label"))
        row["manual_best_branch_A_B"] = normalize_best(row.get("manual_best_branch_A_B"))

        row["manual_inferred_best_branch"] = infer_best_branch_from_labels(row)
        row["B_vs_A"] = compare_b_to_a(row)

        out.append(row)

    return out


# --------------------------------------------------
# Quality checks
# --------------------------------------------------

def collect_invalid_or_incomplete_rows(
    logic_rows: List[Dict[str, Any]],
    clause_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    problems = []

    for r in logic_rows:
        label = normalize_label(r.get("manual_logic_label"))
        issue = str(r.get("manual_logic_issue_type") or "").strip()

        if label and label not in VALID_LOGIC_LABELS:
            problems.append({
                "sheet": "logic",
                "review_id": r.get("review_id"),
                "item_uid": r.get("item_uid"),
                "clause_id": "",
                "problem": "invalid_manual_logic_label",
                "value": label,
            })

        if label in {"partial", "incorrect"} and not issue:
            problems.append({
                "sheet": "logic",
                "review_id": r.get("review_id"),
                "item_uid": r.get("item_uid"),
                "clause_id": "",
                "problem": "missing_logic_issue_type",
                "value": label,
            })

    for r in clause_rows:
        for branch in ["A", "B"]:
            label_field = f"manual_{branch}_leaf_label"
            issue_field = f"manual_{branch}_issue_type"

            label = normalize_label(r.get(label_field))
            issue = str(r.get(issue_field) or "").strip()

            if label and label not in VALID_LEAF_LABELS:
                problems.append({
                    "sheet": "clause",
                    "review_id": r.get("review_id"),
                    "item_uid": r.get("item_uid"),
                    "clause_id": r.get("clause_id"),
                    "branch": branch,
                    "problem": f"invalid_manual_{branch}_leaf_label",
                    "value": label,
                })

            if label in {"partial", "incorrect", "unsupported"} and not issue:
                problems.append({
                    "sheet": "clause",
                    "review_id": r.get("review_id"),
                    "item_uid": r.get("item_uid"),
                    "clause_id": r.get("clause_id"),
                    "branch": branch,
                    "problem": f"missing_manual_{branch}_issue_type",
                    "value": label,
                })

        best = normalize_best(r.get("manual_best_branch_A_B"))
        if r.get("manual_best_branch_A_B") and best not in VALID_BEST_VALUES:
            problems.append({
                "sheet": "clause",
                "review_id": r.get("review_id"),
                "item_uid": r.get("item_uid"),
                "clause_id": r.get("clause_id"),
                "problem": "invalid_manual_best_branch_A_B",
                "value": r.get("manual_best_branch_A_B"),
            })

    return problems


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    ROOT = Path(__file__).resolve().parents[3]

    in_dir = (
        ROOT
        / "outputs"
        / "evaluation"
        / "pre_verification"
        / "semantic_manual_pre_verification_A_B"
    )

    logic_path = in_dir / "semantic_pass1_logic_sheet.csv"
    ab_path = in_dir / "semantic_clause_sheet_A_B.csv"

    out_dir = (
        ROOT
        / "outputs"
        / "evaluation"
        / "pre_verification"
        / "semantic_manual_pre_verification_A_B_summary"
    )

    reviewed_csv_path = (
        out_dir / "reviewed_semantic_clause_labels_A_B.csv"
    )
    invalid_csv_path = (
        out_dir / "invalid_or_incomplete_manual_rows_A_B.csv"
    )
    summary_json_path = (
        out_dir / "semantic_manual_summary_A_B.json"
    )

    required_inputs = {
        "completed Pass 1 logic review sheet": logic_path,
        "completed A/B clause review sheet": ab_path,
    }

    for name, path in required_inputs.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    existing_outputs = [
        path
        for path in [
            reviewed_csv_path,
            invalid_csv_path,
            summary_json_path,
        ]
        if path.exists()
    ]

    if existing_outputs:
        raise FileExistsError(
            "Manual evaluation summaries already exist and will not be overwritten:\n"
            + "\n".join(str(path) for path in existing_outputs)
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    reviewed_csv_path = out_dir / "reviewed_semantic_clause_labels_A_B.csv"
    invalid_csv_path = out_dir / "invalid_or_incomplete_manual_rows_A_B.csv"
    summary_json_path = out_dir / "semantic_manual_summary_A_B.json"

    logic_rows = load_csv(logic_path)
    clause_rows = load_csv(ab_path)

    reviewed_clause_rows = add_comparison_columns(clause_rows)

    invalid_or_incomplete = collect_invalid_or_incomplete_rows(
        logic_rows=logic_rows,
        clause_rows=reviewed_clause_rows,
    )

    reviewed_fieldnames = list(reviewed_clause_rows[0].keys()) if reviewed_clause_rows else []
    write_csv(reviewed_csv_path, reviewed_clause_rows, reviewed_fieldnames)

    invalid_fieldnames = [
        "sheet",
        "review_id",
        "item_uid",
        "clause_id",
        "branch",
        "problem",
        "value",
    ]
    write_csv(invalid_csv_path, invalid_or_incomplete, invalid_fieldnames)

    summary = {
        "stage": "semantic_manual_summary_A_B",
        "inputs": {
            "pass1_logic_sheet": str(logic_path),
            "A_B_clause_sheet": str(ab_path),
        },
        "outputs": {
            "reviewed_clause_labels": str(reviewed_csv_path),
            "invalid_or_incomplete_rows": str(invalid_csv_path),
            "summary_json": str(summary_json_path),
        },
        "interpretation_note": (
            "Clause-level labels compare Pass 2 leaf semantics between Branch A and Branch B. "
            "Pass 1 logic labels evaluate the shared decomposition/logic once. "
            "The inferred best branch is based only on label scores and may differ from the manually selected best branch."
        ),
        "pass1_logic_summary": summarize_pass1_logic(logic_rows),
        "pass1_logic_by_stratum": summarize_pass1_logic_by_stratum(logic_rows),
        "pass2_leaf_semantic_summary": {
            "A": summarize_branch(reviewed_clause_rows, "A"),
            "B": summarize_branch(reviewed_clause_rows, "B"),
        },
        "pass2_leaf_semantic_by_stratum": {
            "A": summarize_branch_by_stratum(reviewed_clause_rows, "A"),
            "B": summarize_branch_by_stratum(reviewed_clause_rows, "B"),
        },
        "branch_comparison": {
            "B_vs_A": summarize_b_vs_a(reviewed_clause_rows),
            "manual_best_branch_A_B": summarize_manual_best(reviewed_clause_rows),
            "inferred_best_branch_from_labels": summarize_inferred_best(reviewed_clause_rows),
        },
        "quality_checks": {
            "n_invalid_or_incomplete_rows": len(invalid_or_incomplete),
        },
    }

    write_json(summary_json_path, summary)

    print("\n===== SEMANTIC MANUAL SUMMARY: A/B =====")
    print("Reviewed clause CSV:", reviewed_csv_path)
    print("Invalid/incomplete rows CSV:", invalid_csv_path)
    print("Summary JSON:", summary_json_path)

    print("\n--- Pass 1 logic summary ---")
    p1 = summary["pass1_logic_summary"]
    print(f"Labeled items: {p1['n_labeled_items']}/{p1['n_total_items']} ({p1['label_coverage_pct']}%)")
    print("Label counts:", p1["label_counts"])
    print("Label %:", p1["label_percentages_among_labeled"])
    print("Issue counts:", p1["issue_counts_among_partial_or_incorrect"])

    print("\n--- Pass 2 leaf semantic summary ---")
    for branch in ["A", "B"]:
        s = summary["pass2_leaf_semantic_summary"][branch]
        print(f"\n{branch}")
        print(f"Labeled rows: {s['n_labeled_rows']}/{s['n_total_rows']} ({s['label_coverage_pct']}%)")
        print("Label counts:", s["label_counts"])
        print("Label %:", s["label_percentages_among_labeled"])
        print("Issue counts:", s["issue_counts_among_non_correct"])

    print("\n--- Branch comparison ---")
    print("B_vs_A:", summary["branch_comparison"]["B_vs_A"])
    print("Manual best:", summary["branch_comparison"]["manual_best_branch_A_B"])
    print("Inferred best:", summary["branch_comparison"]["inferred_best_branch_from_labels"])

    print("\n--- Quality checks ---")
    print("Invalid/incomplete rows:", len(invalid_or_incomplete))


if __name__ == "__main__":
    main()

# Run from the repository root only when intentionally rebuilding summaries:
# python scripts/04_evaluation/01_pre_verification/05_summarize_manual_semantic_labels.py