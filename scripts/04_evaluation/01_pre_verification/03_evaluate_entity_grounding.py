"""
03_evaluate_entity_grounding.py

Evaluate the pre-verification entity grounding of Branch A and Branch B.

The diagnostic checks:
    - whether entity_text is non-empty
    - whether entity_text appears in evidence_text
    - whether clause-level PubMedBERT candidate spans are available
    - whether the extracted entity overlaps those candidate spans
    - whether the entity type agrees with the best candidate span
    - whether the entity text is overly generic

The PubMedBERT candidate spans are used only as evaluation signals.
Branch B did not use them during extraction.

Inputs:
    outputs/extraction/pass2_inputs/
    outputs/extraction/branch_a/pass2_leaves/
    outputs/extraction/branch_b/pass2_leaves_llm/

Outputs:
    outputs/evaluation/pre_verification/
        entity_grounding_pre_verification_A_B/

This script does not call the LLM and does not modify predictions.

Run from the repository root:
python scripts/04_evaluation/01_pre_verification/03_evaluate_entity_grounding.py
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------
# IO helpers
# --------------------------------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# --------------------------------------------------
# Normalization helpers
# --------------------------------------------------

def norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9\+\-]+", norm(text))


def token_overlap(a: str, b: str) -> float:
    ta = set(tokenize(a))
    tb = set(tokenize(b))

    if not ta or not tb:
        return 0.0

    return len(ta & tb) / len(ta | tb)


def contains_normalized(container: str, content: str) -> bool:
    c1 = norm(container)
    c2 = norm(content)

    if not c1 or not c2:
        return False

    return c2 in c1


def rate(x: int, n: int) -> Optional[float]:
    return round(x / n, 4) if n else None


# --------------------------------------------------
# Anchor/type helpers
# --------------------------------------------------

ANCHOR_LABEL_TO_ENTITY_TYPE = {
    "Condition": "condition",
    "Drug": "drug",
    "Procedure": "procedure",
    "Measurement": "lab",
    "Device": "other",
}

GENERIC_ENTITY_TERMS = {
    "treated",
    "active",
    "suspected",
    "acute",
    "chronic",
    "history",
    "dose",
    "only",
    "stable",
    "severe",
    "uncontrolled",
    "unresolved",
}


def choose_best_anchor(anchors: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not anchors:
        return None

    anchors = sorted(
        anchors,
        key=lambda x: (
            -(float(x.get("score")) if x.get("score") is not None else -1.0),
            -len(str(x.get("text") or "")),
        ),
    )

    return anchors[0]


def is_generic_entity(entity_text: str) -> bool:
    t = norm(entity_text)
    return t in GENERIC_ENTITY_TERMS


# --------------------------------------------------
# Compatibility check
# --------------------------------------------------

def build_expected_clause_counts(pass2_inputs_rows: List[Dict[str, Any]]) -> Dict[str, int]:
    expected = {}

    for row in pass2_inputs_rows:
        if row.get("status") != "ok":
            continue

        payload = row.get("pass2_input", {})
        item_uid = payload.get("item_uid")

        if item_uid:
            expected[item_uid] = len(payload.get("clauses", []))

    return expected


def build_observed_clause_counts(pass2_leaves_rows: List[Dict[str, Any]]) -> Dict[str, int]:
    observed = {}

    for row in pass2_leaves_rows:
        if row.get("status") != "ok":
            continue

        payload = row.get("pass2_output", {})
        item_uid = payload.get("item_uid")

        if item_uid:
            observed[item_uid] = len(payload.get("criteria", []))

    return observed


def validate_branch_compatible(
    branch_name: str,
    pass2_inputs_rows: List[Dict[str, Any]],
    pass2_leaves_rows: List[Dict[str, Any]],
) -> None:
    """
    Prevent stale-branch comparisons.
    A and B must use the same current pass2_inputs.
    """
    expected = build_expected_clause_counts(pass2_inputs_rows)
    observed = build_observed_clause_counts(pass2_leaves_rows)

    missing = []
    mismatched = []
    extra = []

    for item_uid, expected_n in expected.items():
        observed_n = observed.get(item_uid)

        if observed_n is None:
            missing.append(item_uid)
        elif observed_n != expected_n:
            mismatched.append(
                {
                    "item_uid": item_uid,
                    "expected": expected_n,
                    "observed": observed_n,
                }
            )

    for item_uid in observed:
        if item_uid not in expected:
            extra.append(item_uid)

    if missing or mismatched or extra:
        raise RuntimeError(
            f"{branch_name} is not compatible with current pass2_inputs. "
            f"Missing={len(missing)}, mismatched={len(mismatched)}, extra={len(extra)}. "
            f"First missing={missing[:5]}, first mismatched={mismatched[:5]}, first extra={extra[:5]}"
        )


# --------------------------------------------------
# Core branch evaluation
# --------------------------------------------------

def evaluate_branch(
    branch_name: str,
    pass2_inputs_path: Path,
    pass2_leaves_path: Path,
    out_root: Path,
) -> Optional[Dict[str, Any]]:
    if not pass2_leaves_path.exists():
        print(f"[SKIP] {branch_name}: file not found: {pass2_leaves_path}")
        return None

    pass2_inputs_rows = [r for r in load_jsonl(pass2_inputs_path) if r.get("status") == "ok"]
    pass2_leaves_rows = [r for r in load_jsonl(pass2_leaves_path) if r.get("status") == "ok"]

    validate_branch_compatible(
        branch_name=branch_name,
        pass2_inputs_rows=pass2_inputs_rows,
        pass2_leaves_rows=pass2_leaves_rows,
    )

    inputs_by_item_uid = {r["item_uid"]: r for r in pass2_inputs_rows}
    leaves_by_item_uid = {r["item_uid"]: r for r in pass2_leaves_rows}

    common_item_uids = sorted(set(inputs_by_item_uid) & set(leaves_by_item_uid))

    detail_rows = []

    n_total_clauses = 0
    n_entity_nonempty = 0
    n_entity_in_evidence = 0
    n_anchor_present = 0
    n_entity_overlaps_any_anchor = 0
    n_best_anchor_type_match = 0
    n_best_anchor_text_match = 0
    n_generic_entity = 0
    n_no_anchor = 0

    # Extra diagnostic:
    # no_anchor = no main clinical BERT anchor
    # no_bert_candidate = no BERT signal at all
    n_any_bert_candidate_present = 0
    n_no_bert_candidate = 0
    n_entity_in_evidence_with_anchor = 0
    n_entity_in_evidence_no_anchor = 0

    n_entity_overlaps_any_anchor_with_anchor = 0
    n_best_anchor_text_match_with_anchor = 0

    issue_counts = Counter()

    for item_uid in common_item_uids:
        input_row = inputs_by_item_uid[item_uid]["pass2_input"]
        leaves_row = leaves_by_item_uid[item_uid]["pass2_output"]

        clauses = {c["clause_id"]: c for c in input_row.get("clauses", [])}
        criteria = leaves_row.get("criteria", [])

        for entry in criteria:
            clause_id = entry.get("clause_id")
            criterion = entry.get("criterion", {})
            clause = clauses.get(clause_id, {})

            entity_text = criterion.get("entity_text", "")
            entity_type = criterion.get("entity_type", "")
            evidence_text = criterion.get("evidence_text", "")
            bert_candidates = clause.get("bert_candidates") or {}

            anchors = bert_candidates.get("anchors") or []
            supports = bert_candidates.get("supports") or []
            others = bert_candidates.get("others") or []

            any_bert_candidate_present = bool(anchors or supports or others)

            best_anchor = choose_best_anchor(anchors)
            best_anchor_text = best_anchor.get("text") if best_anchor else None
            best_anchor_label = best_anchor.get("label") if best_anchor else None
            best_anchor_type = (
                ANCHOR_LABEL_TO_ENTITY_TYPE.get(best_anchor_label)
                if best_anchor_label
                else None
            )

            entity_nonempty = bool(norm(entity_text))
            entity_in_evidence = contains_normalized(evidence_text, entity_text)

            anchor_present = len(anchors) > 0
            any_anchor_overlap = any(
                token_overlap(entity_text, a.get("text", "")) > 0
                for a in anchors
            )
            best_anchor_text_match = token_overlap(entity_text, best_anchor_text or "") > 0
            best_anchor_type_match = (
                best_anchor_type == entity_type
                if best_anchor_type is not None
                else None
            )
            generic_flag = is_generic_entity(entity_text)

            issues = []

            if not entity_nonempty:
                issues.append("missing_entity_text")
            if entity_nonempty and not entity_in_evidence:
                issues.append("entity_not_in_evidence")
            if not anchor_present:
                issues.append("no_anchor_for_clause")
            if anchor_present and not any_anchor_overlap:
                issues.append("entity_not_aligned_with_any_anchor")
            if anchor_present and best_anchor_type_match is False:
                issues.append("entity_type_disagrees_with_best_anchor")
            if generic_flag:
                issues.append("generic_entity_text")

            if entity_nonempty:
                n_entity_nonempty += 1
            if entity_in_evidence:
                n_entity_in_evidence += 1
            if anchor_present:
                n_anchor_present += 1
            else:
                n_no_anchor += 1
            if anchor_present and entity_in_evidence:
                n_entity_in_evidence_with_anchor += 1
            if any_bert_candidate_present:
                n_any_bert_candidate_present += 1
            else:
                n_no_bert_candidate += 1

            if (not anchor_present) and entity_in_evidence:
                n_entity_in_evidence_no_anchor += 1

            if anchor_present and any_anchor_overlap:
                n_entity_overlaps_any_anchor_with_anchor += 1

            if anchor_present and best_anchor_text_match:
                n_best_anchor_text_match_with_anchor += 1
            if any_anchor_overlap:
                n_entity_overlaps_any_anchor += 1
            if best_anchor_text_match:
                n_best_anchor_text_match += 1
            if best_anchor_type_match is True:
                n_best_anchor_type_match += 1
            if generic_flag:
                n_generic_entity += 1

            for issue in issues:
                issue_counts[issue] += 1

            n_total_clauses += 1

            detail_rows.append(
                {
                    "branch": branch_name,
                    "document_id": leaves_row.get("document_id"),
                    "chia_id": leaves_row.get("chia_id"),
                    "item_uid": item_uid,
                    "criterion_type": leaves_row.get("criterion_type"),
                    "clause_id": clause_id,
                    "clause_text": clause.get("clause_text"),
                    "evidence_text": evidence_text,
                    "entity_text": entity_text,
                    "entity_type": entity_type,
                    "best_anchor_text": best_anchor_text,
                    "best_anchor_label": best_anchor_label,
                    "best_anchor_mapped_type": best_anchor_type,
                    "entity_nonempty": entity_nonempty,
                    "entity_in_evidence": entity_in_evidence,
                    "anchor_present": anchor_present,
                    "support_present": len(supports) > 0,
                    "other_present": len(others) > 0,
                    "any_bert_candidate_present": any_bert_candidate_present,
                    "n_anchors": len(anchors),
                    "n_supports": len(supports),
                    "n_others": len(others),
                    "entity_overlaps_any_anchor": any_anchor_overlap,
                    "best_anchor_text_match": best_anchor_text_match,
                    "best_anchor_type_match": best_anchor_type_match,
                    "generic_entity_flag": generic_flag,
                    "issues": issues,
                }
            )

    summary = {
        "evaluation_stage": "entity_grounding_pre_verification_A_B",
        "branch": branch_name,
        "input_pass2_inputs": str(pass2_inputs_path),
        "input_pass2_leaves": str(pass2_leaves_path),
        "n_items_evaluated": len(common_item_uids),
        "n_total_clauses": n_total_clauses,
        "entity_nonempty_rate": rate(n_entity_nonempty, n_total_clauses),
        "entity_in_evidence_rate": rate(n_entity_in_evidence, n_total_clauses),
        "anchor_present_rate": rate(n_anchor_present, n_total_clauses),
        "no_anchor_rate": rate(n_no_anchor, n_total_clauses),
        "any_bert_candidate_present_rate": rate(n_any_bert_candidate_present, n_total_clauses),
        "no_bert_candidate_rate": rate(n_no_bert_candidate, n_total_clauses),
        "entity_in_evidence_rate_when_anchor_present": rate(
            n_entity_in_evidence_with_anchor,
            n_anchor_present,
        ),
        "entity_in_evidence_rate_when_no_anchor": rate(
            n_entity_in_evidence_no_anchor,
            n_no_anchor,
        ),
        "entity_overlaps_any_anchor_rate_when_anchor_present": rate(
            n_entity_overlaps_any_anchor_with_anchor,
            n_anchor_present,
        ),
        "entity_overlaps_best_anchor_rate_when_anchor_present": rate(
            n_best_anchor_text_match_with_anchor,
            n_anchor_present,
        ),
        "entity_overlaps_any_anchor_rate": rate(n_entity_overlaps_any_anchor, n_total_clauses),
        "entity_overlaps_best_anchor_rate": rate(n_best_anchor_text_match, n_total_clauses),
        "best_anchor_type_match_rate": rate(n_best_anchor_type_match, n_anchor_present),
        "generic_entity_rate": rate(n_generic_entity, n_total_clauses),
        "issue_counts": dict(issue_counts),
    }

    out_dir = out_root / branch_name
    out_dir.mkdir(parents=True, exist_ok=True)

    write_json(out_dir / "summary.json", summary)
    write_jsonl(out_dir / "clause_details.jsonl", detail_rows)

    print(f"\n===== ENTITY GROUNDING PRE-VERIFICATION: {branch_name} =====")
    print(f"Pass2 leaves: {pass2_leaves_path}")
    print(f"Items evaluated: {summary['n_items_evaluated']}")
    print(f"Total clauses: {summary['n_total_clauses']}")
    print(f"Entity non-empty rate: {summary['entity_nonempty_rate']}")
    print(f"Entity in evidence rate: {summary['entity_in_evidence_rate']}")
    print(f"Anchor present rate: {summary['anchor_present_rate']}")
    print(f"No-anchor rate: {summary['no_anchor_rate']}")
    print(f"Any BERT candidate present rate: {summary['any_bert_candidate_present_rate']}")
    print(f"No BERT candidate rate: {summary['no_bert_candidate_rate']}")
    print(
        "Entity in evidence rate | with anchor:",
        summary["entity_in_evidence_rate_when_anchor_present"],
    )
    print(
        "Entity in evidence rate | no anchor:",
        summary["entity_in_evidence_rate_when_no_anchor"],
    )
    print(
        "Entity overlaps any anchor rate | with anchor:",
        summary["entity_overlaps_any_anchor_rate_when_anchor_present"],
    )
    print(
        "Entity overlaps best anchor rate | with anchor",
        summary["entity_overlaps_best_anchor_rate_when_anchor_present"],
    )
    print(f"Entity overlaps any anchor rate: {summary['entity_overlaps_any_anchor_rate']}")
    print(f"Best anchor text match rate: {summary['entity_overlaps_best_anchor_rate']}")
    print(f"Best anchor type match rate: {summary['best_anchor_type_match_rate']}")
    print(f"Generic entity rate: {summary['generic_entity_rate']}")
    print(f"Issue counts: {summary['issue_counts']}")

    return summary


# --------------------------------------------------
# Main
# --------------------------------------------------
def main() -> None:
    ROOT = Path(__file__).resolve().parents[3]

    pass2_inputs_path = (
        ROOT
        / "outputs"
        / "extraction"
        / "pass2_inputs"
        / "chia_text_only_200_pass2_inputs.jsonl"
    )

    branch_paths = {
        "A_bert_rules": (
            ROOT
            / "outputs"
            / "extraction"
            / "branch_a"
            / "pass2_leaves"
            / "chia_text_only_200_pass2_leaves.jsonl"
        ),
        "B_llm_pass2": (
            ROOT
            / "outputs"
            / "extraction"
            / "branch_b"
            / "pass2_leaves_llm"
            / "chia_text_only_200_pass2_leaves_llm.jsonl"
        ),
    }

    out_root = (
        ROOT
        / "outputs"
        / "evaluation"
        / "pre_verification"
        / "entity_grounding_pre_verification_A_B"
    )

    required_paths = {
        "Pass 2 inputs": pass2_inputs_path,
        **{
            f"{branch_name} leaves": path
            for branch_name, path in branch_paths.items()
        },
    }

    for name, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    out_root.mkdir(parents=True, exist_ok=True)

    all_summaries = {}

    for branch_name, pass2_leaves_path in branch_paths.items():
        summary = evaluate_branch(
            branch_name=branch_name,
            pass2_inputs_path=pass2_inputs_path,
            pass2_leaves_path=pass2_leaves_path,
            out_root=out_root,
        )

        if summary is not None:
            all_summaries[branch_name] = {
                "n_items_evaluated": summary["n_items_evaluated"],
                "n_total_clauses": summary["n_total_clauses"],
                "entity_nonempty_rate": summary[
                    "entity_nonempty_rate"
                ],
                "entity_in_evidence_rate": summary[
                    "entity_in_evidence_rate"
                ],
                "anchor_present_rate": summary[
                    "anchor_present_rate"
                ],
                "no_anchor_rate": summary["no_anchor_rate"],
                "any_bert_candidate_present_rate": summary[
                    "any_bert_candidate_present_rate"
                ],
                "no_bert_candidate_rate": summary[
                    "no_bert_candidate_rate"
                ],
                "entity_overlaps_any_anchor_rate": summary[
                    "entity_overlaps_any_anchor_rate"
                ],
                "entity_overlaps_best_anchor_rate": summary[
                    "entity_overlaps_best_anchor_rate"
                ],
                "best_anchor_type_match_rate": summary[
                    "best_anchor_type_match_rate"
                ],
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
                "generic_entity_rate": summary[
                    "generic_entity_rate"
                ],
                "issue_counts": summary["issue_counts"],
            }

    all_summary_path = out_root / "all_branch_summary.json"
    write_json(all_summary_path, all_summaries)

    print("\n===== PRE-VERIFICATION ENTITY GROUNDING SUMMARY =====")
    print("Pass 2 inputs:", pass2_inputs_path)
    print("Output:", all_summary_path)


if __name__ == "__main__":
    main()


# Run from the repository root:
# python scripts/04_evaluation/01_pre_verification/03_evaluate_entity_grounding.py