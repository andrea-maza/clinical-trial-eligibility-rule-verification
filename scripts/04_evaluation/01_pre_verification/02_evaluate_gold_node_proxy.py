"""
02_evaluate_gold_node_proxy.py

Evaluate pre-verification Branch A and Branch B leaves against the
CHIA gold annotations using a fair entity span/type proxy.

The proxy:
    - uses entity_text, entity_type, and evidence_text
    - recovers predicted spans from the original criterion text
    - does not use PubMedBERT anchors
    - does not use whole-clause span fallback
    - does not evaluate full semantic correctness

Inputs:
    data/processed/chia_struct_eval_200_gold_graph.jsonl
    outputs/extraction/pass2_inputs/
    outputs/extraction/branch_a/pass2_leaves/
    outputs/extraction/branch_b/pass2_leaves_llm/

Outputs:
    outputs/evaluation/pre_verification/
        gold_nodes_fair_leaf_proxy_pre_verification_A_B/

This script does not call the LLM and does not modify predictions.

Run from the repository root:
python scripts/04_evaluation/01_pre_verification/02_evaluate_gold_node_proxy.py
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter, defaultdict


# ----------------------------
# IO helpers
# ----------------------------

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


# ----------------------------
# Normalization / matching
# ----------------------------

def norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def token_set(text: Any) -> set:
    return set(re.findall(r"[a-z0-9\+\-]+", norm(text)))


def token_jaccard(a: Any, b: Any) -> float:
    ta = token_set(a)
    tb = token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def overlap_len(a_start: Optional[int], a_end: Optional[int],
                b_start: Optional[int], b_end: Optional[int]) -> int:
    if a_start is None or a_end is None or b_start is None or b_end is None:
        return 0
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def safe_div(num: int, den: int) -> Optional[float]:
    if den == 0:
        return None
    return round(num / den, 4)


def f1_score(p: Optional[float], r: Optional[float]) -> Optional[float]:
    if p is None or r is None or p + r == 0:
        return None
    return round(2 * p * r / (p + r), 4)


# ----------------------------
# Type mapping
# ----------------------------

GOLD_KEEP_TYPES = {
    "Condition",
    "Drug",
    "Procedure",
    "Measurement",
    "Observation",
    "Device",
}

PRED_TO_GOLD_TYPE = {
    "condition": "Condition",
    "drug": "Drug",
    "procedure": "Procedure",
    "lab": "Measurement",
    "observation": "Observation",
    # excluded from Level A main metric:
    # demographic, therapy, biomarker, vital, stage, line_of_therapy, other
}


# ----------------------------
# Gold loading
# ----------------------------

def build_gold_index(gold_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Merge inclusion/exclusion gold rows with the same document_id.
    This prevents overwriting one row when CHIA has 200 rows but 100 documents.
    """
    out = {}

    for row in gold_rows:
        doc_id = row.get("document_id")
        if not doc_id:
            continue

        if doc_id not in out:
            out[doc_id] = {
                "document_id": doc_id,
                "nodes": [],
            }

        out[doc_id]["nodes"].extend(row.get("nodes", []))

    return out


def extract_gold_nodes(gold_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    nodes = []

    for node in gold_row.get("nodes", []):
        node_type = node.get("type")
        if node_type not in GOLD_KEEP_TYPES:
            continue

        offsets = node.get("offsets", [])
        if not offsets:
            continue

        # Use first offset span. This is a limitation but keeps the proxy simple.
        start, end = offsets[0]
        texts = node.get("text", [])
        text = texts[0] if texts else ""

        nodes.append({
            "gold_id": node.get("id"),
            "type": node_type,
            "text": text,
            "start": int(start),
            "end": int(end),
        })

    return nodes


# ----------------------------
# Pass 2 input index
# ----------------------------

def build_pass2_input_index(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Key: (item_uid, clause_id)
    Value: clause metadata with item-level offsets.
    """
    out = {}

    for row in rows:
        if row.get("status") != "ok":
            continue

        payload = row.get("pass2_input", {})
        item_uid = payload.get("item_uid")
        item_start_char = payload.get("item_start_char")

        for clause in payload.get("clauses", []):
            clause_id = clause.get("clause_id")

            if item_uid and clause_id:
                out[(item_uid, clause_id)] = {
                    "item_start_char": item_start_char,
                    "item_text": payload.get("item_text"),
                    "clause": clause,
                }

    return out


# ----------------------------
# Fair span recovery
# ----------------------------

def find_span_case_insensitive(container: str, content: str) -> Optional[Tuple[int, int]]:
    """
    Return character span of content inside container without normalizing spaces.
    This keeps returned offsets valid for the original string.
    """
    if not container or not content:
        return None

    idx = container.lower().find(content.lower())
    if idx == -1:
        return None

    return idx, idx + len(content)


def recover_pred_span_fair(
    criterion: Dict[str, Any],
    clause_meta: Dict[str, Any],
) -> Tuple[Optional[int], Optional[int], str]:
    """
    Fair recovery for A/B comparison:
    - Do NOT use BERT anchors.
    - Do NOT fall back to whole clause span for the main metric.
    - Locate evidence_text inside item_text, then entity_text inside evidence_text.
    """
    item_start = clause_meta.get("item_start_char")
    item_text = clause_meta.get("item_text") or ""

    evidence_text = criterion.get("evidence_text", "")
    entity_text = criterion.get("entity_text", "")

    if item_start is None:
        return None, None, "missing_item_start"

    evidence_span = find_span_case_insensitive(item_text, evidence_text)
    if evidence_span is None:
        return None, None, "evidence_not_in_item"

    entity_span_in_evidence = find_span_case_insensitive(evidence_text, entity_text)
    if entity_span_in_evidence is None:
        return None, None, "span_not_recoverable"

    evidence_start, _ = evidence_span
    rel_start, rel_end = entity_span_in_evidence

    doc_start = int(item_start) + evidence_start + rel_start
    doc_end = int(item_start) + evidence_start + rel_end

    return doc_start, doc_end, "entity_in_evidence"


# ----------------------------
# Prediction extraction
# ----------------------------

def extract_pred_nodes(
    pass2_leaf_rows: List[Dict[str, Any]],
    pass2_input_index: Dict[Tuple[str, str], Dict[str, Any]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns:
    - document_id -> comparable predicted nodes
    - flat prediction details
    - coverage stats
    """
    by_doc: Dict[str, List[Dict[str, Any]]] = {}
    details = []

    stats = Counter()
    pred_type_counter = Counter()
    non_comparable_type_counter = Counter()
    span_source_counter = Counter()

    for row in pass2_leaf_rows:
        if row.get("status") != "ok":
            continue

        payload = row.get("pass2_output", {})
        doc_id = payload.get("document_id")
        item_uid = payload.get("item_uid")

        if not doc_id or not item_uid:
            continue

        for entry in payload.get("criteria", []):
            clause_id = entry.get("clause_id")
            criterion = entry.get("criterion", {})

            stats["total_leaves"] += 1

            pred_entity_type = criterion.get("entity_type")
            pred_type_counter[pred_entity_type] += 1

            gold_type = PRED_TO_GOLD_TYPE.get(pred_entity_type)

            detail = {
                "document_id": doc_id,
                "item_uid": item_uid,
                "clause_id": clause_id,
                "criterion_id": criterion.get("criterion_id"),
                "entity_type": pred_entity_type,
                "mapped_gold_type": gold_type,
                "entity_text": criterion.get("entity_text"),
                "evidence_text": criterion.get("evidence_text"),
                "is_comparable": gold_type is not None,
                "span_recoverable": False,
                "span_source": None,
                "start": None,
                "end": None,
            }

            if gold_type is None:
                stats["non_comparable_leaves"] += 1
                non_comparable_type_counter[pred_entity_type] += 1
                details.append(detail)
                continue

            stats["comparable_leaves"] += 1

            clause_meta = pass2_input_index.get((item_uid, clause_id))
            if clause_meta is None:
                stats["missing_clause_meta"] += 1
                detail["span_source"] = "missing_clause_meta"
                details.append(detail)

                by_doc.setdefault(doc_id, []).append({
                    "pred_id": criterion.get("criterion_id"),
                    "type": gold_type,
                    "text": criterion.get("entity_text", ""),
                    "start": None,
                    "end": None,
                    "span_source": "missing_clause_meta",
                })
                continue

            start, end, span_source = recover_pred_span_fair(criterion, clause_meta)
            span_source_counter[span_source] += 1

            detail["span_source"] = span_source
            detail["start"] = start
            detail["end"] = end
            detail["span_recoverable"] = start is not None and end is not None

            if start is not None and end is not None:
                stats["span_recoverable_comparable_leaves"] += 1
            else:
                stats["span_not_recoverable_comparable_leaves"] += 1

            pred_node = {
                "pred_id": criterion.get("criterion_id"),
                "type": gold_type,
                "text": criterion.get("entity_text", ""),
                "start": start,
                "end": end,
                "span_source": span_source,
            }

            by_doc.setdefault(doc_id, []).append(pred_node)
            details.append(detail)

    coverage = {
        "stats": dict(stats),
        "pred_entity_type_counts": dict(pred_type_counter),
        "non_comparable_entity_type_counts": dict(non_comparable_type_counter),
        "span_source_counts": dict(span_source_counter),
    }

    return by_doc, details, coverage


# ----------------------------
# Matching
# ----------------------------

def exact_match(pred: Dict[str, Any], gold: Dict[str, Any], typed: bool) -> bool:
    if pred.get("start") is None or pred.get("end") is None:
        return False
    if typed and pred["type"] != gold["type"]:
        return False
    return pred["start"] == gold["start"] and pred["end"] == gold["end"]


def soft_match(pred: Dict[str, Any], gold: Dict[str, Any], typed: bool) -> bool:
    if pred.get("start") is None or pred.get("end") is None:
        return False
    if typed and pred["type"] != gold["type"]:
        return False
    return overlap_len(pred["start"], pred["end"], gold["start"], gold["end"]) > 0


def greedy_match(
    preds: List[Dict[str, Any]],
    golds: List[Dict[str, Any]],
    matcher,
    typed: bool,
) -> Tuple[int, List[Tuple[str, str]], List[str], List[str]]:
    matched_gold = set()
    matched_pred = set()
    pairs = []

    for p in preds:
        best_idx = None
        best_overlap = -1

        for gi, g in enumerate(golds):
            if gi in matched_gold:
                continue

            if not matcher(p, g, typed=typed):
                continue

            ov = overlap_len(p.get("start"), p.get("end"), g.get("start"), g.get("end"))

            if ov > best_overlap:
                best_overlap = ov
                best_idx = gi

        if best_idx is not None:
            matched_gold.add(best_idx)
            matched_pred.add(p["pred_id"])
            pairs.append((p["pred_id"], golds[best_idx]["gold_id"]))

    tp = len(pairs)
    unmatched_pred = [p["pred_id"] for p in preds if p["pred_id"] not in matched_pred]
    unmatched_gold = [g["gold_id"] for i, g in enumerate(golds) if i not in matched_gold]

    return tp, pairs, unmatched_pred, unmatched_gold


def evaluate_doc(preds: List[Dict[str, Any]], golds: List[Dict[str, Any]]) -> Dict[str, Any]:
    exact_tp_m, exact_pairs_m, exact_unmatched_pred_m, exact_unmatched_gold_m = greedy_match(
        preds, golds, exact_match, typed=False
    )
    soft_tp_m, soft_pairs_m, soft_unmatched_pred_m, soft_unmatched_gold_m = greedy_match(
        preds, golds, soft_match, typed=False
    )
    exact_tp_t, exact_pairs_t, exact_unmatched_pred_t, exact_unmatched_gold_t = greedy_match(
        preds, golds, exact_match, typed=True
    )
    soft_tp_t, soft_pairs_t, soft_unmatched_pred_t, soft_unmatched_gold_t = greedy_match(
        preds, golds, soft_match, typed=True
    )

    n_pred = len(preds)
    n_gold = len(golds)

    return {
        "n_pred_nodes": n_pred,
        "n_gold_nodes": n_gold,

        "exact_mention_matches": exact_pairs_m,
        "soft_mention_matches": soft_pairs_m,
        "exact_typed_matches": exact_pairs_t,
        "soft_typed_matches": soft_pairs_t,

        "exact_typed_unmatched_pred": exact_unmatched_pred_t,
        "exact_typed_unmatched_gold": exact_unmatched_gold_t,

        "exact_mention_tp": exact_tp_m,
        "soft_mention_tp": soft_tp_m,
        "exact_typed_tp": exact_tp_t,
        "soft_typed_tp": soft_tp_t,
    }


def metric_block(tp: int, total_pred: int, total_gold: int) -> Dict[str, Optional[float]]:
    p = safe_div(tp, total_pred)
    r = safe_div(tp, total_gold)
    return {
        "precision": p,
        "recall": r,
        "f1": f1_score(p, r),
    }


def evaluate_predictions(
    pred_by_doc: Dict[str, List[Dict[str, Any]]],
    gold_by_doc: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    common_docs = sorted(set(gold_by_doc) & set(pred_by_doc))

    detail_rows = []

    total_pred = 0
    total_gold = 0

    micro = {
        "exact_mention_tp": 0,
        "soft_mention_tp": 0,
        "exact_typed_tp": 0,
        "soft_typed_tp": 0,
    }

    per_type_counts = defaultdict(lambda: {
        "pred": 0,
        "gold": 0,
        "exact_mention_tp": 0,
        "soft_mention_tp": 0,
        "exact_typed_tp": 0,
        "soft_typed_tp": 0,
    })

    for doc_id in common_docs:
        gold_nodes = extract_gold_nodes(gold_by_doc[doc_id])
        pred_nodes = pred_by_doc.get(doc_id, [])

        doc_eval = evaluate_doc(pred_nodes, gold_nodes)
        doc_eval["document_id"] = doc_id
        detail_rows.append(doc_eval)

        total_pred += doc_eval["n_pred_nodes"]
        total_gold += doc_eval["n_gold_nodes"]

        for key in micro:
            micro[key] += doc_eval[key]

        # Per type evaluation
        all_types = sorted(GOLD_KEEP_TYPES)

        for typ in all_types:
            pred_t = [p for p in pred_nodes if p["type"] == typ]
            gold_t = [g for g in gold_nodes if g["type"] == typ]

            type_eval = evaluate_doc(pred_t, gold_t)

            per_type_counts[typ]["pred"] += type_eval["n_pred_nodes"]
            per_type_counts[typ]["gold"] += type_eval["n_gold_nodes"]
            per_type_counts[typ]["exact_mention_tp"] += type_eval["exact_mention_tp"]
            per_type_counts[typ]["soft_mention_tp"] += type_eval["soft_mention_tp"]
            per_type_counts[typ]["exact_typed_tp"] += type_eval["exact_typed_tp"]
            per_type_counts[typ]["soft_typed_tp"] += type_eval["soft_typed_tp"]

    per_type_metrics = {}

    for typ, counts in per_type_counts.items():
        per_type_metrics[typ] = {
            "n_pred": counts["pred"],
            "n_gold": counts["gold"],
            "exact_mention": metric_block(counts["exact_mention_tp"], counts["pred"], counts["gold"]),
            "soft_mention": metric_block(counts["soft_mention_tp"], counts["pred"], counts["gold"]),
            "exact_typed": metric_block(counts["exact_typed_tp"], counts["pred"], counts["gold"]),
            "soft_typed": metric_block(counts["soft_typed_tp"], counts["pred"], counts["gold"]),
        }

    summary_metrics = {
        "n_common_documents": len(common_docs),
        "n_total_pred_comparable_nodes": total_pred,
        "n_total_gold_nodes": total_gold,
        "micro_exact_mention": metric_block(micro["exact_mention_tp"], total_pred, total_gold),
        "micro_soft_mention": metric_block(micro["soft_mention_tp"], total_pred, total_gold),
        "micro_exact_typed": metric_block(micro["exact_typed_tp"], total_pred, total_gold),
        "micro_soft_typed": metric_block(micro["soft_typed_tp"], total_pred, total_gold),
        "per_type_metrics": per_type_metrics,
    }

    return summary_metrics, detail_rows


# ----------------------------
# Main
# ----------------------------

def run_branch(
    root: Path,
    branch_name: str,
    pass2_leaves_path: Path,
    gold_by_doc: Dict[str, Dict[str, Any]],
    pass2_input_index: Dict[Tuple[str, str], Dict[str, Any]],
    out_root: Path,
) -> Optional[Dict[str, Any]]:
    if not pass2_leaves_path.exists():
        print(f"[SKIP] {branch_name}: file not found: {pass2_leaves_path}")
        return None

    pass2_leaf_rows = load_jsonl(pass2_leaves_path)

    pred_by_doc, pred_details, coverage = extract_pred_nodes(
        pass2_leaf_rows=pass2_leaf_rows,
        pass2_input_index=pass2_input_index,
    )

    metrics, doc_details = evaluate_predictions(
        pred_by_doc=pred_by_doc,
        gold_by_doc=gold_by_doc,
    )

    out_dir = out_root / branch_name
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "evaluation_stage": "level_a_gold_nodes_fair_leaf_proxy_pre_verification_A_B",
        "branch": branch_name,
        "pass2_leaves_file": str(pass2_leaves_path),
        "method_note": (
            "Fair Level A proxy. Uses final Pass 2 leaf entity_text/entity_type/evidence_text only. "
            "Does not use BERT anchors and does not fall back to whole clause span for the main metric."
        ),
        "comparable_gold_types": sorted(GOLD_KEEP_TYPES),
        "pred_to_gold_type_map": PRED_TO_GOLD_TYPE,
        "coverage": coverage,
        **metrics,
    }

    write_json(out_dir / "summary.json", summary)
    write_jsonl(out_dir / "document_details.jsonl", doc_details)
    write_jsonl(out_dir / "prediction_details.jsonl", pred_details)

    print(f"\n===== LEVEL A: CHIA GOLD-NODE FAIR LEAF PROXY: {branch_name} =====")
    print("Pass2 file:", pass2_leaves_path)
    print("Total leaves:", coverage["stats"].get("total_leaves", 0))
    print("Comparable leaves:", coverage["stats"].get("comparable_leaves", 0))
    print("Non-comparable leaves:", coverage["stats"].get("non_comparable_leaves", 0))
    print("Span recoverable comparable leaves:", coverage["stats"].get("span_recoverable_comparable_leaves", 0))
    print("Span NOT recoverable comparable leaves:", coverage["stats"].get("span_not_recoverable_comparable_leaves", 0))
    print("Gold nodes:", metrics["n_total_gold_nodes"])
    print("Pred comparable nodes:", metrics["n_total_pred_comparable_nodes"])
    print("Micro exact mention:", metrics["micro_exact_mention"])
    print("Micro soft mention:", metrics["micro_soft_mention"])
    print("Micro exact typed:", metrics["micro_exact_typed"])
    print("Micro soft typed:", metrics["micro_soft_typed"])

    return summary


def main() -> None:
    ROOT = Path(__file__).resolve().parents[3]

    gold_path = (
        ROOT
        / "data"
        / "processed"
        / "chia_struct_eval_200_gold_graph.jsonl"
    )

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
        / "gold_nodes_fair_leaf_proxy_pre_verification_A_B"
    )

    required_paths = {
        "gold annotations": gold_path,
        "Pass 2 inputs": pass2_inputs_path,
        **{
            f"{branch_name} leaves": path
            for branch_name, path in branch_paths.items()
        },
    }

    for name, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    gold_rows = load_jsonl(gold_path)
    pass2_input_rows = load_jsonl(pass2_inputs_path)

    gold_by_doc = build_gold_index(gold_rows)
    pass2_input_index = build_pass2_input_index(pass2_input_rows)

    all_summaries = {}

    for branch_name, path in branch_paths.items():
        summary = run_branch(
            root=ROOT,
            branch_name=branch_name,
            pass2_leaves_path=path,
            gold_by_doc=gold_by_doc,
            pass2_input_index=pass2_input_index,
            out_root=out_root,
        )

        if summary is not None:
            all_summaries[branch_name] = {
                "total_leaves": summary["coverage"]["stats"].get(
                    "total_leaves", 0
                ),
                "comparable_leaves": summary["coverage"]["stats"].get(
                    "comparable_leaves", 0
                ),
                "span_recoverable_comparable_leaves": summary[
                    "coverage"
                ]["stats"].get(
                    "span_recoverable_comparable_leaves", 0
                ),
                "span_not_recoverable_comparable_leaves": summary[
                    "coverage"
                ]["stats"].get(
                    "span_not_recoverable_comparable_leaves", 0
                ),
                "n_total_pred_comparable_nodes": summary[
                    "n_total_pred_comparable_nodes"
                ],
                "n_total_gold_nodes": summary[
                    "n_total_gold_nodes"
                ],
                "micro_exact_mention": summary[
                    "micro_exact_mention"
                ],
                "micro_soft_mention": summary[
                    "micro_soft_mention"
                ],
                "micro_exact_typed": summary[
                    "micro_exact_typed"
                ],
                "micro_soft_typed": summary[
                    "micro_soft_typed"
                ],
            }

    all_summary_path = out_root / "all_branch_summary.json"
    write_json(all_summary_path, all_summaries)

    print("\n===== PRE-VERIFICATION GOLD-NODE PROXY SUMMARY =====")
    print("Gold annotations:", gold_path)
    print("Output:", all_summary_path)

if __name__ == "__main__":
    main()

# Run from the repository root:
# python scripts/04_evaluation/01_pre_verification/02_evaluate_gold_node_proxy.py