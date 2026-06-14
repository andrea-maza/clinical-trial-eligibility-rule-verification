import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def overlap_len(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def slim_entity(ent: Dict[str, Any], item_start: int) -> Dict[str, Any]:
    """
    Keep only the fields the later LLM needs, and convert offsets
    from row-level chars to item-level chars.
    """
    return {
        "text": ent.get("text"),
        "label": ent.get("label"),
        "start": int(ent["start"]) - item_start,
        "end": int(ent["end"]) - item_start,
        "score": ent.get("score"),
    }


def filter_entities_to_item(
    entities: List[Dict[str, Any]],
    item_start: int,
    item_end: int,
) -> List[Dict[str, Any]]:
    """
    Keep only entities fully contained in the item span.
    For this stage, strict containment is safer than clipping.
    """
    kept = []
    for ent in entities:
        start = int(ent["start"])
        end = int(ent["end"])
        if start >= item_start and end <= item_end:
            kept.append(slim_entity(ent, item_start))
    return kept


def find_substring_span(
    source_text: str,
    target_text: str,
    start_pos: int = 0,
) -> Optional[Tuple[int, int]]:
    """
    Exact substring search first.
    """
    if not target_text:
        return None

    idx = source_text.find(target_text, start_pos)
    if idx != -1:
        return idx, idx + len(target_text)

    return None


def locate_clause_spans(
    item_text: str,
    clauses: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Try to find clause evidence_text inside item_text.
    Fallback: try clause_text.
    """
    located = []
    cursor = 0

    for clause in clauses:
        evidence_text = clause.get("evidence_text", "")
        clause_text = clause.get("clause_text", "")

        span = find_substring_span(item_text, evidence_text, start_pos=cursor)

        if span is None:
            span = find_substring_span(item_text, clause_text, start_pos=cursor)

        if span is None:
            # final fallback: search from start
            span = find_substring_span(item_text, evidence_text, start_pos=0)

        if span is None:
            span = find_substring_span(item_text, clause_text, start_pos=0)

        if span is None:
            clause_start = None
            clause_end = None
        else:
            clause_start, clause_end = span
            cursor = clause_end

        located_clause = dict(clause)
        located_clause["clause_start_char"] = clause_start
        located_clause["clause_end_char"] = clause_end
        located.append(located_clause)

    return located


def assign_candidates_to_clause(
    clause,
    item_anchors,
    item_supports,
    item_others,
):
    """
    Attach only the candidates that overlap the clause span.
    If the clause span is missing, return empty candidate lists.
    """
    c_start = clause.get("clause_start_char")
    c_end = clause.get("clause_end_char")

    clause_out = dict(clause)

    if c_start is None or c_end is None:
        clause_out["bert_candidates"] = {
            "anchors": [],
            "supports": [],
            "others": [],
        }
        return clause_out

    anchors = []
    supports = []
    others = []

    for ent in item_anchors:
        if overlap_len(int(ent["start"]), int(ent["end"]), c_start, c_end) > 0:
            anchors.append(ent)

    for ent in item_supports:
        if overlap_len(int(ent["start"]), int(ent["end"]), c_start, c_end) > 0:
            supports.append(ent)
    
    for ent in item_others:
        if overlap_len(int(ent["start"]), int(ent["end"]), c_start, c_end) > 0:
            others.append(ent)

    clause_out["bert_candidates"] = {
        "anchors": anchors,
        "supports": supports,
        "others": others,
    }
    return clause_out

def audit_bad_bert_offsets(output_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    bad_offsets = []

    for r in output_rows:
        if r.get("status") != "ok":
            continue

        payload = r["pass2_input"]
        item_text = payload["item_text"]

        for clause in payload["clauses"]:
            candidates = clause.get("bert_candidates", {})
            for kind in ["anchors", "supports", "others"]:
                for ent in candidates.get(kind, []):
                    start = ent.get("start")
                    end = ent.get("end")

                    if start is None or end is None:
                        continue

                    if start < 0 or end > len(item_text) or start >= end:
                        bad_offsets.append({
                            "item_uid": r.get("item_uid"),
                            "clause_id": clause.get("clause_id"),
                            "kind": kind,
                            "entity": ent,
                            "item_text": item_text
                        })

    print("\n===== PASS 2 INPUT BERT OFFSET AUDIT =====")
    print("Bad BERT offsets:", len(bad_offsets))

    for x in bad_offsets[:10]:
        print("\nITEM:", x["item_uid"], x["clause_id"], x["kind"])
        print("ENTITY:", x["entity"])
        print("ITEM:", x["item_text"])

    return bad_offsets

def build_bert_index(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Index cleaned BERT candidate rows by chia_id.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        chia_id = row.get("chia_id")
        if chia_id:
            if chia_id in out:
                # You can log or raise here depending on how strict you want to be
                raise ValueError(f"Duplicate chia_id in BERT rows: {chia_id}")
            out[chia_id] = row
    return out


def main() -> None:
    ROOT = Path(__file__).resolve().parents[2]

    pass1_path = ROOT / "outputs" / "extraction" / "pass1_flat" / "chia_text_only_200_pass1_flat.jsonl"
    bert_path = ROOT / "outputs" / "extraction" / "branch_a" / "chia_text_only_200_pubmedbert_entities_cleaned.jsonl"

    out_dir = ROOT / "outputs" / "extraction" / "pass2_inputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "chia_text_only_200_pass2_inputs.jsonl"

    pass1_rows = load_jsonl(pass1_path)
    bert_rows = load_jsonl(bert_path)
    bert_by_chia_id = build_bert_index(bert_rows)

    output_rows: List[Dict[str, Any]] = []

    n_ok = 0
    n_err = 0

    for row in pass1_rows:
        item_uid = row.get("item_uid")
        chia_id = row.get("chia_id")
        doc_id = row.get("document_id")

        if row.get("status") != "ok":
            output_rows.append(
                {
                    "dataset": "CHIA",
                    "stage": "pass2_input_prep",
                    "item_uid": item_uid,
                    "chia_id": chia_id,
                    "document_id": doc_id,
                    "status": "error",
                    "error": f"Pass1 row status is not ok: {row.get('status')}",
                }
            )
            n_err += 1
            continue

        bert_row = bert_by_chia_id.get(chia_id)
        if bert_row is None:
            output_rows.append(
                {
                    "dataset": "CHIA",
                    "stage": "pass2_input_prep",
                    "item_uid": item_uid,
                    "chia_id": chia_id,
                    "document_id": doc_id,
                    "status": "error",
                    "error": "No matching cleaned BERT row found for chia_id.",
                }
            )
            n_err += 1
            continue

        if bert_row.get("status") != "ok":
            output_rows.append(
                {
                    "dataset": "CHIA",
                    "stage": "pass2_input_prep",
                    "item_uid": item_uid,
                    "chia_id": chia_id,
                    "document_id": doc_id,
                    "status": "error",
                    "error": f"Matching BERT row status is not ok: {bert_row.get('status')}",
                }
            )
            n_err += 1
            continue

        item_text = row.get("item_text", "")
        full_text = row.get("full_text", "")
        item_start = row.get("item_start_char")
        item_end = row.get("item_end_char")

        if item_start is None or item_end is None:
            output_rows.append(
                {
                    "dataset": "CHIA",
                    "stage": "pass2_input_prep",
                    "item_uid": item_uid,
                    "chia_id": chia_id,
                    "document_id": doc_id,
                    "status": "error",
                    "error": "Missing item_start_char or item_end_char in Pass1 output.",
                }
            )
            n_err += 1
            continue

        # sanity check: item_text should match the full_text slice
        sliced = full_text[item_start:item_end]
        if sliced != item_text:
            # do not hard-fail, but record warning
            slice_warning = "item_text does not exactly match full_text slice."
        else:
            slice_warning = None

        item_anchors = filter_entities_to_item(
            bert_row.get("anchor_entities", []),
            item_start,
            item_end,
        )
        item_supports = filter_entities_to_item(
            bert_row.get("support_entities", []),
            item_start,
            item_end,
        )
        item_others = filter_entities_to_item(
            bert_row.get("other_entities", []),
            item_start,
            item_end,
        )

        clauses = (row.get("parsed_pass1_with_ids") or {}).get("clauses", [])
        clauses_located = locate_clause_spans(item_text, clauses)
        clauses_with_candidates = [
            assign_candidates_to_clause(clause, item_anchors, item_supports, item_others)
            for clause in clauses_located
        ]

        pass2_input = {
            "trial_id": row.get("document_id"),
            "item_uid": item_uid,
            "chia_id": chia_id,
            "document_id": doc_id,
            "criterion_type": row.get("criterion_type_hint"),
            "item_text": item_text,
            "item_start_char": item_start,
            "item_end_char": item_end,
            "clauses": clauses_with_candidates,
            "item_bert_candidates": {
                "anchors": item_anchors,
                "supports": item_supports,
                "others": item_others,
            },
        }

        rec = {
            "dataset": "CHIA",
            "stage": "pass2_input_prep",
            "item_uid": item_uid,
            "chia_id": chia_id,
            "document_id": doc_id,
            "status": "ok",
            "error": None,
            "warning": slice_warning,
            "pass1_source": "chia_text_only_200_pass1_flat.jsonl",
            "bert_source": "chia_text_only_200_pubmedbert_entities_cleaned.jsonl",
            "pass2_input": pass2_input,
        }

        output_rows.append(rec)
        n_ok += 1

    write_jsonl(out_path, output_rows)

    # after the main loop, before printing OK/ERR
    n_warn = sum(1 for r in output_rows if r.get("warning"))
    bad_offsets = audit_bad_bert_offsets(output_rows)

    print("Warnings (item_text vs full_text mismatch):", n_warn)
    print("Bad BERT offsets:", len(bad_offsets))
    print("Wrote:", out_path)
    print("OK:", n_ok)
    print("ERR:", n_err)


if __name__ == "__main__":
    main()


# Run from the repository root: 
# # python scripts/02_extraction/03_prepare_pass2_inputs.py