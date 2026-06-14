import json
import statistics
from pathlib import Path


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def has_multiple_sentences(text: str) -> bool:
    text = text.strip()
    return text.count(".") >= 1


def is_tricky_item(text: str) -> bool:
    t = text.lower()
    tricky_markers = [
        "unless",
        "except",
        " if ",
        "must have",
        "including",
        "history of",
        "stable for",
        "at least",
        "at most",
        "exactly"
    ]
    return any(m in t for m in tricky_markers) or has_multiple_sentences(text)

def audit_evidence_substrings(ok_rows):
    bad_evidence = []

    for r in ok_rows:
        item_text = r.get("item_text", "")
        clauses = (r.get("parsed_pass1_with_ids") or {}).get("clauses", [])

        for c in clauses:
            evidence = c.get("evidence_text", "")
            if evidence and evidence not in item_text:
                bad_evidence.append({
                    "item_uid": r.get("item_uid"),
                    "clause_id": c.get("clause_id"),
                    "evidence_text": evidence,
                    "item_text": item_text
                })

    print("\n===== PASS 1 EVIDENCE SUBSTRING AUDIT =====")
    print("Evidence_text not exact substring:", len(bad_evidence))

    for x in bad_evidence[:10]:
        print("\nITEM:", x["item_uid"], x["clause_id"])
        print("EVIDENCE:", x["evidence_text"])
        print("ITEM:", x["item_text"])

    return bad_evidence


def audit_rows(rows):
    total = len(rows)
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    err_rows = [r for r in rows if r.get("status") != "ok"]
    bad_evidence = audit_evidence_substrings(ok_rows)

    clause_nonempty = 0
    last_null_ok = 0
    obvious_or_ok = 0
    obvious_and_ok = 0

    tricky = []
    simple = []

    prompt_tokens = []
    completion_tokens = []
    total_tokens = []
    latencies = []

    for r in ok_rows:
        parsed = r.get("parsed_pass1") or {}
        clauses = parsed.get("clauses", [])

        if clauses:
            clause_nonempty += 1

        if clauses and clauses[-1].get("connector_to_next") is None:
            last_null_ok += 1

        item_text = (r.get("item_text") or "").strip()
        lower_item = item_text.lower()

        connectors = [c.get("connector_to_next") for c in clauses]

        # sanity checks for obvious connector cases
        if " or " in lower_item:
            if "OR" in connectors:
                obvious_or_ok += 1

        if " and " in lower_item:
            if "AND" in connectors:
                obvious_and_ok += 1

        if is_tricky_item(item_text):
            tricky.append(r)
        else:
            simple.append(r)

        if r.get("prompt_tokens") is not None:
            prompt_tokens.append(r["prompt_tokens"])
        if r.get("completion_tokens") is not None:
            completion_tokens.append(r["completion_tokens"])
        if r.get("total_tokens") is not None:
            total_tokens.append(r["total_tokens"])
        if r.get("latency_sec") is not None:
            latencies.append(r["latency_sec"])

    def mean_or_none(xs):
        return round(statistics.mean(xs), 2) if xs else None

    def median_or_none(xs):
        return round(statistics.median(xs), 2) if xs else None

    print("\n===== PASS 1 AUDIT SUMMARY =====")
    print(f"Total records: {total}")
    print(f"OK records: {len(ok_rows)}")
    print(f"Error records: {len(err_rows)}")

    if ok_rows:
        print(f"\nAt least one clause: {clause_nonempty}/{len(ok_rows)}")
        print(f"Last connector null: {last_null_ok}/{len(ok_rows)}")

        n_or_items = sum(1 for r in ok_rows if " or " in (r.get('item_text') or "").lower())
        n_and_items = sum(1 for r in ok_rows if " and " in (r.get('item_text') or "").lower())

        if n_or_items:
            print(f"Obvious OR cases with at least one OR connector: {obvious_or_ok}/{n_or_items}")
        if n_and_items:
            print(f"Obvious AND cases with at least one AND connector: {obvious_and_ok}/{n_and_items}")

        print("\nToken / latency stats")
        print(f"Prompt tokens mean/median: {mean_or_none(prompt_tokens)} / {median_or_none(prompt_tokens)}")
        print(f"Completion tokens mean/median: {mean_or_none(completion_tokens)} / {median_or_none(completion_tokens)}")
        print(f"Total tokens mean/median: {mean_or_none(total_tokens)} / {median_or_none(total_tokens)}")
        print(f"Latency sec mean/median: {mean_or_none(latencies)} / {median_or_none(latencies)}")

        print(f"\nSimple items: {len(simple)}")
        print(f"Tricky items: {len(tricky)}")

    if err_rows:
        print("\nFirst 5 errors:")
        for r in err_rows[:5]:
            print(f"- {r.get('item_uid')}: {r.get('error')}")

    if tricky:
        print("\n===== SAMPLE TRICKY ITEMS TO INSPECT =====")
        for r in tricky[:8]:
            print(f"\nITEM UID: {r.get('item_uid')}")
            print(f"TEXT: {r.get('item_text')}")
            clauses = (r.get("parsed_pass1") or {}).get("clauses", [])
            for i, c in enumerate(clauses, start=1):
                print(
                    f"  C{i}: text={c.get('clause_text')!r} | "
                    f"neg={c.get('is_negated')} | "
                    f"conn={c.get('connector_to_next')} | "
                    f"quant={c.get('quantifier')}"
                )


def main():
    ROOT = Path(__file__).resolve().parents[2]
    in_path = ROOT / "outputs" / "extraction" / "pass1_flat" / "chia_text_only_200_pass1_flat.jsonl"

    rows = load_jsonl(in_path)
    audit_rows(rows)


if __name__ == "__main__":
    main()


# Run from the repository root: 
# # python scripts/02_extraction/01a_audit_pass1_flat_outputs.py