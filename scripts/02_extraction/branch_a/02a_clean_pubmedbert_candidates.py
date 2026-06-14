import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ========= 1. Label groups =========

BASE_ANCHOR_LABELS = {
    "Condition",
    "Drug",
    "Procedure",
    "Measurement",
    "Device",
}

SUPPORT_LABELS = {
    "Qualifier",
    "Value",
    "Temporal",
    "Negation",
    "Multiplier",
    "Mood",
    "Observation",
    "Scope",
    "Person",
    "Visit",
}

CLINICAL_SHORT_ANCHORS = {
    "DVT", "HIV", "CNS", "ECG", "EKG", "MRI", "CT", "ALT", "AST", "ULN", "BMI"
}

# ========= 2. JSONL helpers =========

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ========= 3. Small utilities =========

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def entity_length(ent: Dict[str, Any]) -> int:
    return int(ent["end"]) - int(ent["start"])


def overlap_length(a: Dict[str, Any], b: Dict[str, Any]) -> int:
    return max(0, min(int(a["end"]), int(b["end"])) - max(int(a["start"]), int(b["start"])))


def is_contained(inner: Dict[str, Any], outer: Dict[str, Any]) -> bool:
    return int(outer["start"]) <= int(inner["start"]) and int(inner["end"]) <= int(outer["end"])


def has_letter(text: str) -> bool:
    return any(ch.isalpha() for ch in text)


def get_score_mean(ent: Dict[str, Any]) -> Optional[float]:
    if ent.get("score_mean") is not None:
        return float(ent["score_mean"])
    if ent.get("score") is not None:
        return float(ent["score"])
    return None


def get_score_min(ent: Dict[str, Any]) -> Optional[float]:
    if ent.get("score_min") is not None:
        return float(ent["score_min"])
    if ent.get("score") is not None:
        return float(ent["score"])
    return None


def ranking_score(ent: Dict[str, Any]) -> float:
    values = []
    for key in ("score_mean", "score_min", "score"):
        if ent.get(key) is not None:
            values.append(float(ent[key]))
    return max(values) if values else -1.0


def looks_like_broken_fragment(text: str) -> bool:
    t = normalize_text(text)
    if not t:
        return True

    if t.endswith("[") or t.endswith("/"):
        return True
    if t.startswith("]"):
        return True

    # very short unmatched bracket fragments only
    if t.endswith("]") and "[" not in t and len(t) <= 6:
        return True
    if t.startswith("[") and "]" not in t and len(t) <= 6:
        return True

    return False


def has_balanced_brackets(text: str) -> bool:
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack = []
    for ch in text:
        if ch in pairs:
            stack.append(pairs[ch])
        elif ch in pairs.values():
            if not stack or stack[-1] != ch:
                return False
            stack.pop()
    return len(stack) == 0


def has_newline_between(a: Dict[str, Any], b: Dict[str, Any], raw_text: str) -> bool:
    gap_text = raw_text[int(a["end"]):int(b["start"])]
    return "\n" in gap_text


def count_comparison_signals(text: str) -> int:
    lower = normalize_text(text).lower()
    count = 0
    count += len(re.findall(r"<=", lower))
    count += len(re.findall(r">=", lower))
    count += len(re.findall(r"!=", lower))
    count += len(re.findall(r"(?<![<>=!])<(?![=])", lower))
    count += len(re.findall(r"(?<![<>=!])>(?![=])", lower))
    count += len(re.findall(r"(?<![<>=!])=(?![=])", lower))
    lexical_patterns = [
        r"\bless than\b",
        r"\bgreater than\b",
        r"\bat least\b",
        r"\bor older\b",
        r"\bor less\b",
    ]
    for pattern in lexical_patterns:
        count += len(re.findall(pattern, lower))
    return count


def looks_like_complex_nested_value(text: str) -> bool:
    t = normalize_text(text)
    if ("(" in t or ")" in t or "[" in t or "]" in t) and count_comparison_signals(t) >= 2:
        return True
    return False


def get_entity_flags(ent: Dict[str, Any]) -> List[str]:
    flags = []
    text = normalize_text(ent["text"])
    if len(text) >= 25:
        flags.append("long_span")
    if "(" in text or ")" in text or "[" in text or "]" in text:
        flags.append("bracketed_span")
    if count_comparison_signals(text) >= 2:
        flags.append("multi_comparison")
    score_min = get_score_min(ent)
    if score_min is not None and score_min < 0.60:
        flags.append("low_score_min")
    if ent["label"] in {"Condition", "Procedure", "Measurement"} and len(text.split()) == 1:
        flags.append("single_token_base_label")
    return flags


def normalize_entity(ent: Dict[str, Any]) -> Dict[str, Any]:
    score_mean = get_score_mean(ent)
    score_min = get_score_min(ent)
    score = float(ent["score"]) if ent.get("score") is not None else score_mean

    normalized = {
        "start": int(ent["start"]),
        "end": int(ent["end"]),
        "label": str(ent["label"]),
        "text": normalize_text(ent["text"]),
        "score_mean": score_mean,
        "score_min": score_min,
        "score": score,
    }
    normalized["qa_flags"] = get_entity_flags(normalized)
    return normalized


# ========= 4. Anchor / support detection =========

def is_anchor_candidate(ent: Dict[str, Any]) -> bool:
    label = ent["label"]
    text = normalize_text(ent["text"])
    lower = text.lower()

    if label not in BASE_ANCHOR_LABELS:
        return False

    if label == "Condition":
        if text.isdigit():
            return False
        if len(text) < 2:
            return False
        if not has_letter(text):
            return False
        if looks_like_broken_fragment(text):
            return False
        return True

    if label == "Drug":
        if len(text) < 3:
            return False
        if not has_letter(text):
            return False
        return True

    if label == "Procedure":
        if not has_letter(text):
            return False
        if len(text) < 4:
            return False
        if looks_like_broken_fragment(text):
            return False
        return True

    if label == "Measurement":
        if not has_letter(text):
            return False
        if looks_like_broken_fragment(text):
            return False
        return True

    if label == "Device":
        if len(text) < 3:
            return False
        return True
    
    if text.upper() in CLINICAL_SHORT_ANCHORS:
        return True

    return False


def is_support_candidate(ent: Dict[str, Any]) -> bool:
    label = ent["label"]
    text = normalize_text(ent["text"])

    if label not in SUPPORT_LABELS:
        return False

    if not text:
        return False

    if label == "Value":
        # must at least look like a value-ish thing
        lower = text.lower()
        if not (
            any(ch.isdigit() for ch in text)
            or any(op in text for op in ["<", ">", "=", "≤", "≥"])
            or "grade" in lower
            or "negative" in lower
            or "positive" in lower
        ):
            return False

    return True


# ========= 5. Merge broken spans =========

def should_merge_entities(a: Dict[str, Any], b: Dict[str, Any], raw_text: str) -> bool:
    pair_labels = {a["label"], b["label"]}
    allow_value_multiplier_pair = pair_labels <= {"Value", "Multiplier"}

    if a["label"] != b["label"] and not allow_value_multiplier_pair:
        return False

    if int(b["start"]) < int(a["end"]):
        return False

    gap_text = raw_text[int(a["end"]):int(b["start"])]
    a_text = normalize_text(a["text"])
    b_text = normalize_text(b["text"])
    merged_candidate = normalize_text(raw_text[int(a["start"]):int(b["end"])])

    if has_newline_between(a, b, raw_text):
        return False
    if len(gap_text) > 8:
        return False
    if not has_balanced_brackets(merged_candidate):
        return False

    gap_simple = bool(re.fullmatch(r"[\s,;/()-]*", gap_text))
    gap_enum = bool(re.fullmatch(r"[\s,;/()-]*(?:or|and)?[\s,;/()-]*", gap_text))

    # Value/Multiplier combined (e.g., "1" + "mg or less")
    if allow_value_multiplier_pair:
        if int(b["start"]) == int(a["end"]) or gap_enum:
            if looks_like_complex_nested_value(merged_candidate):
                return False
            lower = merged_candidate.lower()
            if (
                any(ch.isdigit() for ch in merged_candidate)
                or re.search(r"[<>!=≤≥]", merged_candidate)
                or re.search(r"\b(?:mg|uln|grade|days?|weeks?|months?|years?)\b", lower)
            ):
                return True
        return False

    if int(b["start"]) != int(a["end"]) and not gap_simple:
        return False

    label = a["label"]

    # Value: handle things like "0,1" + "2"
    if label == "Value":
        if int(b["start"]) == int(a["end"]) or gap_enum:
            if looks_like_complex_nested_value(merged_candidate):
                return False
            if any(ch.isdigit() for ch in a_text) and any(ch.isdigit() for ch in b_text):
                return True
        return False

    # Measurement or Condition: fix broken fragments
    if label in {"Measurement", "Condition"}:
        if int(b["start"]) == int(a["end"]):
            if looks_like_broken_fragment(a_text) or looks_like_broken_fragment(b_text):
                return True
    return False


def merge_two_entities(a: Dict[str, Any], b: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
    start = int(a["start"])
    end = int(b["end"])
    merged_text = normalize_text(raw_text[start:end])

    a_mean = get_score_mean(a)
    b_mean = get_score_mean(b)
    a_min = get_score_min(a)
    b_min = get_score_min(b)

    mean_values = [x for x in [a_mean, b_mean] if x is not None]
    min_values = [x for x in [a_min, b_min] if x is not None]

    merged_score_mean = round(sum(mean_values) / len(mean_values), 6) if mean_values else None
    merged_score_min = round(min(min_values), 6) if min_values else None

    merged_label = "Value" if {a["label"], b["label"]} <= {"Value", "Multiplier"} else a["label"]

    merged = {
        "start": start,
        "end": end,
        "label": merged_label,
        "text": merged_text,
        "score_mean": merged_score_mean,
        "score_min": merged_score_min,
        "score": merged_score_min if merged_score_min is not None else merged_score_mean,
    }
    merged["qa_flags"] = get_entity_flags(merged)
    return merged


def repair_adjacent_fragments(
    entities: List[Dict[str, Any]],
    raw_text: str,
) -> Tuple[List[Dict[str, Any]], int]:
    if not entities:
        return [], 0

    sorted_ents = sorted(
        [normalize_entity(ent) for ent in entities],
        key=lambda x: (x["start"], x["end"], x["label"]),
    )

    repaired: List[Dict[str, Any]] = []
    merge_count = 0

    for ent in sorted_ents:
        if not repaired:
            repaired.append(ent)
            continue
        prev = repaired[-1]
        if should_merge_entities(prev, ent, raw_text):
            repaired[-1] = merge_two_entities(prev, ent, raw_text)
            merge_count += 1
        else:
            repaired.append(ent)

    return repaired, merge_count


# ========= 6. Minimal junk filtering =========

def should_drop_entity_light(ent: Dict[str, Any]) -> bool:
    text = normalize_text(ent["text"])
    label = ent["label"]
    lower = text.lower()

    if not text:
        return True

    # Drop clearly broken fragments
    if looks_like_broken_fragment(text):
        return True

    # Drop single non-operator characters
    if len(text) == 1 and text not in {"<", ">", "="}:
        return True

    # Drop numeric-only "Condition"/"Procedure"/"Measurement"
    if label in {"Condition", "Procedure", "Measurement"} and text.isdigit():
        return True

    # Drop obviously meaningless Measurement entries
    if label == "Measurement" and not has_letter(text):
        return True

    return False


def apply_light_drop(entities: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    kept = []
    dropped = 0
    for ent in entities:
        if should_drop_entity_light(ent):
            dropped += 1
        else:
            kept.append(ent)
    return kept, dropped


# ========= 7. Dedup and overlap =========

def resolve_exact_span_conflicts(entities: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    best_by_span: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
    dropped = 0

    for ent in entities:
        key = (ent["start"], ent["end"], ent["label"])
        if key not in best_by_span:
            best_by_span[key] = ent
        else:
            old = best_by_span[key]
            if ranking_score(ent) > ranking_score(old):
                best_by_span[key] = ent
                dropped += 1
            else:
                dropped += 1

    result = list(best_by_span.values())
    result.sort(key=lambda x: (x["start"], x["end"], x["label"]))
    return result, dropped


def strong_same_label_overlap(a: Dict[str, Any], b: Dict[str, Any], threshold: float) -> bool:
    if a["label"] != b["label"]:
        return False

    ov = overlap_length(a, b)
    if ov == 0:
        return False

    shorter = min(entity_length(a), entity_length(b))
    if shorter <= 0:
        return False

    ratio_on_shorter = ov / shorter
    if ratio_on_shorter >= threshold:
        return True

    return False


def collapse_same_label_overlaps(
    entities: List[Dict[str, Any]],
    overlap_threshold: float,
) -> Tuple[List[Dict[str, Any]], int]:
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for ent in entities:
        by_label.setdefault(ent["label"], []).append(ent)

    final_entities: List[Dict[str, Any]] = []
    dropped = 0

    for label, label_entities in by_label.items():
        sorted_ents = sorted(
            label_entities,
            key=lambda x: (
                -ranking_score(x),
                -entity_length(x),
                x["start"],
                x["end"],
            ),
        )

        kept_for_label: List[Dict[str, Any]] = []

        for ent in sorted_ents:
            conflict = False
            for kept in kept_for_label:
                if strong_same_label_overlap(ent, kept, overlap_threshold):
                    conflict = True
                    dropped += 1
                    break
            if not conflict:
                kept_for_label.append(ent)

        final_entities.extend(kept_for_label)

    final_entities.sort(key=lambda x: (x["start"], x["end"], x["label"]))
    return final_entities, dropped


# ========= 8. Anchor / support / other split =========

def split_entity_groups(
    entities: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    anchor_entities = []
    support_entities = []
    other_entities = []

    for ent in entities:
        if is_anchor_candidate(ent):
            anchor_entities.append(ent)
        elif is_support_candidate(ent):
            support_entities.append(ent)
        else:
            other_entities.append(ent)

    return anchor_entities, support_entities, other_entities


# ========= 9. Main cleaning pipeline =========

def clean_entities_light(
    entities: List[Dict[str, Any]],
    raw_text: str,
    overlap_threshold: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    stats = {
        "input_entities": len(entities),
        "merged_fragment_repairs": 0,
        "dropped_light_junk": 0,
        "dropped_exact_span_conflict": 0,
        "dropped_same_label_overlap": 0,
        "final_entities": 0,
    }

    # 1) Repair obvious fragments
    repaired_entities, merged_count = repair_adjacent_fragments(entities, raw_text)
    stats["merged_fragment_repairs"] = merged_count

    # 2) Very light junk removal
    step2, dropped_light = apply_light_drop(repaired_entities)
    stats["dropped_light_junk"] = dropped_light

    # 3) Deduplicate exact spans
    step3, dropped_exact = resolve_exact_span_conflicts(step2)
    stats["dropped_exact_span_conflict"] = dropped_exact

    # 4) Collapse heavy overlaps of same label
    step4, dropped_overlap = collapse_same_label_overlaps(
        step3,
        overlap_threshold=overlap_threshold,
    )
    stats["dropped_same_label_overlap"] = dropped_overlap

    stats["final_entities"] = len(step4)
    return repaired_entities, step4, stats


# ========= 10. CLI =========

def main() -> None:
    parser = argparse.ArgumentParser(description="Light cleaning of BERT entities before Stage 1 rule creation.")
    parser.add_argument("--input", type=str, required=True, help="Raw candidate JSONL from PubMedBERT inference")
    parser.add_argument("--output", type=str, required=True, help="Cleaned candidate JSONL (light)")
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.80,
        help="Strong overlap threshold on shorter span",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    rows = read_jsonl(input_path)
    output_rows: List[Dict[str, Any]] = []

    print("RUNNING LIGHT STAGE1 CLEANER")

    for row in rows:
        status = row.get("status", "ok")
        predicted_entities = row.get("predicted_entities", [])
        raw_text = row.get("raw_text", "")

        base_output = {
            "chia_id": row.get("chia_id"),
            "document_id": row.get("document_id"),
            "criterion_type": row.get("criterion_type"),
            "raw_text": raw_text,
            "model_id": row.get("model_id"),
            "status": status,
            "error_message": row.get("error_message"),
            "predicted_entities": predicted_entities,
        }

        if "source_n_entities" in row:
            base_output["source_n_entities"] = row["source_n_entities"]
        if "source_n_relations" in row:
            base_output["source_n_relations"] = row["source_n_relations"]

        if status != "ok":
            base_output["repaired_entities"] = []
            base_output["cleaned_entities"] = []
            base_output["anchor_entities"] = []
            base_output["support_entities"] = []
            base_output["other_entities"] = []
            base_output["cleaning_stats"] = {
                "input_entities": len(predicted_entities),
                "merged_fragment_repairs": 0,
                "dropped_light_junk": 0,
                "dropped_exact_span_conflict": 0,
                "dropped_same_label_overlap": 0,
                "final_entities": 0,
            }
            output_rows.append(base_output)
            continue

        repaired_entities, cleaned_entities, stats = clean_entities_light(
            predicted_entities,
            raw_text=raw_text,
            overlap_threshold=args.overlap_threshold,
        )

        anchor_entities, support_entities, other_entities = split_entity_groups(cleaned_entities)

        base_output["repaired_entities"] = repaired_entities
        base_output["cleaned_entities"] = cleaned_entities
        base_output["anchor_entities"] = anchor_entities
        base_output["support_entities"] = support_entities
        base_output["other_entities"] = other_entities
        base_output["cleaning_stats"] = stats

        output_rows.append(base_output)

    write_jsonl(output_path, output_rows)
    print(f"Saved {len(output_rows)} lightly cleaned rows to: {output_path}")


if __name__ == "__main__":
    main()




#& ".\.venv\Scripts\python.exe" scripts/02_extraction_chen/02a_clean_pubmedbert_candidates.py `
#  --input outputs/extraction/candidates/chia_text_only_200_pubmedbert_li_full_entities.jsonl `
#  --output outputs/extraction/candidates/chia_text_only_200_pubmedbert_li_full_entities_cleaned.jsonl