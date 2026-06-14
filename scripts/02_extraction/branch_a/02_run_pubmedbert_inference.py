import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer


CHIA_ID_CANDIDATES = ["chia_id"]
DOCUMENT_ID_CANDIDATES = ["document_id", "id"]
TEXT_CANDIDATES = ["text", "raw_text", "eligibility_text", "criteria_text"]


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


def find_first_existing_key(row: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for key in candidates:
        if key in row:
            return key
    return None


def infer_criterion_type_from_chia_id(chia_id: Optional[str]) -> Optional[str]:
    if chia_id is None:
        return None

    s = str(chia_id).strip()
    if s.endswith("_inc"):
        return "inclusion"
    if s.endswith("_exc"):
        return "exclusion"
    return None


def close_entity(current: Optional[Dict[str, Any]], text: str) -> Optional[Dict[str, Any]]:
    if current is None:
        return None

    current["text"] = text[current["start"]:current["end"]]

    if current["token_scores"]:
        current["score_mean"] = round(
            sum(current["token_scores"]) / len(current["token_scores"]), 6
        )
        current["score_min"] = round(min(current["token_scores"]), 6)

        current["score"] = current["score_min"]
    else:
        current["score_mean"] = None
        current["score_min"] = None
        current["score"] = None

    del current["token_scores"]
    return current


def decode_bio_entities(
    text: str,
    offsets: List[List[int]],
    pred_ids: List[int],
    token_scores: List[float],
    id2label: Dict[int, str],
    special_tokens_mask: List[int],
) -> List[Dict[str, Any]]:
    entities: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    left_boundary_region = True

    for (start, end), pred_id, score, is_special in zip(
        offsets, pred_ids, token_scores, special_tokens_mask
    ):
        if is_special == 1:
            continue

        if end <= start:
            continue

        label = id2label[int(pred_id)]

        if label == "O":
            finished = close_entity(current, text)
            if finished is not None:
                entities.append(finished)
            current = None
            left_boundary_region = False
            continue

        if label.startswith("B-"):
            finished = close_entity(current, text)
            if finished is not None:
                entities.append(finished)

            current = {
                "start": int(start),
                "end": int(end),
                "label": label[2:],
                "token_scores": [float(score)],
            }
            left_boundary_region = False
            continue

        if label.startswith("I-"):
            entity_label = label[2:]

            if current is not None and current["label"] == entity_label:
                current["end"] = int(end)
                current["token_scores"].append(float(score))
                continue

            # Orphan I-tag at the very start of an overflow chunk:
            # skip it instead of creating a fake partial entity.
            if left_boundary_region:
                continue

            # Orphan I-tag inside the chunk:
            # treat it like a fresh B-tag.
            finished = close_entity(current, text)
            if finished is not None:
                entities.append(finished)

            current = {
                "start": int(start),
                "end": int(end),
                "label": entity_label,
                "token_scores": [float(score)],
            }
            continue

        # Fallback for non-BIO labels, just in case
        finished = close_entity(current, text)
        if finished is not None:
            entities.append(finished)

        current = {
            "start": int(start),
            "end": int(end),
            "label": label,
            "token_scores": [float(score)],
        }
        left_boundary_region = False

    finished = close_entity(current, text)
    if finished is not None:
        entities.append(finished)

    return entities


def deduplicate_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best_by_key: Dict[str, Dict[str, Any]] = {}

    for ent in entities:
        key = f"{ent['start']}|{ent['end']}|{ent['label']}|{ent['text']}"
        if key not in best_by_key:
            best_by_key[key] = ent
        else:
            old_score = best_by_key[key].get("score") or -1.0
            new_score = ent.get("score") or -1.0
            if new_score > old_score:
                best_by_key[key] = ent

    deduped = list(best_by_key.values())
    deduped.sort(key=lambda x: (x["start"], x["end"], x["label"]))
    return deduped


def predict_entities_for_text(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForTokenClassification,
    device: torch.device,
    max_length: int,
    stride: int,
) -> List[Dict[str, Any]]:
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
        return_special_tokens_mask=True,
        truncation=True,
        max_length=max_length,
        stride=stride,
        padding=False,
    )

    id2label = {int(k): v for k, v in model.config.id2label.items()}

    all_entities: List[Dict[str, Any]] = []

    for chunk_idx in range(len(encoding["input_ids"])):
        input_ids = torch.tensor([encoding["input_ids"][chunk_idx]], dtype=torch.long, device=device)
        attention_mask = torch.tensor([encoding["attention_mask"][chunk_idx]], dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0]

        probs = torch.softmax(logits, dim=-1)
        pred_ids = torch.argmax(probs, dim=-1).cpu().tolist()
        token_scores = torch.max(probs, dim=-1).values.cpu().tolist()

        offsets = encoding["offset_mapping"][chunk_idx]
        special_tokens_mask = encoding["special_tokens_mask"][chunk_idx]

        chunk_entities = decode_bio_entities(
            text=text,
            offsets=offsets,
            pred_ids=pred_ids,
            token_scores=token_scores,
            id2label=id2label,
            special_tokens_mask=special_tokens_mask,
        )
        all_entities.extend(chunk_entities)

    return deduplicate_entities(all_entities)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PubMedBERT NER inference on CHIA JSONL.")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to trained model directory")
    parser.add_argument("--max-docs", type=int, default=None, help="Optional smoke test limit")
    parser.add_argument("--max-length", type=int, default=512, help="Tokenizer max length")
    parser.add_argument("--stride", type=int, default=128, help="Stride for long documents")
    parser.add_argument("--chia-id-field", type=str, default=None, help="Optional explicit CHIA row id field")
    parser.add_argument("--document-id-field", type=str, default=None, help="Optional explicit document id field")
    parser.add_argument("--text-field", type=str, default=None, help="Optional explicit text field")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    model_dir = Path(args.model_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    rows = read_jsonl(input_path)
    if args.max_docs is not None:
        rows = rows[: args.max_docs]

    if len(rows) == 0:
        raise ValueError("No rows found in input file.")

    sample_row = rows[0]

    chia_id_field = args.chia_id_field or find_first_existing_key(sample_row, CHIA_ID_CANDIDATES)
    document_id_field = args.document_id_field or find_first_existing_key(sample_row, DOCUMENT_ID_CANDIDATES)
    text_field = args.text_field or find_first_existing_key(sample_row, TEXT_CANDIDATES)

    if chia_id_field is None and document_id_field is None:
        raise KeyError(
            "Could not find any id field. "
            f"Tried chia_id candidates: {CHIA_ID_CANDIDATES}, "
            f"document_id candidates: {DOCUMENT_ID_CANDIDATES}. "
            f"Available keys: {list(sample_row.keys())}"
        )

    if document_id_field is None:
        print("WARNING: no document_id field found, document_id will be None for all rows")

    if chia_id_field is None:
        print("WARNING: no chia_id field found, criterion_type cannot be inferred from chia_id")

    if text_field is None:
        raise KeyError(
            f"Could not find a text field. Tried: {TEXT_CANDIDATES}. "
            f"Available keys: {list(sample_row.keys())}"
        )

    print(f"Using chia_id field: {chia_id_field}")
    print(f"Using document_id field: {document_id_field}")
    print(f"Using text field: {text_field}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    output_rows: List[Dict[str, Any]] = []

    for row in tqdm(rows, desc="Running inference"):
        chia_id = row.get(chia_id_field) if chia_id_field is not None else None
        document_id = row.get(document_id_field) if document_id_field is not None else None
        criterion_type = infer_criterion_type_from_chia_id(chia_id)
        raw_text = row.get(text_field, "")

        if raw_text is None or not str(raw_text).strip():
            output_row = {
                "chia_id": chia_id,
                "document_id": document_id,
                "criterion_type": criterion_type,
                "raw_text": raw_text,
                "model_id": str(model_dir),
                "predicted_entities": [],
                "status": "empty_text",
                "error_message": "Input text is empty or whitespace only.",
            }

            if "n_entities" in row:
                output_row["source_n_entities"] = row["n_entities"]
            if "n_relations" in row:
                output_row["source_n_relations"] = row["n_relations"]

            output_rows.append(output_row)
            continue

        try:
            predicted_entities = predict_entities_for_text(
                text=raw_text,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_length=args.max_length,
                stride=args.stride,
            )

            output_row = {
                "chia_id": chia_id,
                "document_id": document_id,
                "criterion_type": criterion_type,
                "raw_text": raw_text,
                "model_id": str(model_dir),
                "predicted_entities": predicted_entities,
                "status": "ok",
                "error_message": None,
            }

            if "n_entities" in row:
                output_row["source_n_entities"] = row["n_entities"]
            if "n_relations" in row:
                output_row["source_n_relations"] = row["n_relations"]

        except Exception as e:
            output_row = {
                "chia_id": chia_id,
                "document_id": document_id,
                "criterion_type": criterion_type,
                "raw_text": raw_text,
                "model_id": str(model_dir),
                "predicted_entities": [],
                "status": "error",
                "error_message": str(e),
            }

            if "n_entities" in row:
                output_row["source_n_entities"] = row["n_entities"]
            if "n_relations" in row:
                output_row["source_n_relations"] = row["n_relations"]

        output_rows.append(output_row)

    write_jsonl(output_path, output_rows)
    print(f"Saved {len(output_rows)} rows to: {output_path}")


if __name__ == "__main__":
    main()



#python .\scripts\02_extraction\branch_a\02_run_pubmedbert_inference.py `
#  --input .\data\processed\chia_text_only_200.jsonl `
#  --output .\outputs\extraction\branch_a\chia_text_only_200_pubmedbert_entities.jsonl `
#  --model-dir .\models\pubmedbert_chia_ner_li_nontest1900_v1
