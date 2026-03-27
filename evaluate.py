import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.extract import InformationExtractor
from src.ocr import OCRPipeline, load_image_from_bytes


PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
LABELS_PATH = DATA_DIR / "labels.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SUMMARY_PATH = OUTPUT_DIR / "evaluation_summary.json"


def normalize_name(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None

    normalized = " ".join(str(value).strip().lower().split())
    return normalized or None


def normalize_date(value: Optional[str], extractor: InformationExtractor) -> Optional[str]:
    if value is None:
        return None

    normalized = extractor._normalize_date(str(value).strip())
    return normalized or None


def normalize_amount(value: Optional[str], extractor: InformationExtractor) -> Optional[str]:
    if value is None:
        return None

    normalized = extractor._normalize_amount(str(value).strip())
    return normalized or None


def compare_field(field_name: str, expected: Optional[str], predicted: Optional[str], extractor: InformationExtractor) -> bool:
    if field_name == "name":
        return normalize_name(expected) == normalize_name(predicted)

    if field_name == "date":
        return normalize_date(expected, extractor) == normalize_date(predicted, extractor)

    if field_name == "amount":
        return normalize_amount(expected, extractor) == normalize_amount(predicted, extractor)

    return str(expected).strip() == str(predicted).strip()


def evaluate_documents(limit: Optional[int] = None) -> Dict[str, Any]:
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with LABELS_PATH.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if limit is not None:
        rows = rows[:limit]

    ocr_pipeline = OCRPipeline(languages=["en"], use_gpu=False)
    extractor = InformationExtractor()

    field_stats = {
        "name": {"correct": 0, "total": 0},
        "date": {"correct": 0, "total": 0},
        "amount": {"correct": 0, "total": 0},
    }
    exact_match_count = 0
    failures: List[Dict[str, Any]] = []
    per_document_results: List[Dict[str, Any]] = []

    for row in rows:
        filename = (row.get("filename") or "").strip()
        image_path = DATA_DIR / filename
        if not filename or not image_path.exists():
            failures.append({
                "filename": filename,
                "reason": "Missing image file",
                "expected": {
                    "name": row.get("name"),
                    "date": row.get("date"),
                    "amount": row.get("amount"),
                },
                "predicted": None,
            })
            continue

        with image_path.open("rb") as image_file:
            image_bytes = image_file.read()

        image = load_image_from_bytes(image_bytes)
        extracted_text = ocr_pipeline.extract_text(image)
        predicted_fields = extractor.extract_all(extracted_text)

        expected_fields = {
            "name": row.get("name"),
            "date": row.get("date"),
            "amount": row.get("amount"),
        }

        field_matches = {}
        for field_name in ("name", "date", "amount"):
            field_stats[field_name]["total"] += 1
            matched = compare_field(field_name, expected_fields[field_name], predicted_fields.get(field_name), extractor)
            field_matches[field_name] = matched
            if matched:
                field_stats[field_name]["correct"] += 1

        exact_match = all(field_matches.values())
        if exact_match:
            exact_match_count += 1
        else:
            failures.append({
                "filename": filename,
                "expected": expected_fields,
                "predicted": predicted_fields,
                "field_matches": field_matches,
                "extracted_text": extracted_text,
            })

        per_document_results.append({
            "filename": filename,
            "expected": expected_fields,
            "predicted": predicted_fields,
            "field_matches": field_matches,
            "exact_match": exact_match,
        })

    total_documents = len(per_document_results)
    field_accuracy = {
        field_name: {
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy": round((stats["correct"] / stats["total"] * 100.0), 2) if stats["total"] else 0.0,
        }
        for field_name, stats in field_stats.items()
    }

    summary = {
        "total_documents": total_documents,
        "exact_match": {
            "correct": exact_match_count,
            "total": total_documents,
            "accuracy": round((exact_match_count / total_documents * 100.0), 2) if total_documents else 0.0,
        },
        "field_accuracy": field_accuracy,
        "failure_count": len(failures),
        "failures": failures,
        "documents": per_document_results,
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, ensure_ascii=False)

    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    print("Evaluation completed")
    print(f"Documents processed: {summary['total_documents']}")
    print(
        "Exact match accuracy: "
        f"{summary['exact_match']['correct']}/{summary['exact_match']['total']} "
        f"({summary['exact_match']['accuracy']}%)"
    )

    for field_name in ("name", "date", "amount"):
        stats = summary["field_accuracy"][field_name]
        print(
            f"{field_name.title()} accuracy: "
            f"{stats['correct']}/{stats['total']} ({stats['accuracy']}%)"
        )

    print(f"Failure count: {summary['failure_count']}")
    print(f"Saved evaluation report to: {SUMMARY_PATH}")


if __name__ == "__main__":
    evaluation_summary = evaluate_documents()
    print_summary(evaluation_summary)
