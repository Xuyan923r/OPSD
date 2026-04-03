import argparse
import json
import re
from pathlib import Path

import pyarrow.parquet as pq


def extract_boxed_answer(text: str) -> str | None:
    """Extract the last \\boxed{...} span, supporting nested braces."""
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None

    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    if depth == 0:
        return text[start : i - 1].strip()
    return None


def extract_fallback_answer(text: str) -> str | None:
    """Fallback extraction for rows that do not end with a boxed answer."""
    search_text = text[text.rfind("</think>") + len("</think>") :] if "</think>" in text else text

    patterns = [
        r"(?is)\*\*Answer:\*\*\s*(.+?)\s*$",
        r"(?is)Answer:\s*(.+?)\s*$",
        r"(?is)The correct answer is\s*(.+?)\s*$",
        r"(?is)Thus, the correct answer is\s*(.+?)\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, search_text.strip())
        if match:
            answer = match.group(1).strip()
            answer = re.sub(r"^[\s:.-]+", "", answer)
            answer = re.sub(r"[\s.]+$", "", answer)
            return answer or None

    lines = [line.strip() for line in search_text.splitlines() if line.strip()]
    if not lines:
        return None

    last_line = lines[-1]
    last_line = re.sub(r"^\*\*Answer:\*\*\s*", "", last_line, flags=re.IGNORECASE)
    last_line = re.sub(r"^Answer:\s*", "", last_line, flags=re.IGNORECASE)
    return last_line or None


def extract_answer(messages: list[dict]) -> str | None:
    assistant_text = None
    for message in reversed(messages):
        if message.get("role") == "assistant":
            assistant_text = message.get("content", "")
            break

    if not assistant_text:
        return None

    answer = extract_boxed_answer(assistant_text)
    if answer is not None:
        return answer

    return extract_fallback_answer(assistant_text)


def convert_parquets_to_jsonl(input_dir: Path, output_path: Path, batch_size: int) -> tuple[int, int]:
    parquet_files = sorted(input_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under: {input_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    extracted_rows = 0

    with output_path.open("w", encoding="utf-8") as output_file:
        for parquet_file in parquet_files:
            parquet = pq.ParquetFile(parquet_file)
            for batch in parquet.iter_batches(batch_size=batch_size):
                for row in batch.to_pylist():
                    answer = extract_answer(row.get("messages", []))
                    row["answer"] = answer
                    if answer is not None:
                        extracted_rows += 1
                    output_file.write(json.dumps(row, ensure_ascii=False))
                    output_file.write("\n")
                    total_rows += 1

    return total_rows, extracted_rows


def main():
    parser = argparse.ArgumentParser(
        description="Convert Mixture-of-Thoughts science parquet shards into a JSONL with an added answer field."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("/idfsdata/yexuyan/OPSD/data/Mixture-of-Thoughts/science"),
        help="Directory containing the science parquet shards.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/idfsdata/yexuyan/OPSD/data/Mixture-of-Thoughts/science_with_answer.jsonl"),
        help="Path to the output JSONL file.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Number of rows to decode per parquet batch.",
    )
    args = parser.parse_args()

    total_rows, extracted_rows = convert_parquets_to_jsonl(args.input_dir, args.output, args.batch_size)
    missing_rows = total_rows - extracted_rows

    print(f"Input directory: {args.input_dir}")
    print(f"Output path: {args.output}")
    print(f"Total rows: {total_rows}")
    print(f"Rows with extracted answer: {extracted_rows}")
    print(f"Rows missing answer: {missing_rows}")


if __name__ == "__main__":
    main()
