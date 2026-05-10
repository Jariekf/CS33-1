"""Dataset intake tools for CSV reading and preview profiling."""

from __future__ import annotations

import csv
import os
from typing import Any

import pandas as pd


def dataset_intake(data_path: str) -> dict[str, Any]:
    """Infer basic file-reading options and record raw intake metadata."""

    sample_bytes = _read_sample_bytes(data_path)
    encoding, encoding_confidence = detect_encoding(sample_bytes)
    sample_text = sample_bytes.decode(encoding, errors="replace")
    delimiter, delimiter_confidence = detect_delimiter(sample_text)
    return {
        "source_path": data_path,
        "data_reading_options": {
            "sep": delimiter,
            "encoding": encoding,
        },
        "encoding_detection": {
            "encoding": encoding,
            "confidence": encoding_confidence,
        },
        "delimiter_detection": {
            "delimiter": delimiter,
            "confidence": delimiter_confidence,
        },
        "sample_size_bytes": len(sample_bytes),
    }


def read_preview(data_path: str, data_reading_options: dict[str, Any], nrows: int = 200) -> pd.DataFrame:
    """Read a bounded CSV preview using previously inferred read options."""

    return pd.read_csv(data_path, nrows=nrows, **data_reading_options)


def build_preview_metadata(data_path: str, preview: pd.DataFrame) -> dict[str, Any]:
    """Build a compact metadata snapshot from a normalized preview."""

    return {
        "source_file_size_bytes": os.path.getsize(data_path),
        "preview_rows": int(len(preview)),
        "preview_columns": int(len(preview.columns)),
        "column_names": list(preview.columns),
        "dtypes": {col: str(dtype) for col, dtype in preview.dtypes.items()},
        "missing_ratio": preview.isna().mean().round(4).to_dict(),
        "numeric_columns": list(preview.select_dtypes(include="number").columns),
        "categorical_columns": list(preview.select_dtypes(exclude="number").columns),
    }


def detect_encoding(sample: bytes) -> tuple[str, str]:
    """Detect a practical CSV encoding from common candidates."""

    if sample.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig", "high"
    for encoding in ("utf-8", "cp1252"):
        try:
            sample.decode(encoding)
            return encoding, "high"
        except UnicodeDecodeError:
            continue
    return "latin-1", "low"


def detect_delimiter(sample_text: str) -> tuple[str, str]:
    """Detect a likely delimiter from a text sample."""

    candidates = [",", ";", "\t", "|"]
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters="".join(candidates))
        if dialect.delimiter in candidates:
            return dialect.delimiter, "high"
    except csv.Error:
        pass

    lines = [line for line in sample_text.splitlines()[:25] if line.strip()]
    if not lines:
        return ",", "low"
    scores: dict[str, tuple[int, int]] = {}
    for delimiter in candidates:
        counts = [len(line.split(delimiter)) for line in lines]
        useful_counts = [count for count in counts if count > 1]
        if not useful_counts:
            scores[delimiter] = (0, 0)
            continue
        scores[delimiter] = (min(useful_counts), sum(useful_counts))
    delimiter, score = max(scores.items(), key=lambda item: item[1])
    return (delimiter, "medium") if score[0] > 1 else (",", "low")


def _read_sample_bytes(data_path: str, sample_size: int = 65536) -> bytes:
    """Read a bounded byte sample for file intake decisions."""

    with open(data_path, "rb") as f:
        return f.read(sample_size)
