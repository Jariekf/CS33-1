"""
Dataset description loader.

Usage:
    text = load_dataset_description("./data/telco_churn/README.md")
    text = load_dataset_description("./data/telco_churn/docs/")   # directory OK too

Design notes:
- Input is always a *path* (file or directory); free-form inline text is intentionally
  not supported, per project decision (dataset description should at minimum be a file).
- Format is not validated — we try hard to extract text from whatever extension appears.
  Unknown / unreadable formats degrade to a short placeholder line rather than raising.
- Optional binary-format readers (pdf, docx, xlsx) are imported lazily so the base
  pipeline keeps running without them installed.
"""
from __future__ import annotations

from pathlib import Path

# Treat these as "just read the bytes as text".
_TEXT_EXTS = {
    ".md",
    ".txt",
    ".rst",
    ".log",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".html",
    ".xml",
}


def _read_text_file(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return f"[{path.name}] (binary or unsupported encoding, skipped)"


def _read_tabular(path: Path) -> str:
    try:
        import pandas as pd

        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        try:
            body = df.head(50).to_markdown(index=False)
        except Exception:
            body = df.head(50).to_string(index=False)
        return f"[{path.name}] (tabular preview; first 50 rows)\n{body}"
    except Exception as e:
        return f"[{path.name}] (failed to parse as table: {e})"


def _read_xlsx(path: Path) -> str:
    try:
        import pandas as pd

        sheets = pd.read_excel(path, sheet_name=None)
        parts: list[str] = []
        for name, df in sheets.items():
            try:
                body = df.head(50).to_markdown(index=False)
            except Exception:
                body = df.head(50).to_string(index=False)
            parts.append(f"## Sheet: {name}\n{body}")
        return f"[{path.name}]\n" + "\n\n".join(parts)
    except ImportError:
        return f"[{path.name}] (xlsx; install `openpyxl` to extract content)"
    except Exception as e:
        return f"[{path.name}] (xlsx read failed: {e})"


def _read_pdf(path: Path) -> str:
    try:
        import pypdf  # optional dependency
    except ImportError:
        return f"[{path.name}] (pdf; `pip install pypdf` to extract text)"
    try:
        reader = pypdf.PdfReader(str(path))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception as e:
        return f"[{path.name}] (pdf read failed: {e})"


def _read_docx(path: Path) -> str:
    try:
        import docx  # optional dependency (python-docx)
    except ImportError:
        return f"[{path.name}] (docx; `pip install python-docx` to extract text)"
    try:
        return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    except Exception as e:
        return f"[{path.name}] (docx read failed: {e})"


def _read_one(path: Path) -> str:
    suf = path.suffix.lower()
    if suf in {".csv", ".tsv"}:
        return _read_tabular(path)
    if suf in {".xlsx", ".xlsm"}:
        return _read_xlsx(path)
    if suf == ".pdf":
        return _read_pdf(path)
    if suf == ".docx":
        return _read_docx(path)
    if suf in _TEXT_EXTS:
        return _read_text_file(path)
    return _read_text_file(path)  # unknown extension → best-effort text read


def load_dataset_description(desc_path: str | None) -> str:
    """
    Load dataset description from a file or a directory.

    Args:
        desc_path: Path to a file or directory containing dataset documentation.
            Empty / None is accepted and returns "" (caller decides whether to warn).

    Returns:
        Plain-text content merged from the input path.
        When the input is a directory, each file is prefixed with a
        ``===== FILE: <name> =====`` marker so the LLM can tell segments apart.

    Raises:
        FileNotFoundError: the path does not exist.
        ValueError: the path exists but is neither a regular file nor a directory,
            or the directory contains no files.
    """
    if not desc_path or not desc_path.strip():
        return ""

    p = Path(desc_path.strip())
    if not p.exists():
        raise FileNotFoundError(f"dataset description path not found: {p}")

    if p.is_file():
        return _read_one(p).strip()

    if p.is_dir():
        chunks: list[str] = []
        for child in sorted(p.iterdir()):
            if child.is_file():
                chunks.append(f"\n===== FILE: {child.name} =====\n{_read_one(child)}")
        if not chunks:
            raise ValueError(f"dataset description directory is empty: {p}")
        return "\n".join(chunks).strip()

    raise ValueError(f"unsupported dataset description path type: {p}")
