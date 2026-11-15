"""Step 1-2: extraction and preprocessing for PDF / DOCX / XLSX / CSV."""
import io
import re
from dataclasses import dataclass

import pandas as pd
from docx import Document
from pypdf import PdfReader


@dataclass
class Segment:
    text: str
    source: str
    location: str = ""   # page number, sheet/rows, etc.


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(\w)", r"\1", text)        # join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(data: bytes, name: str):
    reader = PdfReader(io.BytesIO(data))
    segs = []
    for i, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            segs.append(Segment(text, name, f"page {i}"))
    return segs


def load_docx(data: bytes, name: str):
    doc = Document(io.BytesIO(data))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for t in doc.tables:
        for row in t.rows:
            parts.append(" | ".join(c.text.strip() for c in row.cells))
    text = clean_text("\n".join(parts))
    return [Segment(text, name, "document")] if text else []


def load_tabular(data: bytes, name: str):
    """Returns {sheet_name: DataFrame}."""
    if name.lower().endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(data))
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(data), encoding="latin-1")
        return {"data": df}
    return pd.read_excel(io.BytesIO(data), sheet_name=None)


def table_to_segments(df: pd.DataFrame, name: str, sheet: str, rows_per_chunk: int):
    """Turn rows into text chunks, repeating column names so each chunk is self-contained."""
    segs = []
    cols = list(df.columns)
    for start in range(0, len(df), rows_per_chunk):
        block = df.iloc[start:start + rows_per_chunk]
        lines = [f"File: {name} | Sheet: {sheet} | Columns: {', '.join(map(str, cols))}"]
        for idx, row in block.iterrows():
            lines.append(f"Row {idx + 2}: " + "; ".join(f"{c}={row[c]}" for c in cols))
        segs.append(Segment("\n".join(lines), name, f"{sheet} rows {start + 2}-{start + len(block) + 1}"))
    return segs
