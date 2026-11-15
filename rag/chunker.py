"""Step 3: sentence-aware chunking with overlap."""
import re


def split_text(text: str, size: int = 800, overlap: int = 120):
    sentences = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    pieces = []
    for s in sentences:                       # hard-split very long sentences
        s = s.strip()
        while len(s) > size:
            pieces.append(s[:size])
            s = s[size - overlap:]
        if s:
            pieces.append(s)

    chunks, current = [], ""
    for p in pieces:
        if current and len(current) + len(p) + 1 > size:
            chunks.append(current.strip())
            tail = current[-overlap:] if overlap else ""
            current = tail + " " + p
        else:
            current = f"{current} {p}".strip()
    if current.strip():
        chunks.append(current.strip())
    return chunks
