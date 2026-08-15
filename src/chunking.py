import re
from typing import Dict, List, Any


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n +", "\n", text).strip()


def clean_pdf_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = text.replace("\ufeff", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\n(?=\d+\s*$)", "\n\n", text)
    return normalize_whitespace(text)


def detect_section_heading(lines: List[str]) -> str:
    for line in lines:
        candidate = line.strip()
        if not candidate:
            continue
        if len(candidate.split()) <= 10 and (candidate.isupper() or candidate.endswith(":") or candidate[0].isupper()):
            return candidate
    return "General"


def chunk_policy_document(policy_name: str, source_file: str, pages: List[Dict[str, Any]], chunk_size: int = 700, overlap: int = 100) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for page_entry in pages:
        page_number = int(page_entry.get("page", 1))
        raw_text = clean_pdf_text(page_entry.get("text", ""))
        if not raw_text:
            continue

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", raw_text) if p.strip()]
        if not paragraphs:
            paragraphs = [raw_text]

        section = detect_section_heading(raw_text.splitlines())
        buffer: List[str] = []
        section_buffer: List[str] = []

        for paragraph in paragraphs:
            if len(paragraph.split()) <= 12 and paragraph.lower() not in {"page", "policy"}:
                if section_buffer:
                    section_buffer.append(paragraph)
                else:
                    section_buffer = [paragraph]
                continue

            if section_buffer:
                buffer.extend(section_buffer)
                section_buffer = []

            buffer.append(paragraph)

        text_blocks = [b.strip() for b in buffer if b.strip()]
        if not text_blocks:
            text_blocks = [raw_text]

        current = ""
        for block in text_blocks:
            if not current:
                current = block
                continue
            if len(current) + len(block) <= chunk_size:
                current = f"{current}\n\n{block}"
            else:
                chunks.append({
                    "chunk_id": f"{policy_name}-{source_file}-{page_number}-{len(chunks) + 1}",
                    "policy_name": policy_name,
                    "source_file": source_file,
                    "page": page_number,
                    "section": section,
                    "text": current.strip(),
                    "country": "India",
                })
                current = block

        if current:
            chunks.append({
                "chunk_id": f"{policy_name}-{source_file}-{page_number}-{len(chunks) + 1}",
                "policy_name": policy_name,
                "source_file": source_file,
                "page": page_number,
                "section": section,
                "text": current.strip(),
                "country": "India",
            })

    if overlap > 0 and len(chunks) > 1:
        overlapped = []
        for idx, chunk in enumerate(chunks):
            if idx == 0:
                overlapped.append(chunk)
                continue
            prev = overlapped[-1]["text"]
            current_text = chunk["text"]
            if len(current_text) < overlap:
                chunk["text"] = prev + "\n\n" + current_text
            else:
                chunk["text"] = prev[-overlap:] + "\n\n" + current_text
            overlapped.append(chunk)
        chunks = overlapped

    return [chunk for chunk in chunks if chunk.get("text")]
