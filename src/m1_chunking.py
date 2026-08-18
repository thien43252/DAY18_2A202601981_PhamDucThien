from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Helpers dùng chung cho 3 strategies ────────────────

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?:;])\s+|\n+")

_EMBEDDER = None


def _get_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """Lazy-load + cache embedding model (tránh load lại model mỗi lần gọi)."""
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER = SentenceTransformer(model_name)
    return _EMBEDDER


def _split_sentences(text: str) -> list[str]:
    """Tách câu tiếng Việt đơn giản: theo dấu câu hoặc xuống dòng."""
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s and s.strip()]


def _pack(units: list[str], max_chars: int, joiner: str = " ") -> list[str]:
    """Gộp các đơn vị text thành block ≤ max_chars (đơn vị dài hơn ngưỡng bị cắt cứng)."""
    blocks: list[str] = []
    current = ""
    for unit in units:
        while len(unit) > max_chars:  # 1 đơn vị đã vượt ngưỡng → cắt cứng
            if current:
                blocks.append(current.strip())
                current = ""
            blocks.append(unit[:max_chars].strip())
            unit = unit[max_chars:].strip()
        if not unit:
            continue
        if current and len(current) + len(joiner) + len(unit) > max_chars:
            blocks.append(current.strip())
            current = unit
        else:
            current = f"{current}{joiner}{unit}" if current else unit
    if current.strip():
        blocks.append(current.strip())
    return blocks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None, max_chars: int = 1500) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.

    Thuật toán: embed từng câu → cosine_sim(câu i-1, câu i) < threshold nghĩa là
    chủ đề đã đổi → mở chunk mới; ngược lại gộp câu vào chunk hiện tại.
    """
    from numpy import dot
    from numpy.linalg import norm

    metadata = metadata or {}
    sentences = _split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(text=sentences[0],
                      metadata={**metadata, "chunk_index": 0, "strategy": "semantic",
                                "n_sentences": 1})]

    embeddings = _get_embedder().encode(sentences)

    def cosine_sim(a, b) -> float:
        return float(dot(a, b) / (norm(a) * norm(b) + 1e-9))

    groups: list[list[str]] = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = cosine_sim(embeddings[i - 1], embeddings[i])
        too_long = len(" ".join(groups[-1])) + len(sentences[i]) > max_chars
        if sim < threshold or too_long:
            groups.append([sentences[i]])    # ranh giới ngữ nghĩa → chunk mới
        else:
            groups[-1].append(sentences[i])  # cùng chủ đề → gộp tiếp

    return [
        Chunk(text=" ".join(g),
              metadata={**metadata, "chunk_index": i, "strategy": "semantic",
                        "n_sentences": len(g)})
        for i, g in enumerate(groups)
    ]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return ([], [])

    parents: list[Chunk] = []
    children: list[Chunk] = []

    # Parent: gộp paragraph tới ngưỡng parent_size → giữ ngữ cảnh rộng để trả lời
    for parent_text in _pack(paragraphs, parent_size, joiner="\n\n"):
        pid = f"parent_{len(parents)}"
        parents.append(Chunk(
            text=parent_text,
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid,
                      "chunk_index": len(parents), "strategy": "hierarchical"},
        ))

        # Child: cắt nhỏ theo câu → embedding đặc trưng hơn, retrieve chính xác hơn
        for child_text in _pack(_split_sentences(parent_text), child_size, joiner=" "):
            children.append(Chunk(
                text=child_text,
                metadata={**metadata, "chunk_type": "child", "parent_id": pid,
                          "chunk_index": len(children), "strategy": "hierarchical"},
                parent_id=pid,
            ))

    return (parents, children)


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None,
                          max_chars: int = 2000) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    parts = re.split(r"(^#{1,6}\s+.+$)", text, flags=re.MULTILINE)

    chunks: list[Chunk] = []

    def flush(header: str, body: str) -> None:
        body = body.strip()
        if not header and not body:
            return
        full = f"{header}\n\n{body}".strip() if header else body
        section = header.lstrip("#").strip() if header else "preamble"
        level = len(header) - len(header.lstrip("#")) if header else 0
        # Section dài quá → cắt theo paragraph nhưng lặp lại header ở mỗi mảnh
        pieces = _pack([p.strip() for p in full.split("\n\n") if p.strip()],
                       max_chars, joiner="\n\n") or [full]
        for piece in pieces:
            if header and not piece.startswith(header):
                piece = f"{header}\n\n{piece}"
            chunks.append(Chunk(
                text=piece,
                metadata={**metadata, "section": section, "header_level": level,
                          "chunk_index": len(chunks), "strategy": "structure"},
            ))

    current_header = ""
    buffer = ""
    for part in parts:
        if not part or not part.strip():
            continue
        stripped = part.strip()
        if re.match(r"^#{1,6}\s+", stripped) and "\n" not in stripped:
            flush(current_header, buffer)   # đóng section trước đó
            current_header = stripped
            buffer = ""
        else:
            buffer += part

    flush(current_header, buffer)
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
