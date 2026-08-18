from __future__ import annotations

"""Production RAG Pipeline — ghép M1 + M5 + M2 + M3 + M4 thành 1 luồng end-to-end."""

import json
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K

# Latency của từng bước trong 1 query (ms) — gom lại thành bảng breakdown cuối run
_LATENCY: dict[str, list[float]] = {"search_ms": [], "rerank_ms": [], "llm_ms": [], "total_ms": []}

ANSWER_SYSTEM_PROMPT = """Bạn là trợ lý tra cứu chính sách nội bộ. Quy tắc:
1. CHỈ trả lời dựa trên context được cung cấp. Không suy đoán, không dùng kiến thức ngoài.
2. Nếu context không chứa thông tin → trả lời đúng một câu: "Không tìm thấy."
3. Nếu context có nhiều phiên bản mâu thuẫn (v1/v2, v2023/v2024): dùng phiên bản MỚI NHẤT,
   nêu rõ đó là chính sách hiện hành và ghi chú phiên bản cũ đã bị thay thế.
4. Nếu câu hỏi mang tính phủ định (có được / có nên không) mà context nói KHÔNG,
   hãy trả lời dứt khoát là KHÔNG kèm lý do trong context.
5. Nếu phải TÍNH TOÁN (cộng ngày phép, tính phí phạt, tính mức hoàn trả):
   trích nguyên văn quy định gốc trong context (con số + điều kiện) TRƯỚC,
   rồi mới đưa phép tính và kết quả — mọi con số phải truy được về context.
6. Nếu context thiếu một phần thông tin được hỏi, trả lời phần có trong context và
   nói rõ phần còn lại "không tìm thấy trong tài liệu" — tuyệt đối không đoán.
7. Trả lời bằng câu hoàn chỉnh, nhắc lại chủ thể của câu hỏi (không trả lời cụt lủn),
   ngắn gọn, đúng trọng tâm, bằng tiếng Việt, giữ nguyên con số/đơn vị trong context."""


def build_pipeline():
    """Build production RAG pipeline: chunk → enrich → index → reranker."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    stage_times = {}

    # Step 1: Load & Chunk (M1) — hierarchical: index child, giữ parent để mở rộng context
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    parent_map: dict[tuple[str, str], str] = {}
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        source = doc["metadata"].get("source", "")
        for parent in parents:
            parent_map[(source, parent.metadata.get("parent_id"))] = parent.text
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    stage_times["chunking_s"] = round(time.time() - t0, 1)
    print(f"  ✓ {len(all_chunks)} child chunks / {len(parent_map)} parents "
          f"from {len(docs)} documents ({stage_times['chunking_s']}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        stage_times["enrichment_s"] = round(time.time() - t0, 1)
        print(f"  ✓ Enriched {len(enriched)} chunks ({stage_times['enrichment_s']}s)", flush=True)
    else:
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    search.parent_map = parent_map  # dùng ở run_query: child hit → trả về parent
    stage_times["indexing_s"] = round(time.time() - t0, 1)
    print(f"  ✓ Indexed ({stage_times['indexing_s']}s)", flush=True)

    # Step 4: Reranker (M3) — warm-up để lần rerank đầu không tính cả thời gian load model
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    reranker.rerank("warm up", [{"text": "warm up", "score": 0.0, "metadata": {}}], top_k=1)
    stage_times["reranker_load_s"] = round(time.time() - t0, 1)
    print(f"  ✓ Reranker ready ({stage_times['reranker_load_s']}s)", flush=True)

    search.stage_times = stage_times
    return search, reranker


def _expand_to_parents(reranked, parent_map: dict) -> list[str]:
    """Hierarchical retrieval: match ở child (chính xác) → trả parent (đủ ngữ cảnh)."""
    contexts, seen = [], set()
    for r in reranked:
        meta = r.metadata or {}
        key = (meta.get("source", ""), meta.get("parent_id"))
        parent_text = parent_map.get(key) if parent_map else None
        source = meta.get("source", "")
        text = parent_text or r.text
        if key in seen:
            continue
        seen.add(key)
        # Gắn tên file nguồn để LLM phân biệt được phiên bản (v2023 vs v2024, v1 vs v2)
        contexts.append(f"[Nguồn: {source}]\n{text}" if source else text)
    return contexts


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline: hybrid search → rerank → parent expand → LLM."""
    t_start = time.perf_counter()

    t0 = time.perf_counter()
    results = search.search(query)
    search_ms = (time.perf_counter() - t0) * 1000

    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    t0 = time.perf_counter()
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    rerank_ms = (time.perf_counter() - t0) * 1000

    parent_map = getattr(search, "parent_map", {})
    if reranked:
        contexts = _expand_to_parents(reranked, parent_map)
    else:
        contexts = [r.text for r in results[:RERANK_TOP_K]]

    from config import OPENAI_API_KEY
    llm_ms = 0.0
    if OPENAI_API_KEY and contexts:
        try:
            from openai import OpenAI
            client = OpenAI()
            context_str = "\n\n---\n\n".join(contexts)
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,  # giảm hallucination → faithfulness cao hơn
                messages=[
                    {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
                ],
            )
            llm_ms = (time.perf_counter() - t0) * 1000
            answer = resp.choices[0].message.content
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."

    _LATENCY["search_ms"].append(search_ms)
    _LATENCY["rerank_ms"].append(rerank_ms)
    _LATENCY["llm_ms"].append(llm_ms)
    _LATENCY["total_ms"].append((time.perf_counter() - t_start) * 1000)
    return answer, contexts


def latency_report() -> dict:
    """Bảng latency trung bình/p95 từng bước (bonus: latency breakdown)."""
    def _pct(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))
        return ordered[idx]

    report = {}
    for stage, values in _LATENCY.items():
        if values:
            report[stage] = {
                "avg": round(sum(values) / len(values), 1),
                "p95": round(_pct(values, 0.95), 1),
                "max": round(max(values), 1),
            }
    return report


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    print(f"  ✓ RAGAS done ({time.time() - t0:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)

    # Bổ sung latency breakdown vào report (không đổi schema aggregate/num_questions/failures)
    lat = latency_report()
    stages = getattr(search, "stage_times", {})
    print("\nLATENCY BREAKDOWN (per query, ms)")
    print(f"  {'stage':<12}{'avg':>10}{'p95':>10}{'max':>10}")
    for stage, stats in lat.items():
        print(f"  {stage:<12}{stats['avg']:>10.1f}{stats['p95']:>10.1f}{stats['max']:>10.1f}")
    if stages:
        print("\nBUILD STAGES (s): " + ", ".join(f"{k}={v}" for k, v in stages.items()))

    try:
        with open("ragas_report.json", encoding="utf-8") as f:
            report = json.load(f)
        report["latency_ms"] = lat
        report["build_stages_s"] = stages
        with open("ragas_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"  ⚠️  Không ghi được latency vào report: {e}")

    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
