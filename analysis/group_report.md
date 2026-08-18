# Report — Lab 18: Production RAG

**Bài nộp cá nhân** (theo ASSIGNMENT.md: mỗi học viên implement toàn bộ 5 modules).
**Học viên:** Phạm Đức Thiên — 2A202601981 · AICB-K34
**Ngày:** 18/08/2026

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| Phạm Đức Thiên | M1: Chunking (semantic / hierarchical / structure-aware) | ☑ | 13/13 |
| Phạm Đức Thiên | M2: Hybrid Search (underthesea + BM25 + bge-m3/Qdrant + RRF) | ☑ | 5/5 |
| Phạm Đức Thiên | M3: Reranking (bge-reranker-v2-m3 cross-encoder, + flashrank optional) | ☑ | 5/5 |
| Phạm Đức Thiên | M4: Evaluation (RAGAS 4 metrics + Diagnostic Tree) | ☑ | 4/4 |
| Phạm Đức Thiên | M5: Enrichment (combined 1-call/chunk + 4 technique riêng lẻ) | ☑ | 10/10 |

**Tổng: 37/37 tests pass · 0 TODO còn lại trong `src/m*.py`.**

## Kết quả RAGAS

| Metric | Naive | Production | Δ |
|--------|-------|-----------|---|
| Faithfulness | 0.8278 | **0.8685** | +0.0407 |
| Answer Relevancy | 0.7697 | **0.8485** | +0.0788 |
| Context Precision | 0.9250 | **0.9500** | +0.0250 |
| Context Recall | 0.9250 | **0.9500** | +0.0250 |

Cả 4 metric đều ≥ 0.75; faithfulness ≥ 0.85.
Nguồn: `reports/naive_baseline_report.json`, `reports/ragas_report.json` (một lần chạy `python main.py`, 437.9s).

### Cấu hình 2 nhánh

| | Naive Baseline | Production |
|---|---|---|
| Chunking | paragraph 500 ký tự → 57 chunks | hierarchical 2048/256 → 26 parents / 103 children |
| Enrichment | không | 1 API call/chunk (context + summary + HyQA + metadata), 39.6s |
| Search | dense-only (bge-m3, top-3) | BM25 (underthesea) + dense + RRF, top-20 |
| Rerank | không | cross-encoder bge-reranker-v2-m3, top-20 → top-3 |
| Context cho LLM | 3 child chunk | parent của các child trúng, kèm `[Nguồn: file]` |
| Prompt | "trả lời dựa trên context" | + quy tắc version, quy tắc tính toán, "không đoán", temperature=0 |

### Latency (per query)

| Stage | avg (ms) | p95 (ms) |
|---|---|---|
| Hybrid search | 112.3 | 140.6 |
| Cross-encoder rerank | 7411.1 | 8853.4 |
| LLM answer | 1759.9 | 3036.4 |
| **Tổng** | **9602.5** | 11390.6 |

## Key Findings

1. **Biggest improvement — Answer Relevancy +0.0788.** Phần lớn đến từ prompt: yêu cầu trả lời bằng câu hoàn chỉnh, nhắc lại chủ thể câu hỏi. Baseline hay trả lời cụt ("KHÔNG, ...") khiến RAGAS chấm relevancy = 0 cho chính câu trả lời đúng.
2. **Biggest challenge — faithfulness ở câu numeric/multi-hop.** Retrieval đã hoàn hảo (precision/recall = 1.0) mà faithfulness vẫn 0.0–0.29 vì RAGAS phạt các claim suy diễn (phép tính) không có nguyên văn trong context. Fix bằng cách buộc LLM trích quy định gốc trước rồi mới tính: 0.8417 → 0.8786.
3. **Surprise finding — baseline đã rất mạnh (precision/recall 0.925).** Corpus nhỏ (26 tài liệu, ~21k ký tự) và câu hỏi bám sát tài liệu nên dense-only đã đủ; giá trị của hybrid + rerank ở đây không nằm ở "tìm được tài liệu" mà ở **chọn đúng phiên bản** và **giữ đủ ngữ cảnh** (parent expansion). 2/5 failure còn lại đều là chunk phiên bản cũ (v2023 / mật khẩu v1) chen vào top-k — vấn đề metadata, không phải vấn đề embedding.
4. **Rerank là bottleneck vận hành:** 7.4s/query trên CPU, chiếm 77% tổng latency, trong khi hybrid search chỉ 112ms.

## Presentation Notes (5 phút)

1. **RAGAS scores (naive vs production):** 0.8278/0.7697/0.925/0.925 → 0.8685/0.8485/0.95/0.95 — cả 4 metric cùng tăng, không đánh đổi.
2. **Biggest win — module nào, tại sao:** M5 enrichment + prompt engineering ở tầng generation. Enrichment gắn câu mô tả có nêu phiên bản vào mỗi chunk trước khi embed; prompt biến "trả lời đúng nhưng cụt/suy diễn" thành "trả lời đúng có trích dẫn".
3. **Case study — 1 failure, Error Tree walkthrough:** câu "Senior 9 năm thâm niên: bao nhiêu ngày phép và lương khoảng nào?" → output thiếu vế lương → context thiếu `bang_luong_2024.md` (recall 0.5) → nguyên nhân: câu hỏi 2 mệnh đề nhưng chỉ 1 vector truy vấn → fix ở tầng query understanding (decomposition), không phải ở generation. Điểm tốt: hệ thống nói "không tìm thấy" thay vì bịa.
4. **Next optimization nếu có thêm 1 giờ:** (a) version-aware filtering bằng metadata `status: superseded` — xử lý 2/5 failure; (b) hạ latency rerank bằng flashrank/GPU; (c) query decomposition cho câu multi-hop; (d) OCR 2 PDF scan đang bị loại khỏi index.
