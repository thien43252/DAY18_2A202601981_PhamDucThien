# Reflection — Lab 18: Production RAG Pipeline

**Học viên:** Phạm Đức Thiên · **Mã:** 2A202601981 · **Lớp:** AICB-K34 — Ngày 18
**Môi trường chạy:** Windows 11, Python 3.11.9 (uv venv), Qdrant qua Docker, `gpt-4o-mini` cho generation + RAGAS judge, embedding `BAAI/bge-m3`, reranker `BAAI/bge-reranker-v2-m3` (CPU).

**Kết quả cuối (chạy `python main.py`, 20 câu test):**

| Metric | Naive Baseline | Production | Δ |
|---|---|---|---|
| Faithfulness | 0.8278 | **0.8685** | +0.0407 |
| Answer Relevancy | 0.7697 | **0.8485** | +0.0788 |
| Context Precision | 0.9250 | **0.9500** | +0.0250 |
| Context Recall | 0.9250 | **0.9500** | +0.0250 |

> Ghi chú: Phần 3 (Action Plan) viết cho project cá nhân "Trợ lý tra cứu quy định nội bộ tiếng Việt"; nếu project của bạn khác, phần tên project và mốc tuần cần chỉnh lại cho đúng.

---

## Phần 1: Mapping bài giảng → code

| Lecture Concept | Module | Hàm cụ thể | Observation (số liệu thật khi chạy repo này) |
|---|---|---|---|
| Semantic chunking (ngắt theo ranh giới ngữ nghĩa) | M1 | `chunk_semantic()` | Với `threshold=0.85` trên toàn corpus (20.976 ký tự): **481 chunks, avg 42 ký tự** so với basic **51 chunks, avg 410**. Ngưỡng 0.85 quá gắt với văn bản chính sách tiếng Việt (câu ngắn, nhiều gạch đầu dòng) → chunk vụn, mất ngữ cảnh. Muốn dùng thật thì phải hạ threshold xuống ~0.5–0.6 hoặc ép `max_chars` tối thiểu. |
| Hierarchical / parent-child retrieval | M1 + pipeline | `chunk_hierarchical()` + `_expand_to_parents()` | Corpus sinh **26 parents / 103 children** (parent 2048, child 256). Search khớp ở child (embedding đặc trưng hơn) nhưng context trả cho LLM là parent → giữ đủ điều khoản xung quanh. Đây là strategy tôi chọn cho pipeline production. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | **106 chunks, avg 197**, mỗi chunk giữ nguyên header (`section` trong metadata). Điểm mạnh: chunk tự mang tiêu đề mục → LLM biết đang đọc điều khoản nào; điểm yếu: 1 section bảng dài vẫn ra chunk 788 ký tự. |
| Vietnamese word segmentation | M2 | `segment_vietnamese()` | underthesea nối từ ghép bằng `_` ("nghỉ_phép"). Nếu giữ nguyên, BM25 tokenize query "nghỉ phép" thành 2 token → **không khớp token nào**. Sau khi `replace("_", " ")` + lowercase đồng nhất giữa index và query, BM25 trả đúng doc nghỉ phép. |
| BM25 + Dense fusion (RRF) | M2 | `reciprocal_rank_fusion()` | RRF chỉ dùng **thứ hạng**, không dùng score thô → không phải chuẩn hoá giữa BM25 (score không chặn trên) và cosine (0–1). Dense bắt được câu hỏi diễn đạt khác từ vựng tài liệu, BM25 bắt chính xác con số/mã ("2%/tháng", "MFA"); hợp nhất bằng `1/(60 + rank + 1)`. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Rerank top-20 → top-3. Latency đo được: **avg ~6.9s/query trên CPU** (chiếm ~77% tổng 9.0s/query), search chỉ ~107ms. Precision tăng nhưng đây là bottleneck rõ ràng — production phải chạy GPU hoặc đổi sang flashrank (`FlashrankReranker` đã implement sẵn để so sánh). |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Metric thấp nhất là **faithfulness** ở các câu multi-hop có tính toán: RAGAS tách answer thành từng claim, claim "18 ngày = 15 + 3" hay "50.000 VNĐ" không xuất hiện nguyên văn trong context → bị chấm là không có căn cứ, dù kết quả đúng với ground truth. |
| Diagnostic Tree / failure analysis | M4 | `failure_analysis()` | Map metric yếu nhất → nguyên nhân → fix: faithfulness→siết prompt, context_recall→chunking/top-k, context_precision→rerank/lọc metadata, answer_relevancy→prompt template. Bottom-5 xuất ra thẳng `ragas_report.json` để viết `failure_analysis.md`. |
| Contextual embeddings (Anthropic style) | M5 | `contextual_prepend()` / `_enrich_single_call()` | Mỗi chunk được prepend 1 câu mô tả "chunk này nằm ở đâu, nói về gì" + các câu hỏi giả định (HyQA) trước khi embed. Với corpus có 2 phiên bản song song (v2023/v2024, mật khẩu v1/v2), câu context nêu rõ version giúp chunk phiên bản mới không bị chìm dưới chunk phiên bản cũ. |
| Enrichment cost optimization | M5 | `_enrich_single_call()` | Combined mode: **1 API call/chunk** thay vì 4 (summary + HyQA + context + metadata trong 1 JSON response) → giảm ~75% call. 103 chunks enrich hết **~40–46s** nhờ `ThreadPoolExecutor(max_workers=8)`. |
| Query → answer prompt engineering | pipeline | `ANSWER_SYSTEM_PROMPT` | Prompt v1 (chỉ "trả lời dựa trên context") cho faithfulness 0.8417 / relevancy 0.8030. Prompt v2 (thêm quy tắc: trích nguyên văn quy định trước khi tính toán, trả lời bằng câu hoàn chỉnh, nói rõ phần thiếu) → **0.8786 / 0.8553**. Cùng retrieval, chỉ đổi prompt. |

---

## Phần 2: Khó khăn & cách giải quyết

### 2.1. `UnicodeEncodeError` — pipeline chết ngay dòng print đầu tiên

```
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f4cc'
in position 2: character maps to <undefined>   (Lib/encodings/cp1252.py)
```

- **Nguyên nhân:** console Windows mặc định cp1252, trong khi code print emoji (📌, ⚠️) và tiếng Việt có dấu.
- **Debug:** traceback trỏ thẳng vào `cp1252.py` → không phải lỗi logic RAG mà là lỗi encoding stdout.
- **Fix:** ép UTF-8 ngay khi load config (`config.py`): `sys.stdout.reconfigure(encoding="utf-8")` cho cả stdout/stderr, và `import config` sớm trong `main.py` (vì `main.py` print emoji **trước** khi import module nào chạm config → vẫn crash nếu không import sớm).
- **Bài học:** với script có log tiếng Việt, encoding phải được set ở entry point, không phải ở module load muộn.

### 2.2. BM25 không trả về kết quả cho query tiếng Việt

- **Triệu chứng:** `bm25.search("nghỉ phép")` trả list rỗng dù corpus có câu "Nhân viên được nghỉ phép năm 12 ngày".
- **Debug:** in `corpus_tokens[0]` → thấy `['nhân viên', 'được', 'nghỉ_phép', ...]`, còn query tokenize ra `['nghỉ', 'phép']` → giao nhau bằng 0 → mọi score = 0 và bị lọc bởi điều kiện `score > 0`.
- **Fix:** `segment_vietnamese()` phải `replace("_", " ")`, và dùng **một hàm tokenize duy nhất** (`BM25Search._tokenize`) cho cả index lẫn query, kèm `lower()` để không lệch vì viết hoa đầu câu.

### 2.3. `recreate_collection` deprecated trên qdrant-client 1.19

- Scaffold gợi ý `recreate_collection()`; bản 1.19 vẫn còn nhưng đã deprecated và sẽ bỏ.
- **Fix:** dùng `collection_exists()` → `delete_collection()` → `create_collection()`, và upsert theo batch 128 point để tránh request quá lớn. Search dùng `query_points()` (không phải `search()` đã deprecated).

### 2.4. `No module named pytest` trong venv do uv quản lý

- venv được tạo bằng `uv` nên không có `pip`; `python -m pip install pytest` báo `No module named pip`.
- **Fix:** `uv add --dev pytest` → 37/37 test pass.

### 2.5. Faithfulness thấp ở câu multi-hop có tính toán (khó nhất)

- **Triệu chứng run #1:** câu "tạm ứng 15 triệu, trả sau 20 ngày bị phạt bao nhiêu?" → answer đúng nhưng `faithfulness = 0.33`; câu "Senior 9 năm thâm niên" → `faithfulness = 0.5`.
- **Debug bằng Error Tree:** context đúng (context_precision = 1.0, recall = 1.0) → lỗi không nằm ở retrieval mà ở **generation**: LLM đưa ra con số suy diễn (15+3 = 18 ngày, 15tr × 2% × 5/30 = 50.000 VNĐ) mà context không chứa nguyên văn → RAGAS coi claim đó không có căn cứ.
- **Fix:** thêm quy tắc 5 và 6 vào `ANSWER_SYSTEM_PROMPT` — bắt buộc trích nguyên văn quy định gốc trước khi tính, và nói rõ phần nào không có trong tài liệu thay vì đoán; đồng thời set `temperature=0`.
- **Kết quả:** faithfulness 0.8417 → **0.8786**, answer_relevancy 0.8030 → **0.8553** (cùng retrieval, chỉ đổi prompt). Lần chạy cuối qua `main.py` cho 0.8685 / 0.8485 — chênh ±0.01 so với lần đo trên là do LLM judge của RAGAS không tất định, nên khi so sánh cấu hình phải chạy lại **cả hai** nhánh trong cùng một lần chạy như `main.py` đang làm.

### 2.6. Kiến thức còn thiếu → cách bổ sung

| Thiếu gì | Bổ sung thế nào |
|---|---|
| Cách RAGAS chấm faithfulness (claim decomposition) | Đọc docs RAGAS 0.1 + in `result.to_pandas()` từng dòng để xem claim nào bị đánh trượt; hiểu rằng metric phạt cả suy luận đúng nếu không truy được về context |
| Chọn threshold cho semantic chunking | Thực nghiệm: chạy `compare_strategies()` với nhiều threshold, xem phân phối độ dài chunk trước khi chốt |
| Reranker phục vụ ở production (latency) | Cần thử flashrank / ONNX / GPU và benchmark; hiện chỉ có số liệu CPU |
| Xử lý PDF scan (BCTC.pdf, Nghị định 13/2023) | 2 file này bị bỏ qua vì không có text layer → cần OCR (paddleocr/tesseract) mới đưa vào index được; đây là khoảng trống dữ liệu đã biết của lab này |

---

## Phần 3: Action Plan cho project

## Project: Trợ lý tra cứu quy định nội bộ tiếng Việt (RAG assistant)

### Hiện tại
- RAG pipeline hiện tại: chunk theo paragraph cố định → embed → dense search top-3 → LLM trả lời. Đúng bằng baseline `naive_baseline.py` của lab này.
- Known issues:
  1. Tài liệu có nhiều phiên bản (quy định v1/v2, năm 2023/2024) → assistant trả lời theo bản cũ.
  2. Câu hỏi chứa mã/số cụ thể (mã khoản mục, ngưỡng tiền) hay trượt vì dense search không khớp từ khoá hiếm.
  3. Không có cách đo chất lượng — chỉ đánh giá cảm tính khi demo.
  4. Tài liệu PDF scan chưa dùng được.

### Plan áp dụng
1. [ ] **Chunking strategy:** hierarchical (parent 2048 / child 256) làm mặc định — retrieve ở child, trả parent cho LLM; với tài liệu markdown/quy chế có heading rõ thì chồng thêm structure-aware để mỗi chunk mang tên điều khoản. **Không** dùng semantic threshold 0.85 (đã chứng minh vụn chunk: 481 chunks avg 42 ký tự).
2. [ ] **Search:** hybrid BM25 (underthesea + `replace("_"," ")`) + dense bge-m3, hợp nhất bằng RRF k=60, top-20. Lý do: câu hỏi nội bộ trộn cả ngôn ngữ tự nhiên lẫn mã/số — dense một mình trượt token hiếm, BM25 một mình trượt cách diễn đạt khác.
3. [ ] **Reranking:** có — cross-encoder `bge-reranker-v2-m3` top-20 → top-3. Nhưng phải giải quyết latency 6.9s/query trên CPU: chạy GPU hoặc dùng flashrank cho tier realtime, giữ cross-encoder cho tier chất lượng cao (batch/offline).
4. [ ] **Evaluation:** RAGAS 4 metrics làm cổng chất lượng (ngưỡng chấp nhận: faithfulness ≥ 0.85, 3/4 metric ≥ 0.75) + bộ test 30–50 câu tự viết theo 6 loại (lookup, version, negation, multi-hop, numeric, ambiguous), chạy lại mỗi lần đổi prompt/chunking. Bổ sung metric custom "version-correctness": câu trả lời có trích đúng phiên bản hiện hành không.
5. [ ] **Enrichment:** contextual prepend + HyQA gộp trong **1 call/chunk** (`_enrich_single_call`) — rẻ nhất trên mỗi đơn vị cải thiện, và chính câu context là thứ giúp phân biệt bản v2024 với v2023. Auto metadata (`category`, `topic`) để về sau lọc theo phòng ban.

### Timeline

| Tuần | Việc | Đầu ra đo được |
|---|---|---|
| Tuần 1 | Dựng test set 40 câu + chạy baseline hiện tại qua RAGAS | Bảng điểm baseline (4 metrics) làm mốc so sánh |
| Tuần 2 | Thay chunking sang hierarchical + structure-aware, index lại | So sánh context_recall trước/sau |
| Tuần 3 | Thêm BM25 + RRF, bật enrichment 1-call/chunk | context_precision & recall tăng; log chi phí enrichment/1000 chunk |
| Tuần 4 | Thêm reranker + tối ưu latency (flashrank vs cross-encoder GPU) | Bảng latency breakdown p95 < 2s/query |
| Tuần 5 | Vòng lặp prompt: quy tắc version, quy tắc tính toán, quy tắc "không đoán" | faithfulness ≥ 0.85 |
| Tuần 6 | OCR tài liệu PDF scan → đưa vào index, đánh giá lại toàn bộ | Số tài liệu phủ được tăng; RAGAS không tụt |

### Nguyên tắc rút ra từ lab
- **Đo trước, tối ưu sau:** baseline chạy trước cho phép nói chính xác module nào đem lại bao nhiêu điểm.
- **Retrieval tốt chưa đủ:** context_precision/recall đã 0.95 mà faithfulness vẫn 0.84 — nút thắt nằm ở prompt generation.
- **Rẻ mà hiệu quả trước:** một dòng quy tắc trong system prompt đem lại +0.037 faithfulness, không tốn thêm 1 đồng infra nào.
