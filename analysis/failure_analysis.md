# Failure Analysis — Lab 18: Production RAG

**Thực hiện:** Phạm Đức Thiên (2A202601981) — bài cá nhân, tự implement cả 5 modules M1–M5.
**Ngày chạy:** 18/08/2026 · **Test set:** 20 câu (`test_set.json`) · **Judge:** `gpt-4o-mini` qua RAGAS 0.1.22
**Nguồn số liệu:** `reports/naive_baseline_report.json` và `reports/ragas_report.json` (sinh bởi `python main.py`).

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8278 | **0.8685** | +0.0407 |
| Answer Relevancy | 0.7697 | **0.8485** | +0.0788 |
| Context Precision | 0.9250 | **0.9500** | +0.0250 |
| Context Recall | 0.9250 | **0.9500** | +0.0250 |

**Naive** = paragraph chunking (57 chunks) + dense-only top-3, không rerank, không enrichment.
**Production** = hierarchical chunking (26 parents / 103 children) + enrichment 1-call/chunk + hybrid BM25/dense + RRF top-20 + cross-encoder rerank top-3 + parent expansion + prompt có quy tắc version/tính toán.

### Latency breakdown (per query, ms)

| Stage | avg | p95 | max |
|---|---|---|---|
| Hybrid search (BM25 + dense + RRF) | 112.3 | 140.6 | 144.2 |
| Cross-encoder rerank (20 → 3) | 7411.1 | 8853.4 | 9309.4 |
| LLM answer (gpt-4o-mini) | 1759.9 | 3036.4 | 3082.8 |
| **Tổng / query** | **9602.5** | 11390.6 | 11513.5 |

Build một lần: chunking 0.0s · enrichment 103 chunks 39.6s · indexing 43.2s · load reranker 6.6s.
→ **Rerank chiếm 77% latency** (CPU). Đây là điểm tối ưu số 1 nếu đưa lên production.

---

## Bottom-5 Failures

### #1 — "Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?"
- **Expected:** Quá hạn 5 ngày → phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng, pro-rata ≈ 50.000 VNĐ.
- **Got:** Trích đúng điều khoản "2%/tháng trên số tiền chưa hoàn ứng", tính 15.000.000 × 2% × (5/30) = **50.000 VNĐ** — tức là **đúng ground truth**.
- **Worst metric:** faithfulness = 0.00 (recall 1.0, precision 1.0, relevancy 0.83; avg 0.708).
- **Error Tree:** Output sai? → **Không, output đúng** → Context đúng? → **Đúng (precision/recall = 1.0)** → Query OK? → **OK** → vậy lỗi nằm ở **cách metric chấm**, không ở pipeline.
- **Root cause:** RAGAS tách answer thành các claim rồi kiểm tra từng claim có xuất hiện trong context không. Các claim số học suy diễn ("5 ngày = 1/6 tháng", "= 50.000 VNĐ") không có nguyên văn trong `tam_ung.md` → bị chấm 0 dù suy luận hợp lệ. Đây là **false negative của faithfulness với câu numeric/multi-hop**.
- **Suggested fix:** (a) tách metric: dùng faithfulness cho câu lookup, dùng answer_correctness/so khớp số cho câu tính toán; (b) nếu vẫn muốn điểm faithfulness cao: bắt LLM chỉ trả lời công thức + con số gốc và đặt phép tính trong ngoặc "suy ra"; (c) bổ sung vào tài liệu ví dụ tính mẫu để phép tính có căn cứ nguyên văn.

### #2 — "Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?"
- **Expected:** 15 + 3 = 18 ngày phép (v2024) **và** lương Senior P3–P4: 20–35 triệu/tháng.
- **Got:** "18 ngày phép năm (15 + 3)… Về lương, **thông tin không tìm thấy trong tài liệu**."
- **Worst metric:** context_recall = 0.50 (faithfulness 0.667, precision 1.0, relevancy 0.863; avg 0.758).
- **Error Tree:** Output thiếu vế lương → Context có vế lương không? → **Không** (chỉ có 2 chunk nghỉ phép v2024/v2023, thiếu `bang_luong_2024.md`) → Query OK? → Query là **multi-hop 2 chủ đề** (phép + lương) nhưng chỉ có 1 embedding → retrieval bị kéo về chủ đề trội nhất.
- **Root cause:** top-3 sau rerank + parent dedupe không đủ chỗ cho 2 chủ đề; không có bước tách câu hỏi đa ý.
- **Suggested fix:** query decomposition (tách "phép" và "lương" thành 2 truy vấn con, gộp context), hoặc tăng `RERANK_TOP_K` lên 5 cho câu dài/nhiều mệnh đề (adaptive top-k), hoặc thêm MMR để đa dạng hoá chủ đề trong top-k.
- **Điểm tích cực:** answer **không bịa** phần lương — quy tắc 6 trong `ANSWER_SYSTEM_PROMPT` ("phần thiếu phải nói rõ không tìm thấy") hoạt động đúng. Ở prompt v1, chính câu này bịa ra "lương… có lương cho 15 ngày phép" (faithfulness 0.5).

### #3 — "Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng. Phải hoàn trả bao nhiêu?"
- **Expected:** Cam kết 1 năm; nghỉ trước hạn → hoàn trả 100% = 25.000.000 VNĐ.
- **Got:** Đúng: trích cam kết 1 năm → 8 tháng < 1 năm → hoàn 100% = 25.000.000 VNĐ.
- **Worst metric:** faithfulness = 0.286 (recall 1.0, precision 1.0, relevancy 0.764; avg 0.763).
- **Error Tree:** Output sai? → **Không** → Context đúng? → **Đúng** → lỗi lại ở bước chấm claim suy diễn ("8 tháng chưa đủ 1 năm", "= 25.000.000 VNĐ").
- **Root cause:** giống #1 — chuỗi suy luận multi-hop (điều kiện → áp dụng vào số của câu hỏi) không có nguyên văn trong context.
- **Suggested fix:** như #1; ngoài ra có thể tách "phần trích dẫn" và "phần tính toán" thành 2 trường trong output có cấu trúc, chỉ chấm faithfulness trên phần trích dẫn.

### #4 — "Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?"
- **Expected:** v2024: từ 3 năm, mỗi 3 năm +1 ngày (v2023 cũ: 5 năm).
- **Got:** Đúng — trả lời theo v2024 và nói rõ bản cũ đã bị thay thế.
- **Worst metric:** context_precision = 0.50 (3 metric còn lại 1.0 / 1.0 / 0.766; avg 0.817).
- **Error Tree:** Output đúng → Context đúng nhưng **có nhiễu**: chunk `nghi_phep_nam_v2023.md` xếp **trên** chunk v2024 → precision phạt vì chunk hạng 1 không phải chunk hữu ích.
- **Root cause:** hai phiên bản gần như trùng từ vựng → cả BM25 lẫn dense đều cho điểm gần bằng nhau; cross-encoder cũng không biết văn bản nào còn hiệu lực vì "hiệu lực" là **metadata nghiệp vụ**, không phải ngữ nghĩa câu.
- **Suggested fix:** version-aware filtering — đánh dấu `status: superseded` cho v2023/v1 ngay khi ingest (M5 `extract_metadata` đã sinh sẵn `topic/category`, chỉ cần thêm trường version/status), rồi lọc hoặc hạ trọng số ở tầng search thay vì để LLM tự chọn. Đây là fix quan trọng nhất về mặt sản phẩm: hiện độ đúng phiên bản đang phụ thuộc vào prompt.

### #5 — "Nhân viên được nghỉ bao nhiêu ngày phép năm?"
- **Expected:** 15 ngày (v2024 hiện hành); v2023 là 12 ngày, đã bị thay thế.
- **Got:** Đúng: "15 ngày… đã thay thế hoàn toàn phiên bản 2023 (12 ngày)".
- **Worst metric:** context_precision = 0.50 (avg 0.825).
- **Error Tree / Root cause / Fix:** hoàn toàn giống #4 — cùng cặp tài liệu v2023/v2024 cạnh tranh nhau. Nói cách khác, **2/5 failure là cùng một nguyên nhân gốc: thiếu quản lý phiên bản tài liệu ở tầng retrieval.**

---

## Tổng hợp nguyên nhân gốc

| Nhóm nguyên nhân | Số câu trong bottom-5 | Tầng bị lỗi | Fix ưu tiên |
|---|---|---|---|
| RAGAS phạt claim suy diễn ở câu numeric/multi-hop | #1, #3 | Đo lường (không phải pipeline) | Đổi metric cho nhóm câu tính toán; buộc trích dẫn trước khi tính |
| Chunk phiên bản cũ chen vào top-k | #4, #5 | Retrieval (thiếu metadata version) | Gắn `status: superseded` khi ingest → lọc/hạ trọng số |
| Câu hỏi đa chủ đề chỉ lấy được 1 chủ đề | #2 | Retrieval (top-k + query đơn) | Query decomposition / adaptive top-k / MMR |

---

## Case Study (cho presentation)

**Question chọn phân tích:** "Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?"

**Error Tree walkthrough:**
1. **Output đúng?** → Đúng một nửa: phần ngày phép chính xác (18 ngày), phần lương trả về "không tìm thấy trong tài liệu".
2. **Context đúng?** → Không đủ: context chỉ chứa `nghi_phep_nam_v2024.md` + `nghi_phep_nam_v2023.md`, **không có** `bang_luong_2024.md` → context_recall = 0.5. Vậy đây là **lỗi retrieval**, không phải lỗi generation (faithfulness của phần trả lời được vẫn ổn, và LLM không bịa).
3. **Query rewrite OK?** → Chưa có bước này. Câu hỏi gồm 2 mệnh đề độc lập ("bao nhiêu ngày phép" + "lương khoảng nào") nhưng chỉ được embed thành **một** vector → chủ đề "nghỉ phép" áp đảo, chủ đề "lương" bị đẩy khỏi top-20 trước cả khi rerank.
4. **Fix ở bước:** **Retrieval / query understanding** — tách câu hỏi đa mệnh đề thành các truy vấn con, chạy hybrid search cho từng phần rồi hợp nhất bằng RRF; đồng thời cho phép top-k linh hoạt (3 → 5) khi phát hiện câu hỏi nhiều mệnh đề.

**So sánh trước/sau khi sửa prompt (cùng retrieval, chỉ đổi `ANSWER_SYSTEM_PROMPT`):**

| Lần chạy | Faithfulness | Answer Relevancy | Ghi chú |
|---|---|---|---|
| Prompt v1 ("chỉ trả lời dựa trên context") | 0.8417 | 0.8030 | Câu Senior **bịa** phần lương |
| Prompt v2 (+ quy tắc trích dẫn trước khi tính, + "phần thiếu phải nói rõ", + câu trả lời hoàn chỉnh, temperature = 0) | **0.8786** | **0.8553** | Câu Senior nói rõ "không tìm thấy" |
| Prompt v2 — lần chạy cuối qua `main.py` | 0.8685 | 0.8485 | Dao động ±0.01 giữa các lần chạy do LLM judge |

**Nếu có thêm 1 giờ, sẽ optimize:**
1. **Version-aware retrieval** (fix 2/5 failure): thêm `version` + `status` vào metadata khi ingest, lọc `status != superseded` ở tầng search — vừa tăng context_precision, vừa loại rủi ro trả lời theo quy định hết hiệu lực.
2. **Giảm latency rerank** (7.4s → mục tiêu < 1s): thử `FlashrankReranker` (đã implement sẵn) và giới hạn rerank ở top-10 thay vì top-20; đo lại bằng `benchmark_reranker()`.
3. **Query decomposition** cho câu multi-hop (fix failure #2).
4. **OCR 2 file PDF scan** (`BCTC.pdf`, `Nghi_dinh_13-2023.pdf`) hiện đang bị bỏ khỏi index → đang có khoảng trống dữ liệu mà test set chưa chạm tới.
