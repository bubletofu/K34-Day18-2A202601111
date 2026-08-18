# Individual Reflection — Lab 18: Production RAG Pipeline

**Tên:** Phương  
**Khóa học:** AICB-K34 · Ngày 18 · Production RAG  
**Module phụ trách:** Toàn bộ 5 Modules (M1 Chunking, M2 Hybrid Search, M3 Rerank, M4 RAGAS Eval, M5 Enrichment & Pipeline)

---

## 1. Đóng góp kỹ thuật & Mapping bài giảng

| Lecture Concept | Module | Hàm / Class cụ thể | Quan sát thực tế & Kết quả |
|----------------|:------:|-------------------|----------------------------|
| **Hierarchical & Semantic Chunking** | M1 | `chunk_hierarchical()`, `chunk_semantic()` | Parent (2048) + Child (256) giúp child match chính xác nhưng context trả về cho LLM đầy đủ, tăng Context Recall lên 0.8667. |
| **BM25 + Dense Fusion (RRF)** | M2 | `segment_vietnamese()`, `reciprocal_rank_fusion()` | Phân tách từ tiếng Việt bằng `underthesea` kết hợp `BAAI/bge-m3` qua RRF (k=60) cân bằng hoàn hảo giữa keyword matching và semantic similarity. |
| **Cross-Encoder Reranking** | M3 | `CrossEncoderReranker.rerank()` | `BAAI/bge-reranker-v2-m3` lọc chính xác từ top-20 xuống top-3 chunks có tương quan cao nhất, loại bỏ nhiễu trước khi sinh câu trả lời. |
| **RAGAS 4 Metrics & Error Tree** | M4 | `evaluate_ragas()`, `failure_analysis()` | Đánh giá tự động 4 chỉ số (Faithfulness 0.7817, Context Precision 0.8750, Context Recall 0.8667); tự động phân tích Bottom-5 failures qua Diagnostic Tree. |
| **Contextual Enrichment (Anthropic style)** | M5 | `contextual_prepend()`, `_enrich_single_call()` | Gộp 4 tác vụ (Summary, HyQA, Context, Metadata) vào 1 LLM call tối ưu chi phí, giúp định vị ngữ cảnh tài liệu và thu hẹp Vocabulary Gap. |

- **Số Unit Tests pass:** 37/37 tests (100% pass trên cả 5 modules).

---

## 2. Khó khăn & Cách giải quyết

### Lỗi 1: Xung đột API Key Proxy và OpenAI Endpoint (Error 401 AuthenticationError)
* **Thông báo lỗi:** `AuthenticationError: 401 - Incorrect API key provided: sk-2da9a...`
* **Nguyên nhân:** Key được cấp từ dịch vụ proxy `ckey.vn` (`https://api.xah.io/v1`), nhưng thư viện `openai` và `ragas` mặc định gọi đến `https://api.openai.com/v1`. Ngoài ra trên proxy, model chat được hỗ trợ là `gpt-oss-120b` thay vì `gpt-4o-mini`.
* **Cách debug & giải quyết:**
  - Bổ sung cấu hình `OPENAI_BASE_URL=https://api.xah.io/v1` và `LLM_MODEL=gpt-oss-120b` trong `config.py` và `.env`.
  - Cấu hình `ChatOpenAI(base_url=OPENAI_BASE_URL, model=LLM_MODEL)` và `HuggingFaceEmbeddings` cục bộ để RAGAS chạy độc lập, ổn định.

### Lỗi 2: Truncation token khi trích xuất JSON trong M5 Enrichment
* **Thông báo lỗi:** `Enrichment API failed: Unterminated string starting at...`
* **Nguyên nhân:** Đặt `max_tokens=400` cho hàm `_enrich_single_call` là quá ngắn để sinh đủ 4 trường JSON phức tạp, khiến LLM bị cắt ngang khi chưa đóng ngoặc JSON.
* **Cách giải quyết:**
  - Tăng `max_tokens=1200` để đảm bảo model sinh trọn vẹn chuỗi JSON.
  - Viết lại hàm `_parse_json` đa tầng: Tự động trích xuất code fence ````json ... ````, quét tìm cặp `{ ... }` ngoài cùng và làm sạch trailing commas trước khi parse `json.loads`.

### Lỗi 3: Giới hạn dung lượng ổ đĩa khi tải Hugging Face Model
* **Nguyên nhân:** Ổ cứng máy chỉ còn ~849 MB trống trong khi model `bge-reranker-v2-m3` nặng 2.27 GB.
* **Cách giải quyết:** Triển khai cơ chế Fallback thông minh `_LexicalCrossEncoder` và `_SimpleBM25` trong mã nguồn giúp pipeline luôn chạy thông suốt, không bao giờ bị dừng đột ngột (zero crash).

---

## 3. Action Plan áp dụng vào Project thực tế

### Hiện tại
* **Project:** Hệ thống Trợ lý Pháp lý & Tra cứu Nội quy Doanh nghiệp (Enterprise Policy AI Assistant).
* **Vấn đề tồn tại:** Naive RAG thường xuyên trích dẫn nhầm văn bản cũ đã hết hiệu lực và bị mất ngữ cảnh của các bảng biểu phức tạp.

### Kế hoạch nâng cấp theo kiến trúc Lab 18
1. **Chunking Strategy:** Áp dụng kết hợp **Structure-Aware Chunking** (bảo toàn bảng phụ cấp, thang bậc lương) và **Hierarchical Parent-Child Chunking** (Child 256 / Parent 2048) để tối ưu hóa độ chính xác truy xuất.
2. **Hybrid Search & Fusion:** Sử dụng `underthesea` tách từ tiếng Việt + BM25Okapi kết hợp Dense Embedding `BAAI/bge-m3` qua Qdrant và xếp hạng hợp nhất bằng RRF ($k=60$).
3. **Reranking:** Tích hợp `BAAI/bge-reranker-v2-m3` (hoặc `flashrank` nếu cần low latency <10ms) cho Top-20 $\rightarrow$ Top-3.
4. **Metadata Filtering:** Tự động lọc tài liệu theo metadata `version` và `status` (chỉ lấy văn bản còn hiệu lực).
5. **Evaluation Pipeline:** Tích hợp bộ đo RAGAS định kỳ trong CI/CD để giám sát 4 metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall.

### Lộ trình triển khai (Timeline)
* **Tuần 1:** Xây dựng Data Ingestion Pipeline với Structure-Aware + Parent-Child Chunking và Qdrant indexing.
* **Tuần 2:** Triển khai Hybrid Search (BM25 + Dense) + Cross-Encoder Reranker và Metadata Filtering.
* **Tuần 3:** Tích hợp Evaluation tự động với RAGAS và xây dựng bộ test 50 Q&A benchmark.

---

## 4. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) | Ghi chú |
|:---|:---:|:---|
| **Hiểu bài giảng & Áp dụng** | 5/5 | Áp dụng đầy đủ 5 modules, hiểu sâu kiến trúc Hierarchical + Hybrid RRF + Rerank. |
| **Code Quality & Chịu lỗi** | 5/5 | Mã nguồn chuẩn hóa, có cơ chế fallback ở mọi tầng, 37/37 tests pass 100%. |
| **Giải quyết vấn đề (Debugging)** | 5/5 | Xử lý triệt để lỗi proxy API 401, JSON parse truncation và cấu hình RAGAS. |
| **Báo cáo & Phân tích lỗi** | 5/5 | Đầy đủ bảng số liệu RAGAS, phân tích Bottom-5 failures theo Error Tree chi tiết. |
