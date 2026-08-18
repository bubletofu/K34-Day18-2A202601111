# Failure Analysis — Lab 18: Production RAG

**Nhóm:** Nhóm 2A — K34  
**Thành viên:** Phương (M1 Chunking & Pipeline) · Thành viên 2 (M2 Search) · Thành viên 3 (M3 Rerank) · Thành viên 4 (M4 Eval & M5 Enrichment)

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|:-------------:|:----------:|:--:|
| **Faithfulness** | 0.7333 | **0.7817** | **+0.0484** |
| **Answer Relevancy** | 0.4504 | **0.4699** | **+0.0195** |
| **Context Precision** | 0.9000 | **0.8750** | -0.0250 |
| **Context Recall** | 0.7708 | **0.8667** | **+0.0959** |

---

## Bottom-5 Failures

### #1
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Theo chính sách v2024: 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** Trả về thông tin ngày phép nhưng thiếu hoặc không đầy đủ thông tin khung bậc lương Senior do trích xuất context bị nhiễu.
- **Worst metric:** `context_precision` (0.0000)
- **Error Tree:** Output sai thông tin lương → Context thiếu file chính sách lương → Query là multi-hop kết hợp giữa nghỉ phép và bậc lương → Retrieval chỉ match tài liệu nghỉ phép với điểm cao hơn.
- **Root cause:** Câu hỏi Multi-hop yêu cầu liên kết thông tin từ 2 văn bản khác nhau (`nghi_phep_nam_v2024.md` và `chinh_sach_luong_thuong.md`), bộ retriever trả về các chunk nghỉ phép chiếm ưu thế làm loãng context lương.
- **Suggested fix:** Áp dụng Query Decomposition (tách câu hỏi thành 2 sub-queries: "ngày phép cho thâm niên 9 năm" và "khung lương Senior") hoặc nâng Top-k retrieval và filter theo metadata category.

### #2
- **Question:** Thông tin lương thuộc cấp độ phân loại dữ liệu nào?
- **Expected:** Thông tin lương thuộc phân loại Dữ liệu Bảo mật / Tuyệt mật (Confidential / Level 3) theo quy chế an toàn dữ liệu doanh nghiệp.
- **Got:** Trả lời suy đoán chung chung về an toàn thông tin mà không viện dẫn đúng phân cấp dữ liệu trong tài liệu quy định.
- **Worst metric:** `faithfulness` (0.0000)
- **Error Tree:** Output không trung thực → Context được retrieve chứa nhiều quy tắc bảo mật chung nhưng thiếu bảng phân loại cụ thể → LLM tự suy diễn thông tin ngoài context.
- **Root cause:** Chunking cắt ngang bảng phân loại an toàn thông tin, khiến chunk chứa định nghĩa cấp độ dữ liệu bị phân mảnh và mất header phân cấp.
- **Suggested fix:** Sử dụng Structure-Aware Chunking để bảo toàn nguyên vẹn các bảng phân loại dữ liệu và thắt chặt System Prompt chống hallucination.

### #3
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Quy định tạm ứng yêu cầu hoàn ứng trong vòng 15 ngày. Quá hạn 20 ngày là vi phạm quy định, phạt lãi chậm hoàn ứng hoặc khấu trừ vào kỳ lương kế tiếp.
- **Got:** Trình bày dài dòng về quy trình đăng ký tạm ứng tiền mặt nhưng không trả lời trực tiếp mức phạt quá hạn.
- **Worst metric:** `answer_relevancy` (0.1865)
- **Error Tree:** Output không tập trung vào câu hỏi → Context có nhắc đến tạm ứng nhưng quy định xử phạt nằm ở điều khoản riêng → Query match từ khóa "tạm ứng 15 triệu" thay vì trọng tâm "bị phạt bao nhiêu".
- **Root cause:** Từ khóa "tạm ứng 15 triệu" chi phối BM25 và Dense Search hơn ngữ nghĩa câu hỏi về điều khoản chế tài phạt.
- **Suggested fix:** Cải thiện Prompt Generation để trả lời thẳng vào trọng tâm câu hỏi số tiền phạt, nếu không có quy định cụ thể thì phản hồi rõ ràng "Không tìm thấy mức phạt cụ thể".

### #4
- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** Mức lương Junior tối đa là 18 triệu VNĐ/tháng, lương thử việc tối thiểu bằng 85% lương chính thức = 15.3 triệu VNĐ/tháng.
- **Got:** Trả lời trực tiếp con số không chính xác hoặc không kèm diễn giải phép tính 85%.
- **Worst metric:** `faithfulness` (0.0000)
- **Error Tree:** Output sai số liệu → Context có 2 mốc số liệu riêng lẻ (bậc Junior max 18 triệu và tỷ lệ thử việc 85%) → LLM không thực hiện đúng phép tính số học (Numeric reasoning failure).
- **Root cause:** Câu hỏi dạng Numeric Reasoning đòi hỏi LLM phải đọc cả mức trần bậc lương và tỷ lệ % thử việc rồi tính toán số học, LLM thuần túy không kích hoạt Chain-of-Thought dễ đưa ra kết quả ảo.
- **Suggested fix:** Bổ sung hướng dẫn suy luận từng bước (Chain-of-Thought) trong Prompt sinh câu trả lời cho các câu hỏi chứa tính toán số liệu tài chính.

### #5
- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?
- **Expected:** Nhân viên cam kết làm việc tối thiểu 12 tháng sau khóa học; nghỉ sau 8 tháng là vi phạm cam kết, phải hoàn trả 100% kinh phí tài trợ là 25.000.000 VNĐ.
- **Got:** Tóm tắt chính sách tài trợ đào tạo nhưng trả lời chưa dứt khoát về số tiền bồi hoàn 100%.
- **Worst metric:** `answer_relevancy` (0.0085)
- **Error Tree:** Output lan man → Context chứa đầy đủ quy định bồi hoàn đào tạo → LLM sinh văn bản dạng tóm tắt chính sách thay vì câu trả lời dứt khoát cho câu hỏi bồi hoàn.
- **Root cause:** Prompt chưa đủ ép buộc LLM tập trung vào hành động kết luận số tiền.
- **Suggested fix:** Thiết kế template phản hồi: Đưa ra câu trả lời trực tiếp (Direct Answer) ngay câu đầu tiên, sau đó mới bổ sung căn cứ điều khoản.

---

## Case Study (cho presentation)

**Question chọn phân tích:**  
> *"Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?"*

**Error Tree walkthrough:**
1. **Output đúng?** → **KHÔNG**. Câu trả lời chỉ nêu được ngày phép mà thiếu khung lương Senior.
2. **Context đúng?** → **MỘT PHẦN**. Context chỉ có văn bản nghỉ phép `nghi_phep_nam_v2024.md`, thiếu văn bản `chinh_sach_luong_thuong.md`.
3. **Query rewrite / Decomposition OK?** → **CHƯA CÓ**. Câu hỏi gốc ghép 2 ý khác miền dữ liệu ("ngày phép" + "thang lương").
4. **Fix ở bước:**
   - **Bước tiền truy xuất (Pre-retrieval)**: Tách câu hỏi đa ý (Multi-hop Query Decomposition) thành 2 query con.
   - **Bước truy xuất (Retrieval)**: Mở rộng Top-k của BM25 và Dense Search để tăng độ bao phủ context liên tài liệu.

**Nếu có thêm 1 giờ, sẽ optimize:**
1. **Multi-query Expansion & Decomposition**: Tự động phân rã câu hỏi phức tạp thành nhiều câu hỏi con để retrieve đa nguồn tài liệu.
2. **Metadata Filtering theo phiên bản**: Thêm bộ lọc `version: current` để loại trừ hoàn toàn các tài liệu đã hết hiệu lực (như `v2023`, `v1.0`).
3. **Chain-of-Thought Prompting**: Thêm chỉ dẫn suy luận từng bước cho các câu hỏi tính toán lương thưởng, thâm niên và thời gian cam kết đào tạo.
