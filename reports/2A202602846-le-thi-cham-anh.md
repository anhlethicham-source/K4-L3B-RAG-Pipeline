# Individual contribution report

## Thông tin

- Họ và tên: Lê Thị Châm Anh
- Mã học viên: 2A202602846
- Nhóm: Làm cá nhân (không có nhóm)
- Repository/branch: https://github.com/anhlethicham-source/K4-L3B-RAG-Pipeline · `main`

## Phần việc đã thực hiện

Làm cá nhân nên toàn bộ các module do tôi phụ trách. Có sử dụng Claude Code (AI coding assistant) hỗ trợ viết code và phân tích.

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| Chọn đề tài & thu thập dữ liệu (Task 1–2) | Đề tài thuế hộ kinh doanh 2026; tải 4 văn bản pháp luật, crawl 5 bài viết (VnExpress, Chinhphu.vn, MISA, Einvoice) kèm manifest nguồn | `src/task1_collect_legal_docs.py`, `src/task2_crawl_news.py`, `data/landing/` · `2b6cd91` | Done |
| Chuẩn hoá Markdown (Task 3) | OCR PDF scan bằng EasyOCR, chuẩn hoá lỗi OCR mã mẫu biểu, làm sạch boilerplate và ký tự Cyrillic trong bài crawl | `src/task3_convert_markdown.py`, `data/standardized/` · `2b6cd91` | Done |
| Chunking, embedding, index (Task 4) | Recursive 800/120, bge-m3, ChromaDB cosine, upsert idempotent + xoá chunk cũ | `src/task4_chunking_indexing.py` · `2b6cd91` | Done |
| Dense, BM25, RRF (Task 5–7) | BM25 với IDF kiểu Lucene; RRF k=60 dedupe theo ID | `src/task5_*.py`, `src/task6_*.py`, `src/task7_*.py` · `2b6cd91` | Done |
| Fallback & pipeline (Task 8–9) | PageIndex client có timeout, lỗi → `[]`; calibrate threshold 0.54 | `src/task8_*.py`, `src/task9_*.py`, `group_project/evaluation/calibrate_threshold.py` · `2b6cd91` | Partial (PageIndex chưa có API key nên chưa chạy thật) |
| Generation & UI (Task 10) | Citation `[n]` map về `sources`, safe refusal, dispatch OpenAI/OpenRouter/Gemini/Claude; Streamlit hiển thị nguồn, score, method | `src/task10_generation.py`, `app.py` · `2b6cd91` | Done |
| Evaluation | 18 câu golden, 4 metric LLM-judge + evidence hit/MRR, A/B dense vs hybrid, phân tích lỗi | `group_project/evaluation/` · `2b6cd91` | Done |

## Quyết định kỹ thuật quan trọng

1. **Quyết định:** OCR các PDF pháp luật bằng EasyOCR thay vì chỉ dùng MarkItDown, và thêm bước sửa mã mẫu biểu (`OI/CNKD` → `01/CNKD`).
   **Lý do/evidence:** Cả 4 PDF là bản scan, MarkItDown trả về 0–149 ký tự cho 2–80 trang → file Markdown rỗng. Sau OCR, corpus pháp luật có ~315KB text; OCR đọc nhầm số 0/1 thành O/I/L làm BM25 không khớp truy vấn theo mã mẫu.
   **Trade-off:** OCR trên CPU mất ~1 giờ cho 140 trang và vẫn còn lỗi dấu, trộn cột ở văn bản in hai cột (QĐ 180/TTg).

2. **Quyết định:** Fallback threshold = 0.54 trên dense cosine gốc, không dùng RRF score.
   **Lý do/evidence:** 18 query in-domain có best cosine 0.638–0.809, 8 query ngoài domain 0.289–0.437; hai phân bố tách rời nên lấy trung điểm.
   **Trade-off:** Chỉ calibrate trên 26 query; câu hỏi gần domain (vd thuế doanh nghiệp) chưa được kiểm tra.

## Kiểm thử và kết quả

- Test hoặc query tôi đã dùng: `pytest -q` (20/20 pass); 18 câu golden cho A/B; query ngoài domain "Công thức nấu phở bò gồm gì?" → safe refusal.
- Kết quả trước/sau nếu có: Dense-only average 0.955 vs hybrid + RRF 0.945; context recall 0.926 vs 0.870; evidence hit@5 0.944 vs 0.889.
- Lỗi đã phát hiện và cách xử lý: BM25Okapi cho IDF = 0 khi corpus nhỏ → đổi sang IDF kiểu Lucene; hybrid làm mất evidence ở q10 do RRF không trọng số (ghi vào RESULT.md, đề xuất weighted RRF/reranker).

## Điều còn hạn chế

- Một hạn chế cụ thể của phần tôi làm: Judge dùng cùng model với generator nên chấm dễ (faithfulness 1.0 mọi câu, kể cả câu trả lời lạc đề ở q10); corpus còn 2 văn bản lạc đề (NĐ 73/2016, QĐ 180/TTg).
- Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện: Thay RRF bằng weighted RRF hoặc cross-encoder reranker và chunk văn bản pháp luật theo từng Điều, rồi chạy lại A/B.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 26/09/2026
- Tên thành viên: Lê Thị Châm Anh
