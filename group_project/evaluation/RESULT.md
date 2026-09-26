# RAG evaluation results

## Run information

| Field                              | Value |
| ---------------------------------- | ----- |
| Evaluation date                    | 2026-09-26 |
| Framework and version              | LLM-as-judge tự viết theo định nghĩa metric của RAGAS (`group_project/evaluation/run_evaluation.py`); retrieval metrics (evidence hit / MRR) tính bằng token overlap, không dùng LLM |
| Evaluator model                    | `openai/gpt-4o-mini` qua OpenRouter, temperature 0.3 |
| Generator model                    | `openai/gpt-4o-mini` qua OpenRouter, temperature 0.3, top_p 0.9 |
| Embedding model                    | `BAAI/bge-m3` (1024 chiều, sentence-transformers, local CPU) |
| Corpus version/commit              | Base commit `a23df34` + corpus hiện tại: 4 văn bản pháp luật (OCR) + 5 bài viết → 9 documents, 596 chunks (800/120 ký tự) |
| Golden dataset size                | 18 câu (`golden_dataset.json`) |
| `top_k`                            | 5 (ứng viên mỗi retriever: 10) |
| Fallback threshold and calibration | 0.54 trên best dense cosine. 18 query in-domain: 0.638–0.809; 8 query ngoài domain: 0.289–0.437; lấy trung điểm (`calibrate_threshold.py`). PageIndex chưa có API key nên fallback trả `[]` và pipeline dùng hybrid results; câu ngoài domain được LLM safe-refuse |

## Configurations

- **Config A — dense-only:** `retrieve(query, top_k=5, use_reranking=False)` — top 5 theo cosine của bge-m3 trên ChromaDB.
- **Config B — hybrid + RRF:** `retrieve(query, top_k=5, use_reranking=True)` — top 10 dense + top 10 BM25 (IDF kiểu Lucene) → RRF (k=60) một lần → top 5.

Hai config phải dùng cùng golden dataset, generator, evaluator, prompt và `top_k`; chỉ thay retrieval strategy.

## Overall scores

| Metric            | Config A | Config B | Delta B−A |
| ----------------- | -------: | -------: | --------: |
| Faithfulness      |    1.000 |    1.000 |    +0.000 |
| Answer relevance  |    1.000 |    0.989 |    −0.011 |
| Context recall    |    0.926 |    0.870 |    −0.056 |
| Context precision |    0.895 |    0.919 |    +0.024 |
| **Average**       |    0.955 |    0.945 |    −0.011 |

Retrieval metrics không dùng LLM (cùng lần chạy):

| Metric                                   | Config A | Config B | Delta |
| ---------------------------------------- | -------: | -------: | ----: |
| Source hit@5 (đúng tài liệu)              |    1.000 |    1.000 | +0.000 |
| Evidence hit@5 (chunk chứa ≥60% expected_context) | 0.944 | 0.889 | −0.056 |
| Evidence MRR                             |    0.875 |    0.817 | −0.058 |
| Retrieval latency (s/query, CPU)         |    0.147 |    0.147 | +0.000 |

## A/B comparison

- Cấu hình tốt hơn: **Config A (dense-only)** trên corpus hiện tại, chênh lệch nhỏ (−0.011 average, −0.056 context recall cho B).
- Evidence: B thắng rõ ở q01 (evidence rank 4 → 1) và q17 (2 → 1) — các câu có con số/mốc cụ thể mà BM25 khớp tốt; context precision của B cao hơn (+0.024). Nhưng B làm mất evidence ở q10 (rank 1 → không có trong top 5) và đẩy q08 từ rank 1 xuống 5. Ở q10, chunk đúng đứng #1 dense nhưng không nằm trong top 10 BM25, nên RRF (1/61 ≈ 0.0164) xếp nó dưới các chunk xuất hiện ở cả hai list (≈ 0.032). Kết quả là câu trả lời B-q10 lạc đề.
- Trade-off về latency/cost: BM25 in-memory trên 596 chunks gần như không tốn thêm thời gian (0.147 s/query cả hai, chủ yếu là embedding query trên CPU); không phát sinh API cost. Hybrid bền hơn với query chứa mã văn bản/mẫu biểu (ví dụ `01/TB-ĐĐKD`) nhờ bước chuẩn hoá OCR, nhưng với 18 câu hiện tại lợi ích đó chưa thể hiện thành điểm.

Lưu ý độ tin cậy: faithfulness = 1.000 ở mọi mẫu và answer relevance B-q10 = 0.80 dù câu trả lời lạc đề cho thấy judge (cùng model với generator) chấm dễ. Các kết luận trên dựa chủ yếu vào context recall và evidence hit/MRR — hai metric đối chiếu với ground truth.

## Worst performers

|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |
| --: | -------- | ------ | -----------: | --------: | -----: | --------: | ------------- | ---------- |
|   1 | q10 — Chi phí được trừ khi tính thuế theo lợi nhuận cần điều kiện gì? | B | 1.00 | 0.80 | 0.00 | 0.83 | retrieval | RRF không trọng số: chunk đúng là #1 dense nhưng vắng mặt ở BM25 top 10 nên bị các chunk "có mặt ở cả hai list" vượt qua. BM25 bị kéo lệch bởi NĐ 73/2016 (bảo hiểm) có nhiều từ "chi phí được trừ". Answer lạc sang thuế suất 15%; judge không bắt được lỗi |
|   2 | q13 — Thông tư 18/2026/TT-BTC quy định về những nội dung gì? | A & B | 1.00 | 1.00 | 0.00 | 1.00 | data (chunking) | Chunk-0 (trích yếu tiêu đề) khớp câu hỏi mạnh nhất, còn Điều 1 "Phạm vi điều chỉnh" nằm ở chunk khác không vào top 5. Answer chỉ nêu chung chung "hồ sơ, thủ tục quản lý thuế", thiếu danh sách thủ tục |
|   3 | q12 — Người bán trên sàn TMĐT và Facebook/TikTok nộp thuế thế nào? | A & B | 1.00 | 1.00 | 0.67 | 1.00 | retrieval/generation | Ý "số thuế sàn đã khấu trừ được trừ khi quyết toán TNCN cuối năm" nằm ở article_02 nhưng chunk đó không vào top 5; model chỉ tổng hợp từ article_01 |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | ------ | ------------------------------ | --------------- | ------------- |
|        1 | Dùng weighted RRF (trọng số dense > BM25, ví dụ 1.0/0.5) hoặc cross-encoder reranker (bge-reranker-v2-m3) trên union top 20 | q10, q08: chunk #1 dense bị RRF đẩy xuống/ra khỏi top 5 | Giữ lợi ích của BM25 ở q01/q17 mà không mất q08/q10; evidence hit B ≥ 0.944 | Chạy lại `run_evaluation.py`, so sánh evidence hit/MRR và context recall với bảng trên |
|        2 | Chunk theo cấu trúc văn bản pháp luật (tách theo `Điều`, gắn tiêu đề văn bản + số Điều vào mỗi chunk) | q13: Điều 1 tách khỏi chunk tiêu đề; OCR gộp nhiều Điều trên một dòng nên splitter không tách được | Câu hỏi "văn bản X quy định gì / Điều Y" lấy đúng điều khoản | Thêm 3–5 câu golden dạng "Điều N của văn bản X", đo evidence hit |
|        3 | Làm sạch corpus: thay NĐ 73/2016 (kinh doanh bảo hiểm) và QĐ 180/TTg (1992) bằng NĐ 68/2026/NĐ-CP; dùng judge khác model generator (hoặc RAGAS với model mạnh hơn) | q10: BM25 top 1 là chunk NĐ 73/2016; faithfulness 1.00 ở mọi mẫu kể cả answer lạc đề | Giảm nhiễu lexical; metric generation phân biệt được câu tốt/xấu | So sánh phân bố điểm judge mới với đánh giá tay 5 câu |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| ---------- | -------- | -----------: | -----------------: | ---------- |
| Không thực hiện bonus (HyDE/reranker/memory) trong lần chạy này. UI có tô đậm citation `[n]` và đánh dấu nguồn được trích dẫn (✅) | — | — | — | Chưa đủ điều kiện tính bonus vì chưa có phép đo so sánh |
