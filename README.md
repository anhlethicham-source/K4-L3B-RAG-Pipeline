# Day 8 — RAG Pipeline

## Đề tài của nhóm: Trợ lý tra cứu thuế hộ kinh doanh 2026

Chatbot trả lời câu hỏi về chính sách thuế, kê khai và kế toán cho hộ kinh doanh, cá nhân kinh doanh sau khi bỏ thuế khoán từ 1/1/2026.

| Thành phần | Lựa chọn |
| --- | --- |
| Corpus | 4 văn bản pháp luật (TT 18/2026/TT-BTC, TT 152/2025/TT-BTC, NĐ 73/2016/NĐ-CP, QĐ 180/TTg) — PDF scan, OCR bằng EasyOCR; 5 bài viết (VnExpress ×2, Chinhphu.vn, MISA, Einvoice) — crawl + làm sạch boilerplate |
| Chunking | `RecursiveCharacterTextSplitter`, 800 ký tự, overlap 120, ưu tiên tách theo heading / `Điều` |
| Embedding / vector DB | `BAAI/bge-m3` (1024 chiều, local) · ChromaDB cosine |
| Retrieval | Dense + BM25 (IDF kiểu Lucene) → RRF (k=60) → PageIndex fallback khi best dense cosine < `SCORE_THRESHOLD` |
| Generation | Gemini / OpenAI / Claude theo `LLM_PROVIDER`, citation `[n]` map về `sources[n-1]`, safe refusal |
| Evaluation | `group_project/evaluation/run_evaluation.py` — 18 câu golden, 4 metric (LLM-judge theo định nghĩa RAGAS) + hit@k, A/B dense vs hybrid |

Chạy nhanh (sau khi điền `.env`):

```bash
python -m src.task3_convert_markdown        # OCR PDF scan lần đầu mất ~1 giờ trên CPU
python -m src.task4_chunking_indexing
python -m group_project.evaluation.calibrate_threshold
python -m group_project.evaluation.run_evaluation
streamlit run app.py
```

## Mục tiêu

Mỗi nhóm xây dựng một chatbot RAG trả lời câu hỏi từ bộ tài liệu do nhóm thu thập. Sản phẩm phải có hybrid retrieval, citation, giao diện chat và báo cáo đánh giá.

Nhóm tự chọn bài toán và thu thập dữ liệu phù hợp; repo không cung cấp dữ liệu mẫu.

## Sản phẩm phải nộp

- Repository nhóm chạy được.
- Tối thiểu 3 tài liệu chính sách và 5 bài viết/page do nhóm tự thu thập.
- Pipeline: convert → chunk → index → dense + BM25 → RRF → fallback → generation có citation.
- Chatbot Streamlit hiển thị câu trả lời và nguồn đã dùng.
- Golden dataset tối thiểu 15 câu; đánh giá 4 metric và so sánh A/B.
- `group_project/evaluation/RESULT.md`.
- Mỗi thành viên nộp báo cáo cá nhân theo template trong `group_project/ịndividual/INDIVIDUAL_REPORT.md`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
python -m playwright install chromium
cp .env.example .env
```

Điền API key cần dùng trong `.env`; không commit file này.

```bash
# 1. Thu thập và chuẩn hoá
python -m src.task1_collect_legal_docs
python -m src.task2_crawl_news
python -m src.task3_convert_markdown

# 2. Index và kiểm tra contract
python -m src.task4_chunking_indexing
pytest -q

# 3. Chạy sản phẩm
streamlit run app.py
```

## Lộ trình 3 giờ

| Mốc                  | Thời gian | Kết quả cần có                           |
| -------------------- | --------: | ---------------------------------------- |
| 0. Setup             |   10 phút | Môi trường và `.env` sẵn sàng            |
| 1. Data              |   25 phút | ≥3 legal, ≥5 news, Markdown đã chuẩn hoá |
| 2. Index & search    |   30 phút | ChromaDB, dense search và BM25 chạy được |
| 3. Fusion & fallback |   25 phút | RRF và fallback tuân thủ contract        |
| 4. Generation & UI   |   30 phút | Chatbot trả lời có citation              |
| 5. Evaluation        |   30 phút | 15+ Q&A, 4 metric, A/B comparison        |
| 6. Demo & handoff    |   30 phút | Test, report, demo và push repository    |

## Lưu ý quy tắc để có code quality tốt:

- Dense và BM25 nên cùng trả về `SearchResult` theo một schema.
- RRF chỉ nên dùng để gộp thứ hạng và chỉ chạy một lần.
- Fallback dùng cosine score gốc của dense retrieval.
- Threshold phải được hiệu chỉnh trên query in domain và out of domain, không có một con số đúng cho mọi corpus.

## Tài liệu

- [Module contracts](docs/MODULE_CONTRACTS.md): schema, interface và invariant mà code/test nên tuân theo.
- [Step-by-step guide](docs/STEP_BY_STEP.md): thứ tự triển khai và tiêu chí hoàn thành từng bước.
- [Grading rubric](docs/GRADING_RUBRIC.md): Rubric thang điểm.
- [Individual report](group_project/ịndividual/INDIVIDUAL_REPORT.md): template báo cáo cá nhân.
- [Suggested topics](docs/SUGGESTED_TOPICS.md): danh sách chủ đề tham khảo, không bắt buộc.

## Kiểm tra

```bash
# Contract tests
pytest tests/test_contracts.py -q

# Acceptance tests
pytest tests/test_acceptance.py -q

# Toàn bộ
pytest -q
```
