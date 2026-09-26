"""
Task 9 — Retrieval pipeline hoàn chỉnh.

Luồng xử lý:
    1. Chạy semantic_search và lexical_search.
    2. Fuse hai danh sách bằng RRF đúng một lần.
    3. Lấy best cosine score gốc từ dense results.
    4. Nếu score dưới threshold, thử PageIndex fallback.
    5. Nếu fallback lỗi, trả hybrid results thay vì crash.

Không so sánh threshold với RRF score vì hai thang đo khác nhau.
"""

import os

from dotenv import load_dotenv

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


load_dotenv()

# Calibrate bằng group_project/evaluation/calibrate_threshold.py (bge-m3):
# 18 query in-domain có best dense cosine 0.638–0.809, 8 query ngoài domain
# 0.289–0.437 -> lấy trung điểm 0.54.
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD") or 0.54)
DEFAULT_TOP_K = 5
CANDIDATE_MULTIPLIER = 2


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Trả về hybrid (hoặc dense khi use_reranking=False) hoặc pageindex SearchResult."""
    candidates = top_k * CANDIDATE_MULTIPLIER
    dense = semantic_search(query, top_k=candidates)
    if use_reranking:
        sparse = lexical_search(query, top_k=candidates)
        ranked = rerank_rrf([dense, sparse], top_k=top_k)
    else:
        ranked = dense[:top_k]

    # Fallback dựa trên cosine score gốc của dense, không dùng RRF score.
    best_dense_score = dense[0]["score"] if dense else 0.0
    if best_dense_score < score_threshold:
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                return fallback[:top_k]
        except Exception:
            pass  # provider lỗi -> dùng lại kết quả hybrid, Task 10 tự safe refusal
    return ranked[:top_k]


def best_dense_score(query: str) -> float:
    """Tiện ích cho UI/calibration: cosine cao nhất của dense search."""
    dense = semantic_search(query, top_k=1)
    return dense[0]["score"] if dense else 0.0


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "Hộ kinh doanh kê khai thuế như thế nào từ 2026?"
    for result in retrieve(query, top_k=3):
        print(f"{result['score']:.4f}  [{result['retrieval_method']}]  {result['id']}")
