"""
Task 7 — Reciprocal Rank Fusion.

RRF gộp nhiều bảng xếp hạng mà không cộng trực tiếp cosine score với BM25
score. Công thức: RRF(d) = sum(1 / (k + rank)), rank bắt đầu từ 1.

Lưu ý: RRF score chỉ phản ánh thứ hạng, không dùng để quyết định fallback.

-> Dùng Jina hoặc self host hoặc bất cứ công cụ nào bạn quen
"""


def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """Fuse nhiều ranked lists và trả hybrid SearchResult."""
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for ranked_list in ranked_lists:
        seen: set[str] = set()
        for rank, item in enumerate(ranked_list, 1):
            item_id = item["id"]
            if item_id in seen:  # một list chỉ góp một lần cho mỗi ID
                continue
            seen.add(item_id)
            scores[item_id] = scores.get(item_id, 0.0) + 1 / (k + rank)
            items.setdefault(item_id, item)

    # Tie-break theo ID để thứ tự ổn định giữa các lần chạy.
    ranked_ids = sorted(scores, key=lambda item_id: (-scores[item_id], item_id))
    results = []
    for item_id in ranked_ids[:max(top_k, 0)]:
        result = dict(items[item_id])
        result["score"] = scores[item_id]
        result["retrieval_method"] = "hybrid"
        results.append(result)
    return results


if __name__ == "__main__":
    import sys

    from .task5_semantic_search import semantic_search
    from .task6_lexical_search import lexical_search

    query = " ".join(sys.argv[1:]) or "Hộ kinh doanh có doanh thu dưới 500 triệu có phải nộp thuế không?"
    fused = rerank_rrf([semantic_search(query, 10), lexical_search(query, 10)], top_k=5)
    for result in fused:
        print(f"{result['score']:.4f}  {result['id']}")
