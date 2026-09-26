"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5. BM25 phù hợp với từ khóa chính xác, mã tài
liệu và tên riêng. Output phải theo SearchResult và sort score giảm dần.
"""

import re


# Để trống thì lazy-load toàn bộ chunks từ Chroma collection của Task 4.
CORPUS: list[dict] = []

_TOKEN = re.compile(r"\w+", re.UNICODE)
_cache: dict = {"key": None, "bm25": None}


def tokenize(text: str) -> list[str]:
    """Tách âm tiết tiếng Việt (giữ dấu), bỏ dấu câu; giữ mã văn bản như 152/2025."""
    return _TOKEN.findall(text.lower())


def build_bm25_index(corpus: list[dict]):
    """Tạo BM25 index từ cùng corpus chunks của Task 4."""
    import math

    from rank_bm25 import BM25Okapi

    class LuceneIdfBM25(BM25Okapi):
        # IDF của BM25Okapi = 0 hoặc âm khi term xuất hiện ở >= nửa số chunk
        # (tiếng Việt có nhiều âm tiết phổ biến như "thuế", "kinh", "doanh").
        # Dùng IDF kiểu Lucene: log(1 + (N - n + 0.5) / (n + 0.5)) > 0.
        def _calc_idf(self, nd):
            self.idf = {
                word: math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
                for word, freq in nd.items()
            }

    return LuceneIdfBM25([tokenize(item["content"]) for item in corpus])


def load_corpus_from_vectorstore() -> list[dict]:
    from .task4_chunking_indexing import get_collection
    from .task5_semantic_search import _from_chroma_metadata

    data = get_collection().get(include=["documents", "metadatas"])
    return [
        {"id": item_id, "content": content, "metadata": _from_chroma_metadata(metadata)}
        for item_id, content, metadata in zip(data["ids"], data["documents"], data["metadatas"])
    ]


def _get_index() -> tuple[list[dict], object]:
    global CORPUS
    if not CORPUS:
        CORPUS = load_corpus_from_vectorstore()
    key = (id(CORPUS), len(CORPUS))
    if _cache["key"] != key:
        _cache["bm25"] = build_bm25_index(CORPUS) if CORPUS else None
        _cache["key"] = key
    return CORPUS, _cache["bm25"]


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    tokens = tokenize(query)
    if top_k <= 0 or not tokens:
        return []
    corpus, bm25 = _get_index()
    if bm25 is None:
        return []

    scores = bm25.get_scores(tokens)
    ranked = sorted(range(len(corpus)), key=lambda i: scores[i], reverse=True)
    results = []
    for index in ranked[:top_k]:
        if scores[index] <= 0:
            break
        item = corpus[index]
        results.append({
            "id": item["id"],
            "content": item["content"],
            "score": float(scores[index]),
            "metadata": item["metadata"],
            "retrieval_method": "bm25",
        })
    return results


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "Thông tư 152/2025/TT-BTC"
    for result in lexical_search(query, top_k=3):
        print(f"{result['score']:.3f}  {result['id']}\n    {result['content'][:150]!r}")
