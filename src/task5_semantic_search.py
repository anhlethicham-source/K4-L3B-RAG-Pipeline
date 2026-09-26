"""
Task 5 — Semantic search.

Embed query bằng chính hàm của Task 4, query ChromaDB và đổi cosine distance
thành similarity. Output phải theo SearchResult, sort giảm dần và không quá top_k.
"""

from .task4_chunking_indexing import embed_texts, get_collection


def _from_chroma_metadata(metadata: dict) -> dict:
    metadata = dict(metadata)
    if not metadata.get("url"):
        metadata["url"] = None
    return metadata


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về dense SearchResult theo score giảm dần."""
    if top_k <= 0 or not query.strip():
        return []
    query_vector = embed_texts([query])[0]
    response = get_collection().query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    results = {}
    for item_id, content, metadata, distance in zip(
        response["ids"][0],
        response["documents"][0],
        response["metadatas"][0],
        response["distances"][0],
    ):
        results[item_id] = {
            "id": item_id,
            "content": content,
            # cosine distance = 1 - cosine similarity; giữ score gốc cho fallback.
            "score": max(0.0, 1.0 - float(distance)),
            "metadata": _from_chroma_metadata(metadata),
            "retrieval_method": "dense",
        }
    return sorted(results.values(), key=lambda item: item["score"], reverse=True)[:top_k]


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "Hộ kinh doanh nộp thuế như thế nào từ năm 2026?"
    for result in semantic_search(query, top_k=3):
        print(f"{result['score']:.3f}  {result['id']}\n    {result['content'][:150]!r}")
