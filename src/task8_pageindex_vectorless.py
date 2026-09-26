"""
Task 8 — PageIndex vectorless fallback.

Hướng dẫn:
    1. Đọc PAGEINDEX_API_KEY từ .env.
    2. Upload tài liệu ở định dạng PageIndex hỗ trợ.
    3. Cache document IDs để không upload lại.
    4. Parse kết quả thành SearchResult có method pageindex.

PageIndex là dịch vụ ngoài: cần timeout và xử lý lỗi để pipeline không crash.

PageIndex chỉ nhận PDF, nên chỉ upload các PDF pháp luật gốc trong
data/landing/legal (đã đăng ký trong sources.json). Khi thiếu API key hoặc
dịch vụ lỗi/timeout, pageindex_search trả [] để Task 9 dùng lại hybrid results.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
LEGAL_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
DOC_ID_CACHE = Path(__file__).parent.parent / "data" / "pageindex_docs.json"

REQUEST_TIMEOUT_S = 30
POLL_INTERVAL_S = 2


def _client():
    from pageindex import PageIndexClient

    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def _load_manifest() -> dict:
    path = LEGAL_DIR / "sources.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_doc_ids() -> dict:
    """Mapping source PDF -> PageIndex doc_id đã upload."""
    if DOC_ID_CACHE.exists():
        return json.loads(DOC_ID_CACHE.read_text(encoding="utf-8"))
    return {}


def upload_documents() -> None:
    """Upload tài liệu và lưu document IDs để tái sử dụng."""
    if not PAGEINDEX_API_KEY:
        print("PAGEINDEX_API_KEY chưa được cấu hình; bỏ qua upload.")
        return
    client = _client()
    doc_ids = load_doc_ids()
    for name in _load_manifest():
        path = LEGAL_DIR / name
        if name in doc_ids or not path.exists() or path.suffix.lower() != ".pdf":
            continue
        response = client.submit_document(str(path))
        doc_ids[name] = response["doc_id"]
        DOC_ID_CACHE.write_text(json.dumps(doc_ids, indent=2), encoding="utf-8")
        print(f"Uploaded {name} -> {response['doc_id']}")


def _parse_nodes(retrieval: dict) -> list[dict]:
    """Chuẩn hoá retrieved nodes; tên field được kiểm tra thay vì giả định."""
    nodes = retrieval.get("retrieved_nodes") or retrieval.get("nodes") or []
    parsed = []
    for node in nodes:
        contents = node.get("relevant_contents") or []
        texts = [
            part.get("relevant_content") or part.get("content") or ""
            for part in contents if isinstance(part, dict)
        ]
        text = "\n\n".join(t for t in texts if t) or node.get("text") or node.get("content") or ""
        pages = [part.get("page_index") for part in contents if isinstance(part, dict)]
        parsed.append({
            "node_id": str(node.get("node_id") or node.get("id") or len(parsed)),
            "title": node.get("title") or "",
            "content": text.strip(),
            "page": next((p for p in pages if p is not None), None),
        })
    return [node for node in parsed if node["content"]]


def _query_document(client, doc_id: str, query: str, deadline: float) -> dict:
    retrieval_id = client.submit_query(doc_id, query)["retrieval_id"]
    while time.monotonic() < deadline:
        result = client.get_retrieval(retrieval_id)
        if result.get("status") == "completed":
            return result
        if result.get("status") == "failed":
            return {}
        time.sleep(POLL_INTERVAL_S)
    return {}


def _search(query: str, top_k: int) -> list[dict]:
    client = _client()
    manifest = _load_manifest()
    deadline = time.monotonic() + REQUEST_TIMEOUT_S
    results = []
    for source, doc_id in load_doc_ids().items():
        if time.monotonic() >= deadline:
            break
        info = manifest.get(source, {})
        for node in _parse_nodes(_query_document(client, doc_id, query, deadline)):
            results.append({
                "id": f"pageindex::{doc_id}::{node['node_id']}",
                "content": node["content"],
                "metadata": {
                    "source": Path(source).with_suffix(".md").name,
                    "title": info.get("title") or node["title"] or source,
                    "doc_type": "legal",
                    "url": info.get("url"),
                    "chunk_index": int(node["page"] or 0),
                },
                "retrieval_method": "pageindex",
            })
    # API không trả score: gán score giảm dần theo rank.
    unique = list({item["id"]: item for item in results}.values())[:top_k]
    for rank, item in enumerate(unique, 1):
        item["score"] = 1.0 / rank
    return unique


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult; lỗi/timeout/thiếu key -> []."""
    if not PAGEINDEX_API_KEY or not load_doc_ids() or top_k <= 0:
        return []
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_search, query, top_k)
    try:
        return future.result(timeout=REQUEST_TIMEOUT_S + 5)
    except (FutureTimeout, Exception) as error:  # dịch vụ ngoài không được làm crash UI
        print(f"PageIndex fallback unavailable: {error!r}")
        return []
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    upload_documents()
