"""
Task 4 — Chunking, embedding và indexing.

Hướng dẫn:
    1. Đọc toàn bộ Markdown trong data/standardized/.
    2. Chia văn bản bằng strategy đã chọn.
    3. Embed chunks bằng một provider duy nhất.
    4. Upsert vào ChromaDB với cosine distance.

Mỗi document/chunk phải theo docs/MODULE_CONTRACTS.md. ID cần ổn định để
chạy lại pipeline không tạo dữ liệu trùng. Task 5 phải dùng chung embed_texts().
"""

import os
import re
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Văn bản pháp luật tiếng Việt chia theo Điều/Khoản; 800 ký tự (~150-200 từ)
# thường chứa trọn một khoản, overlap 120 ký tự giữ ngữ cảnh khi cắt giữa khoản.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
CHUNKING_METHOD = "recursive"
SEPARATORS = ["\n## ", "\n### ", "\nĐiều ", "\n\n", "\n", ". ", " ", ""]

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or "BAAI/bge-m3"
EMBEDDING_DIM = 1024
EMBED_BATCH_SIZE = 16

COLLECTION_NAME = "rag_documents"

_HEADER_TITLE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_HEADER_SOURCE = re.compile(r"^\*\*Source:\*\*\s*(\S+)", re.MULTILINE)


@lru_cache(maxsize=1)
def _sentence_transformer():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts bằng provider trong EMBEDDING_PROVIDER (vector đã normalize)."""
    if not texts:
        return []
    if EMBEDDING_PROVIDER == "sentence_transformers":
        vectors = _sentence_transformer().encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > EMBED_BATCH_SIZE,
        )
        return vectors.tolist()
    if EMBEDDING_PROVIDER == "openai":
        from openai import OpenAI

        response = OpenAI().embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [item.embedding for item in response.data]
    if EMBEDDING_PROVIDER == "gemini":
        from google import genai

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.embed_content(model=EMBEDDING_MODEL, contents=texts)
        return [item.values for item in response.embeddings]
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}")


def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def load_documents() -> list[dict]:
    """Đọc Markdown và trả về danh sách Document."""
    documents = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        title_match = _HEADER_TITLE.search(content)
        source_match = _HEADER_SOURCE.search(content)
        url = source_match.group(1) if source_match else None
        documents.append({
            "id": path.relative_to(STANDARDIZED_DIR).as_posix(),
            "content": content,
            "metadata": {
                "source": path.name,
                "title": title_match.group(1).strip() if title_match else path.stem,
                "doc_type": "legal" if "legal" in path.parts else "news",
                "url": url if url and url.startswith("http") else None,
            },
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )
    chunks = []
    for document in documents:
        texts = [text.strip() for text in splitter.split_text(document["content"])]
        for index, text in enumerate(t for t in texts if t):
            chunks.append({
                "id": f"{document['id']}::chunk-{index}",
                "content": text,
                "metadata": {**document["metadata"], "chunk_index": index},
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk (trả bản sao, không sửa input)."""
    vectors = embed_texts([chunk["content"] for chunk in chunks])
    return [{**chunk, "embedding": vector} for chunk, vector in zip(chunks, vectors)]


def _to_chroma_metadata(metadata: dict) -> dict:
    # Chroma không lưu None; Task 5 đổi "" về None khi đọc ra.
    return {key: ("" if value is None else value) for key, value in metadata.items()}


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB và xoá chunk cũ không còn trong corpus."""
    collection = get_collection()
    new_ids = {chunk["id"] for chunk in chunks}
    stale_ids = [item for item in collection.get(include=[])["ids"] if item not in new_ids]
    if stale_ids:
        collection.delete(ids=stale_ids)

    batch = 500
    for start in range(0, len(chunks), batch):
        part = chunks[start:start + batch]
        collection.upsert(
            ids=[chunk["id"] for chunk in part],
            documents=[chunk["content"] for chunk in part],
            embeddings=[chunk["embedding"] for chunk in part],
            metadatas=[_to_chroma_metadata(chunk["metadata"]) for chunk in part],
        )


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    documents = load_documents()
    chunks = chunk_documents(documents)
    embedded_chunks = embed_chunks(chunks)
    index_to_vectorstore(embedded_chunks)
    print(f"Indexed {len(embedded_chunks)} chunks from {len(documents)} documents")


if __name__ == "__main__":
    run_pipeline()
