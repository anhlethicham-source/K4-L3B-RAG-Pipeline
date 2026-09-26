"""
Task 10 — Generation có citation.

Hướng dẫn:
    1. Retrieve top-k chunks.
    2. Reorder để giảm lost-in-the-middle.
    3. Format context kèm title và source.
    4. Gọi provider được chọn trong .env.
    5. Trả answer, sources và retrieval_source.

Nếu context không đủ hoặc provider lỗi, trả safe refusal; không bịa thông tin.

Citation: mỗi chunk được gắn số [n] theo thứ tự trong `sources` (score giảm dần)
trước khi reorder, nên [n] trong answer luôn trỏ về sources[n-1].
"""

import os
import re

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve


load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-haiku-4-5",
}
LLM_MODEL = os.getenv("LLM_MODEL") or DEFAULT_MODELS.get(LLM_PROVIDER, "")

REFUSAL = "Tôi không thể xác minh thông tin này từ nguồn hiện có."

SYSTEM_PROMPT = f"""Bạn là trợ lý tra cứu chính sách thuế cho hộ kinh doanh, cá nhân kinh doanh tại Việt Nam.
Quy tắc:
- Chỉ trả lời dựa trên các tài liệu trong phần Context. Không dùng kiến thức bên ngoài.
- Mỗi câu khẳng định phải kèm citation dạng [n], trong đó n là số của tài liệu trong Context (ví dụ [1], [2][3]).
- Chỉ dùng các số citation có trong Context.
- Nếu Context không chứa đủ thông tin để trả lời, chỉ trả lời đúng câu: "{REFUSAL}"
- Trả lời ngắn gọn, bằng tiếng Việt."""

_CITATION = re.compile(r"\[(\d+)\]")


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đưa chunks quan trọng về đầu và cuối context (không sửa list gốc)."""
    if len(chunks) <= 2:
        return list(chunks)
    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


def format_context(chunks: list[dict]) -> str:
    """Tạo context có số citation, title và source label."""
    parts = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk["metadata"]
        number = chunk.get("citation_id", index)
        url = f" | URL: {metadata['url']}" if metadata.get("url") else ""
        parts.append(
            f"[{number}] Title: {metadata['title']} | Source: {metadata['source']}{url}\n"
            f"{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi OpenAI, Gemini hoặc Anthropic theo cấu hình; trả text thuần."""
    if LLM_PROVIDER == "openai":
        from openai import OpenAI

        # OPENAI_BASE_URL cho phép dùng endpoint tương thích OpenAI (vd OpenRouter).
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
            timeout=60,
        )
        response = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content or ""
    if LLM_PROVIDER == "gemini":
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            ),
        )
        return response.text or ""
    if LLM_PROVIDER == "anthropic":
        import anthropic

        response = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=60).messages.create(
            model=LLM_MODEL,
            max_tokens=1024,
            temperature=TEMPERATURE,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return "".join(block.text for block in response.content if block.type == "text")
    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


def _refusal(sources: list[dict], retrieval_source: str) -> dict:
    return {"answer": REFUSAL, "sources": sources, "retrieval_source": retrieval_source}


def generate_from_chunks(query: str, chunks: list[dict]) -> dict:
    """Sinh câu trả lời từ chunks đã retrieve (dùng chung cho app và evaluation)."""
    if not chunks:
        return _refusal([], "none")

    sources = [{**chunk, "citation_id": index} for index, chunk in enumerate(chunks, 1)]
    retrieval_source = "pageindex" if sources[0]["retrieval_method"] == "pageindex" else "hybrid"
    context = format_context(reorder_for_llm(sources))
    user_message = f"Context:\n{context}\n\nQuestion: {query}"
    try:
        answer = call_llm(SYSTEM_PROMPT, user_message).strip()
    except Exception as error:  # provider lỗi -> safe refusal, không crash UI
        print(f"LLM provider error: {error!r}")
        return _refusal(sources, retrieval_source)
    if not answer:
        return _refusal(sources, retrieval_source)

    # Bỏ citation không map được về sources.
    answer = _CITATION.sub(
        lambda m: m.group(0) if 1 <= int(m.group(1)) <= len(sources) else "",
        answer,
    )
    return {"answer": answer, "sources": sources, "retrieval_source": retrieval_source}


def cited_sources(result: dict) -> list[dict]:
    """Các source thực sự được trích dẫn trong answer."""
    numbers = {int(n) for n in _CITATION.findall(result["answer"])}
    return [s for s in result["sources"] if s.get("citation_id") in numbers]


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Trả về GenerationResult."""
    return generate_from_chunks(query, retrieve(query, top_k=top_k))


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "Từ năm 2026 hộ kinh doanh nộp thuế theo phương pháp nào?"
    result = generate_with_citation(query)
    print(result["answer"])
    for source in result["sources"]:
        print(f"  [{source['citation_id']}] {source['metadata']['title']} ({source['id']})")
