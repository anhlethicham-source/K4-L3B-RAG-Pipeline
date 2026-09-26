import html
import re

import streamlit as st
from dotenv import load_dotenv


load_dotenv()
st.set_page_config(
    page_title="Trợ lý thuế hộ kinh doanh",
    page_icon="📑",
    layout="wide",
)

from src.task10_generation import (  # noqa: E402  (load_dotenv trước khi đọc config)
    LLM_MODEL,
    LLM_PROVIDER,
    cited_sources,
    generate_from_chunks,
)
from src.task9_retrieval_pipeline import SCORE_THRESHOLD, retrieve  # noqa: E402

CITATION = re.compile(r"\[(\d+)\]")
METHOD_LABELS = {
    "hybrid": "Hybrid (Dense + BM25 → RRF)",
    "dense": "Dense only",
    "pageindex": "PageIndex fallback",
    "none": "Không có nguồn",
}

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("📑 Trợ lý thuế hộ kinh doanh")
    st.caption(
        "Hỏi đáp về chính sách thuế và quản lý thuế đối với hộ kinh doanh, "
        "cá nhân kinh doanh từ năm 2026 (Thông tư 18/2026, Thông tư 152/2025, "
        "Nghị định 73/2016 và các bài viết hướng dẫn)."
    )
    top_k = st.slider("Số chunks", 3, 10, 5)
    use_reranking = st.toggle("Hybrid retrieval (BM25 + RRF)", value=True)
    st.caption(f"LLM: `{LLM_PROVIDER}/{LLM_MODEL}` · fallback threshold: {SCORE_THRESHOLD}")
    if st.button("Xoá hội thoại"):
        st.session_state.messages = []
        st.rerun()


def highlight_citations(answer: str) -> str:
    """Tô đậm [n] để người dùng đối chiếu với danh sách nguồn."""
    return CITATION.sub(r"**[\1]**", answer)


def render_sources(result: dict) -> None:
    sources = result["sources"]
    if not sources:
        return
    cited_ids = {source["id"] for source in cited_sources(result)}
    st.caption(
        f"Retrieval: **{METHOD_LABELS.get(result['retrieval_source'], result['retrieval_source'])}** · "
        f"{len(cited_ids)}/{len(sources)} nguồn được trích dẫn"
    )
    for source in sources:
        metadata = source["metadata"]
        marker = "✅" if source["id"] in cited_ids else "▫️"
        label = (
            f"{marker} [{source['citation_id']}] {metadata['title']} — "
            f"{source['retrieval_method']} score {source['score']:.4f}"
        )
        with st.expander(label, expanded=False):
            st.markdown(
                f"**Source:** `{metadata['source']}` · chunk {metadata['chunk_index']} · "
                f"{metadata['doc_type']}"
            )
            if metadata.get("url"):
                st.markdown(f"[Mở nguồn gốc]({metadata['url']})")
            st.markdown(
                f"<div style='white-space:pre-wrap;font-size:0.9em'>{html.escape(source['content'])}</div>",
                unsafe_allow_html=True,
            )


st.title("Trợ lý tra cứu thuế hộ kinh doanh")
st.caption(
    "Mọi câu trả lời đều kèm citation [n] trỏ tới nguồn bên dưới. "
    "Nếu tài liệu không đủ căn cứ, trợ lý sẽ từ chối trả lời."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(highlight_citations(message["content"]))
            render_sources(message["result"])
        else:
            st.markdown(message["content"])

query = st.chat_input("Ví dụ: Hộ kinh doanh có doanh thu dưới 500 triệu có phải nộp thuế không?")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm và tổng hợp..."):
            try:
                chunks = retrieve(query, top_k=top_k, use_reranking=use_reranking)
                result = generate_from_chunks(query, chunks)
            except Exception as error:  # không để UI crash
                st.error(f"Lỗi pipeline: {error}")
                result = {
                    "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
                    "sources": [],
                    "retrieval_source": "none",
                }
        st.markdown(highlight_citations(result["answer"]))
        render_sources(result)

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "result": result}
    )
