"""Streamlit chat UI.

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

from ytrag import RAGConfig, RAGPipeline
from ytrag.youtube import TranscriptError

st.set_page_config(page_title="YouTube Q&A", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner=False)
def get_pipeline(retrieval: str, k: int, rerank: bool, router: bool) -> RAGPipeline:
    config = RAGConfig().with_overrides(retrieval=retrieval, k=k, rerank=rerank, router=router)
    return RAGPipeline(config=config)


state = st.session_state
state.setdefault("messages", [])  # [{role, content, sources, route}]
state.setdefault("player", None)  # (video_id, start_seconds)

# ---------- sidebar: settings, ingestion, library ----------
with st.sidebar:
    st.title("🎬 YouTube Q&A")
    st.caption("Chat with YouTube videos. Answers cite clickable timestamps.")

    with st.expander("Retrieval settings"):
        retrieval = st.selectbox("Mode", ["hybrid", "mmr", "similarity"], help="hybrid = BM25 + FAISS (RRF)")
        k = st.slider("Chunks (k)", 1, 10, RAGConfig().k)
        rerank = st.toggle("LLM rerank", value=False)
        router = st.toggle("Summary router (map-reduce)", value=True)

    try:
        pipeline = get_pipeline(retrieval, k, rerank, router)
    except RuntimeError as exc:  # missing API key
        st.error(str(exc))
        st.stop()

    with st.form("ingest", clear_on_submit=True):
        url = st.text_input("Add a video", placeholder="https://www.youtube.com/watch?v=...")
        if st.form_submit_button("Ingest", use_container_width=True) and url:
            with st.spinner("Fetching transcript and building index..."):
                try:
                    info = pipeline.ingest(url)
                    st.success(f"Indexed '{info['title']}' ({info['chunks']} chunks)")
                except (TranscriptError, ValueError) as exc:
                    st.error(str(exc))

    videos = pipeline.list_videos()
    labels = {v["video_id"]: f"{v['title']} ({v['duration'] / 60:.0f} min)" for v in videos}
    selected = st.multiselect(
        "Chat with", options=list(labels), default=list(labels)[:1], format_func=labels.get
    )

    if st.button("Clear chat", use_container_width=True):
        state.messages, state.player = [], None
        st.rerun()

# ---------- main: chat + player ----------
chat_col, player_col = st.columns([3, 2], gap="large")


def render_sources(sources: list[dict], key_prefix: str) -> None:
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            cols = st.columns([1, 6])
            if cols[0].button(f"▶ {s['timestamp']}", key=f"{key_prefix}-{s['n']}"):
                state.player = (s["video_id"], int(s["start"]))
                st.rerun()
            cols[1].markdown(f"**[{s['n']}] {s['title']}** · [open on YouTube]({s['url']})")
            cols[1].caption(s["text"][:400] + ("..." if len(s["text"]) > 400 else ""))


with player_col:
    if state.player:
        vid, start = state.player
        st.video(f"https://www.youtube.com/watch?v={vid}", start_time=start)
    elif selected:
        st.video(f"https://www.youtube.com/watch?v={selected[0]}")
    else:
        st.info("Add a video in the sidebar to get started.")

with chat_col:
    for i, msg in enumerate(state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources(msg["sources"], key_prefix=f"m{i}")

    if question := st.chat_input("Ask about the video...", disabled=not selected):
        with st.chat_message("user"):
            st.markdown(question)
        history = [{"role": m["role"], "content": m["content"]} for m in state.messages]

        with st.chat_message("assistant"):
            with st.spinner("Searching the transcript..."):
                prepared = pipeline.prepare(question, selected, history)
            if prepared.route == "summary":
                st.caption("Summary question detected: summarizing the whole video (map-reduce).")
            answer = st.write_stream(prepared.stream())
            sources = [s.__dict__ for s in prepared.sources]
            render_sources(sources, key_prefix=f"m{len(state.messages) + 1}")

        state.messages += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer, "sources": sources, "route": prepared.route},
        ]
