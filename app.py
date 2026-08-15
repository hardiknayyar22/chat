"""Streamlit UI for the Coforge India HR Policy Assistant POC."""

import json
import subprocess
import sys
from pathlib import Path

import streamlit as st

import config
from src.pipeline import answer_query
from src.policy_registry import build_policy_registry
from src.retriever import load_index_bundle

st.set_page_config(page_title="Coforge India HR Policy Assistant", page_icon="💬")


@st.cache_resource(show_spinner="Loading policy indexes...")
def load_resources():
    registry = build_policy_registry(config.POLICY_DOCS_DIR)
    bundle = load_index_bundle()
    return registry, bundle


registry, bundle = load_resources()

st.title("Coforge India HR Policy Assistant")
st.caption("AI-powered support for your HR policy questions.")

with st.sidebar:
    st.subheader("Available policies")
    if registry:
        for policy_name in sorted(registry):
            st.write(f"- {policy_name}")
    else:
        st.write("No policy PDFs detected yet.")

    st.markdown("---")
    if st.button("Rebuild index"):
        with st.spinner("Rebuilding the local policy index..."):
            result = subprocess.run([sys.executable, "-m", "src.ingestion"], capture_output=True, text=True)
        if result.returncode == 0:
            st.success("Index rebuilt successfully.")
            st.code(result.stdout.strip() or "Index rebuild completed.")
        else:
            st.error("Index rebuild failed.")
            st.code(result.stderr.strip() or result.stdout.strip() or "Unknown error")

if not bundle or not bundle.get("chunks"):
    st.error("No policy index found. Place PDFs in data/policies/ and run: python -m src.ingestion")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Ask a policy question about India HR matters...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Checking the relevant policy and retrieving the best matches..."):
            response = answer_query(
                question=user_query,
                chunks=bundle["chunks"],
                registry=registry,
                conversation_state=st.session_state.get("conversation_state", {}),
                bm25_indexes=bundle.get("bm25_indexes"),
            )

        answer = response["answer"]
        st.markdown(answer)

        if response.get("sources"):
            with st.expander("Sources"):
                for source in response["sources"]:
                    title = source.get("policy_name") or "Unknown policy"
                    section = source.get("section") or "General"
                    page = source.get("page")
                    source_label = f"{title} | Section: {section}"
                    if page:
                        source_label += f" | Page: {page}"
                    st.write(f"- {source_label}")

        st.caption(
            "Routing: " + response["route"]["route"] +
            " | Policy: " + (response["route"].get("policy_id") or "all policies")
        )

    st.session_state["conversation_state"] = response["conversation_state"]
    st.session_state.messages.append({"role": "assistant", "content": answer})

st.sidebar.button("Clear conversation", key="clear_chat", on_click=lambda: st.session_state.__setitem__("messages", []))
