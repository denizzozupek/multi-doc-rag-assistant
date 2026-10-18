import hashlib
import json
import logging
import os
import sqlite3
import tempfile
import uuid
from pathlib import Path
from typing import Any

import streamlit as st
from langgraph.checkpoint.sqlite import SqliteSaver

from src.config import (
    DEFAULT_LLM,
    DEFAULT_RERANK_TOP_K,
    LOCAL_CHAT_HISTORY_FILE,
    RETRIEVAL_FETCH_K,
    RETRIEVER_SEARCH_TYPE,
    SQLITE_DB_PATH,
)
from src.graph.main import graph_pipeline
from src.ingestion.ingestion import ingestion_pipeline
from src.utils import delete_file, get_existing_file_names_with_hashes

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@st.cache_resource
def get_compiled_app():
    """Initializes the SQLite checkpointer, builds the workflow graph, and returns the compiled application."""
    Path(SQLITE_DB_PATH.parent).mkdir(parents=True, exist_ok=True)

    try:
        conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
        saver = SqliteSaver(conn)

        workflow = graph_pipeline(
            llm=DEFAULT_LLM,
            top_k=DEFAULT_RERANK_TOP_K,
        )

        app = workflow.compile(checkpointer=saver)
        logger.info(
            "LangGraph workflow compiled successfully with SQLite checkpointer."
        )
        return app
    except Exception as e:
        logger.error(
            "Failed to compile LangGraph workflow: %s", e, exc_info=True
        )
        raise


def start_chat_session():
    """Initializes chat state lazily without creating dummy conversations on disk."""
    if "chats" not in st.session_state:
        st.session_state.chats = load_chat_metadata()

    if "active_chat" not in st.session_state:
        if st.session_state.chats:
            st.session_state.active_chat = next(iter(st.session_state.chats))
        else:
            st.session_state.active_chat = None


def load_chat_history(thread_id: str | None, app: Any) -> list:
    """Loads previous messages for a specific thread from the LangGraph checkpointer."""
    if not thread_id:
        return []

    config = {"configurable": {"thread_id": thread_id}}
    try:
        state_snapshot = app.get_state(config=config)
        values = getattr(state_snapshot, "values", None) or {}
        return values.get("messages", [])
    except Exception as e:
        logger.error(
            "Error loading chat history for thread_id '%s': %s",
            thread_id,
            e,
            exc_info=True,
        )
        return []


def load_chat_metadata() -> dict:
    """Loads chat sidebar metadata from the local JSON file."""
    chat_file = Path(LOCAL_CHAT_HISTORY_FILE)
    if chat_file.exists():
        try:
            with open(chat_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to load chat metadata: %s", e, exc_info=True)
            return {}
    return {}


def save_chat_history():
    """Persists chat metadata dictionary from session state to the JSON file."""
    chat_history = st.session_state.get("chats", {})
    chat_file = Path(LOCAL_CHAT_HISTORY_FILE)
    chat_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(chat_file, "w", encoding="utf-8") as f:
            json.dump(chat_history, f, ensure_ascii=False, indent=4)
            logger.info("Chat history saved successfully.")
    except Exception as e:
        logger.error("Failed to save chat history: %s", e, exc_info=True)


def delete_chat(chat_id_to_delete: str):
    """Deletes a chat from state and JSON. Manages active chat pointers cleanly."""
    if chat_id_to_delete in st.session_state.chats:
        st.session_state.chats.pop(chat_id_to_delete, None)
        save_chat_history()

        # If deleted chat was active, switch to another or set to None
        if st.session_state.active_chat == chat_id_to_delete:
            if st.session_state.chats:
                st.session_state.active_chat = next(
                    iter(st.session_state.chats)
                )
            else:
                st.session_state.active_chat = None


def process_uploaded_file(uploaded_file):
    """Calculates file hash and triggers ingestion only when content is new or changed."""
    try:
        file_bytes = uploaded_file.getvalue()
        current_hash = hashlib.sha256(file_bytes).hexdigest()
        needs_ingestion = False

        if uploaded_file.name not in st.session_state.processed_files:
            needs_ingestion = True
        elif st.session_state.processed_files[uploaded_file.name] != current_hash:
            delete_file(uploaded_file.name)
            needs_ingestion = True

        if needs_ingestion:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(file_bytes)
                ingestion_pipeline(pdf_path=temp_file_path)

            st.session_state.processed_files[uploaded_file.name] = current_hash
            st.toast(f"'{uploaded_file.name}' processed successfully.")
        else:
            logger.info(
                f"'{uploaded_file.name}' has already been processed and is up-to-date."
            )
            st.toast(
                f"'{uploaded_file.name}' has already been processed and is up-to-date."
            )

    except Exception as e:
        logger.error(
            f"Error processing uploaded file '{uploaded_file.name}': {e}",
            exc_info=True,
        )
        st.error(f"'{uploaded_file.name}' could not be processed: {e}")


# --- Application Setup & Session Initialization ---
st.set_page_config(page_title="Document Assistant", layout="wide")
st.title("Conversational Document Assistant")
st.caption("Conversational RAG with LangGraph")

start_chat_session()
app = get_compiled_app()

if "processed_files" not in st.session_state:
    st.session_state.processed_files = get_existing_file_names_with_hashes()

# --- Sidebar UI ---
with st.sidebar:
    st.header("Document Management")
    upload_files = st.file_uploader(
        "Upload PDF files", type=["pdf"], accept_multiple_files=True
    )

    if upload_files:
        for uploaded_file in upload_files:
            process_uploaded_file(uploaded_file)

    selected_files = st.multiselect(
        "Select files for retrieval",
        options=list(st.session_state.processed_files.keys()),
        help="If no files are selected, the system will use all processed files for retrieval.",
    )

    if st.button("Delete Selected Files", disabled=not selected_files):
        for file_name in selected_files:
            if delete_file(file_name):
                st.session_state.processed_files.pop(file_name, None)
        st.success("Selected files deleted successfully.")
        st.rerun()

    st.title("Chats")

    if st.button("+ New Chat", use_container_width=True):
        st.session_state.active_chat = None
        st.rerun()

    st.divider()

    # List conversations with inline clean delete action
    for chat_id, chat_data in list(st.session_state.chats.items()):
        is_active = chat_id == st.session_state.active_chat
        col_btn, col_del = st.columns([0.85, 0.15])

        with col_btn:
            btn_label = (
                f"**{chat_data['title']}**" if is_active else chat_data["title"]
            )
            if st.button(
                btn_label,
                key=f"chat_{chat_id}",
                use_container_width=True,
                disabled=is_active,
            ):
                st.session_state.active_chat = chat_id
                st.rerun()

        with col_del:
            if st.button("×", key=f"del_{chat_id}", help="Delete chat"):
                delete_chat(chat_id)
                st.rerun()

# --- Main Chat Interface ---
chat_history = load_chat_history(st.session_state.active_chat, app)

# Render existing messages
for message in chat_history:
    role = "user" if message.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# Handle user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Lazy creation on the very first prompt
    if not st.session_state.active_chat:
        new_chat_id = str(uuid.uuid4())
        chat_title = prompt[:24] + ("..." if len(prompt) > 24 else "")
        st.session_state.chats[new_chat_id] = {"title": chat_title}
        st.session_state.active_chat = new_chat_id
        save_chat_history()

    active_chat_id = st.session_state.active_chat

    with st.chat_message("user"):
        st.markdown(prompt)

    config = {"configurable": {"thread_id": active_chat_id}}
    inputs = {
        "messages": [("user", prompt)],
        "selected_files": selected_files,
    }

    with st.chat_message("assistant"):

        async def stream_graph_response():
            async for chunk in app.astream(
                inputs, config=config, stream_mode="updates"
            ):
                for node_name, node_output in chunk.items():
                    if "messages" in node_output and node_output["messages"]:
                        yield node_output["messages"][-1].content

        st.write_stream(stream_graph_response)

    st.rerun()