import os
import uuid
import streamlit as st
import tempfile
import logging
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any


from langchain_core.runnables import Runnable
from langgraph.checkpoint.sqlite import SqliteSaver

from src.graph.main import graph_pipeline
from src.generation.chain import conversation_history
from src.ingestion.ingestion import ingestion_pipeline
from src.retrieval.retriever import get_retriever
from src.utils import delete_file, get_existing_file_names_with_hashes
from src.config import (
    RETRIEVER_SEARCH_TYPE,
    SQLITE_DB_PATH,
    DEFAULT_LLM,
    RETRIEVAL_FETCH_K,
    DEFAULT_RERANK_TOP_K,
    RETRIEVAL_TO_RERANK_TOP_K,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SQLITE_DB_PATH = Path(SQLITE_DB_PATH)
Path(SQLITE_DB_PATH.parent).mkdir(parents=True, exist_ok=True)


@st.cache_resource
def get_compiled_app():
    """Initializes the SQLite checkpointer, builds the workflow graph, and returns the compiled application."""
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
        saver = SqliteSaver(conn)

        workflow = graph_pipeline(
            llm=DEFAULT_LLM,
            top_k=DEFAULT_RERANK_TOP_K,
        )

        app = workflow.compile(checkpointer=saver)
        logger.info("LangGraph workflow compiled successfully with SQLite checkpointer.")
        return app
    except Exception as e:
        logger.error("Failed to compile LangGraph workflow: %s", e, exc_info=True)
        raise

def start_chat_session():
    """Initializes the chat session state in Streamlit."""
    if "chats" not in st.session_state:
        st.session_state.chats = []
        loaded_chat_history = load_chat_history()
        if loaded_chat_history:
            pass

        
def load_chat_history(thread_id: str, app : Any) -> list:
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state_snapshot = app.get_state(config=config)
        values = getattr(state_snapshot, "values", None) or {}
        state_data = values.get("messages", [])
    except Exception as e:
        logger.error("Error loading chat history for thread_id '%s': %s", thread_id, e, exc_info=True)
        state_data = []
    return state_data

    
def save_chat_history():
    chat_history = st.session_state.chats
    chat_history_file = "data/chat_history.json"
    os.makedirs(os.path.dirname(chat_history_file), exist_ok=True)
    with open(chat_history_file, "w", encoding="utf-8") as f:
        json.dump(chat_history, f, ensure_ascii=False, indent=4)


def process_uploaded_file(uploaded_file):
    try:
        # 1. Calculate the hash of the uploaded file without saving it to disk
        file_bytes = uploaded_file.getvalue()
        current_hash = hashlib.sha256(file_bytes).hexdigest()
        needs_ingestion = False

        # Scenario 1:  File has not been processed before
        if uploaded_file.name not in st.session_state.processed_files:
            needs_ingestion = True
        # Scenario 2: File has been processed before, but the content has changed
        elif st.session_state.processed_files[uploaded_file.name] != current_hash:
            delete_file(uploaded_file.name)
            needs_ingestion = True
        # Scenario 3: File has been processed before and the content has not changed
        if needs_ingestion:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = f"{temp_dir}/{uploaded_file.name}"
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
            f"Error processing uploaded file '{uploaded_file.name}': {e}", exc_info=True
        )
        st.error(f"'{uploaded_file.name}' could not be processed: {e}")


def get_chain(selected_files: list[str] | None = None) -> Runnable | None:

    active_retriever = get_retriever(
        k=15,
        fetch_k=20,
        selected_files=selected_files,
        search_type=RETRIEVER_SEARCH_TYPE,
    )
    if active_retriever is None:
        logger.warning("No retriever could be created. Returning None for the chain.")
        return None
    return conversation_history(retriever=active_retriever)


st.title("Conversational Document Assistant")
st.caption("Conversational RAG with LCEL")

# 1. Initialize session state for chats and active chat
if "chats" not in st.session_state:
    loaded_chat_history = load_chat_history()
    if loaded_chat_history:
        st.session_state.chats = loaded_chat_history
        st.session_state.active_chat = next(iter(loaded_chat_history), None)
    else:
        default_chat_id = str(uuid.uuid4())
        st.session_state.active_chat = default_chat_id
        st.session_state.chats = {
            default_chat_id: {"title": "Default Chat", "messages": []}
        }

if "processed_files" not in st.session_state:
    st.session_state.processed_files = get_existing_file_names_with_hashes()

    # 2. Sidebar for chat management
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
        options=list(st.session_state.processed_files),
        help="If no files are selected, the system will use all processed files for retrieval.",
    )

    if st.button("Delete Selected Files", disabled=not selected_files):
        for file_name in selected_files:
            if delete_file(file_name):
                st.session_state.processed_files.pop(file_name, None)
        st.success("Selected files deleted successfully.")
        save_chat_history()
        st.rerun()

    chain = get_chain(selected_files)

    st.title("Chats")

    if st.button("+ New Chat", use_container_width=True):
        new_chat_id = str(uuid.uuid4())
        st.session_state.active_chat = new_chat_id
        st.session_state.chats[new_chat_id] = {
            "title": f"Chat {len(st.session_state.chats) + 1}",
            "messages": [],
        }
        save_chat_history()
        st.rerun()

    st.divider()

    for chat_id, chat in st.session_state.chats.items():
        is_active = chat_id == st.session_state.active_chat
        btn_label = f"**{chat['title']}**" if is_active else chat["title"]
        if st.button(
            btn_label, key=chat_id, use_container_width=True, disabled=is_active
        ):
            st.session_state.active_chat = chat_id
            st.rerun()

# 3. Get the active chat and display its messages
active_chat_id = st.session_state.active_chat
current_chat = st.session_state.chats[active_chat_id]

# 4. Display active chat past messages into the main area
for message in current_chat["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# 5. Input area and generate assistant response
user_prompt = st.chat_input("Type your message here...")


if user_prompt:
    if chain is None:
        st.warning(
            "No retriever is available. Please upload and process at least one PDF file to enable the retrieval-based chat functionality."
        )
        st.stop()

    is_first_message = len(current_chat["messages"]) == 0
    if is_first_message:
        current_chat["title"] = (
            user_prompt[:25] + "..." if len(user_prompt) > 25 else user_prompt
        )

    # User message
    with st.chat_message("user"):
        st.write(user_prompt)
    current_chat["messages"].append({"role": "user", "content": user_prompt})

    # Generate assistant response (Streaming + dynamic session_id)
    with st.chat_message("assistant"):
        response = chain.stream(
            {"input": user_prompt},
            config={"configurable": {"session_id": active_chat_id}},
        )
        assistant_response = st.write_stream(response)

    current_chat["messages"].append(
        {"role": "assistant", "content": assistant_response}
    )
    save_chat_history()
    if is_first_message:
        st.rerun()
