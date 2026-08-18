import uuid
import streamlit as st
from langchain_core.runnables import Runnable
from src.chain import conversation_history
from src.ingestion import ingestion_pipeline, compute_pdf_hash
from src.retriever import (
    get_retriever,
    delete_file_from_vector_db,
    get_existing_file_names_with_hashes,
)
import tempfile
import logging

logger = logging.getLogger(__name__)


def process_uploaded_file(uploaded_file):

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = f"{temp_dir}/{uploaded_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            current_hash = compute_pdf_hash(temp_file_path)

            # SCENARIO 1: File is new and not processed before
            if uploaded_file.name not in st.session_state.processed_files:
                ingestion_pipeline(pdf_path=temp_file_path)
                st.session_state.processed_files[uploaded_file.name] = current_hash

            #SCENARIO 2: File has been processed before but has changed (hash mismatch)
            elif st.session_state.processed_files[uploaded_file.name] != current_hash:
                delete_file_from_vector_db(uploaded_file.name)
                ingestion_pipeline(pdf_path=temp_file_path)
                st.session_state.processed_files[uploaded_file.name] = current_hash
    except Exception as e:
        logger.error(f"Error processing uploaded file '{uploaded_file.name}': {e}", exc_info=True)
        st.error(f"Error processing uploaded file '{uploaded_file.name}': {e}")


def get_chain(selected_files_tuple: list[str] | None = None) -> Runnable | None:
    selected_files = selected_files_tuple if selected_files_tuple else None
    active_retriever = get_retriever(k=3, selected_files=selected_files)
    if active_retriever is None:
        logger.warning("No retriever could be created. Returning None for the chain.")
        return None
    return conversation_history(retriever=active_retriever)


st.title("Conversational Document Assistant")
st.caption("Conversational RAG with LCEL")

# 1. Initialize session state for chats and active chat
if "chats" not in st.session_state:
    st.session_state.chats = {}

if "active_chat" not in st.session_state:
    default_chat_id = str(uuid.uuid4())
    st.session_state.active_chat = default_chat_id
    st.session_state.chats[default_chat_id] = {"title": "Default Chat", "messages": []}

if "processed_files" not in st.session_state:
    st.session_state.processed_files = get_existing_file_names_with_hashes()

    # 2. Sidebar for chat management
with st.sidebar:

    upload_files = st.file_uploader(
        "Upload PDF files", type=["pdf"], accept_multiple_files=True
    )

    if upload_files:
        for uploaded_file in upload_files:
            process_uploaded_file(uploaded_file)

    selected_files = st.multiselect(
        "Select files for retrieval", options=list(st.session_state.processed_files)
    )

    if st.button("Delete Selected Files", disabled=not selected_files):
        for file_name in selected_files:
            if delete_file_from_vector_db(file_name):
                st.session_state.processed_files.pop(file_name, None)
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
            "No retriever is available. Please upload and process at least one PDF file to enable the retrieval-based chat functionality.")
        st.stop()
    if not current_chat["messages"]:
        current_chat["title"] = (
            user_prompt[:20] + "..." if len(user_prompt) > 20 else user_prompt
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
