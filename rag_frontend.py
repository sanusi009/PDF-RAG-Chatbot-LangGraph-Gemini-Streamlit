import uuid
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from rag_backend import (
    workflow,
    ingest_pdf,
    retriever_all_threads,
    thread_document_metadata,
)

# ===================== Utilities =====================
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread_to_history(st.session_state["thread_id"])
    st.session_state["message_history"] = []

def add_thread_to_history(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    state = workflow.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])

# ======================= Session Initialization ====================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retriever_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

add_thread_to_history(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]
selected_thread = None

# =================== Sidebar =====================
st.sidebar.title("PDF LangGraph Chatbot")
st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using '{latest_doc.get('filename')}' — "
        f"{latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"'{uploaded_pdf.name}' already processed for this chat.")
    else:
        with st.sidebar.status(f"Indexing '{uploaded_pdf.name}'...", expanded=False):
            file_bytes = uploaded_pdf.read()
            summary = ingest_pdf(file_bytes, thread_id=thread_key, filename=uploaded_pdf.name)
            thread_docs[uploaded_pdf.name] = summary
        st.sidebar.success(f"Indexed '{uploaded_pdf.name}' — {summary['chunks']} chunks.")
        st.rerun()

st.sidebar.header("Conversation History")
for thread_id in threads:
    label = str(thread_id)
    if st.sidebar.button(label, key=f"thread_{label}"):
        st.session_state["thread_id"] = thread_id
        loaded_messages = load_conversation(thread_id)

        temp_messages = []
        for message in loaded_messages:
            if isinstance(message, HumanMessage):
                temp_messages.append({'role': 'user', 'content': message.content})
            elif isinstance(message, AIMessage) and message.content:
                temp_messages.append({'role': 'assistant', 'content': message.content})

        st.session_state["message_history"] = temp_messages
        st.rerun()

# =================== Main chat UI =====================
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state["message_history"].append({'role': 'user', 'content': user_input})
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {'configurable': {'thread_id': st.session_state["thread_id"]}}

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = workflow.invoke(
                {'messages': [HumanMessage(content=user_input)]},
                config=CONFIG
            )
        ai_message = response['messages'][-1].content
        st.text(ai_message)

    st.session_state["message_history"].append({'role': 'assistant', 'content': ai_message})