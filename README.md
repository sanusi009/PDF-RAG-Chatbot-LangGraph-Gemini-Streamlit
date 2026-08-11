# PDF RAG Chatbot — LangGraph + Gemini + Streamlit

A multi-tool conversational agent that lets you upload a PDF per chat thread and ask questions grounded in that document — alongside general-purpose tools for web search, calculation, and live stock prices. Built with LangGraph for orchestration, Gemini for reasoning, and Streamlit for the UI, with per-thread conversation history persisted to SQLite.

## Features

- **Per-thread PDF ingestion** — upload a PDF in the sidebar; it's chunked, embedded, and indexed in FAISS scoped to that specific conversation thread, so different chats can reference different documents without cross-contamination.
- **Tool-using agent** — the LLM decides when to call:
  - `rag_tool` — retrieves relevant chunks from the thread's indexed PDF
  - `search_tool` — general web search via DuckDuckGo
  - `calculator` — arithmetic operations
  - `get_stock_price` — live stock quotes via Alpha Vantage
- **Persistent conversation history** — every thread is checkpointed to a local SQLite database (`chat_history.db`), so past conversations survive an app restart and are browsable from the sidebar.
- **New Chat / thread switching** — start a fresh thread or revisit any previous one, each with its own message history and (if uploaded) its own PDF context.

## Architecture

```
Streamlit UI (rag_frontend.py)
        │
        ▼
LangGraph workflow (rag_backend.py)
        │
   ┌────┴────┐
   │chat_node│──(LLM decides)──▶ tool_node ──▶ back to chat_node
   └─────────┘                       │
                          ┌──────────┼──────────┬─────────────┐
                       rag_tool  search_tool  calculator  get_stock_price
                          │
                    FAISS retriever
                    (per-thread, in-memory)
```

Conversation state is checkpointed via `SqliteSaver`, keyed by `thread_id` (a UUID generated per chat session).

## Stack

- [LangGraph](https://github.com/langchain-ai/langgraph) — agent orchestration (tool-calling loop)
- [`langchain-google-genai`](https://pypi.org/project/langchain-google-genai/) — Gemini (`gemini-2.5-flash`) as the reasoning model
- [`langchain-huggingface`](https://pypi.org/project/langchain-huggingface/) + `sentence-transformers` — local embeddings (`BAAI/bge-small-en-v1.5`), no API quota
- [FAISS](https://github.com/facebookresearch/faiss) — per-thread vector index
- [Streamlit](https://streamlit.io/) — chat UI
- [`ddgs`](https://pypi.org/project/ddgs/) — web search
- [Alpha Vantage](https://www.alphavantage.co/) — stock price data

## Setup

1. **Clone and create a virtual environment**
   ```bash
   python -m venv myenv
   myenv\Scripts\Activate.ps1        # Windows PowerShell
   # source myenv/bin/activate       # macOS/Linux
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your `.env` file**
   ```
   GOOGLE_API_KEY=your_gemini_api_key
   ALPHAVANTAGE_API_KEY=your_alphavantage_api_key
   ```
   - Get a free Gemini key at [Google AI Studio](https://aistudio.google.com/apikey).
   - Get a free Alpha Vantage key at [alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key).

4. **Run the app**
   ```bash
   streamlit run rag_frontend.py
   ```
   Never run this file with plain `python rag_frontend.py` — Streamlit apps must be launched via `streamlit run`.

## Usage

1. Open the app in your browser (Streamlit will print a local URL, typically `http://localhost:8501`).
2. Upload a PDF from the sidebar — you'll see a confirmation with the number of pages and chunks indexed.
3. Ask questions about the PDF, or ask general questions (web search, math, stock prices) — the agent picks the right tool automatically.
4. Use **New Chat** to start a fresh thread; previous threads remain listed in the sidebar and can be reopened at any time.

## Files

| File | Purpose |
|---|---|
| `rag_backend.py` | LangGraph workflow: LLM, tools, PDF ingestion/retrieval, SQLite checkpointing |
| `rag_frontend.py` | Streamlit chat interface: thread management, PDF upload, message rendering |
| `requirements.txt` | Pinned dependencies |

## Notes & limitations

- PDF indexes are held **in memory** (`_THREAD_RETRIEVERS` dict) — they're rebuilt if the app restarts, even though chat history persists in SQLite. Re-upload the PDF after a restart to resume document Q&A on that thread.
- Local embeddings (`BAAI/bge-small-en-v1.5`) trade a small amount of retrieval quality for zero API cost/quota — swap to a hosted embedding model if higher accuracy is needed.
- Web search quality depends on the underlying DuckDuckGo scraping backend and may occasionally return empty or irrelevant results for obscure queries.

---
*Author: Sanusi — M.Sc. Data Science, University of Leoben.*
