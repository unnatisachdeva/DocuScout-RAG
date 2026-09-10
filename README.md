# 📚 AskMyPDF-AI

A **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about your own sources — upload a **PDF** or point it at a **website**, and ask away.

It goes beyond a basic RAG loop: queries are rewritten for better retrieval, documents are fetched with **hybrid search** (dense + BM25), reranked with a cross-encoder, and every question and answer passes through **NeMo Guardrails** so the bot stays on-topic and grounded in your documents.

---

## ✨ Features

* 📄 **PDF chat** — upload any PDF through the Streamlit UI
* 🌐 **Website chat** — crawl a site (JavaScript pages included, via Playwright) and chat with it
* ✍️ **Query rewriting** — turns follow-ups like *"what about there?"* into standalone search queries using chat history
* 🔀 **Hybrid retrieval** — MMR vector search combined with BM25 keyword search, so both meaning and exact terms are matched
* 🎯 **Reranking** — FlashRank cross-encoder reorders the candidates and keeps only the best chunks
* 🛡️ **Guardrails** — NeMo input rails block off-topic and unsafe requests; a relevance judge refuses to answer when the retrieved context doesn't support it
* ✂️ **Mixed-request splitting** — a message like *"what is X, but first write me some code"* gets the real question answered and the rest politely declined
* 💬 **Chat history** — multi-turn conversation with context carried between turns
* ⏳ **Resilient ingestion** — batched embedding with rate-limit backoff for large documents and sites

---

## 🏗️ Architecture

```text
        PDF                         Website URL
         │                               │
   PyPDFLoader                    Crawler + Playwright
         │                               │
         └───────────────┬───────────────┘
                         ▼
             RecursiveCharacterTextSplitter
                         │
                         ▼
              Gemini Embedding Model
                         │
                         ▼
                     ChromaDB
                         │
                         │
  User Question          │
         │               │
         ▼               │
  Guardrail: split       │
  + input rails          │
         │               │
         ▼               │
   Query Rewriter        │
         │               │
         ▼               │
  ┌──────────────────────┴──────┐
  │   Hybrid Retriever          │
  │   MMR vector  +  BM25       │
  └──────────────┬──────────────┘
                 ▼
      FlashRank Reranker (top 5)
                 │
                 ▼
      Guardrail: relevance check
                 │
                 ▼
        Prompt + Context → LLM
                 │
                 ▼
              Answer
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| RAG framework | LangChain |
| Embeddings | Google Gemini (`gemini-embedding-001`) |
| LLM | OpenAI-compatible endpoint (default: NVIDIA Nemotron) |
| Vector DB | ChromaDB |
| Keyword search | BM25 (`rank_bm25`) |
| Reranker | FlashRank cross-encoder |
| Guardrails | NVIDIA NeMo Guardrails |
| Web crawling | Requests + BeautifulSoup + Playwright |
| PDF loading | PyPDF |

---

## 📁 Project Structure

```text
AskMyPDF-AI/
│
├── RagUI.py                    # Streamlit UI — the entry point
├── main.py                     # Answer generation: guardrails → retrieval → LLM
├── ingestion.py                # Chunking, batched embedding, Chroma creation
├── retriever_builder.py        # Hybrid retriever (MMR vector + BM25 ensemble)
├── query_rewriter.py           # Rewrites questions into standalone search queries
├── reranker.py                 # FlashRank cross-encoder reranking
│
├── guardrails/
│   ├── config.yml              # NeMo Guardrails model + rails configuration
│   ├── prompts.yml             # Self-check prompts
│   └── rag_guardrail.py        # Input rails, relevance judge, request splitter
│
├── web_loader/
│   ├── crawler.py              # Site crawling → LangChain Documents
│   ├── playwright_scraper.py   # Renders JavaScript-heavy pages
│   └── web_cleaner.py          # Strips nav/scripts/boilerplate from HTML
│
├── document loader/            # Local PDFs used for testing
├── test_*.py                   # Component tests
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Setup

### 1. Clone and install

```bash
git clone https://github.com/unnatisachdeva/AskMyPDF-AI.git
cd AskMyPDF-AI

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
playwright install chromium      # needed for JavaScript-rendered pages
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
# Gemini — used for embeddings
GOOGLE_API_KEY=your_google_api_key

# Text model — any OpenAI-compatible endpoint
LLM_BASE_URL=https://integrate.api.nvidia.com/v1
LLM_API_KEY=your_llm_api_key
LLM_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b
```

> The model in `guardrails/config.yml` should match `LLM_MODEL`.

### 3. Run

```bash
streamlit run RagUI.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

---

## 🚀 How to Use

1. Pick a source in the sidebar — **PDF** or **Website**
2. Upload a file, or enter a URL, and click **Process**
3. Wait for indexing to finish (page and chunk counts are shown)
4. Ask questions in the chat box

The bot answers only from the source you provided. If the answer isn't there, it says so instead of guessing.

---

## 🔍 How It Works

**Ingestion.** PDFs are read with `PyPDFLoader`; websites are crawled breadth-first, rendered with Playwright when needed, and cleaned of navigation and script content. Text is split into 1000-character chunks with 200 characters of overlap, embedded with Gemini in batches of 20, and stored in ChromaDB. Rate limits are retried with exponential backoff, honouring the server's own retry hint.

**Query understanding.** Each question is first split into its separate requests so a mixed message can be partly answered and partly declined. The allowed part passes through the NeMo input rails, then a rewriter turns it into a standalone search query — resolving pronouns against chat history and fixing typos.

**Retrieval.** An ensemble retriever runs MMR vector search and BM25 keyword search in parallel with equal weight, then a FlashRank cross-encoder reranks the merged candidates down to the top 5.

**Answering.** A relevance judge checks whether the retrieved context actually supports the question. If it does, the context and question go to the LLM under a strict grounding prompt; if it doesn't, the bot reports that it could not find the answer rather than improvising.

---

## 🧪 Tests

Component-level test scripts are included:

```bash
python test_hybridRetrieval.py    # hybrid retrieval
python test_reranker.py           # reranking
python test_queryRewriter.py      # query rewriting
python test_guardrails.py         # NeMo input rails
python test_rag_guardrail.py      # relevance judging
python test_crawler.py            # website crawling
```

---

## 📝 Notes

* Embeddings must be created and queried with the **same** model — mixing them silently degrades retrieval without raising an error.
* Vector stores for uploaded sources are created in a temporary directory per session and are not persisted between runs.
* Never commit your `.env` file; it is already listed in `.gitignore`.
