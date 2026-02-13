# Agentic RAG

Agentic RAG is a Streamlit application that orchestrates a multi-agent CrewAI workflow to answer user questions with retrieval-augmented generation (RAG). A Groq-hosted LLM coordinates specialized agents that first mine a local FAISS knowledge base, then fall back to Tavily web search when fresh context is required. The result is a grounded, citation-heavy answer stream that can run locally or inside Docker.

---

## Features

- **Multi-agent pipeline**: Researcher → Writer → Critic → Reviser sequence ensures grounded and polished responses.
- **Hybrid retrieval**: FAISS-backed local search plus optional Tavily-powered web search for recency.
- **Embeddings toolbox**: `SentenceTransformer` ingestion scripts, deduplicated uploads, and reusable loaders.
- **Streamlit UX**: Chat-style question history with cached resources for quick iteration.
- **Docker-ready**: Container image for reproducible deployments.

---

## Architecture

1. Users submit a question through the Streamlit UI (`app.py`).
2. Cached bootstrap loads the Groq LLM, FAISS index (`vectorstore/`), and CrewAI agents (`llm_and_crew_functions.py`).
3. `LocalFAISSSearchTool` retrieves relevant chunks; if needed, `TavilyWebSearchTool` supplements with web context.
4. CrewAI agents pass evidence, draft answers, critiques, and revisions sequentially until a final Markdown response is produced.
5. Streamlit renders the answer and logs the exchange for reference.

---

## Requirements

- Python 3.10+ (tested with 3.11)
- pip / virtualenv (or Conda)
- Docker (optional, for containerized runs)
- API credentials
	- Groq (LLM execution)
	- OpenAI (fallback usage within CrewAI stack)
	- Tavily (web search)

---

## Quickstart

```bash
# 1. Clone and enter the repository
git clone https://github.com/<your-org>/Agentic-RAG.git
cd Agentic-RAG

# 2. Create & activate a virtual environment
python -m venv .venv
.\.venv\Scripts\activate  # (Linux/macOS: source .venv/bin/activate)

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Launch the Streamlit app
streamlit run app.py
```

The first run caches the CrewAI resources; subsequent queries reuse the same Crew for faster responses.

---

## Credentials & Configuration

Store API keys as JSON files under `keys/` (already gitignored):

```
keys/
	groq.json
	openai.json
	tavily.json
```

Each file must expose one of `api_key`, `groq_api_key` / `openai_api_key` / `tavily_api_key`, `key`, or `token`.

Example (`keys/groq.json`):

```json
{
	"api_key": "gsk_your_secret"
}
```

`app.py` loads these via the helpers in `load_keys.py` and sets the corresponding environment variables before CrewAI initializes.

---

## Building / Updating the Local Vector Store

The application reads an on-disk FAISS index from `vectorstore/`. To create or refresh it:

1. Place source documents inside `documents/` (PDF, Markdown, or plain text).
2. Use `docstore_functions.py` utilities (or the exploratory `docstore_notebook.ipynb`) to ingest:

```python
from docstore_functions import create_faiss_store_from_documents, path_upload_document_to_vectorstore

# Example: load documents with LangChain loaders and persist an index
store = create_faiss_store_from_documents(documents=my_docs, index_dir="vectorstore")

# Later, append more files with deduplication
store = path_upload_document_to_vectorstore(["documents/new_report.pdf"], store)
```

- Deduplication occurs via content hashes, so repeated uploads skip identical chunks.
- `retrieve_local()` offers a normalized schema you can reuse in notebooks.

If the directory is missing, the Streamlit app will raise a `FileNotFoundError` prompting you to build the index first.

---

## Running with Docker

The included `Dockerfile` builds a self-contained image:

```bash
docker build -t agentic-rag .
docker run --rm -p 8501:8501 \
	-v %cd%/keys:/app/keys \
	-v %cd%/vectorstore:/app/vectorstore \
	agentic-rag
```

Mount the `keys/` and `vectorstore/` directories (read-only is fine) so the container can access credentials and embeddings.

---

## Using the App

1. Open the Streamlit URL (default `http://localhost:8501`).
2. Enter a research question or prompt.
3. Click **Retrieve answer**.
4. Review the streamed response; each exchange is stored in the sidebar chat history.

If the crew cannot find adequate local evidence, it explicitly states the gap and suggests what additional documents are required.

---

## Repository Highlights

- `app.py`: Streamlit UI + resource bootstrap.
- `docstore_functions.py`: loaders, FAISS ingestion, deduplication, and retrieval helpers.
- `llm_and_crew_functions.py`: CrewAI agent definitions, local and web tools, and Groq LLM utilities.
- `docstore_notebook.ipynb` / `llm_and_crewai_notebook.ipynb`: interactive sandboxes for experimenting with ingestion and crew prompts.

---

## Troubleshooting

- **Missing FAISS index**: Run the ingestion step to recreate `vectorstore/`.
- **Credential errors**: Ensure JSON files exist and contain the correct key field names.
- **Rate limits / model errors**: Groq API sometimes responds with `429`; rerun after a short delay or lower parallel usage.
- **Streamlit cache issues**: Use the sidebar reload button (uncommented in `app.py`) or restart the app to clear stale resources.

---

## Contributing

Issues and pull requests are welcome. Please add tests or notebook snippets that showcase new capabilities, and document any new configuration options in this README.

---

## License

Distributed under the MIT License. See `LICENSE` for details.