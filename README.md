# myfirstRAG 🔎🧠

A small **local RAG** (Retrieval-Augmented Generation) app: it reads the PDFs in `data/`
and answers questions about them from the terminal. Everything runs on your machine —
retrieval with **ChromaDB**, orchestration with **LangChain**, and generation with a
local LLM served by **Ollama**.

## How it works

```
PDFs ──load──▶ chunks ──embeddings──▶ ChromaDB ──retrieve top-k──▶ prompt + LLM ──▶ answer
```

1. **Load & split** — `PyPDFLoader` reads every PDF in `data/`; the text is cut into chunks.
2. **Index** — each chunk is embedded (`all-MiniLM-L6-v2`) and stored in a local ChromaDB (`chroma_db/`).
3. **Retrieve** — for each question, the most similar chunks are fetched.
4. **Generate** — the chunks are passed as context to the LLM (`qwen3:8b` via Ollama),
   which answers *only* from that context (and says `I don't know` otherwise).

## Project structure

```
myfirstRAG/
├── app.py             # entry point: LLM, prompt, RAG chain and interactive loop
├── functions.py       # load/split, ChromaDB + retriever, helpers
├── requirements.txt   # Python dependencies
├── Dockerfile         # image recipe
├── compose.yaml       # containerized run (mounts data/, wires Ollama)
├── .dockerignore
└── data/              # the source PDFs
    ├── visitor_guide.pdf
    └── history_and_rules.pdf
```
> `chroma_db/` is generated on first run (the vector index).

## Requirements

- **Python 3.9+**
- **[Ollama](https://ollama.com)** installed and running, with the model pulled:
  ```bash
  ollama pull qwen3:8b
  ```
- Some PDFs in `data/`.

## Run locally (virtual environment)

```bash
python -m venv .venv
# Windows:  .\.venv\Scripts\Activate.ps1     |  Linux/Mac:  source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Run with Docker

The app is interactive, so use `run` (not `up`) to get a console:

```bash
docker compose run --rm rag
```

Because the LLM (Ollama) runs on the **host**, the container reaches it via
`OLLAMA_URL=http://host.docker.internal:11434` (already set in `compose.yaml`).
If you get a connection error, make the host's Ollama listen on all interfaces:

```bash
# on the host, then restart Ollama
# Windows (PowerShell):  $env:OLLAMA_HOST = "0.0.0.0:11434"
# Linux/Mac:             export OLLAMA_HOST=0.0.0.0:11434
```

## Configuration (environment variables)

| Variable        | Default                         | Meaning                                  |
|-----------------|---------------------------------|------------------------------------------|
| `CHUNK_SIZE`    | `500`                           | Characters per chunk                     |
| `CHUNK_OVERLAP` | `100`                           | Overlap between consecutive chunks       |
| `K`             | `4`                             | Number of chunks retrieved per question  |
| `TEMPERATURE`   | `0`                             | LLM temperature (0 = deterministic)      |
| `OLLAMA_URL`    | `http://localhost:11434`        | Ollama server URL                        |

## Usage

```
Welcome to the RAG system! Type your question or 'exit' to quit.

How much is an adult ticket?
An adult ticket costs 39.90 euros.

Sources: data/visitor_guide.pdf p.1
--------------------------------------------------
```

Type `exit` to quit.

## Notes

- **Re-indexing:** the index is built only if `chroma_db/` doesn't exist. If you add or
  change a PDF in `data/`, **delete `chroma_db/`** so it rebuilds with the new content.
- **Model cache:** the embedding model is downloaded on the first run and cached in
  `hf_cache/` (mounted at `/root/.cache/huggingface`), so it isn't re-downloaded on later runs.
- **`qwen3:8b`** is a reasoning model; if answers include `<think>...</think>` blocks,
  add `/no_think` to the prompt or switch to `qwen2.5:7b` / `llama3.1:8b`.
