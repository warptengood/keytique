# Keytique

> *To criticize your keys, muahahah!*

Keytique turns a musician's recording into **teacher-quality feedback**. You give it an
audio performance and the reference score (MIDI); it analyzes what was played against
what was written, retrieves relevant practice pedagogy, and asks an LLM to write
specific, actionable feedback — the kind a good teacher would give.

> ⚠️ **Work in progress.** This is an experimental research project, not a finished
> product. Interfaces, corpus, and models are expected to change. See
> [Roadmap](#roadmap).

---

## How it works

The system is built around a hard separation between **Music Information Extraction
(MIE)** and the **LLM pipeline**. They communicate only through a fixed data contract
(`AnalysisResult`), so either side can be swapped or tested independently.

```
audio.wav + reference.mid
        │
        ▼
┌───────────────────┐
│  MIE (basic-pitch │  audio → MIDI → note/timing comparison (mir_eval)
│   + mir_eval)     │
└───────────────────┘
        │  AnalysisResult  (raw: every missed/extra note, every timing deviation)
        ▼
┌───────────────────┐
│    Processor      │  aggregates raw data into pedagogically meaningful features
└───────────────────┘
        │  ProcessedAnalysisResult  (pitch-class histogram, error-time clusters,
        │                            timing tendency, …)
        ▼
┌───────────────────────────── LLM pipeline (2 stages) ────────────────────────────┐
│                                                                                  │
│  Stage 1 — Query builder:  ProcessedAnalysisResult -> LLM -> retrieval query     │
│                                                                                  │
│  RAG:  query -> embed -> Qdrant search -> top-k pedagogy excerpts (Context)      │
│                                                                                  │
│  Stage 2 — Feedback:  ProcessedAnalysisResult + Context + (user note) -> LLM     │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                                    feedback text
```

**Why two LLM calls?** The raw `AnalysisResult` is not a good retrieval query, so
Stage 1 turns the analysis into a natural-language query phrased to match the pedagogy
corpus. Stage 2 then writes the feedback grounded in both the analysis and the
retrieved material. The query-builder step can also run from a deterministic template
instead of the LLM — that comparison is part of ongoing evaluation.

### The data contract

- **`AnalysisResult`** (raw) — piece/instrument metadata, note accuracy (F1, precision,
  recall, full lists of missed/extra notes), and full timing deviations.
- **`ProcessedAnalysisResult`** (aggregated) — the LLM-facing view. Instead of hundreds
  of raw notes it carries derived features:
  - `most_missed_pitches` / `most_missed_pitch_classes` (the latter surfaces
    key-signature struggles, e.g. consistently missing F♯/C♯)
  - `error_time_clusters` — where in the piece mistakes bunch up
  - `timing` — `rushing` / `dragging` / `steady` tendency plus the worst deviations

---

## Tech stack

| Layer | Tool |
|---|---|
| Audio → MIDI | [`basic-pitch`](https://github.com/spotify/basic-pitch) (Spotify) |
| MIDI parsing | `pretty_midi` |
| Note/timing comparison | `mir_eval` |
| Vector DB | Qdrant (local, via Docker) |
| Embeddings | Hugging Face `transformers` (BGE / MiniLM / E5) |
| LLM gateway | [LiteLLM](https://github.com/BerriAI/litellm) (OpenAI, Ollama, …) |
| Schemas | Pydantic v2 |
| Package manager | [uv](https://github.com/astral-sh/uv) |

---

## Project structure

```
keytique/
├── src/
│   ├── mie/                     # Music Information Extraction (no LLM imports)
│   │   ├── basic_pitch_analyzer.py   # audio+midi → AnalysisResult
│   │   ├── processor.py              # AnalysisResult → ProcessedAnalysisResult
│   │   ├── schemas.py                # the data contract
│   │   └── tests/
│   ├── rag/
│   │   ├── indexer.py                # build the Qdrant index from the corpus
│   │   ├── retriever.py              # query → top-k pedagogy excerpts
│   │   ├── embedding_model.py        # pluggable HF embedding models
│   │   ├── corpus/{pitch,timing}/    # markdown pedagogy docs (with frontmatter)
│   │   └── schemas.py
│   └── llm/
│       ├── client.py                 # LiteLLM wrapper: query-builder + feedback
│       └── prompts/                  # query-builder.md, feedback.md
├── cli.py                       # end-to-end command-line entry point
└── pyproject.toml
```

---

## Setup

**Requirements:** Python ≥ 3.12, [uv](https://github.com/astral-sh/uv), Docker.

```bash
# 1. Install dependencies
uv sync

# 2. Configure environment
cp .env_example .env
# then edit .env with your keys:
#   HF_TOKEN=...          (Hugging Face — for downloading embedding models)
#   OPENAI_API_KEY=...    (if using openai/* providers)
#   OLLAMA_API_BASE=...   (e.g. http://localhost:11434, if using ollama/* providers)
```

### Start Qdrant (required)

> **You must have the Qdrant service running before using the pipeline.** The client
> connects to `http://localhost:6333`.

```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage" \
    qdrant/qdrant
```

---

## Usage

Use `cli.py` to test the initial end-to-end capability:

```bash
uv run python cli.py path/to/performance.mp3 path/to/reference.mid
```

Full options:

| Flag | Default | Description |
|---|---|---|
| `input` (positional) | — | Two paths: audio file, then MIDI file |
| `--method` | `basic_pitch` | MIE backend |
| `--embedding-model` | `bge-small-en-v1.5` | RAG embedding model (see below) |
| `--llm-provider` | `openai/gpt-4o` | Any LiteLLM provider, e.g. `ollama/qwen2.5:3b-instruct` |
| `--user-input` | `None` | Optional student note/question passed to the feedback stage |

Example with a local model:

```bash
uv run python cli.py perf.mp3 ref.mid \
    --llm-provider ollama/qwen2.5:3b-instruct \
    --embedding-model bge-small-en-v1.5 \
    --user-input "I struggled with the left hand — any tips?"
```

> Note: the CLI currently **rebuilds the Qdrant index on every run** (drops and
> re-indexes the corpus). Fine for experimentation; a persistent-index path is future
> work.

---

## Embedding models

Three Hugging Face embedding models are supported out of the box, selectable via
`--embedding-model`:

| Key | Model |
|---|---|
| `bge-small-en-v1.5` ⭐ | `BAAI/bge-small-en-v1.5` |
| `all-MiniLM-L6-v2` | `sentence-transformers/all-MiniLM-L6-v2` |
| `e5-small-v2` | `intfloat/e5-small-v2` |

**Finding:** among the three tested, **`bge-small-en-v1.5` performed best** and is the
default. Each model handles its own query/passage prefixing and pooling (CLS vs. mean).
Adding another Hugging Face embedding model is straightforward — subclass
`BasicEmbeddingModel` and register it in the `EMBEDDING_MODELS` map in
[`src/rag/embedding_model.py`](src/rag/embedding_model.py).

---

## RAG corpus

The pedagogy corpus lives in [`src/rag/corpus/`](src/rag/corpus/), split into two skill
areas — `pitch` and `timing`. Each document is a markdown file with YAML frontmatter
(title, source, license, `skill_area`, etc.); the retriever can optionally filter by
skill area.

> **The current corpus is small and not curated** — it exists purely for
> experimentation and to exercise the pipeline. It is meant to be expanded and cleaned
> up. Corpus documents retain the licenses of their original sources.

---

## Testing

```bash
uv run pytest                        # full suite
uv run pytest -k "not real_sample"   # skip the slow end-to-end basic-pitch test
```

---

## Notes

- The **RAG corpus is small and uncurated** — experimental only, and expandable.
- This is **not the final version.** Much more is planned.

---

## Roadmap

Upcoming / planned work:

- **More comprehensive audio analysis** (richer MIE features beyond pitch and onset timing)
- **Agentic workflow** with skills / tools orchestration
- **UI** for uploading recordings and viewing feedback
- **gRPC** service interface
- Baseline vs. fine-tuned LLM evaluation
