# OpenExtract

OpenExtract is an open-source **retrieval-augmented generation (RAG)** pipeline for structured data extraction from scientific PDFs. It retrieves the most relevant text chunks for each question and queries a large language model (via [OpenRouter](https://openrouter.ai/)) to produce answers. The default sample use case is data extraction for a systematic literature review in digital health, but the pipeline is use-case agnostic: define your own questions and prompts.

---

## Looking for the paper code?

> **This branch is under active development (OpenExtract 2.0).**  
> If you want the exact codebase associated with the published paper, use the frozen release below.

| Resource | Link |
| --- | --- |
| **Paper** | [OpenExtract: Automated Data Extraction for Systematic Reviews in Health](https://doi.org/10.3233/shti260316) (DOI: `10.3233/shti260316`) |
| **Code release for the paper** | [`openextract-mie-2025`](https://github.com/JimAchterbergLUMC/OpenExtract/releases/tag/openextract-mie-2025) |
| **Full commit history of that snapshot** | [commits on `openextract-mie-2025`](https://github.com/JimAchterbergLUMC/OpenExtract/commits/openextract-mie-2025) |

Checkout the paper snapshot:

```bash
git clone https://github.com/JimAchterbergLUMC/OpenExtract.git
cd OpenExtract
git checkout openextract-mie-2025
```

Paper reproduction instructions for that snapshot are in its own [README](https://github.com/JimAchterbergLUMC/OpenExtract/blob/openextract-mie-2025/README.md).

---

## What this repository contains (current `main`)

The current branch improves on the paper release with a focus on parsing quality, chunking that respects the embedder’s context window, and a disposable content-addressed cache:

| Area | Paper release (`openextract-mie-2025`) | Current `main` (2.0, WIP) |
| --- | --- | --- |
| PDF parsing | `pypdf` plain text | Default **pymupdf4llm** (layout-aware Markdown); optional **docling**; `pypdf` fallback |
| Chunking | Fixed token windows (`tiktoken`), default 1000 / overlap 500 | Section-aware packing; defaults **300 / 50**, capped to the embedder’s max length |
| Cache | None | Content-addressed `cache/` (parsed text, chunks, embeddings) |
| Questions | Multiple-choice | Multiple-choice **and** free-text (open questions) |

---

## How it works

1. **Parse** each PDF into text (Markdown when using pymupdf/docling).
2. **Clean** boilerplate (headers/footers) and optionally strip noisy reference sections.
3. **Chunk** by section boundaries, packing sentences within a section; tables stay atomic when possible.
4. **Embed** chunks with a Hugging Face sentence-transformers model (default: `neuml/pubmedbert-base-embeddings`).
5. **Retrieve** the top-k chunks for each question via dense similarity.
6. **Ask** an OpenRouter LLM, constrained by the retrieved context and your prompts.
7. **Write** one JSON answer file per paper.

Artifacts from steps 1–4 are cached under `--cache-dir` so re-runs with the same PDF and config skip recompute.

---

## Requirements

- Python 3.10+ recommended
- An [OpenRouter](https://openrouter.ai/) API key
- Dependencies in `requirements.txt` (installs `pymupdf4llm`, `sentence-transformers`, etc.)

Optional high-fidelity parser:

```bash
pip install docling   # then run with --parser docling
```

---

## Quick start

```bash
git clone https://github.com/JimAchterbergLUMC/OpenExtract.git
cd OpenExtract
pip install -r requirements.txt
```

Put your OpenRouter key in a text file (e.g. `key.txt`) or export `OPENROUTER_API_KEY`.

Place PDFs in a directory (e.g. `./papers`), then:

```bash
python main.py \
  --papers-dir ./papers \
  --output-dir ./answers \
  --questions-file ./questions.json \
  --model qwen/qwen-2.5-7b-instruct \
  --dense-model neuml/pubmedbert-base-embeddings \
  --top-k 3 \
  --chunk-tokens 300 \
  --chunk-overlap 50 \
  --api-key-file ./key.txt
```

Each paper produces `{paper_stem}_answers.json` in the output directory, containing the paper name, answers, and runtime.

### Try it with bundled dev examples

The repository ships a small shared corpus and pre-run outputs so you can inspect the pipeline without preparing your own PDFs first:

```bash
python main.py \
  --papers-dir ./papers_dev \
  --output-dir ./answers \
  --questions-file ./questions_uncan.json \
  --model qwen/qwen-2.5-7b-instruct \
  --api-key-file ./key.txt
```

Compare your fresh output against the reference files in `answers_dev/` (see [Development and troubleshooting](#development-and-troubleshooting)).

---

## Configuring a use case

### Questions (`questions.json`)

Questions are a JSON object with a `"questions"` array. Each item has `id` and `text`. A `choices` array is optional:

- **Closed (multiple-choice):** include `"choices"`. The model is instructed to return a JSON array of option IDs from that list (e.g. `["Descriptive: ..."]`). Answers are normalized to valid choice IDs; unparseable replies become `"Unknown from this paper"`.
- **Open (free-text):** omit `"choices"`. The model returns an answer based only on the retrieved context.

Example:

```json
{
  "questions": [
    {
      "id": "Q1",
      "text": "Which data type is used in this study?",
      "choices": ["Tabular", "Time-series", "Images", "Text", "Video", "Audio", "Multi-modal"]
    },
    {
      "id": "Q3",
      "text": "In what way is the cohort described in this study?"
    }
  ]
}
```

Bundled question sets:

| File | Role |
| --- | --- |
| `questions.json` | Full digital-health SLR extraction schema (MIE2026 paper use case) |
| `questions_dev.json` | Mixed closed/open questions for experimentation |

#### How open and closed questions differ at runtime

Both question types share the same retrieval step (top-k dense similarity over section-aware chunks). They diverge only in the LLM call and post-processing:

| Aspect | Closed (with `choices`) | Open (no `choices`) |
| --- | --- | --- |
| Prompt | `prompts.json` → `user` | `prompts.json` → `user_open` |
| Expected model output | JSON array of choice IDs | Plain prose |
| `answer` field in output JSON | List of resolved choice IDs | Free-text string |
| `answer_label` | Human-readable labels for chosen IDs | `null` |
| `choices_ids` | The valid option list | `null` |
| Structured output (`--use-structured-output`) | Supported (model-dependent) | Not used |

For open questions, `raw_answer` is kept verbatim (same as `answer` unless you post-process externally). The open prompt explicitly tells the model not to invent option letters and to reply `Unknown from this paper` when the context is insufficient.

### Prompts (`prompts.json`)

Edit `prompts.json` to change the system (`base`) and user instructions the LLM sees:

- `base` — shared system prompt for all questions
- `user` — instructions for closed questions (JSON array of choice IDs)
- `user_open` — instructions for open questions (concise prose, context-only)

---

## CLI reference

| Argument | Default | Description |
| --- | --- | --- |
| `--papers-dir` | `./papers` | Directory of PDF files |
| `--output-dir` | `./answers` | Where answer JSON files are written |
| `--questions-file` | `./questions.json` | Questions JSON |
| `--model` | `openai/gpt-oss-120b:free` | OpenRouter model id |
| `--parser` | `pymupdf` | `pymupdf`, `docling`, or `pypdf` |
| `--cache-dir` | `./cache` | Content-addressed cache root |
| `--no-cache` | off | Ignore existing cache (still writes new artifacts) |
| `--top-k` | `3` | Chunks retrieved per question |
| `--chunk-tokens` | `300` | Max tokens per chunk (embedder tokenizer; capped at model max length) |
| `--chunk-overlap` | `50` | Approximate overlap between consecutive chunks |
| `--dense-model` | `neuml/pubmedbert-base-embeddings` | Hugging Face sentence-transformers id |
| `--dense-device` | auto | `cpu` or `cuda` |
| `--dense-batch-size` | `8` | Embedding batch size |
| `--api-key-file` | — | File containing the OpenRouter key (else `OPENROUTER_API_KEY`) |
| `--referer` | — | Optional `HTTP-Referer` for OpenRouter |
| `--use-structured-output` | off | Force OpenRouter structured outputs (model-dependent) |
| `--random-subset` | — | Randomly select N papers |
| `--random-seed` | `42` | Seed for `--random-subset` |
| `--stop-after-n-papers` | — | Process at most N papers (after selection) |

### Cache (`cache/`)

Parsing, chunking, and embedding are the slow steps. OpenExtract caches their results under `--cache-dir` (default `./cache`) so re-runs with the same PDF and configuration skip recompute. The cache is **disposable**: delete it anytime; the next run will rebuild what it needs.

**What is cached**

| Artifact | Purpose |
| --- | --- |
| `meta.json` | Provenance: source filename, content hash, parser metadata |
| `text.{parser}.md` | Cleaned Markdown/plain text after PDF parsing |
| `chunks.jsonl` | One JSON object per line — the best file to inspect chunk quality |
| `embeddings.npy` | Float32 matrix; row *i* corresponds to chunk *i* |
| `config.json` | Exact chunk/parser/embedder settings that produced the index |

**How keys work**

- **`paper_id`** — first 16 hex characters of the SHA-256 hash of the PDF bytes. Renaming a file does not change its cache entry; changing the file content does.
- **`index_key`** — human-readable directory name encoding parser version, chunk token/overlap settings, and embedder name (e.g. `pymupdf4llm-1.28.0_t300_o50_neuml-pubmedbert-base-embeddings`). A config change creates a new subdirectory instead of silently reusing stale chunks or embeddings.

**Inspecting the cache**

```bash
python -m pipeline.paper_store ./cache              # list cached papers
python -m pipeline.paper_store ./cache/{paper_id}   # detail + chunk preview
```

Each answer JSON also records `paper_id`, `chunks_id`, `chunks_str`, `chunks_section`, and `chunks_score`, so you can trace a given answer back to the exact cached chunks that were retrieved.

Cache layout:

```
cache/
└── {paper_id}/
    ├── meta.json
    ├── text.{parser}.md
    └── {index_key}/
        ├── config.json
        ├── chunks.jsonl
        └── embeddings.npy
```

Use `--no-cache` to force a full rebuild while still writing fresh artifacts (useful when debugging parsing or chunking). The `cache/` directory itself is gitignored; only your local runs populate it.

---

## Development and troubleshooting

Two tracked directories provide a shared mini-corpus and reference outputs for local development:

| Directory | Contents |
| --- | --- |
| `papers_dev/` | Three example PDFs (digital-health papers used during 2.0 development) |
| `answers_dev/` | Reference `{paper_stem}_answers.json` files produced from those PDFs with `questions_uncan.json` |

**Typical workflow**

1. Run against `papers_dev/` and `questions_uncan.json` (see [Quick start](#try-it-with-bundled-dev-examples)).
2. Open the matching file in `answers_dev/` and compare answers, retrieved chunks (`chunks_str`), and scores (`chunks_score`).
3. If retrieval looks wrong, inspect `cache/{paper_id}/…/chunks.jsonl` to judge whether the problem is parsing, chunking, or the retriever/LLM.
4. If parsing or chunking is the issue, adjust `--parser`, `--chunk-tokens`, or `--chunk-overlap`, or use `--no-cache` and re-run.

The dev answer files include both closed questions (lists in `answer`, populated `choices_ids`) and open questions (prose in `answer`, `choices_ids: null`), making them useful for validating either path.


## Citation

If you use OpenExtract in academic work, please cite:

> Achterberg, J., van Dijk, B., Meng, J., et al. (2026). *OpenExtract: Automated Data Extraction for Systematic Reviews in Health*. Studies in Health Technology and Informatics. https://doi.org/10.3233/shti260316

```bibtex
@inproceedings{achterberg2026openextract,
  title     = {OpenExtract: Automated Data Extraction for Systematic Reviews in Health},
  author    = {Achterberg, Jim and van Dijk, Bram and Meng, Jing and others},
  booktitle = {Studies in Health Technology and Informatics},
  year      = {2026},
  doi       = {10.3233/shti260316},
  url       = {https://doi.org/10.3233/shti260316}
}
```

**Code corresponding to that paper:** [`openextract-mie-2025` release](https://github.com/JimAchterbergLUMC/OpenExtract/releases/tag/openextract-mie-2025).

---

## License and acknowledgements

OpenExtract is developed in the context of digital-health systematic review research (Leiden University Medical Center and collaborators). Some components like the embedding model and LLM API are third-party services/models subject to their own terms.
