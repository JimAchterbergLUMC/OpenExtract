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

Paper reproduction instructions for that snapshot are in its own [README](https://github.com/JimAchterbergLUMC/OpenExtract/blob/openextract-mie-2025/README.md) and are summarized again in [Reproducing the paper results](#reproducing-the-paper-results) below.

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

---

## Configuring a use case

### Questions (`questions.json`)

Questions are a JSON object with a `"questions"` array. Each item has `id` and `text`. A `choices` array is optional:

- **Closed (multiple-choice):** include `"choices"`. The model returns option IDs / labels from that list.
- **Open (free-text):** omit `"choices"`. The model returns a free-text answer.

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
| `questions.json` | Full digital-health SLR extraction schema (paper use case) |
| `questions_brief.json` | Shorter variant for quicker runs |
| `questions_uncan.json` | Mixed closed/open questions for experimentation |

### Prompts (`prompts.json`)

Edit `prompts.json` to change the system (`base`) and user instructions the LLM sees. Defaults steer the model toward careful, context-only answers and JSON-shaped choice lists for closed questions.

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

Inspect the cache:

```bash
python -m pipeline.paper_store ./cache
python -m pipeline.paper_store ./cache/{paper_id}
```

Cache layout (one directory per PDF content hash):

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

The index key encodes parser version, chunk settings, and embedder name, so config changes never silently reuse stale embeddings.

---

## Reproducing the paper results

Use the **frozen** tag, not necessarily current `main`:

```bash
git checkout openextract-mie-2025
pip install -r requirements.txt
```

Then run with the paper settings (chunk sizes and defaults differ from 2.0):

```bash
python main.py \
  --papers-dir {PAPERS} \
  --output-dir {OUTPUTS} \
  --questions-file {QUESTIONS} \
  --model {MODEL} \
  --dense-model neuml/pubmedbert-base-embeddings \
  --top-k 3 \
  --chunk-tokens 1000 \
  --chunk-overlap 500 \
  --api-key-file {KEY} \
  --random-subset 50 \
  --random-seed 42 \
  --stop-after-n-papers 10
```

Replace `{MODEL}` with an OpenRouter model used in the paper (e.g. larger Qwen or DeepSeek variants). See the paper for reported precision/recall.

### Manual extraction and evaluation

- Excel macros and researcher annotations: [`manual_extraction/`](manual_extraction/)
- Scoring notebook (Cohen’s kappa, precision/recall): [`evaluation.ipynb`](evaluation.ipynb)

Follow the notebook to build a manual-extraction workbook and to compare OpenExtract outputs to human labels as described in the paper.

---

## Repository layout

```
OpenExtract/
├── main.py                 # CLI entry point
├── pipeline/               # Parse → clean → chunk → retrieve → answer
│   ├── pdf_parsing.py
│   ├── text_cleaning.py
│   ├── section_chunking.py
│   ├── paper_store.py
│   ├── retrieval.py
│   ├── qa_orchestrator.py
│   └── ...
├── questions.json          # Example closed-question SLR schema
├── prompts.json            # LLM system/user prompts
├── evaluation.ipynb        # Manual extraction + paper metrics
├── manual_extraction/      # Human labels and Excel macros
├── papers/                 # Sample / working PDF corpus (local)
├── cache/                  # Disposable parse/chunk/embedding cache
├── answers_*/              # Example or run outputs
└── devdocs/                # Design notes for 2.0 work
```

---

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

OpenExtract is developed in the context of digital-health systematic review research (Leiden University Medical Center and collaborators). The default embedding model and LLMs are third-party services/models subject to their own terms. Note that **pymupdf** is AGPL-licensed; that is fine for research and open-source use—check licensing if you redistribute commercially.
