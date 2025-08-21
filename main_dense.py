from __future__ import annotations

#!/usr/bin/env python3
"""
Paper Q&A (per PDF) via OpenRouter — dense retrieval + per-question choices (ID-based)
-------------------------------------------------------------------------------------

Behavior
- If a question includes "choices", the model must output exactly one *choice ID* from that list.
  Supported forms:
    1) ["tabular","time-series","images","text","video","audio"]   (IDs are the strings)
    2) [{"id":"ts","label":"time-series"}, {"id":"img","label":"images"}, ...]
- If a question has no "choices", the model answers in concise free text.
- If evidence isn't in the retrieved context, the model must output: "Unknown from this paper".

Retrieval
- Dense semantic retrieval only (SentenceTransformers).
- Cosine similarity via L2-normalized embeddings.

Requirements:
  pip install pypdf tiktoken requests sentence-transformers numpy

Usage:
  export OPENROUTER_API_KEY=...  # or use --api-key-file path/to/key.txt
  python main.py \
    --papers-dir ./papers \
    --questions-file ./questions_free_text.json \
    --output-dir ./answers_free_text \
    --model "qwen/qwen2.5-vl-32b-instruct:free" \
    --chunk-tokens 800 --chunk-overlap 160 --top-k 3 \
    --dense-model "thenlper/gte-small" --dense-batch-size 8
"""

import argparse
import json
import os
import time
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Any, Dict, Union

import numpy as np
import requests

# Optional deps
try:
    import tiktoken  # type: ignore
except Exception:
    tiktoken = None  # fallback to char-based splitting

try:
    from pypdf import PdfReader  # type: ignore
except Exception:
    PdfReader = None

try:
    import torch  # type: ignore
except Exception:
    torch = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# ------------------ Data model ------------------

ChoiceType = Union[str, Dict[str, str]]


@dataclass
class Question:
    id: str
    text: str
    choices: Optional[List[ChoiceType]] = None  # strings or {"id":..., "label":...}


def load_questions(path: Path) -> List[Question]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    qs = []
    for q in raw.get("questions", []):
        qs.append(
            Question(
                id=q["id"],
                text=q["text"],
                choices=q.get("choices") or None,
            )
        )
    return qs


# ------------------ PDF extraction ------------------


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF using pypdf."""
    if PdfReader is None:
        raise RuntimeError(
            "pypdf is not installed. Run `pip install pypdf` (or use pdfplumber/pdfminer)."
        )
    reader = PdfReader(str(pdf_path))
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    text = "\n\n".join(pages)
    text = text.replace("\u00ad", "")  # soft hyphens
    return text


# ------------------ Chunking ------------------


def _split_by_tokens(
    text: str,
    chunk_tokens: int,
    overlap_tokens: int,
    model_name: str = "gpt-4o-mini",
) -> List[str]:
    if tiktoken is None:
        approx_chars = chunk_tokens * 4
        approx_overlap = overlap_tokens * 4
        chunks = []
        i = 0
        n = len(text)
        while i < n:
            end = min(n, i + approx_chars)
            chunks.append(text[i:end])
            if end == n:
                break
            i = max(i + approx_chars - approx_overlap, end)
        return chunks

    try:
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
        enc = tiktoken.get_encoding("o200k_base")
    ids = enc.encode(text)

    chunks = []
    i = 0
    n = len(ids)
    while i < n:
        end = min(n, i + chunk_tokens)
        chunk_ids = ids[i:end]
        chunks.append(enc.decode(chunk_ids))
        if end == n:
            break
        i = i + chunk_tokens - overlap_tokens
        if i <= 0:
            i = end
    return chunks


def chunk_text(
    text: str,
    chunk_tokens: int,
    chunk_overlap: int,
    model_name: str,
) -> List[str]:
    chunk_tokens = max(128, int(chunk_tokens))
    chunk_overlap = max(0, int(chunk_overlap))
    chunk_overlap = min(chunk_overlap, chunk_tokens // 2)
    return _split_by_tokens(text, chunk_tokens, chunk_overlap, model_name)


# ------------------ Dense retrieval ------------------


class DenseRetriever:
    def __init__(
        self,
        chunks: List[str],
        model_name: str = "thenlper/gte-small",
        device: Optional[str] = None,
        batch_size: int = 64,
    ):
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )
        if device is None:
            if torch is not None and torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = int(batch_size)
        self.chunks = chunks
        # Encode once per paper
        self.emb = self._encode(chunks)
        self.emb = self._l2norm(self.emb)  # L2 normalize for cosine via dot

    def _encode(self, texts: List[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False,  # we'll normalize ourselves
            )
        )

    @staticmethod
    def _l2norm(x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / n

    def top_k(self, query: str, k: int) -> List[Tuple[int, float]]:
        q = self.model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        sims = self.emb @ q[0]  # cosine similarity
        # Return top-k (idx, score)
        idx = np.argpartition(-sims, kth=min(k, len(sims) - 1))[:k]
        pairs = [(int(i), float(sims[i])) for i in idx]
        return sorted(pairs, key=lambda x: x[1], reverse=True)


# ------------------ Choice prep & parsing ------------------


def normalize_choices(choices: List[ChoiceType]) -> Tuple[List[str], Dict[str, str]]:
    """
    Returns:
        ids: list of canonical IDs the model must choose from
        id_to_label: optional label mapping (may be same as ids if choices were strings)
    """
    ids: List[str] = []
    id_to_label: Dict[str, str] = {}
    if not choices:
        return ids, id_to_label
    if isinstance(choices[0], str):
        ids = [str(c) for c in choices]
        id_to_label = {c: c for c in ids}
    else:
        for obj in choices:  # type: ignore[assignment]
            cid = str(obj.get("id"))
            lab = str(obj.get("label", cid))
            ids.append(cid)
            id_to_label[cid] = lab
    return ids, id_to_label


def extract_choice_id(ans: str, valid_ids: List[str]) -> Optional[str]:
    """Parse the model output and return a valid choice ID if present; else None."""
    if not ans:
        return None
    s = ans.strip()

    # Try fenced JSON
    fence = re.search(r"```json\s*(\{.*?\}|\s*\".*?\"\s*)\s*```", s, flags=re.DOTALL)
    if fence:
        block = fence.group(1).strip()
        try:
            j = json.loads(block)
            if isinstance(j, dict) and "id" in j and j["id"] in valid_ids:
                return j["id"]
            if isinstance(j, str) and j in valid_ids:
                return j
        except Exception:
            pass

    # Try direct JSON
    if s.startswith("{") or s.startswith('"'):
        try:
            j = json.loads(s)
            if isinstance(j, dict) and "id" in j and j["id"] in valid_ids:
                return j["id"]
            if isinstance(j, str) and j in valid_ids:
                return j
        except Exception:
            pass

    # Exact bare token
    if s in valid_ids:
        return s

    # Regex lookaround match for any ID
    if valid_ids:
        alt = "|".join(re.escape(v) for v in sorted(valid_ids, key=len, reverse=True))
        m = re.search(rf"(?<![A-Za-z0-9_])({alt})(?![A-Za-z0-9_])", s)
        if m:
            return m.group(1)

    return None


# ------------------ OpenRouter call ------------------


def call_openrouter(
    model: str,
    question: str,
    paper_title: str,
    context_chunks: List[str],
    api_key: str,
    referer: Optional[str] = None,
    x_title: Optional[str] = "Paper QA",
    choice_ids: Optional[List[str]] = None,
    id_to_label: Optional[Dict[str, str]] = None,
) -> str:
    context = "\n\n---\n\n".join(context_chunks)

    base_rules = (
        "You are a meticulous research assistant. Answer strictly from the provided paper excerpts (context). "
        "If the answer is not present in the context, reply exactly: 'Unknown from this paper'."
    )

    if choice_ids:
        if id_to_label:
            listed = "\n".join(
                f"- {cid} → {id_to_label.get(cid, cid)}" for cid in choice_ids
            )
            listing_block = f"Choices (id → label):\n{listed}"
        else:
            listing_block = "Choices (ids): " + ", ".join(choice_ids)

        classification_rules = (
            "\nYou must output exactly one of the listed IDs, with no extra text. "
            "If the answer is not present in the context, output exactly: Unknown from this paper."
        )
        system_prompt = base_rules + classification_rules
        user_suffix = "Return only the ID string. Do not include explanations."
        extra_content = f"\n\n{listing_block}\n"
    else:
        system_prompt = base_rules
        user_suffix = "Return only the answer (max ~12 words). Be concise."
        extra_content = ""

    user_prompt = (
        f"Paper: {paper_title}\n\n"
        f"Question: {question}\n\n"
        f"Context (selected excerpts):\n{context}\n"
        f"{extra_content}\n"
        f"{user_suffix}"
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if referer:
        headers["HTTP-Referer"] = referer
    if x_title:
        headers["X-Title"] = x_title

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        # No temperature/max_tokens per your constraints.
    }

    # JSON schema (many models on OpenRouter honor; safe to include, otherwise ignored)
    if choice_ids:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "single_choice",
                "schema": {
                    "type": "object",
                    "properties": {"id": {"type": "string", "enum": choice_ids}},
                    "required": ["id"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

    for attempt in range(3):
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            try:
                return data["choices"][0]["message"]["content"].strip()
            except Exception:
                return "Unknown from this paper"
        time.sleep(1.5 * (attempt + 1))
    return "Unknown from this paper"


# ------------------ Orchestration ------------------


def answer_questions_for_paper(
    pdf_path: Path,
    questions: List[Question],
    model: str,
    api_key: str,
    chunk_tokens: int,
    chunk_overlap: int,
    top_k: int,
    referer: Optional[str],
    dense_model: str,
    dense_device: Optional[str],
    dense_batch_size: int,
) -> List[dict]:
    raw_text = extract_pdf_text(pdf_path)
    if not raw_text.strip():
        return [
            {"id": q.id, "question": q.text, "answer": "Unknown from this paper"}
            for q in questions
        ]

    chunks = chunk_text(raw_text, chunk_tokens, chunk_overlap, model_name=model)

    # Build dense retriever once per paper
    retriever = DenseRetriever(
        chunks=chunks,
        model_name=dense_model,
        device=dense_device,
        batch_size=dense_batch_size,
    )

    answers = []
    title = pdf_path.name
    for q in questions:
        print(f"Processing question: {q.id}")
        idx_scores = retriever.top_k(q.text, k=top_k)
        selected = [chunks[idx] for idx, _ in idx_scores]

        # Prepare per-question choices
        q_choice_ids: Optional[List[str]] = None
        q_id_to_label: Optional[Dict[str, str]] = None
        if q.choices and len(q.choices) > 0:
            ids, id_to_label = normalize_choices(q.choices)
            if ids:
                q_choice_ids = ids
                q_id_to_label = id_to_label

        raw_answer = call_openrouter(
            model=model,
            question=q.text,
            paper_title=title,
            context_chunks=selected,
            api_key=api_key,
            referer=referer,
            x_title="Paper QA",
            choice_ids=q_choice_ids,
            id_to_label=q_id_to_label,
        )

        if q_choice_ids:
            cid = extract_choice_id(raw_answer, q_choice_ids)
            final_answer = cid if cid is not None else "Unknown from this paper"
        else:
            final_answer = raw_answer  # free text

        record = {
            "id": q.id,
            "question": q.text,
            "answer": final_answer,
            "chunks_used": [idx for idx, _ in idx_scores],
        }
        if q_choice_ids:
            record["raw_answer"] = raw_answer  # auditing
            record["choices_ids"] = q_choice_ids
            if q_id_to_label and final_answer in q_id_to_label:
                record["answer_label"] = q_id_to_label[final_answer]
        answers.append(record)

    return answers


def main():
    parser = argparse.ArgumentParser(
        description="Answer JSON questions per PDF using OpenRouter (dense retrieval)."
    )
    parser.add_argument(
        "--papers-dir", required=True, type=Path, help="Directory with PDF files."
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Directory to write answers."
    )
    parser.add_argument(
        "--questions-file", required=True, type=Path, help="JSON with questions."
    )
    parser.add_argument(
        "--model",
        default="qwen/qwen2.5-vl-32b-instruct:free",
        help="OpenRouter model id.",
    )
    parser.add_argument(
        "--top-k", type=int, default=3, help="# of chunks to retrieve per question."
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=800,
        help="Approx tokens per chunk (or char/4).",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=160, help="Overlap tokens between chunks."
    )
    parser.add_argument(
        "--referer", default=None, help="Optional HTTP-Referer header for OpenRouter."
    )
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=None,
        help="Optional file with the OpenRouter API key. If not provided, uses OPENROUTER_API_KEY env var.",
    )
    # Dense retrieval params
    parser.add_argument(
        "--dense-model",
        default="thenlper/gte-small",
        help="SentenceTransformers embedding model.",
    )
    parser.add_argument(
        "--dense-device",
        default=None,
        help="Force device for dense model: 'cpu' or 'cuda'. Default: auto.",
    )
    parser.add_argument(
        "--dense-batch-size",
        type=int,
        default=8,
        help="Batch size for encoding chunks.",
    )

    args = parser.parse_args()

    # API key
    if args.api_key_file:
        api_key = args.api_key_file.read_text(encoding="utf-8").strip()
    else:
        api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Provide OpenRouter API key via --api-key-file or OPENROUTER_API_KEY env var."
        )

    # Output dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Questions
    questions = load_questions(args.questions_file)

    # Papers
    papers = sorted(
        [p for p in args.papers_dir.iterdir() if p.suffix.lower() == ".pdf"]
    )
    if not papers:
        raise RuntimeError(f"No PDFs found in {args.papers_dir}")

    for pdf in papers:
        print(f"Processing: {pdf.name}")
        answers = answer_questions_for_paper(
            pdf_path=pdf,
            questions=questions,
            model=args.model,
            api_key=api_key,
            chunk_tokens=args.chunk_tokens,
            chunk_overlap=args.chunk_overlap,
            top_k=args.top_k,
            referer=args.referer,
            dense_model=args.dense_model,
            dense_device=args.dense_device,
            dense_batch_size=args.dense_batch_size,
        )

        clean_name = pdf.stem.replace(" ", "_").replace("-", "_")
        clean_name = "".join(c for c in clean_name if c.isalnum() or c in "._-")
        output_file = args.output_dir / f"{clean_name}_answers.json"

        result = {"paper": pdf.name, "answers": answers}

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Wrote {output_file}")


if __name__ == "__main__":
    main()
