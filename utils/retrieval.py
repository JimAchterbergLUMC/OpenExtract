"""
Dense retrieval utilities for semantic search.

This module provides the DenseRetriever class for performing semantic search
over text chunks using sentence transformers and cosine similarity.
"""

from typing import List, Optional, Tuple

import numpy as np

# Optional dependency handling
try:
    import torch  # type: ignore
except ImportError:
    torch = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:
    SentenceTransformer = None


# Global cache of models by name
_model_cache = {}


def get_sentence_transformer(model_name: str, device: str):
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name, device=device)
    return _model_cache[model_name]


class DenseRetriever:
    """
    Dense retrieval system using sentence transformers for semantic search.

    This class encodes text chunks into dense embeddings and provides
    efficient similarity-based retrieval using cosine similarity.
    """

    def __init__(
        self,
        chunks: List[str],
        model_name: str = "thenlper/gte-small",
        device: Optional[str] = None,
        batch_size: int = 64,
    ) -> None:
        """
        Initialize the dense retriever with text chunks.

        Args:
            chunks: List of text chunks to index for retrieval
            model_name: Name of the sentence transformer model to use
            device: Device to run the model on ('cpu', 'cuda', or None for auto)
            batch_size: Batch size for encoding operations
        Raises:
            RuntimeError: If sentence-transformers is not installed
        """
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
        # Auto-detect device if not specified
        if device is None:
            if torch is not None and torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = int(batch_size)
        self.chunks = chunks

        # Encode all chunks once during initialization
        self.embeddings = self._encode(chunks)
        self.embeddings = self._l2_normalize(self.embeddings)

    def _encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into embeddings.

        Args:
            texts: List of texts to encode

        Returns:
            Numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False,  # We normalize manually
            )
        )

    @staticmethod
    def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
        """
        L2 normalize embeddings for cosine similarity computation.

        Args:
            embeddings: Array of embeddings to normalize

        Returns:
            L2-normalized embeddings
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        return embeddings / norms

    def top_k(self, query: str, k: int) -> List[Tuple[int, float]]:
        """
        Retrieve the top-k most similar chunks for a given query.

        Args:
            query: Query string to search for
            k: Number of top results to return

        Returns:
            List of (chunk_index, similarity_score) tuples, sorted by score descending
        """
        # Encode and normalize the query
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        query_embedding = query_embedding / (
            np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-12
        )

        # Compute cosine similarities using dot product (since embeddings are normalized)
        similarities = self.embeddings @ query_embedding[0]

        # Get top-k indices
        k = min(k, len(similarities))
        top_indices = np.argpartition(-similarities, kth=k - 1)[:k]

        # Create (index, score) pairs and sort by score descending
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]
        return sorted(results, key=lambda x: x[1], reverse=True)
