from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from recalld.llm.context import estimate_tokens
from recalld.pipeline.align import LabelledTurn


@dataclass
class ChunkStrategy:
    strategy: str  # "single" or "map_reduce"
    chunks: list[list[LabelledTurn]]
    topic_count: int


def detect_topics(turns: list[LabelledTurn], threshold: float = 0.3) -> list[int]:
    """
    Use sentence-transformers to find semantic topic boundaries between turns.
    Returns list of turn indices where a new topic begins (index > 0).
    Falls back to empty list (no splits) if sentence-transformers unavailable.
    """
    if len(turns) < 2:
        return []
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        return []

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [t.text for t in turns]
    embeddings = model.encode(texts, show_progress_bar=False)

    boundaries = []
    for i in range(1, len(embeddings)):
        a = embeddings[i - 1]
        b = embeddings[i]
        # Cosine similarity
        sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        if sim < threshold:
            boundaries.append(i)
    return boundaries


def _turns_to_text(turns: list[LabelledTurn]) -> str:
    return "\n".join(f"{t.speaker}: {t.text}" for t in turns)


def _split_at_boundaries(turns: list[LabelledTurn], boundaries: list[int]) -> list[list[LabelledTurn]]:
    if not boundaries:
        return [turns]
    chunks = []
    prev = 0
    for b in sorted(boundaries):
        chunks.append(turns[prev:b])
        prev = b
    chunks.append(turns[prev:])
    return [c for c in chunks if c]


def _split_oversized_chunk(chunk: list[LabelledTurn], budget: int) -> list[list[LabelledTurn]]:
    """Split a chunk that exceeds budget at speaker turn boundaries."""
    result = []
    current: list[LabelledTurn] = []
    for turn in chunk:
        test = current + [turn]
        if estimate_tokens(_turns_to_text(test)) > budget and current:
            result.append(current)
            current = [turn]
        else:
            current = test
    if current:
        result.append(current)
    return result if result else [chunk]


def chunk_transcript(turns: list[LabelledTurn], token_budget: int, topic_threshold: float = 0.3) -> ChunkStrategy:
    full_text = _turns_to_text(turns)
    if estimate_tokens(full_text) <= token_budget:
        return ChunkStrategy(strategy="single", chunks=[turns], topic_count=1)

    boundaries = detect_topics(turns, threshold=topic_threshold)
    raw_chunks = _split_at_boundaries(turns, boundaries)

    final_chunks: list[list[LabelledTurn]] = []
    for chunk in raw_chunks:
        if estimate_tokens(_turns_to_text(chunk)) > token_budget:
            final_chunks.extend(_split_oversized_chunk(chunk, token_budget))
        else:
            final_chunks.append(chunk)

    return ChunkStrategy(
        strategy="map_reduce",
        chunks=final_chunks,
        topic_count=len(final_chunks),
    )
