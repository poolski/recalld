from recalld.pipeline.align import LabelledTurn
from recalld.llm.chunking import detect_topics, chunk_transcript, ChunkStrategy


def _turns(texts: list[str]) -> list[LabelledTurn]:
    turns = []
    t = 0.0
    for i, text in enumerate(texts):
        turns.append(LabelledTurn(speaker=f"SPEAKER_0{i%2}", start=t, end=t+10, text=text))
        t += 10
    return turns


def test_chunk_strategy_single_when_fits():
    turns = _turns(["hello world"] * 5)
    strategy = chunk_transcript(turns, token_budget=10000)
    assert strategy.strategy == "single"
    assert len(strategy.chunks) == 1


def test_chunk_strategy_map_reduce_when_over_budget():
    # Create enough turns to exceed a tiny budget
    turns = _turns(["word " * 50] * 20)  # ~1000 words total
    strategy = chunk_transcript(turns, token_budget=100)
    assert strategy.strategy == "map_reduce"
    assert len(strategy.chunks) > 1


def test_each_chunk_within_budget():
    turns = _turns(["word " * 100] * 10)
    strategy = chunk_transcript(turns, token_budget=200)
    for chunk in strategy.chunks:
        from recalld.llm.context import estimate_tokens
        text = "\n".join(f"{t.speaker}: {t.text}" for t in chunk)
        assert estimate_tokens(text) <= 200


def test_detect_topics_returns_boundaries():
    # Without real embeddings, just verify it returns a list of ints
    turns = _turns(["hello world", "goodbye moon", "hello again"])
    boundaries = detect_topics(turns, threshold=0.99)  # high threshold = many boundaries
    assert isinstance(boundaries, list)
    assert all(isinstance(b, int) for b in boundaries)
