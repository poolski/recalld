# Testing

Default test runs must not depend on external local services such as LM Studio.

## Default suite

Run:

```bash
uv run pytest
```

The default suite uses mocks for:

- LM Studio chat requests
- LLM model metadata requests
- post-processing client calls

## Opt-in audio fixture checks

The fixture-backed transcription smoke test for `tests/fixtures/conversation.wav` is intentionally excluded from the default suite because it depends on local model execution.

To run it explicitly:

```bash
RECALLD_RUN_AUDIO_FIXTURES=1 uv run pytest tests/pipeline/test_conversation_fixture.py
```
