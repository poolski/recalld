from types import ModuleType, SimpleNamespace
import sys
import wave
from array import array

from recalld.pipeline.diarise import diarise


class _FakeSegment:
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end


class _FakeDiarizeOutput:
    def __init__(self):
        self.speaker_diarization = [
            {"segment": _FakeSegment(0.0, 1.25), "speaker": "SPEAKER_00"},
            {"segment": _FakeSegment(1.25, 2.5), "speaker": "SPEAKER_01"},
        ]


class _FakePipeline:
    def to(self, device):
        return self

    def __call__(self, wav_path: str):
        return _FakeDiarizeOutput()


def test_diarise_accepts_output_without_itertracks(monkeypatch, tmp_path):
    fake_torch = ModuleType("torch")
    fake_torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    fake_torch.device = lambda value: value

    fake_pyannote = ModuleType("pyannote.audio")
    fake_pyannote.Pipeline = type(
        "Pipeline",
        (),
        {"from_pretrained": staticmethod(lambda model, token: _FakePipeline())},
    )

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_pyannote)

    wav_path = tmp_path / "sample.wav"
    _write_pcm_wav(wav_path, duration_seconds=1.0)

    turns = diarise(wav_path, "token")

    assert [(turn.start, turn.end, turn.speaker) for turn in turns] == [
        (0.0, 1.25, "SPEAKER_00"),
        (1.25, 2.5, "SPEAKER_01"),
    ]


class _FakeMixedDiarizeOutput:
    def __init__(self):
        self.speaker_diarization = [
            {"segment": _FakeSegment(0.0, 1.0), "speaker": "SPEAKER_00"},
            {"segment": _FakeSegment(1.0, 2.0)},
            {"speaker": "SPEAKER_01"},
            {"segment": _FakeSegment(2.0, 3.0), "speaker": "SPEAKER_01"},
        ]


def test_diarise_skips_incomplete_speaker_turns(monkeypatch, tmp_path):
    fake_torch = ModuleType("torch")
    fake_torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    fake_torch.device = lambda value: value

    fake_pyannote = ModuleType("pyannote.audio")
    fake_pyannote.Pipeline = type(
        "Pipeline",
        (),
        {"from_pretrained": staticmethod(lambda model, token: _FakePipeline())},
    )

    class _MixedPipeline(_FakePipeline):
        def __call__(self, wav_path: str):
            return _FakeMixedDiarizeOutput()

    fake_pyannote.Pipeline = type(
        "Pipeline",
        (),
        {"from_pretrained": staticmethod(lambda model, token: _MixedPipeline())},
    )

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_pyannote)

    wav_path = tmp_path / "sample.wav"
    _write_pcm_wav(wav_path, duration_seconds=1.0)

    turns = diarise(wav_path, "token")

    assert [(turn.start, turn.end, turn.speaker) for turn in turns] == [
        (0.0, 1.0, "SPEAKER_00"),
        (2.0, 3.0, "SPEAKER_01"),
    ]


class _FakeAnnotation:
    def __init__(self, tracks):
        self._tracks = tracks

    def itertracks(self, yield_label=False):
        for segment, track, speaker in self._tracks:
            if yield_label:
                yield segment, track, speaker
            else:
                yield segment, track


class _FakeStructuredDiarizeOutput:
    def __init__(self):
        self.speaker_diarization = _FakeAnnotation([
            (_FakeSegment(0.0, 1.0), "track-0", "SPEAKER_00"),
        ])
        self.exclusive_speaker_diarization = _FakeAnnotation([
            (_FakeSegment(0.0, 0.8), "track-0", "SPEAKER_00"),
            (_FakeSegment(0.8, 1.7), "track-1", "SPEAKER_01"),
        ])


def test_diarise_uses_exclusive_annotation_from_diarize_output(monkeypatch, tmp_path):
    fake_torch = ModuleType("torch")
    fake_torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    fake_torch.device = lambda value: value

    class _StructuredPipeline(_FakePipeline):
        def __call__(self, wav_path: str):
            return _FakeStructuredDiarizeOutput()

    fake_pyannote = ModuleType("pyannote.audio")
    fake_pyannote.Pipeline = type(
        "Pipeline",
        (),
        {"from_pretrained": staticmethod(lambda model, token: _StructuredPipeline())},
    )

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_pyannote)

    wav_path = tmp_path / "sample.wav"
    _write_pcm_wav(wav_path, duration_seconds=1.0)

    turns = diarise(wav_path, "token")

    assert [(turn.start, turn.end, turn.speaker) for turn in turns] == [
        (0.0, 0.8, "SPEAKER_00"),
        (0.8, 1.7, "SPEAKER_01"),
    ]


def _write_pcm_wav(path, *, duration_seconds: float, sample_rate: int = 16_000, amplitude: int = 12_000):
    total_frames = int(duration_seconds * sample_rate)
    samples = array("h", [amplitude] * total_frames)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())


def test_diarise_chunks_long_audio_and_offsets_turns(monkeypatch, tmp_path):
    fake_torch = ModuleType("torch")
    fake_torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    fake_torch.device = lambda value: value

    class _ChunkedDiarizeOutput:
        def __init__(self):
            self.speaker_diarization = [
                {"segment": _FakeSegment(0.0, 1.25), "speaker": "SPEAKER_00"},
            ]

    class _ChunkedPipeline(_FakePipeline):
        def __init__(self):
            self.calls: list[str] = []

        def __call__(self, wav_path: str):
            self.calls.append(wav_path)
            return _ChunkedDiarizeOutput()

    pipeline = _ChunkedPipeline()

    fake_pyannote = ModuleType("pyannote.audio")
    fake_pyannote.Pipeline = type(
        "Pipeline",
        (),
        {"from_pretrained": staticmethod(lambda model, token: pipeline)},
    )

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_pyannote)
    monkeypatch.setattr("recalld.pipeline.diarise.MAX_CHUNK_SECONDS", 2.0)

    wav_path = tmp_path / "sample.wav"
    _write_pcm_wav(wav_path, duration_seconds=5.0)

    turns = diarise(wav_path, "token")

    assert len(pipeline.calls) == 3
    assert all(call != str(wav_path) for call in pipeline.calls)
    assert [(turn.start, turn.end, turn.speaker) for turn in turns] == [
        (0.0, 1.25, "SPEAKER_00"),
        (2.0, 3.25, "SPEAKER_00"),
        (4.0, 5.25, "SPEAKER_00"),
    ]


def test_diarise_emits_progress_messages_for_chunked_audio(monkeypatch, tmp_path):
    fake_torch = ModuleType("torch")
    fake_torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    fake_torch.device = lambda value: value

    class _ChunkedDiarizeOutput:
        def __init__(self):
            self.speaker_diarization = [
                {"segment": _FakeSegment(0.0, 1.25), "speaker": "SPEAKER_00"},
            ]

    class _ChunkedPipeline(_FakePipeline):
        def __call__(self, wav_path: str):
            return _ChunkedDiarizeOutput()

    fake_pyannote = ModuleType("pyannote.audio")
    fake_pyannote.Pipeline = type(
        "Pipeline",
        (),
        {"from_pretrained": staticmethod(lambda model, token: _ChunkedPipeline())},
    )

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_pyannote)
    monkeypatch.setattr("recalld.pipeline.diarise.MAX_CHUNK_SECONDS", 2.0)

    wav_path = tmp_path / "sample.wav"
    _write_pcm_wav(wav_path, duration_seconds=5.0)

    progress = []
    diarise(wav_path, "token", progress_cb=progress.append)

    assert progress[0] == "Loading diarisation model."
    assert any(msg == "Diarisation model loaded." for msg in progress)
    assert any(msg.startswith("Planned 3 diarisation chunk(s)") for msg in progress)
    assert any(msg.startswith("Diarising chunk 2/3") for msg in progress)
    assert any(msg.startswith("Finished chunk 3/3") for msg in progress)
    assert progress[-1].startswith("Completed diarisation in ")


def test_diarise_loads_community_model(monkeypatch, tmp_path):
    fake_torch = ModuleType("torch")
    fake_torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    fake_torch.device = lambda value: value

    loaded = {}

    class _ModelPipeline(_FakePipeline):
        pass

    def fake_from_pretrained(model, token):
        loaded["model"] = model
        loaded["token"] = token
        return _ModelPipeline()

    fake_pyannote = ModuleType("pyannote.audio")
    fake_pyannote.Pipeline = type(
        "Pipeline",
        (),
        {"from_pretrained": staticmethod(fake_from_pretrained)},
    )

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_pyannote)

    wav_path = tmp_path / "sample.wav"
    _write_pcm_wav(wav_path, duration_seconds=1.0)

    diarise(wav_path, "token")

    assert loaded["model"] == "pyannote/speaker-diarization-community-1"
    assert loaded["token"] == "token"
