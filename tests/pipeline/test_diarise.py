from types import ModuleType, SimpleNamespace
import sys

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
    wav_path.write_bytes(b"wav")

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
    wav_path.write_bytes(b"wav")

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
    wav_path.write_bytes(b"wav")

    turns = diarise(wav_path, "token")

    assert [(turn.start, turn.end, turn.speaker) for turn in turns] == [
        (0.0, 0.8, "SPEAKER_00"),
        (0.8, 1.7, "SPEAKER_01"),
    ]
