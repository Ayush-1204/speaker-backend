import uuid

import numpy as np

from app.utils import feature_extractor
from app.utils import prd_services_db as svc
from app.utils.verification_pipeline import enrollment_rejection_message


def _fake_embed(monkeypatch):
    lengths = []

    def fake(chunk, sr=16000):
        lengths.append(len(chunk))
        return np.ones(192, dtype=np.float32)

    monkeypatch.setattr(feature_extractor, "_embed_waveform_array", fake)
    return lengths


def test_enrollment_drops_short_padded_tail(monkeypatch):
    lengths = _fake_embed(monkeypatch)
    sr = 16000
    wave = np.random.default_rng(0).normal(0, 0.1, int(3.4 * sr)).astype(np.float32)

    embs = feature_extractor.embed_waveform_chunks(wave, sr, segments=[(0, len(wave))], drop_short_tails=True)

    # 3.4 s -> one 3 s chunk; the 0.4 s remainder would have been reflect-padded.
    assert len(embs) == 1
    assert lengths == [48000]


def test_inference_chunking_unchanged_by_default(monkeypatch):
    lengths = _fake_embed(monkeypatch)
    sr = 16000
    wave = np.random.default_rng(0).normal(0, 0.1, int(3.4 * sr)).astype(np.float32)

    embs = feature_extractor.embed_waveform_chunks(wave, sr, segments=[(0, len(wave))])

    assert len(embs) == 2
    assert lengths == [48000, 16000]


def test_only_short_speech_is_still_embedded(monkeypatch):
    _fake_embed(monkeypatch)
    wave = np.zeros(8000, dtype=np.float32) + 0.01
    embs = feature_extractor.embed_waveform_chunks(wave, 16000, segments=[(0, 8000)], drop_short_tails=True)
    assert len(embs) == 1


def test_household_thresholds_default_to_field_tuned_globals():
    class ExplodingDb:
        def query(self, *a, **k):
            raise AssertionError("DB must not be queried when household thresholds are disabled")

    assert svc.USE_HOUSEHOLD_THRESHOLDS is False
    assert svc.get_household_thresholds(ExplodingDb(), str(uuid.uuid4())) == (svc.T_HIGH, svc.T_LOW)


def test_rejection_codes_map_to_plain_messages():
    msg = enrollment_rejection_message("enroll_rejected_voiced_too_short | voiced_ms=2400 | min=3000")
    assert "Speak continuously" in msg
    assert "try again" in enrollment_rejection_message("something_unexpected").lower()


def test_enrollment_concatenates_short_phrases_into_full_chunks(monkeypatch):
    lengths = _fake_embed(monkeypatch)
    sr = 16000
    wave = np.random.default_rng(1).normal(0, 0.1, 10 * sr).astype(np.float32)
    # Ten 0.6 s phrases separated by pauses: 6 s of speech in total.
    segments = [(int(i * sr), int(i * sr + 0.6 * sr)) for i in range(10)]

    embs = feature_extractor.embed_waveform_chunks(
        wave, sr, segments=segments, drop_short_tails=True, concat_segments=True
    )

    # Without concatenation every 0.6 s phrase would be padded into its own chunk.
    assert len(embs) == 2
    assert lengths == [48000, 48000]
