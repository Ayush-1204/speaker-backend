"""The offline replay must reset the stranger streak at a stream gap, as the server does."""

from app.services.decision_engine import DecisionParams
from scripts.eval.replay import simulate


def _row(ts_ms: int) -> dict:
    return {
        "device_id": "dev-1",
        "parent_id": "p-1",
        "ts_ms": ts_ms,
        "tier3_pass": True,
        "score": 0.20,  # stranger band, above the fast-track score
        "closest_speaker_id": "spk-1",
        "closest_speaker_role": "familiar_adult",
        "tier2_category": "human_speech",
        "tier2_confidence": 0.95,
        "centroid": None,
    }


def test_contiguous_stranger_windows_alert():
    params = DecisionParams.from_env()
    rows = [_row(1_000_000 + i * 1500) for i in range(params.alert_trigger_streak)]
    assert len(simulate(rows, params)) == 1


def test_gap_resets_streak():
    params = DecisionParams.from_env()
    k = params.alert_trigger_streak
    before = [_row(1_000_000 + i * 1500) for i in range(k - 1)]
    after_start = before[-1]["ts_ms"] + 5 * 60 * 1000
    after = [_row(after_start + i * 1500) for i in range(k - 1)]
    assert simulate(before + after, params) == []
