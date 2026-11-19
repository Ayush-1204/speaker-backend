"""decide() must reproduce the field-tuned April (ff78b72) decision rules.

`_april_reference` is a line-for-line port of the inline decision block from
app/main.py at commit ff78b72, kept here as the parity oracle.
"""

import random

import numpy as np

from app.services.decision_engine import DecisionParams, decide, mark_identity_alert_fired
from app.utils import prd_services_db as svc


APRIL = dict(
    T_HIGH=0.47, T_LOW=0.31, ALERT_TRIGGER_STREAK=3, FORCE_ALERT_SCORE=0.15, FORCE_ALERT_MIN_STREAK=2,
    FAMILIAR_GRACE_SEC=20, FAMILIAR_HOLD_FLOOR=0.30, FAMILIAR_GRACE_MARGIN=0.05, FAMILIAR_GRACE_MIN_CONF=0.65,
    FAMILIAR_RESET_MIN_STREAK=2, SOFT_STRANGER_MARGIN=0.03, SOFT_STRANGER_MIN_CONF=0.30,
    STRANGER_MIN_TIER2_CONF=0.70, STRANGER_UNCERTAIN_CONF_FLOOR=0.88, DEBOUNCE_SEC=60, SPEAKER_CHANGE_THRESHOLD=0.30,
)


def _april_params() -> DecisionParams:
    return DecisionParams(t_high=0.47, t_low=0.31, alert_trigger_streak=3)


def _cos(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))


def _april_reference(session, score, closest_speaker_id, tier2_category, tier2_confidence, centroid, now):
    c = APRIL
    allow_stranger_increment = (
        (tier2_category == "human_speech" and tier2_confidence >= c["STRANGER_MIN_TIER2_CONF"])
        or (tier2_category == "uncertain" and tier2_confidence >= c["STRANGER_UNCERTAIN_CONF_FLOOR"])
    )
    if score >= c["T_HIGH"]:
        session.familiar_recovery_streak += 1
        if session.stranger_streak > 0 and session.familiar_recovery_streak < c["FAMILIAR_RESET_MIN_STREAK"]:
            decision = "uncertain_recovery"
        else:
            session.stranger_streak = 0
            session.last_familiar_ms = now
            session.last_familiar_speaker_id = str(closest_speaker_id) if closest_speaker_id else None
            svc.clear_stranger_identity(session)
            decision = "familiar"
    elif score <= c["T_LOW"]:
        session.familiar_recovery_streak = 0
        within_familiar_grace = session.last_familiar_ms > 0 and (now - session.last_familiar_ms) <= c["FAMILIAR_GRACE_SEC"] * 1000
        same_recent_speaker = (
            bool(closest_speaker_id) and bool(session.last_familiar_speaker_id)
            and str(closest_speaker_id) == str(session.last_familiar_speaker_id)
        )
        mild_post_familiar_dip = score >= max(c["FAMILIAR_HOLD_FLOOR"], c["T_LOW"] - c["FAMILIAR_GRACE_MARGIN"])
        strong_human_signal = tier2_category == "human_speech" and tier2_confidence >= c["FAMILIAR_GRACE_MIN_CONF"]
        if within_familiar_grace and same_recent_speaker and session.stranger_streak == 0 and mild_post_familiar_dip and strong_human_signal:
            session.stranger_streak = 0
            decision = "uncertain_post_familiar"
        elif not allow_stranger_increment:
            decision = "uncertain_noise"
        else:
            session.stranger_streak += 1
            decision = "stranger_candidate"
            svc.record_confirmed_stranger_window(session, centroid)
    else:
        session.familiar_recovery_streak = 0
        soft_limit = c["T_LOW"] + c["SOFT_STRANGER_MARGIN"]
        likely_human_speech = tier2_category == "human_speech" and tier2_confidence >= c["SOFT_STRANGER_MIN_CONF"]
        if score <= soft_limit and likely_human_speech:
            session.stranger_streak += 1
            decision = "stranger_candidate_soft"
            svc.record_confirmed_stranger_window(session, centroid)
        else:
            decision = "uncertain"

    if score <= c["FORCE_ALERT_SCORE"] and session.stranger_streak >= c["FORCE_ALERT_MIN_STREAK"]:
        session.stranger_streak = max(session.stranger_streak, c["ALERT_TRIGGER_STREAK"])

    alert = False
    if session.stranger_streak >= c["ALERT_TRIGGER_STREAK"]:
        debounce_elapsed = (now - session.last_alert_ms) >= c["DEBOUNCE_SEC"] * 1000
        if not debounce_elapsed:
            reference = session.active_stranger_embedding
            if reference is not None and centroid is not None and _cos(centroid, reference) < c["SPEAKER_CHANGE_THRESHOLD"]:
                session.last_alert_ms = 0
                debounce_elapsed = True
        if debounce_elapsed:
            alert = True
            session.last_alert_ms = now
            session.stranger_streak = 0
            if not session.recent_confirmed_stranger_embeddings:
                svc.record_confirmed_stranger_window(session, centroid)
            svc.set_active_stranger_embedding(session)
    return decision, alert


def _random_window(rng: random.Random, voices):
    score = rng.choice([rng.uniform(-0.1, 0.2), rng.uniform(0.25, 0.36), rng.uniform(0.36, 0.5), rng.uniform(0.45, 0.9)])
    category = rng.choice(["human_speech", "human_speech", "uncertain", "reject"])
    confidence = rng.uniform(0.0, 1.0)
    speaker = rng.choice(["mom", "dad", None])
    centroid = voices[rng.randrange(len(voices))] + np.random.default_rng(rng.randrange(10**6)).normal(0, 0.05, 192).astype(np.float32)
    return score, speaker, category, confidence, centroid


def test_decide_matches_april_logic_on_random_sequences():
    voices = [np.random.default_rng(i).normal(size=192).astype(np.float32) for i in range(3)]
    for seed in range(25):
        rng = random.Random(seed)
        live = svc.SessionState(parent_id="p", device_id="d")
        ref = svc.SessionState(parent_id="p", device_id="d")
        now = 1_700_000_000_000
        params = _april_params()
        for _ in range(400):
            now += rng.choice([250, 250, 250, 1000, 30_000])
            score, speaker, category, confidence, centroid = _random_window(rng, voices)

            expected_decision, expected_alert = _april_reference(ref, score, speaker, category, confidence, centroid, now)
            result = decide(
                live, score=score, closest_speaker_id=speaker, closest_speaker_role="familiar_adult",
                tier2_category=category, tier2_confidence=confidence, current_stranger_centroid=centroid,
                now_ms=now, params=params,
            )
            if result.should_alert:
                mark_identity_alert_fired(live, now, centroid)

            assert result.decision == expected_decision, (seed, now)
            assert result.should_alert == expected_alert, (seed, now)
            assert live.stranger_streak == ref.stranger_streak, (seed, now)
            assert live.last_alert_ms == ref.last_alert_ms, (seed, now)


def _window(session, score, now, params, category="human_speech", confidence=0.95, centroid=None):
    return decide(
        session, score=score, closest_speaker_id=None, closest_speaker_role="familiar_adult",
        tier2_category=category, tier2_confidence=confidence, current_stranger_centroid=centroid,
        now_ms=now, params=params,
    )


def test_three_stranger_windows_alert_then_debounce():
    params = _april_params()
    s = svc.SessionState(parent_id="p", device_id="d")
    voice = np.ones(192, dtype=np.float32)
    t0 = 1_700_000_000_000
    assert not _window(s, 0.25, t0, params, centroid=voice).should_alert
    assert not _window(s, 0.25, t0 + 250, params, centroid=voice).should_alert
    r = _window(s, 0.25, t0 + 500, params, centroid=voice)
    assert r.should_alert
    mark_identity_alert_fired(s, t0 + 500, voice)
    assert s.stranger_streak == 0 and s.stranger_weight_sum == 0.0

    # Same voice again inside the debounce window is held back.
    for t in (t0 + 1_000, t0 + 1_250, t0 + 1_500):
        r = _window(s, 0.25, t, params, centroid=voice)
    assert not r.should_alert and r.block_reason == "debounced"

    # A clearly different voice inside the debounce window does alert.
    other = -np.ones(192, dtype=np.float32)
    r = _window(s, 0.25, t0 + 1_750, params, centroid=other)
    assert r.should_alert and r.debounce_reset_similarity is not None


def test_low_confidence_uncertain_audio_never_counts():
    params = _april_params()
    s = svc.SessionState(parent_id="p", device_id="d")
    for t in range(10):
        r = _window(s, 0.05, 1_700_000_000_000 + t * 250, params, category="uncertain", confidence=0.5)
        assert r.decision == "uncertain_noise"
    assert s.stranger_streak == 0


def test_weighted_streak_is_opt_in():
    s = svc.SessionState(parent_id="p", device_id="d")
    params = DecisionParams(t_high=0.47, t_low=0.31, weighted_streak=True, alert_weight_threshold=2.5)
    # Scores just under T_LOW carry little weight, so three of them do not alert.
    for t in (1_700_000_000_000, 1_700_000_000_250, 1_700_000_000_500):
        r = _window(s, 0.28, t, params)
    assert not r.should_alert
    assert s.stranger_streak == 3


def test_child_self_match_resets_streak_without_opening_grace():
    params = _april_params()
    s = svc.SessionState(parent_id="p", device_id="d")
    _window(s, 0.2, 1_000, params)
    _window(s, 0.6, 1_250, params)
    r = decide(
        s, score=0.6, closest_speaker_id="kid", closest_speaker_role="child_self",
        tier2_category="human_speech", tier2_confidence=0.9, current_stranger_centroid=None,
        now_ms=1_500, params=params,
    )
    assert r.decision == "child_self_match" and r.child_self_match
    assert s.stranger_streak == 0
    assert s.last_familiar_ms == 0
