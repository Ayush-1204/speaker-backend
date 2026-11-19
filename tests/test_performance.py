"""Phase 3 regressions: matrix scoring parity, gallery cache, session lifecycle."""

import os
import time
import uuid

import numpy as np
import pytest

from app.database.session import SessionLocal, init_db
from app.utils import prd_services_db as svc


def _legacy_score(candidates, query_embs):
    """The pre-optimisation scoring loop, kept as the reference."""
    best, best_sid = -1.0, None
    for sid, entry in candidates.items():
        per_chunk = []
        for q in query_embs:
            chunk_best = max((svc._cos(q, ref) for ref in entry["embs"]), default=-1.0)
            if chunk_best >= 0.0:
                per_chunk.append(float(chunk_best))
        if not per_chunk:
            continue
        top = sorted(per_chunk, reverse=True)[:2]
        score = float(0.7 * np.median(per_chunk) + 0.3 * np.mean(top))
        if score > best:
            best, best_sid = score, sid
    return (0.0, None) if best < 0 else (best, best_sid)


def test_matrix_scoring_matches_legacy_loop():
    rng = np.random.default_rng(7)
    for _ in range(50):
        candidates = {
            f"s{i}": {"embs": [rng.normal(size=192).astype(np.float32) * rng.uniform(0.5, 3)
                               for _ in range(rng.integers(1, 8))], "role": "familiar_adult"}
            for i in range(rng.integers(1, 5))
        }
        queries = [rng.normal(size=192).astype(np.float32) for _ in range(rng.integers(1, 4))]
        expected_score, expected_sid = _legacy_score(candidates, queries)
        score, sid, _ = svc._score_against_candidates(candidates, queries)
        assert sid == expected_sid
        assert score == pytest.approx(expected_score, abs=1e-5)


def test_gallery_cache_refreshes_after_enrollment_changes():
    init_db()
    db = SessionLocal()
    try:
        parent = svc.register_parent_with_email(db, f"g_{uuid.uuid4().hex}@example.com", "password123", "G")
        pid = str(parent.id)
        speaker = svc.create_speaker(db, pid, "Mom")
        svc.save_speaker_embedding(db, pid, str(speaker.id), np.ones(192, dtype=np.float32))

        first = svc.load_parent_embeddings(db, pid)
        assert len(first[str(speaker.id)]["embs"]) == 1
        first.pop(str(speaker.id))  # callers may mutate their copy
        assert str(speaker.id) in svc.load_parent_embeddings(db, pid)

        svc.save_speaker_embedding(db, pid, str(speaker.id), np.full(192, 2.0, dtype=np.float32))
        assert len(svc.load_parent_embeddings(db, pid)[str(speaker.id)]["embs"]) == 2

        svc.delete_speaker(db, pid, str(speaker.id))
        assert svc.load_parent_embeddings(db, pid) == {}
    finally:
        db.close()


def test_embedding_files_never_overwrite_after_a_gap():
    init_db()
    db = SessionLocal()
    try:
        parent = svc.register_parent_with_email(db, f"f_{uuid.uuid4().hex}@example.com", "password123", "F")
        pid = str(parent.id)
        speaker = svc.create_speaker(db, pid, "Dad")
        sdir = svc.get_speaker_embedding_dir(pid, str(speaker.id))
        os.makedirs(sdir, exist_ok=True)
        np.save(os.path.join(sdir, "emb_2.npy"), np.ones(192, dtype=np.float32))  # gap: no emb_1
        path = svc.save_speaker_embedding(db, pid, str(speaker.id), np.zeros(192, dtype=np.float32) + 3)
        assert os.path.basename(path) == "emb_3.npy"
        assert np.load(os.path.join(sdir, "emb_2.npy"))[0] == 1.0
    finally:
        db.close()


def test_idle_sessions_are_evicted_and_state_is_restored(monkeypatch):
    stored = {}
    monkeypatch.setattr(svc, "persist_session", lambda p, d, m: stored.update({(p, d): dict(m)}))
    monkeypatch.setattr(svc, "load_session", lambda p, d: stored.get((p, d)))

    pid, did = str(uuid.uuid4()), str(uuid.uuid4())
    session = svc.get_or_create_session(pid, did)
    session.stranger_streak = 2
    session.last_alert_ms = 123456
    session.clip_buffer = np.zeros(160000, dtype=np.float32)
    svc.persist_session_state(session)

    session.last_used_monotonic = time.monotonic() - 3600
    assert svc.evict_idle_sessions(max_idle_sec=600) >= 1

    restored = svc.get_or_create_session(pid, did)
    assert restored is not session
    assert restored.stranger_streak == 2 and restored.last_alert_ms == 123456
    assert restored.clip_buffer is None  # audio is not kept for idle devices


def test_stranger_episode_does_not_accumulate_audio():
    session = svc.SessionState(parent_id="p", device_id="d")
    svc.start_stranger_segment(session, np.zeros(16000, dtype=np.float32), 1000)
    for _ in range(100):
        svc.append_stranger_segment(session, np.zeros(16000, dtype=np.float32))
    assert session.stranger_segment is None
    assert session.stranger_segment_started_ms == 1000


def test_unreachable_redis_is_skipped_after_first_failure(monkeypatch):
    from app.services import session_redis as sr

    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1")
    monkeypatch.setattr(sr, "_CLIENT", None)
    monkeypatch.setattr(sr, "_UNAVAILABLE_UNTIL", 0.0)
    assert sr.get_redis_client() is None  # pays one connect attempt
    t = time.perf_counter()
    for _ in range(20):
        sr.save_session_metadata("p", "d", {"x": 1})
    assert time.perf_counter() - t < 0.2  # backoff: no further connect attempts
