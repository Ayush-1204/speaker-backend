"""Adaptive symmetric score normalisation (AS-norm) in the production scorer."""

import numpy as np
import pytest

from app.utils import prd_services_db as svc
from app.utils import score_normaliser as sn


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture(autouse=True)
def _reset_asnorm():
    yield
    sn.set_asnorm(None)


def _setup(seed=0):
    rng = np.random.default_rng(seed)
    cohort = rng.normal(size=(60, 192)).astype(np.float32)
    spk = _unit(rng.normal(size=192))
    gallery = {"a": {"embs": [_unit(spk + 0.3 * rng.normal(size=192)) for _ in range(3)], "role": "familiar_adult"}}
    query = _unit(spk + 0.3 * rng.normal(size=192))
    return cohort, gallery, query


def test_disabled_by_default_gives_raw_cosine():
    sn.set_asnorm(None)
    _, gallery, query = _setup()
    s, sid, _ = svc._score_against_candidates(gallery, [query])
    refs = np.stack(gallery["a"]["embs"])
    assert sid == "a"
    assert s == pytest.approx(float((refs @ query).max()), abs=1e-5)


def test_asnorm_matches_formula_and_calibration():
    cohort, gallery, query = _setup()
    norm = sn.AsNorm(cohort, top_n=10, scale=0.1, offset=0.3)
    sn.set_asnorm(norm)
    s, sid, _ = svc._score_against_candidates(gallery, [query])
    refs = np.stack(gallery["a"]["embs"])
    raw = float((refs @ query).max())
    c = norm.cohort
    e = np.sort((c @ refs.T).max(axis=1))[-10:]
    q = np.sort(c @ query)[-10:]
    z = 0.5 * ((raw - e.mean()) / e.std() + (raw - q.mean()) / q.std())
    assert sid == "a"
    assert s == pytest.approx(0.1 * z + 0.3, abs=1e-4)


def test_enroll_stats_are_cached_per_norm_instance():
    cohort, gallery, query = _setup()
    norm = sn.AsNorm(cohort, top_n=10)
    sn.set_asnorm(norm)
    svc._score_against_candidates(gallery, [query])
    assert gallery["a"]["_asnorm_stats"][0] is norm
    other = sn.AsNorm(cohort, top_n=20)
    sn.set_asnorm(other)
    svc._score_against_candidates(gallery, [query])
    assert gallery["a"]["_asnorm_stats"][0] is other


def test_save_and_load_roundtrip(tmp_path):
    cohort, _, _ = _setup()
    sn.AsNorm(cohort, top_n=15, scale=0.08, offset=0.25).save(str(tmp_path))
    loaded = sn.AsNorm.load(str(tmp_path))
    assert loaded.top_n == 15 and loaded.scale == pytest.approx(0.08) and loaded.offset == pytest.approx(0.25)
    assert loaded.cohort.shape == (60, 192)
