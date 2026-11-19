import numpy as np
import pytest

from app.utils.calibration import find_eer_threshold


def _brute_force_eer(target, impostor):
    t = np.asarray(target)
    i = np.asarray(impostor)
    grid = np.linspace(min(t.min(), i.min()), max(t.max(), i.max()), 5000)
    gaps = [abs(np.mean(i >= g) - np.mean(t < g)) for g in grid]
    k = int(np.argmin(gaps))
    return float(grid[k]), float((np.mean(i >= grid[k]) + np.mean(t < grid[k])) / 2)


@pytest.mark.parametrize("target_mean,impostor_mean", [(0.70, 0.25), (0.60, 0.35), (0.55, 0.45)])
def test_eer_threshold_matches_brute_force(target_mean, impostor_mean):
    rng = np.random.default_rng(0)
    target = list(rng.normal(target_mean, 0.07, 200))
    impostor = list(rng.normal(impostor_mean, 0.08, 400))

    thr, eer = find_eer_threshold(target, impostor)
    ref_thr, ref_eer = _brute_force_eer(target, impostor)

    assert abs(eer - ref_eer) < 0.02
    # FAR and FRR at the returned threshold must be roughly balanced.
    far = np.mean(np.asarray(impostor) >= thr)
    frr = np.mean(np.asarray(target) < thr)
    assert abs(far - frr) < 0.03
    assert abs(thr - ref_thr) < 0.05


def test_eer_threshold_lies_between_class_means():
    rng = np.random.default_rng(1)
    target = list(rng.normal(0.70, 0.06, 30))
    impostor = list(rng.normal(0.25, 0.08, 27))
    thr, _ = find_eer_threshold(target, impostor)
    # The pre-fix implementation returned ~0.71 here, above the target mean.
    assert 0.25 < thr < 0.70
