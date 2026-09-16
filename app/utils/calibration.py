"""EER threshold computation (Change 5 – auto-calibration dependency).

find_eer_threshold() locates the threshold where FAR ≈ FRR on a set of
target and impostor scores.  It is used by auto_calibrate_thresholds() in
prd_services_db.py to set per-household T_HIGH.
"""

from typing import List, Tuple
import numpy as np


def find_eer_threshold(
    target_scores: List[float],
    impostor_scores: List[float],
    n_thresholds: int = 1000,
) -> Tuple[float, float]:
    """Return (threshold_at_eer, eer_fraction).

    Args:
        target_scores   : cosine similarities where ground truth = same speaker
        impostor_scores : cosine similarities where ground truth = different speaker
        n_thresholds    : number of candidate thresholds to test

    Returns:
        (eer_threshold, eer)  — eer is in [0, 1] (not percent)

    Notes
    -----
    - FAR (false accept rate) = fraction of impostor scores >= threshold
    - FRR (false reject rate) = fraction of target scores  <  threshold
    - EER is the threshold where |FAR - FRR| is minimised.
    - With very few scores the EER estimate is noisy; callers should only
      invoke this when len(target_scores) >= 3 and len(impostor_scores) >= 3.
    """
    if not target_scores or not impostor_scores:
        raise ValueError("find_eer_threshold requires non-empty target and impostor score lists")

    all_scores = sorted(set(target_scores + impostor_scores))
    # Sample n_thresholds evenly across the score range
    mn, mx = float(all_scores[0]), float(all_scores[-1])
    thresholds = np.linspace(mn - 1e-6, mx + 1e-6, n_thresholds)

    target_arr = np.asarray(target_scores, dtype=np.float32)
    impostor_arr = np.asarray(impostor_scores, dtype=np.float32)

    best_threshold = float(np.mean(thresholds))
    best_eer = 1.0

    for t in thresholds:
        far = float(np.mean(impostor_arr >= t))
        frr = float(np.mean(target_arr < t))
        # EER is where FAR == FRR; minimise |FAR - FRR|
        if abs(far - frr) < abs(best_eer - 0.5) + 0.5:  # track the crossover
            best_threshold = float(t)
            best_eer = (far + frr) / 2.0

    return best_threshold, best_eer
