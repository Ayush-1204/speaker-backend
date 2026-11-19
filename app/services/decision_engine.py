"""Per-window stranger decision logic.

`decide()` is the single source of truth for turning a scored window into a
decision.  It mutates only the session's decision counters and never touches
the database, audio buffers or notifications, so the live /detect path and the
offline replay tool (scripts/eval/replay.py) run exactly the same logic.

Defaults reproduce the rules field-tuned on the worn device (backend ff78b72,
April 2026).  Experimental rules are opt-in via environment flags.
"""

import datetime
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from app.utils import prd_services_db as svc

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, "true" if default else "false").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class DecisionParams:
    t_high: float = svc.T_HIGH
    t_low: float = svc.T_LOW
    stranger_min_tier2_conf: float = 0.70
    stranger_uncertain_conf_floor: float = 0.88
    familiar_grace_sec: int = 20
    familiar_hold_floor: float = 0.30
    familiar_grace_margin: float = 0.05
    familiar_grace_min_conf: float = 0.65
    familiar_reset_min_streak: int = 2
    soft_stranger_margin: float = 0.03
    soft_stranger_min_conf: float = 0.30
    force_alert_score: float = 0.15
    force_alert_min_streak: int = 2
    alert_trigger_streak: int = svc.STRANGER_CONFIRM_COUNT
    debounce_sec: int = 60
    speaker_change_threshold: float = 0.30
    # Opt-in: alert on accumulated confidence weight instead of window count.
    weighted_streak: bool = False
    alert_weight_threshold: float = 2.5

    @classmethod
    def from_env(cls) -> "DecisionParams":
        d = cls()
        return cls(
            t_high=_env_float("SAFEEAR_T_HIGH", d.t_high),
            t_low=_env_float("SAFEEAR_T_LOW", d.t_low),
            stranger_min_tier2_conf=_env_float("SAFEEAR_STRANGER_MIN_TIER2_CONF", d.stranger_min_tier2_conf),
            stranger_uncertain_conf_floor=_env_float("SAFEEAR_STRANGER_UNCERTAIN_CONF_FLOOR", d.stranger_uncertain_conf_floor),
            familiar_grace_sec=_env_int("SAFEEAR_FAMILIAR_GRACE_SEC", d.familiar_grace_sec),
            familiar_hold_floor=_env_float("SAFEEAR_FAMILIAR_HOLD_FLOOR", d.familiar_hold_floor),
            familiar_grace_margin=_env_float("SAFEEAR_FAMILIAR_GRACE_MARGIN", d.familiar_grace_margin),
            familiar_grace_min_conf=_env_float("SAFEEAR_FAMILIAR_GRACE_MIN_CONF", d.familiar_grace_min_conf),
            familiar_reset_min_streak=_env_int("SAFEEAR_FAMILIAR_RESET_MIN_STREAK", d.familiar_reset_min_streak),
            soft_stranger_margin=_env_float("SAFEEAR_SOFT_STRANGER_MARGIN", d.soft_stranger_margin),
            soft_stranger_min_conf=_env_float("SAFEEAR_SOFT_STRANGER_MIN_CONF", d.soft_stranger_min_conf),
            force_alert_score=_env_float("SAFEEAR_FORCE_ALERT_SCORE", d.force_alert_score),
            force_alert_min_streak=_env_int("SAFEEAR_FORCE_ALERT_MIN_STREAK", d.force_alert_min_streak),
            alert_trigger_streak=_env_int("SAFEEAR_ALERT_TRIGGER_STREAK", d.alert_trigger_streak),
            debounce_sec=_env_int("SAFEEAR_DEBOUNCE_SEC", d.debounce_sec),
            speaker_change_threshold=_env_float("SAFEEAR_SPEAKER_CHANGE_THRESHOLD", d.speaker_change_threshold),
            weighted_streak=_env_bool("SAFEEAR_WEIGHTED_STREAK", d.weighted_streak),
            alert_weight_threshold=_env_float("SAFEEAR_ALERT_WEIGHT_THRESHOLD", d.alert_weight_threshold),
        )


@dataclass
class DecisionResult:
    decision: str
    should_alert: bool = False
    block_reason: Optional[str] = None
    child_self_match: bool = False
    debounce_reset_similarity: Optional[float] = None


def _normalized_embedding(vec: Optional[Any]) -> Optional[np.ndarray]:
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    n = float(np.linalg.norm(arr))
    if n <= 1e-12:
        return None
    return (arr / n).astype(np.float32)


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    an = _normalized_embedding(a)
    bn = _normalized_embedding(b)
    if an is None or bn is None:
        return 0.0
    return float(np.dot(an, bn))


def _has_speaker_changed(
    session: svc.SessionState,
    current_embedding: Optional[Any],
    speaker_change_threshold: float,
) -> Tuple[bool, Optional[float]]:
    """Whether the current stranger differs from the one that last alerted.

    Only compares against the stranger that actually triggered the last alert.
    Falling back to the running centroid of the current episode lets noisy
    1.5 s embeddings of the same person reset the debounce and re-alert.
    """
    reference = session.active_stranger_embedding
    if reference is None or current_embedding is None:
        return False, None

    sim = _cos_sim(current_embedding, reference)
    return sim < speaker_change_threshold, sim


def _streak_weight(score: float, t_low: float, t_high: float, tier2_confidence: Optional[float]) -> float:
    """Continuous weight for a stranger-candidate window (opt-in weighted streak)."""
    if t_low <= 0:
        return 1.0
    score_weight = max(0.0, min(1.0, (t_low - score) / t_low)) if score <= t_low else 0.0
    conf_weight = min(1.0, tier2_confidence / 0.90) if tier2_confidence else 1.0
    return score_weight * conf_weight


def _should_alert(session: svc.SessionState, now_ms: int) -> Tuple[bool, Optional[str]]:
    """Context gates (party mode, quiet hours, home geofence).

    Only active when the parent has configured them in HouseholdConfig.
    """
    cfg = session.household_config
    if not cfg:
        return True, None

    party = cfg.get("party_mode_until")
    if party:
        party_dt = datetime.datetime.fromisoformat(party.replace("Z", "+00:00"))
        if party_dt.tzinfo is None:
            party_dt = party_dt.replace(tzinfo=datetime.timezone.utc)
        if datetime.datetime.now(datetime.timezone.utc) < party_dt:
            return False, "party_mode"

    start = cfg.get("quiet_hours_start")
    end = cfg.get("quiet_hours_end")
    if start is not None and end is not None:
        h = datetime.datetime.fromtimestamp(now_ms / 1000.0).hour
        if start <= end:
            if start <= h < end:
                return False, "quiet_hours"
        elif h >= start or h < end:
            return False, "quiet_hours"

    home_lat = cfg.get("home_lat")
    home_lon = cfg.get("home_lon")
    radius = cfg.get("home_radius_m") or 100.0
    if home_lat is not None and home_lon is not None and session.lat is not None and session.lon is not None:
        r_earth = 6371000
        phi1, phi2 = math.radians(home_lat), math.radians(session.lat)
        dphi = math.radians(session.lat - home_lat)
        dlambda = math.radians(session.lon - home_lon)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        dist = 2 * r_earth * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        if dist <= radius:
            return False, "geofence_home"

    return True, None


def _add_stranger_window(
    session: svc.SessionState,
    score: float,
    limit: float,
    params: DecisionParams,
    tier2_confidence: float,
    centroid: Optional[np.ndarray],
) -> None:
    session.stranger_streak += 1
    session.stranger_weight_sum += _streak_weight(score, limit, params.t_high, tier2_confidence)
    svc.record_confirmed_stranger_window(session, centroid)


def decide(
    session: svc.SessionState,
    *,
    score: float,
    closest_speaker_id: Optional[str],
    closest_speaker_role: Optional[str],
    tier2_category: str,
    tier2_confidence: float,
    current_stranger_centroid: Optional[np.ndarray],
    now_ms: int,
    params: DecisionParams,
) -> DecisionResult:
    """Classify one tier-3-scored window and decide whether to raise an alert.

    The caller is responsible for creating the alert when `should_alert` is
    true and must then call `mark_identity_alert_fired()`.
    """
    role = closest_speaker_role or "familiar_adult"
    allow_stranger_increment = (
        (tier2_category == "human_speech" and tier2_confidence >= params.stranger_min_tier2_conf)
        or (tier2_category == "uncertain" and tier2_confidence >= params.stranger_uncertain_conf_floor)
    )
    child_self_match = False

    if score >= params.t_high:
        session.familiar_recovery_streak += 1
        if session.stranger_streak > 0 and session.familiar_recovery_streak < params.familiar_reset_min_streak:
            decision = "uncertain_recovery"
        else:
            session.stranger_streak = 0
            session.stranger_weight_sum = 0.0
            session.familiar_recovery_streak = 0
            if role == "child_self":
                # The child's own voice resets the streak but does not open the
                # familiar grace window: it says nothing about who else is near.
                decision = "child_self_match"
                child_self_match = True
            else:
                session.last_familiar_ms = now_ms
                session.last_familiar_speaker_id = str(closest_speaker_id) if closest_speaker_id else None
                svc.clear_stranger_identity(session)
                decision = "familiar"
    elif score <= params.t_low:
        session.familiar_recovery_streak = 0
        within_familiar_grace = (
            session.last_familiar_ms > 0
            and (now_ms - session.last_familiar_ms) <= params.familiar_grace_sec * 1000
        )
        same_recent_speaker = (
            bool(closest_speaker_id)
            and bool(session.last_familiar_speaker_id)
            and str(closest_speaker_id) == str(session.last_familiar_speaker_id)
        )
        mild_post_familiar_dip = score >= max(params.familiar_hold_floor, params.t_low - params.familiar_grace_margin)
        strong_human_signal = tier2_category == "human_speech" and tier2_confidence >= params.familiar_grace_min_conf

        if within_familiar_grace and same_recent_speaker and session.stranger_streak == 0 and mild_post_familiar_dip and strong_human_signal:
            session.stranger_streak = 0
            session.stranger_weight_sum = 0.0
            decision = "uncertain_post_familiar"
        elif not allow_stranger_increment:
            decision = "uncertain_noise"
        else:
            _add_stranger_window(session, score, params.t_low, params, tier2_confidence, current_stranger_centroid)
            decision = "stranger_candidate"
    else:
        session.familiar_recovery_streak = 0
        soft_limit = params.t_low + params.soft_stranger_margin
        likely_human_speech = tier2_category == "human_speech" and tier2_confidence >= params.soft_stranger_min_conf
        if score <= soft_limit and likely_human_speech:
            _add_stranger_window(session, score, soft_limit, params, tier2_confidence, current_stranger_centroid)
            decision = "stranger_candidate_soft"
        else:
            decision = "uncertain"

    # Clearly low scores fast-track only after a minimum number of stranger windows.
    if score <= params.force_alert_score and session.stranger_streak >= params.force_alert_min_streak:
        session.stranger_streak = max(session.stranger_streak, params.alert_trigger_streak)
        session.stranger_weight_sum = max(session.stranger_weight_sum, params.alert_weight_threshold)

    result = DecisionResult(decision=decision, child_self_match=child_self_match)

    triggered = (
        session.stranger_weight_sum >= params.alert_weight_threshold
        if params.weighted_streak
        else session.stranger_streak >= params.alert_trigger_streak
    )
    if not triggered:
        result.block_reason = "insufficient_stranger_streak"
        return result

    can_alert, gate_reason = _should_alert(session, now_ms)
    if not can_alert:
        session.stranger_streak = 0
        session.stranger_weight_sum = 0.0
        svc.clear_stranger_segment(session)
        svc.clear_stranger_identity(session)
        result.block_reason = gate_reason
        return result

    debounce_elapsed = (now_ms - session.last_alert_ms) >= params.debounce_sec * 1000
    if not debounce_elapsed:
        speaker_changed, similarity = _has_speaker_changed(
            session, current_stranger_centroid, params.speaker_change_threshold
        )
        if speaker_changed:
            session.last_alert_ms = 0
            debounce_elapsed = True
            result.debounce_reset_similarity = similarity

    if debounce_elapsed:
        result.should_alert = True
    else:
        result.block_reason = "debounced"
    return result


def mark_identity_alert_fired(
    session: svc.SessionState,
    now_ms: int,
    current_stranger_centroid: Optional[np.ndarray] = None,
) -> None:
    """Session bookkeeping after an unenrolled-contact alert has been created."""
    session.last_alert_ms = now_ms
    session.stranger_streak = 0
    session.stranger_weight_sum = 0.0
    if not session.recent_confirmed_stranger_embeddings:
        svc.record_confirmed_stranger_window(session, current_stranger_centroid)
    svc.set_active_stranger_embedding(session)
