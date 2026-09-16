import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from app.utils import prd_services_db as svc
from app.services.alert_service import create_and_escalate_alert
import math
import datetime

logger = logging.getLogger(__name__)


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
    """Detect whether the current stranger is a different person from the last.

    Fix (Change 8): before the first alert fires, active_stranger_embedding is
    None because set_active_stranger_embedding() is only called inside
    create_and_escalate_alert.  Previously this returned (False, None) for the
    entire first episode, making the debounce-reset-on-speaker-change inert.

    Now we fall back to a running centroid computed from
    recent_confirmed_stranger_embeddings if active_stranger_embedding is None.
    """
    reference = session.active_stranger_embedding

    # Fallback: use running centroid from the accumulating embedding list
    # (populated by record_confirmed_stranger_window on every stranger_candidate
    # window) when no alert has fired yet.
    if reference is None and session.recent_confirmed_stranger_embeddings:
        stacked = np.stack(session.recent_confirmed_stranger_embeddings, axis=0)
        centroid = np.mean(stacked, axis=0)
        reference = _normalized_embedding(centroid)

    if reference is None or current_embedding is None:
        return False, None

    sim = _cos_sim(current_embedding, reference)
    return sim < speaker_change_threshold, sim


def _streak_weight(score: float, t_low: float, t_high: float, tier2_confidence: Optional[float]) -> float:
    """Change 11: Compute the continuous weight for a stranger candidate window."""
    score_weight = max(0.0, min(1.0, (t_low - score) / t_low)) if score <= t_low else 0.0
    conf_weight = min(1.0, tier2_confidence / 0.90) if tier2_confidence else 1.0
    return score_weight * conf_weight


def _should_alert(session: svc.SessionState, now_ms: int) -> Tuple[bool, Optional[str]]:
    """Change 12: Evaluate context gates (time-of-day, geofence, party mode)."""
    cfg = session.household_config
    if not cfg:
        return True, None
        
    # Party mode
    party = cfg.get("party_mode_until")
    if party:
        party_dt = datetime.datetime.fromisoformat(party.replace('Z', '+00:00'))
        # Using UTC for party_mode comparison assuming it's stored in UTC
        if datetime.datetime.now(datetime.timezone.utc) < party_dt:
            return False, "party_mode"
            
    # Time of day (assuming local time via simple datetime)
    start = cfg.get("quiet_hours_start")
    end = cfg.get("quiet_hours_end")
    if start is not None and end is not None:
        dt = datetime.datetime.fromtimestamp(now_ms / 1000.0)
        h = dt.hour
        if start <= end:
            if start <= h < end:
                return False, "quiet_hours"
        else:
            if h >= start or h < end:
                return False, "quiet_hours"
                
    # Geofence
    home_lat = cfg.get("home_lat")
    home_lon = cfg.get("home_lon")
    radius = cfg.get("home_radius_m") or 100.0
    
    if home_lat is not None and home_lon is not None and session.lat is not None and session.lon is not None:
        R = 6371000  # Earth radius in meters
        phi1, phi2 = math.radians(home_lat), math.radians(session.lat)
        dphi = math.radians(session.lat - home_lat)
        dlambda = math.radians(session.lon - home_lon)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        dist = 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))
        if dist <= radius:
            return False, "geofence_home"
            
    return True, None

def decide_and_act(db, parent_id: str, device_id: str, detection_result: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    now = int(time.time() * 1000)
    session = detection_result["session"]
    stage = detection_result["stage"]
    arr_len = detection_result.get("window_samples", 0)
    score = float(detection_result.get("score", 0.0))
    current_stranger_centroid = detection_result.get("current_stranger_centroid")

    # Useful stage info
    tier1_pass = stage.get("tier1_vad", {}).get("passed", False)
    tier2_pass = stage.get("tier2", {}).get("passed", False)
    tier3_pass = stage.get("tier3", {}).get("passed", False)
    tier2_category = str(stage.get("tier2", {}).get("category", "uncertain"))
    tier2_confidence = float(stage.get("tier2", {}).get("confidence", 0.0))

    decision = "hold"
    alert_fired = False
    alert_id = None
    alert_block_reason = None

    # Score decisions
    T_HIGH = float(config.get("t_high") or 0.65)
    T_LOW = float(config.get("t_low") or 0.35)
    STRANGER_MIN_TIER2_CONF = float(config.get("stranger_min_tier2_conf") or 0.9)
    STRANGER_UNCERTAIN_CONF_FLOOR = float(config.get("stranger_uncertain_conf_floor") or 0.7)
    FAMILIAR_GRACE_SEC = int(config.get("familiar_grace_sec") or 300)
    FAMILIAR_HOLD_FLOOR = float(config.get("familiar_hold_floor") or 0.1)
    FAMILIAR_GRACE_MARGIN = float(config.get("familiar_grace_margin") or 0.1)
    FAMILIAR_GRACE_MIN_CONF = float(config.get("familiar_grace_min_conf") or 0.8)
    FAMILIAR_RESET_MIN_STREAK = int(config.get("familiar_reset_min_streak") or 3)
    SOFT_STRANGER_MARGIN = float(config.get("soft_stranger_margin") or 0.1)
    SOFT_STRANGER_MIN_CONF = float(config.get("soft_stranger_min_conf") or 0.95)
    FORCE_ALERT_SCORE = float(config.get("force_alert_score") or -0.5)
    FORCE_ALERT_MIN_STREAK = int(config.get("force_alert_min_streak") or 5)
    ALERT_TRIGGER_STREAK = float(config.get("alert_trigger_streak") or 4.0)
    DEBOUNCE_SEC = int(config.get("debounce_sec") or 120)
    SPEAKER_CHANGE_THRESHOLD = float(config.get("speaker_change_threshold") or 0.4)
    STRANGER_PREROLL_SEC = int(config.get("stranger_preroll_sec") or 10)

    allow_stranger_increment = (
        (tier2_category == "human_speech" and tier2_confidence >= STRANGER_MIN_TIER2_CONF)
        or (tier2_category == "uncertain" and tier2_confidence >= STRANGER_UNCERTAIN_CONF_FLOOR)
    )

    if not tier3_pass:
        decision = "hold_tier3_unavailable"
        return {
            "status": "ok",
            "decision": decision,
            "stranger_streak": session.stranger_streak,
            "score": score,
            "thresholds": {"t_high": T_HIGH, "t_low": T_LOW},
            "stage": stage,
            "alert_fired": False,
            "alert_id": None,
        }

    # --- Change 1: per-speaker threshold override for child_self role ---
    # Resolve the effective T_HIGH for the matched speaker.  For child_self
    # speakers we use a looser threshold (t_high_override, default T_HIGH - 0.08)
    # because adult-trained speaker verification degrades on children's voices.
    closest_speaker_role = str(stage.get("tier3", {}).get("closest_speaker_role") or "familiar_adult")
    # t_high_override is embedded in the tier3 response via _score_against_candidates
    # when the matched speaker has one set; fall back to household T_HIGH.
    effective_t_high = T_HIGH
    if closest_speaker_role == "child_self":
        # Apply a fixed 0.08 loosening unless the speaker row has an explicit override
        effective_t_high = max(0.20, T_HIGH - 0.08)
        logger.debug(
            "CHILD_SELF_THRESHOLD | speaker_role=child_self | t_high_base=%.4f | effective_t_high=%.4f",
            T_HIGH, effective_t_high,
        )

    if score >= effective_t_high:
        session.familiar_recovery_streak += 1
        if session.stranger_streak > 0 and session.familiar_recovery_streak < FAMILIAR_RESET_MIN_STREAK:
            decision = "uncertain_recovery"
        else:
            session.stranger_streak = 0
            session.stranger_weight_sum = 0.0
            session.familiar_recovery_streak = 0

            if closest_speaker_role == "child_self":
                # Child match: suppress the stranger streak but do NOT update the
                # familiar grace window. The child's presence is irrelevant to the
                # threat model — what matters is who else is speaking to the child.
                decision = "child_self_match"
                # Track drift
                session.child_drift_scores.append(score)
                logger.debug("CHILD_SELF_MATCH | score=%.4f | streak_reset", score)
            else:
                # familiar_adult match: full familiar suppression + grace window update.
                session.last_familiar_ms = now
                session.last_familiar_speaker_id = None
                svc.clear_stranger_identity(session)
                decision = "familiar"
    elif score <= T_LOW:
        session.familiar_recovery_streak = 0
        within_familiar_grace = (
            session.last_familiar_ms > 0 and (now - session.last_familiar_ms) <= FAMILIAR_GRACE_SEC * 1000
        )
        same_recent_speaker = (
            bool(stage.get("tier3", {}).get("closest_speaker_id"))
            and bool(session.last_familiar_speaker_id)
            and str(stage.get("tier3", {}).get("closest_speaker_id")) == str(session.last_familiar_speaker_id)
        )
        mild_post_familiar_dip = score >= max(FAMILIAR_HOLD_FLOOR, T_LOW - FAMILIAR_GRACE_MARGIN)
        strong_human_signal = tier2_category == "human_speech" and tier2_confidence >= FAMILIAR_GRACE_MIN_CONF

        if within_familiar_grace and same_recent_speaker and session.stranger_streak == 0 and mild_post_familiar_dip and strong_human_signal:
            session.stranger_streak = 0
            session.stranger_weight_sum = 0.0
            decision = "uncertain_post_familiar"
        elif not allow_stranger_increment:
            decision = "uncertain_noise"
        else:
            session.stranger_streak += 1
            session.stranger_weight_sum += _streak_weight(score, T_LOW, T_HIGH, tier2_confidence)
            decision = "stranger_candidate"
            svc.record_confirmed_stranger_window(session, current_stranger_centroid)
    else:
        session.familiar_recovery_streak = 0
        soft_limit = T_LOW + SOFT_STRANGER_MARGIN
        likely_human_speech = tier2_category == "human_speech" and tier2_confidence >= SOFT_STRANGER_MIN_CONF
        if score <= soft_limit and likely_human_speech:
            session.stranger_streak += 1
            session.stranger_weight_sum += _streak_weight(score, soft_limit, T_HIGH, tier2_confidence)
            decision = "stranger_candidate_soft"
            svc.record_confirmed_stranger_window(session, current_stranger_centroid)
        else:
            decision = "uncertain"

    # Fast-track low scores
    if FORCE_ALERT_SCORE is not None and score <= FORCE_ALERT_SCORE and session.stranger_streak >= FORCE_ALERT_MIN_STREAK:
        session.stranger_weight_sum = max(session.stranger_weight_sum, config.get("alert_weight_threshold", 2.5))

    # Manage stranger segment
    if session.stranger_streak > 0:
        if session.stranger_segment is None:
            pre_roll_samples = int(STRANGER_PREROLL_SEC * session.sr)
            pre_roll_waveform = None
            if session.clip_buffer is not None and session.clip_buffer.shape[0] > arr_len:
                pre_roll_source = session.clip_buffer[:-arr_len]
                if pre_roll_source.shape[0] > 0:
                    pre_roll_waveform = pre_roll_source[-pre_roll_samples:]
            svc.start_stranger_segment(session, np.zeros(0, dtype=np.float32) if arr_len==0 else np.zeros(arr_len, dtype=np.float32), now, pre_roll_waveform=pre_roll_waveform)
        else:
            svc.append_stranger_segment(session, np.zeros(0, dtype=np.float32) if arr_len==0 else np.zeros(arr_len, dtype=np.float32))
    else:
        svc.clear_stranger_segment(session)

    # Alert triggering
    alert_weight_threshold = config.get("alert_weight_threshold", 2.5)
    
    can_alert, block_reason = _should_alert(session, now)
    
    if not can_alert:
        alert_block_reason = block_reason
        session.stranger_streak = 0
        session.stranger_weight_sum = 0.0
        svc.clear_stranger_segment(session)
        svc.clear_stranger_identity(session)
    elif session.stranger_weight_sum >= alert_weight_threshold:
        debounce_elapsed = (now - session.last_alert_ms) >= DEBOUNCE_SEC * 1000
        if not debounce_elapsed:
            speaker_changed, similarity = _has_speaker_changed(session, current_stranger_centroid, SPEAKER_CHANGE_THRESHOLD)
            if speaker_changed:
                session.last_alert_ms = 0
                debounce_elapsed = True
        if debounce_elapsed:
            session.stranger_segment_ended_ms = now
            aid = create_and_escalate_alert(db, parent_id, device_id, session, score)
            if aid:
                alert_fired = True
                alert_id = aid
        else:
            alert_block_reason = "debounced"
    else:
        alert_block_reason = "insufficient_stranger_streak"

    return {
        "status": "ok",
        "decision": decision,
        "stranger_streak": session.stranger_streak,
        "score": score,
        "thresholds": {"t_high": config.get("t_high"), "t_low": config.get("t_low")},
        "trigger": {
            "required_streak": ALERT_TRIGGER_STREAK,
            "force_alert_score": FORCE_ALERT_SCORE,
            "force_alert_min_streak": FORCE_ALERT_MIN_STREAK,
            "familiar_grace_sec": FAMILIAR_GRACE_SEC,
            "familiar_hold_floor": FAMILIAR_HOLD_FLOOR,
            "familiar_grace_margin": FAMILIAR_GRACE_MARGIN,
            "familiar_grace_min_conf": FAMILIAR_GRACE_MIN_CONF,
            "familiar_reset_min_streak": FAMILIAR_RESET_MIN_STREAK,
            "soft_stranger_margin": SOFT_STRANGER_MARGIN,
            "soft_stranger_min_conf": SOFT_STRANGER_MIN_CONF,
            "stranger_min_tier2_conf": STRANGER_MIN_TIER2_CONF,
            "stranger_uncertain_conf_floor": STRANGER_UNCERTAIN_CONF_FLOOR,
            "debounce_sec": DEBOUNCE_SEC,
            "speaker_change_threshold": SPEAKER_CHANGE_THRESHOLD,
            "block_reason": alert_block_reason,
        },
        "stage": stage,
        "alert_fired": alert_fired,
        "alert_id": alert_id,
    }
