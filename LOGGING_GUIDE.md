# SafeEar Backend Logging Guide

## Overview
The backend now logs comprehensive score and acknowledgement information at each stage of the detection pipeline, making it easy to track and debug the system.

## Log Entry Format
All logs follow this pattern:
```
YYYY-MM-DD HH:MM:SS,milliseconds - app.<module> - INFO - [<function>] <LOG_TYPE> | field1=value1 | field2=value2
```

## Log Types by Stage

### 1. Audio Reception
**Log Entry:** `DETECT_CHUNK_RECEIVED`
- **When:** Audio chunk arrives at `/detect/chunk`
- **Format:** `parent_id={id} | device_id={id} | audio_len={samples} | sr={sample_rate}`
- **Example:**
```
2026-03-27 10:30:45,123 - app.main - INFO - [detect_chunk] DETECT_CHUNK_RECEIVED | parent_id=123e4567-e89b-12d3-a456-426614174000 | device_id=98765432-f89b-12d3-a456-426614174111 | audio_len=24000 | sr=16000
```

### 2. Acknowledgements (Warmup/No-Hop)
**Log Entries:** `ACK`
- **When:** System is warming up or chunk too small
- **Format for Warming Up:** `warming_up | ring_samples={curr} | required={needed}`
- **Format for No Hop:** `no_hop | chunk_samples={curr} | required_hop={needed}`
- **Examples:**
```
2026-03-27 10:30:45,234 - app.main - INFO - [detect_chunk] ACK | warming_up | ring_samples=12000 | required=24000
2026-03-27 10:30:46,100 - app.main - INFO - [detect_chunk] ACK | no_hop | chunk_samples=3600 | required_hop=4000
```

### 3. Tier-1: Voice Activity Detection (VAD)
**Rejection Log:** `TIER1_VAD_REJECTED`
- **Format:** `parent_id={id} | reason={reason} | Metric=value`
- **Reasons:** `low_rms`, `insufficient_voice`
- **Examples:**
```
2026-03-27 10:30:46,500 - app.utils.prd_services_db - INFO - [evaluate_window] TIER1_VAD_REJECTED | parent_id=123e4567-e89b-12d3-a456-426614174000 | reason=low_rms | rms=0.000123
2026-03-27 10:30:47,100 - app.utils.prd_services_db - INFO - [evaluate_window] TIER1_VAD_REJECTED | parent_id=123e4567-e89b-12d3-a456-426614174000 | reason=insufficient_voice | voiced_ms=250 | threshold_ms=500
```

**Pass Log:** `TIER1_VAD_PASSED`
- **Format:** `parent_id={id} | voiced_ms={duration} | rms={energy}`
- **Example:**
```
2026-03-27 10:30:47,600 - app.utils.prd_services_db - INFO - [evaluate_window] TIER1_VAD_PASSED | parent_id=123e4567-e89b-12d3-a456-426614174000 | voiced_ms=1200 | rms=0.050234
```

### 4. Tier-2: Speech Classification
**Evaluation Log:** `TIER2_EVALUATION`
- **When:** YAMNet and speech metrics are computed
- **Format:** `parent_id={id} | speech_ok=bool | yamnet_category={cat} | yamnet_confidence=float | rms=float | flatness=float | centroid_hz=float`
- **Example:**
```
2026-03-27 10:30:48,200 - app.utils.prd_services_db - INFO - [evaluate_window] TIER2_EVALUATION | parent_id=123e4567-e89b-12d3-a456-426614174000 | speech_ok=True | yamnet_category=speech | yamnet_confidence=0.9123 | rms=0.045600 | flatness=0.3200 | centroid_hz=2500.50
```

**Tier-2 Rejection:** `TIER2_REJECTED`
- **When:** Audio fails speech quality checks
- **Format:** `parent_id={id} | speech_ok={bool} | yamnet_category={cat} | yamnet_confidence={conf}`
- **Example:**
```
2026-03-27 10:30:48,700 - app.utils.prd_services_db - INFO - [evaluate_window] TIER2_REJECTED | parent_id=123e4567-e89b-12d3-a456-426614174000 | speech_ok=False | yamnet_category=reject | yamnet_confidence=0.8500
```

**Tier-2 Pass:** `TIER2_PASSED`
- **Format:** `parent_id={id} | proceeding_to_tier3_speaker_match`
- **Example:**
```
2026-03-27 10:30:49,100 - app.utils.prd_services_db - INFO - [evaluate_window] TIER2_PASSED | parent_id=123e4567-e89b-12d3-a456-426614174000 | proceeding_to_tier3_speaker_match
```

### 5. Tier-3: Speaker Matching
**Speaker Scoring Log:** `SPEAKER_SCORING` (DEBUG level)
- **Format:** `parent_id={id} | scores={speaker_name1:score1, speaker_name2:score2, ...}`
- **Example:**
```
2026-03-27 10:30:49,500 - app.utils.prd_services_db - DEBUG - [score_against_parent] SPEAKER_SCORING | parent_id=123e4567-e89b-12d3-a456-426614174000 | scores={Ayush:0.8234, Mom:0.2100}
```

**No Speaker Match:** `NO_SPEAKER_MATCH`
- **Format:** `parent_id={id} | score=0.0`
- **Example:**
```
2026-03-27 10:30:50,000 - app.utils.prd_services_db - INFO - [score_against_parent] NO_SPEAKER_MATCH | parent_id=123e4567-e89b-12d3-a456-426614174000 | score=0.0
```

**Best Speaker Match:** `BEST_SPEAKER_MATCHED`
- **Format:** `parent_id={id} | speaker_id={speaker_id} | score={score}` (e.g. score=0.8234)
- **Example:**
```
2026-03-27 10:30:50,400 - app.utils.prd_services_db - INFO - [score_against_parent] BEST_SPEAKER_MATCHED | parent_id=123e4567-e89b-12d3-a456-426614174000 | speaker_id=12345678-1234-5678-1234-567812345678 | score=0.8234
```

**Tier-3 Scored:** `TIER3_SCORED`
- **Format:** `parent_id={id} | score={score} | closest_speaker_id={id} | t_high={threshold} | t_low={threshold}`
- **Example:**
```
2026-03-27 10:30:50,800 - app.utils.prd_services_db - INFO - [evaluate_window] TIER3_SCORED | parent_id=123e4567-e89b-12d3-a456-426614174000 | score=0.8234 | closest_speaker_id=12345678-1234-5678-1234-567812345678 | t_high=0.72 | t_low=0.60
```

### 6. Score Decision
**Log Entry:** `STAGE_RESULT`
- **When:** All three tiers have been evaluated
- **Format:** `score={score} | tier1_vad={bool} | tier2={bool} | tier3={bool} | stage_decision={decision}`
- **Example:**
```
2026-03-27 10:30:51,100 - app.main - INFO - [detect_chunk] STAGE_RESULT | score=0.8234 | tier1_vad=True | tier2=True | tier3=True | stage_decision=tier3_scored
```

**Log Entry:** `SCORE_DECISION`
- **When:** Score is compared against thresholds
- **Format:** `score={score} [comparison] t_high/t_low={threshold} | decision={decision} | [streak_info]`
- **Examples:**
```
2026-03-27 10:30:51,300 - app.main - INFO - [detect_chunk] SCORE_DECISION | score=0.8234 >= t_high=0.72 | decision=familiar | streak_reset
2026-03-27 10:30:51,400 - app.main - INFO - [detect_chunk] SCORE_DECISION | score=0.5500 <= t_low=0.60 | decision=stranger_candidate | streak_incremented=1
2026-03-27 10:30:51,500 - app.main - INFO - [detect_chunk] SCORE_DECISION | t_low < score=0.6500 < t_high | decision=uncertain
```

### 7. Alert Trigger Logic
**Fast Track Alert:** `FAST_TRACK_ALERT`
- **When:** Score is very low (≤ force_alert_score threshold)
- **Format:** `score={score} <= force_alert_score={threshold} | streak_updated={old_streak}->{new_streak}`
- **Example:**
```
2026-03-27 10:30:52,100 - app.main - INFO - [detect_chunk] FAST_TRACK_ALERT | score=0.4500 <= force_alert_score=0.50 | streak_updated=1->3
```

### 8. Alert Creation & Escalation
**Alert Created:** `ALERT_CREATED`
- **When:** Alert database record is created
- **Format:** `alert_id={id} | parent_id={id} | device_id={id} | confidence_score={score} | location=({lat}, {lng})`
- **Example:**
```
2026-03-27 10:30:52,500 - app.utils.prd_services_db - INFO - [create_alert] ALERT_CREATED | alert_id=87654321-4321-8765-4321-876543218765 | parent_id=123e4567-e89b-12d3-a456-426614174000 | device_id=98765432-f89b-12d3-a456-426614174111 | confidence_score=0.4200 | location=(37.7749, -122.4194)
```

**Alert Fired:** `ALERT_FIRED`
- **When:** Alert threshold is met and debounce period has passed
- **Format:** `alert_id={id} | score={score} | streak_satisfied={required_streak} | clip_path={path}`
- **Example:**
```
2026-03-27 10:30:53,100 - app.main - INFO - [detect_chunk] ALERT_FIRED | alert_id=87654321-4321-8765-4321-876543218765 | score=0.4200 | streak_satisfied=3 | clip_path=app/data/tenants/123e4567-e89b-12d3-a456-426614174000/alerts/87654321-4321-8765-4321-876543218765/clip.wav
```

**Alert Escalation Start:** `ALERT_ESCALATION_START`
- **When:** Notification thread is started for FCM/SMS/Email
- **Format:** `alert_id={id} | parent_email={email} | fcm_token_available={bool}`
- **Example:**
```
2026-03-27 10:30:53,400 - app.main - INFO - [detect_chunk] ALERT_ESCALATION_START | alert_id=87654321-4321-8765-4321-876543218765 | parent_email=parent@example.com | fcm_token_available=True
```

### 9. Alert Blocking Reasons
**Alert Blocked:** `ALERT_BLOCKED`
- **Reasons:**
  - `insufficient_stranger_streak`: Not enough consecutive low-score windows
  - `debounced`: Alert fired recently (within debounce period)
- **Examples:**
```
2026-03-27 10:30:54,100 - app.main - INFO - [detect_chunk] ALERT_BLOCKED | alert_id=None | reason=insufficient_stranger_streak | current_streak=1 | required=3
2026-03-27 10:30:54,300 - app.main - INFO - [detect_chunk] ALERT_BLOCKED | alert_id=None | reason=debounced | last_alert_ms=1711449653100 | now=1711449653200 | debounce_sec=60
```

## Log Levels
- **INFO**: Normal operation, important decisions, scores, and alerts
- **DEBUG**: Detailed scoring information (speaker scores comparison)

## Grep Commands for Tracking

### Find all decisions for a parent:
```bash
grep "parent_id=<parent_uuid>" /path/to/logs | grep SCORE_DECISION
```

### Track a specific alert:
```bash
grep "alert_id=<alert_uuid>" /path/to/logs
```

### Find all tier rejections:
```bash
grep "TIER.*REJECTED" /path/to/logs
```

### See streaming audio acknowledgements:
```bash
grep "ACK |" /path/to/logs
```

### Track all alerts that fired:
```bash
grep "ALERT_FIRED" /path/to/logs
```

## Configuration via Environment Variables

Thresholds and triggers can be tuned via environment variables:

```bash
export SAFEEAR_T_HIGH=0.72              # Familiar score threshold (default: 0.72)
export SAFEEAR_T_LOW=0.60               # Stranger score threshold (default: 0.60)
export SAFEEAR_ALERT_TRIGGER_STREAK=3   # Consecutive windows before alert (default: 3)
export SAFEEAR_FORCE_ALERT_SCORE=0.50   # Fast-track threshold (default: 0.50)
export SAFEEAR_DEBOUNCE_SEC=60          # Seconds between alerts (default: 60)
```

## Example Full Detection Flow

Here's what a complete detection flow looks like in the logs:

```
DETECT_CHUNK_RECEIVED | parent_id=... | device_id=... | audio_len=24000 | sr=16000
ACK | warming_up | ring_samples=12000 | required=24000
DETECT_CHUNK_RECEIVED | parent_id=... | device_id=... | audio_len=24000 | sr=16000
TIER1_VAD_PASSED | parent_id=... | voiced_ms=1200 | rms=0.050234
TIER2_EVALUATION | parent_id=... | speech_ok=True | yamnet_category=speech | yamnet_confidence=0.9123 | rms=0.045600 | flatness=0.3200 | centroid_hz=2500.50
TIER2_PASSED | parent_id=... | proceeding_to_tier3_speaker_match
TIER3_SCORED | parent_id=... | score=0.8234 | closest_speaker_id=... | t_high=0.72 | t_low=0.60
STAGE_RESULT | score=0.8234 | tier1_vad=True | tier2=True | tier3=True | stage_decision=tier3_scored
SCORE_DECISION | score=0.8234 >= t_high=0.72 | decision=familiar | streak_reset
```

And for an alert scenario:

```
TIER3_SCORED | parent_id=... | score=0.2500 | closest_speaker_id=... | t_high=0.72 | t_low=0.60
STAGE_RESULT | score=0.2500 | tier1_vad=True | tier2=True | tier3=True | stage_decision=tier3_scored
SCORE_DECISION | score=0.2500 <= t_low=0.60 | decision=stranger_candidate | streak_incremented=1
TIER3_SCORED | parent_id=... | score=0.1800 | closest_speaker_id=... | t_high=0.72 | t_low=0.60
STAGE_RESULT | score=0.1800 | tier1_vad=True | tier2=True | tier3=True | stage_decision=tier3_scored
SCORE_DECISION | score=0.1800 <= t_low=0.60 | decision=stranger_candidate | streak_incremented=2
TIER3_SCORED | parent_id=... | score=0.0500 | closest_speaker_id=... | t_high=0.72 | t_low=0.60
STAGE_RESULT | score=0.0500 | tier1_vad=True | tier2=True | tier3=True | stage_decision=tier3_scored
SCORE_DECISION | score=0.0500 <= t_low=0.60 | decision=stranger_candidate | streak_incremented=3
FAST_TRACK_ALERT | score=0.0500 <= force_alert_score=0.50 | streak_updated=3->3
ALERT_CREATED | alert_id=... | parent_id=... | device_id=... | confidence_score=0.0500 | location=(37.7749, -122.4194)
ALERT_FIRED | alert_id=... | score=0.0500 | streak_satisfied=3 | clip_path=...
ALERT_ESCALATION_START | alert_id=... | parent_email=... | fcm_token_available=True
```
