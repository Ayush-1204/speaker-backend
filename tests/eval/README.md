# Evaluation Harness — SafeEar (Change 9)

## Purpose
This directory contains the evaluation harness for measuring system reliability.
Without quantitative evaluation, every change to the pipeline is a guess.

## Required Metrics

| Metric | Primary? | What it measures |
|--------|----------|-----------------|
| False-alarms-per-hour | **YES** | User experience tolerance threshold |
| Detection latency P95 | **YES** | Safety relevance (time-to-alert) |
| EER | No | Speaker verification quality summary |
| minDCF | No | Weighted cost (miss >> false alarm) |

## Minimum Viable Dataset

```
eval_data/
  {speaker_id}/           # e.g., "parent_a", "child", "sibling_b"
    enrollment/           # clips used for enrollment (not for evaluation)
      clip_01.wav         # ≥ 4 s voiced, 16 kHz mono
      clip_02.wav
      ...
    inference/
      quiet/              # household quiet room
        clip_01.wav       # ≥ 1.5 s voiced
        ...
      moderate/           # background TV / music at ~60 dB
        clip_01.wav
        ...
      crowded/            # outdoor / playground noise
        clip_01.wav
        ...
  impostor_{n}/           # ≥ 20 different unenrolled speakers
    inference/
      quiet/
        clip_01.wav
      moderate/
      crowded/
  edge_cases/
    overlapping_speech/
    music_with_lyrics/
    clothing_rustle/
    phone_speaker/
    distant_speech/
```

**Minimum quantities:**
- ≥ 3 enrolled speakers × 5 enrollment clips × ≥ 4 s
- ≥ 3 enrolled speakers × 20 inference clips × 3 conditions
- ≥ 2 child speakers × 10 inference clips
- ≥ 20 impostor speakers × 5 clips × 3 conditions
- 10 edge-case clips per category

## Consent Requirements

All recordings require explicit informed consent.  For child speakers,
written parental consent is mandatory.  Store consent records (name, date,
purpose) alongside the data.  Do not commit recordings to version control.
This is a legal requirement, not a best practice.  Verify requirements in
your jurisdiction with a qualified professional.

## Running the harness

```bash
cd c:\Users\AYUSH VERMA\Documents\sem6\speaker-backend
python -m tests.eval.harness --data-root eval_data/ --enrolled parent_a parent_b
```

## EER target (operational guidance)

Do not set an EER target without first running the harness on real data and
establishing a baseline.  Choosing a number before measurement is arbitrary.
Once you have the baseline, the goal is: FAR-per-hour on familiar-only audio ≤ 2.
