# Evaluation Harness — SafeEar (Change 9)

> **Start here:** if you have no dataset, follow [RECORDING_PROTOCOL.md](RECORDING_PROTOCOL.md).
> It captures labelled sessions from your own device (`SAFEEAR_EVAL_CAPTURE_DIR`) and
> scores them with `scripts/eval/replay.py`. `scripts/eval/public_benchmark.py` measures
> EER on public corpora across languages and accents (below).

## Public multi-language / multi-accent benchmark

Build one manifest from any mix of corpora, then score it.  Results are reported
per group (language or accent), with impostors drawn from the same group.

| Corpus | Covers | Where | Adapter |
|---|---|---|---|
| Kathbath (clean + noisy test) | 12 Indian languages | `objectstore.e2enetworks.net/indic-superb/` (CC0, no login) | `kathbath` |
| LibriSpeech test-clean | US English | openslr.org/12 | `librispeech` |
| OpenSLR 83 / 70 | UK + Irish dialects, Nigerian English | openslr.org/83, /70 | `crowdsourced` |
| Common Voice | 100+ languages, accent tags | commonvoice.mozilla.org (accept terms) | `commonvoice` |
| VoxCeleb1 test | many nationalities, in-the-wild | Hugging Face mirror (accept terms) | `voxceleb1 --meta vox1_meta.csv` |
| VCTK 0.92 | English accents | datashare.ed.ac.uk | `vctk` |
| NISP, your own recordings | anything | `<root>/<speaker>/[<language>/]*.wav` | `folder` |

```bash
python -m scripts.eval.build_manifest kathbath <dir>/kb_data_clean_m4a --out m.csv
python -m scripts.eval.build_manifest kathbath <dir>/kb_data_noisy_m4a --out m.csv --append
python -m scripts.eval.build_manifest librispeech <dir>/LibriSpeech/test-clean --out m.csv --append
python -m scripts.eval.public_benchmark --manifest m.csv --kathbath-meta <dir>/meta_data \
    --noise-dir <ESC-50 audio> --out results/
# Parallel runs (--groups Hindi Tamil ...) can be merged afterwards:
python -m scripts.eval.public_benchmark --from-scores results_*/scores.csv --out results_all/
```

Two protocols: `pairs` is the standard utterance-vs-utterance EER (Kathbath uses the
official IndicSUPERB trial lists), comparable with published numbers.  `safeear`
enrolls each speaker from 3 utterances and scores 1.5 s windows, clean, with an
overlapping talker and with background noise, which is what the product does.
Speakers recorded in two languages also get a cross-language group.  None of these
corpora contain children, so confirm thresholds on your own recordings.

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
