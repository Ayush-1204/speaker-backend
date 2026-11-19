"""Simulated households from public speech, scored by the real detection pipeline.

For each language/accent group in a manifest (scripts/eval/build_manifest.py):
  - a household enrolls 3 speakers from 3 utterances each (as the app does);
  - a familiar-only session streams several minutes of those speakers talking;
  - stranger sessions stream an unenrolled speaker of the same group, alone and
    taking turns with a family member;
  - each session is clean or mixed with background noise (--noise-dir, e.g. ESC-50).
Audio is cut into contiguous 1.5 s windows, like the child device sends, and each
window goes through the production evaluate_window() (VAD, YAMNet, embedding,
scoring) with the household's gallery.  The output is in the server's eval-capture
format plus a labelled sessions.csv, so replay.py measures false alerts per hour,
detection rate and time to alert, and --sweep tunes T_HIGH / T_LOW / streak:

    python -m scripts.eval.simulate_households --manifest m.csv --noise-dir <ESC-50/audio> --out sim
    python -m scripts.eval.replay --capture sim/capture --labels sim/sessions.csv --sweep

Adults only, and studio/phone recordings: treat the result as a starting point and
confirm on recordings from the real device before changing production thresholds.
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.utils import prd_services_db as svc  # noqa: E402
from app.utils import vad  # noqa: E402
from scripts.eval.datasets import SR, Utterance, load_audio, read_manifests  # noqa: E402
from scripts.eval.public_benchmark import Embedder, read_exclusions  # noqa: E402

WINDOW = int(1.5 * SR)
SESSION_GAP_MS = 5 * 60 * 1000  # longer than the debounce and the stream-gap reset
CONDITIONS = ["clean", "noise 10dB", "noise 5dB"]

_GALLERIES: Dict[str, Dict[str, Dict]] = {}


def _gallery_for(_db, parent_id: str) -> Dict[str, Dict]:
    return _GALLERIES[parent_id]


def _thresholds(_db, _parent_id: str) -> Tuple[float, float]:
    return svc.T_HIGH, svc.T_LOW


def build_stream(clips: List[np.ndarray], rng: random.Random, seconds: float) -> np.ndarray:
    """Clips joined with short pauses until `seconds` of audio."""
    parts: List[np.ndarray] = []
    total = 0
    while total < seconds * SR and clips:
        clip = rng.choice(clips)
        pause = np.zeros(int(rng.uniform(0.3, 1.5) * SR), dtype=np.float32)
        parts += [clip, pause]
        total += len(clip) + len(pause)
    return np.concatenate(parts)[: int(seconds * SR)] if parts else np.zeros(0, dtype=np.float32)


def turn_taking(a: List[np.ndarray], b: List[np.ndarray], rng: random.Random, seconds: float) -> np.ndarray:
    """Alternate one clip from each speaker (a conversation)."""
    parts: List[np.ndarray] = []
    total = 0
    turn = 0
    while total < seconds * SR:
        clip = rng.choice(a if turn % 2 == 0 else b)
        pause = np.zeros(int(rng.uniform(0.3, 1.0) * SR), dtype=np.float32)
        parts += [clip, pause]
        total += len(clip) + len(pause)
        turn += 1
    return np.concatenate(parts)[: int(seconds * SR)]


def add_noise(stream: np.ndarray, noise_clips: List[np.ndarray], snr_db: float, rng: random.Random) -> np.ndarray:
    if not noise_clips or not len(stream):
        return stream
    noise = np.concatenate([rng.choice(noise_clips) for _ in range(int(len(stream) / (5 * SR)) + 2)])
    start = rng.randrange(0, len(noise) - len(stream) + 1)
    noise = noise[start:start + len(stream)]
    voiced = stream[np.abs(stream) > 1e-4]
    p_speech = float(np.mean(voiced ** 2)) if len(voiced) else 1e-6
    p_noise = float(np.mean(noise ** 2)) + 1e-12
    return (stream + np.sqrt(p_speech / (p_noise * 10 ** (snr_db / 10.0))) * noise).astype(np.float32)


def score_stream(stream: np.ndarray, household: str, device: str, t0_ms: int, out_fh) -> int:
    """Contiguous 1.5 s windows through evaluate_window(); one capture record each."""
    n = 0
    for i in range(0, len(stream) - WINDOW + 1, WINDOW):
        stage = svc.evaluate_window(None, household, stream[i:i + WINDOW], sr=SR)
        tier2 = stage.get("tier2", {}) or {}
        tier3 = stage.get("tier3", {}) or {}
        centroid = stage.get("_query_centroid")
        record = {
            "ts_ms": t0_ms + ((i + WINDOW) * 1000) // SR,  # the server sees a window when it ends
            "parent_id": household,
            "device_id": device,
            "tier1_pass": bool((stage.get("tier1_vad") or {}).get("passed", False)),
            "tier2_pass": bool(tier2.get("passed", False)),
            "tier2_category": str(tier2.get("category", "uncertain")),
            "tier2_confidence": float(tier2.get("confidence", 0.0) or 0.0),
            "tier3_pass": bool(tier3.get("passed", False)),
            "score": float(tier3.get("score", 0.0) or 0.0),
            "closest_speaker_id": tier3.get("closest_speaker_id"),
            "closest_speaker_role": tier3.get("closest_speaker_role"),
            "speaker_scores": {k: round(float(v), 4) for k, v in (tier3.get("speaker_scores") or {}).items()},
            "centroid": [round(float(x), 5) for x in np.asarray(centroid).reshape(-1)] if centroid is not None else None,
        }
        out_fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        n += 1
    return n


def simulate_group(group: str, rows: List[Utterance], args, rng: random.Random, noise: List[np.ndarray],
                   capture_dir: str, sessions: List[Dict]) -> int:
    by_spk: Dict[str, List[str]] = defaultdict(list)
    for r in rows:
        by_spk[r.speaker].append(r.path)
    eligible = sorted(s for s, f in by_spk.items() if len(f) >= args.enroll_utts + 4)
    rng.shuffle(eligible)
    if len(eligible) < args.family + 2:
        print(f"    skipped: {len(eligible)} usable speakers")
        return 0

    emb = Embedder()

    def clips(spk: str, files: List[str]) -> List[np.ndarray]:
        return [w for w in (emb.load(f) for f in files) if w is not None and len(w) > SR]

    windows = 0
    n_house = min(args.households, len(eligible) // args.family)
    for h in range(n_house):
        family = eligible[h * args.family:(h + 1) * args.family]
        strangers = [s for s in eligible if s not in family]
        household = f"{group}#{h}"
        device = household.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
        test_clips: Dict[str, List[np.ndarray]] = {}
        gallery: Dict[str, Dict] = {}
        for spk in family:
            files = sorted(by_spk[spk])
            rng.shuffle(files)
            embs = [e for f in files[: args.enroll_utts] for e in emb.enrollment(f)]
            if embs:
                gallery[spk] = {"embs": embs, "role": "familiar_adult"}
                test_clips[spk] = clips(spk, files[args.enroll_utts: args.enroll_utts + 12])
        if len(gallery) < 2:
            continue
        _GALLERIES[household] = gallery

        t = 1_700_000_000_000 + rng.randrange(0, 10 ** 9)
        plan: List[Tuple[str, str, np.ndarray]] = []
        family_clips = [c for spk in gallery for c in test_clips[spk]]
        for _ in range(args.familiar_sessions):
            plan.append(("familiar_only", "family talking", build_stream(family_clips, rng, args.familiar_sec)))
        for k in range(args.stranger_sessions):
            stranger = strangers[(h * args.stranger_sessions + k) % len(strangers)]
            s_clips = clips(stranger, sorted(by_spk[stranger])[:10])
            if not s_clips:
                continue
            if k % 2 == 0:
                plan.append(("stranger_present", "stranger alone", build_stream(s_clips, rng, args.stranger_sec)))
            else:
                plan.append(("stranger_present", "stranger with family",
                             turn_taking(family_clips, s_clips, rng, args.stranger_sec)))

        with open(os.path.join(capture_dir, f"{device}.jsonl"), "w", encoding="utf-8") as fh:
            for label, kind, stream in plan:
                cond = rng.choice(CONDITIONS if noise else ["clean"])
                if cond != "clean":
                    stream = add_noise(stream, noise, float(cond.split()[1][:-2]), rng)
                windows += score_stream(stream, household, device, t, fh)
                end = t + (len(stream) * 1000) // SR
                sessions.append({"start": t, "end": end, "label": label, "device_id": device,
                                 "notes": f"{group}|{cond}|{kind}"})
                t = end + SESSION_GAP_MS
    return windows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", nargs="*", default=None)
    ap.add_argument("--households", type=int, default=3, help="Households per group")
    ap.add_argument("--family", type=int, default=3, help="Enrolled speakers per household")
    ap.add_argument("--enroll-utts", type=int, default=3)
    ap.add_argument("--familiar-sessions", type=int, default=2)
    ap.add_argument("--familiar-sec", type=float, default=120.0)
    ap.add_argument("--stranger-sessions", type=int, default=2)
    ap.add_argument("--stranger-sec", type=float, default=40.0)
    ap.add_argument("--noise-dir", default=None)
    ap.add_argument("--noise-clips", type=int, default=300, help="Noise clips sampled from --noise-dir")
    ap.add_argument("--embedding-cache", default=None, help="Folder to keep enrollment embeddings across runs")
    ap.add_argument("--exclude-speakers", default=None, help="audit_speakers.py output (IDs with two voices)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    Embedder.cache_dir = args.embedding_cache
    excluded = read_exclusions(args.exclude_speakers)
    rows = [r for r in read_manifests(args.manifest) if r.speaker not in excluded]
    by_group: Dict[str, List[Utterance]] = defaultdict(list)
    for r in rows:
        by_group[r.group].append(r)
    groups = sorted(g for g in by_group if not args.groups or any(k.lower() in g.lower() for k in args.groups))

    noise: List[np.ndarray] = []
    if args.noise_dir:
        files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(args.noise_dir)
                       for f in fs if f.endswith((".wav", ".flac")))
        random.Random(args.seed).shuffle(files)
        noise = [n for n in (load_audio(f) for f in files[: args.noise_clips]) if np.sqrt(np.mean(n ** 2)) > 1e-3]
        print(f"Loaded {len(noise)} noise clips")

    svc.load_parent_embeddings = _gallery_for
    svc.get_household_thresholds = _thresholds
    vad.get_speech_segments(np.zeros(SR, dtype=np.float32), sr=SR, use_silero=True)
    vad.restore_torch_threads()

    capture_dir = os.path.join(args.out, "capture")
    os.makedirs(capture_dir, exist_ok=True)
    sessions: List[Dict] = []
    for g in groups:
        t0 = time.time()
        print(f"[{g}]", flush=True)
        n = simulate_group(g, by_group[g], args, random.Random(f"{args.seed}:{g}"), noise, capture_dir, sessions)
        print(f"    {n} windows in {time.time() - t0:.0f}s", flush=True)
        with open(os.path.join(args.out, "sessions.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["start", "end", "label", "device_id", "notes"])
            w.writeheader()
            w.writerows(sessions)
    hours = sum((s["end"] - s["start"]) for s in sessions if s["label"] == "familiar_only") / 3.6e6
    print(f"\n{len(sessions)} sessions ({hours:.1f} h familiar-only). Next:\n"
          f"  python -m scripts.eval.replay --capture {capture_dir} --labels {args.out}/sessions.csv --sweep")


if __name__ == "__main__":
    main()
