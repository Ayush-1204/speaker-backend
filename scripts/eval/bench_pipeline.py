"""Measure detection latency per 1.5 s window, per stage and end to end.

Feeds overlapping windows from WAV files through the real models and the real
evaluate_window() (with an in-memory gallery, so no database is needed) and
reports p50/p95 milliseconds.  --legacy reproduces the pre-fix pipeline
(energy VAD, one embedding per VAD fragment, torch on 1 thread) for comparison.

Usage (from speaker-backend):
    python -m scripts.eval.bench_pipeline --wav clip1.wav clip2.wav
    python -m scripts.eval.bench_pipeline --wav clip1.wav --legacy
"""

import argparse
import os
import sys
import time
from typing import Callable, Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def _load(path: str) -> np.ndarray:
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    if sr != 16000:
        import librosa

        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
    return data.astype(np.float32)


def _windows(wavs: List[np.ndarray], n: int, win: int = 24000, hop: int = 16000) -> List[np.ndarray]:
    out = []
    for w in wavs:
        for s in range(0, max(1, len(w) - win + 1), hop):
            out.append(w[s:s + win])
    return out[:n] if n else out


def _time(fn: Callable[[], object], runs: List[float]) -> object:
    t = time.perf_counter()
    result = fn()
    runs.append((time.perf_counter() - t) * 1000.0)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav", nargs="+", required=True, help="Speech recordings to slice into 1.5 s windows")
    ap.add_argument("--windows", type=int, default=40)
    ap.add_argument("--speakers", type=int, default=4, help="Synthetic enrolled speakers in the gallery")
    ap.add_argument("--refs", type=int, default=10, help="Voiceprints per enrolled speaker")
    ap.add_argument("--legacy", action="store_true", help="Pre-fix pipeline for comparison")
    args = ap.parse_args()

    if args.legacy:
        os.environ["SAFEEAR_VAD_BACKEND"] = "energy"
        os.environ["SAFEEAR_INFERENCE_EMBED"] = "segments"
        os.environ["SAFEEAR_TORCH_THREADS"] = "1"

    import torch

    from app.utils import feature_extractor as fx
    from app.utils import prd_services_db as svc
    from app.utils import vad
    from app.utils import yamnet_classifier as yc

    fx._get_redimnet()
    yc._get_yamnet_model()
    vad._get_silero()
    vad.restore_torch_threads()
    if args.legacy:
        svc.INFERENCE_EMBED_MODE = "segments"

    rng = np.random.default_rng(0)
    gallery: Dict[str, Dict] = {}
    for i in range(args.speakers):
        embs = [rng.normal(size=192).astype(np.float32) for _ in range(args.refs)]
        gallery[f"speaker{i}"] = {"embs": embs, "role": "familiar_adult", "matrix": svc._unit_rows(embs)}
    svc.load_parent_embeddings = lambda db, pid: dict(gallery)  # no database needed

    windows = _windows([_load(p) for p in args.wav], args.windows)
    if not windows:
        sys.exit("No 1.5 s windows in the given files.")

    # Warm-up (first calls pay one-time setup costs).
    for w in windows[:2]:
        svc.evaluate_window(None, "bench", w)

    stages: Dict[str, List[float]] = {k: [] for k in ("vad", "yamnet", "embedding", "scoring", "evaluate_window")}
    outcomes: Dict[str, int] = {}
    query = None
    for w in windows:
        segs = _time(lambda: vad.get_speech_segments(w), stages["vad"])
        voiced = np.concatenate([w[s:e] for s, e in segs]) if segs else w
        _time(lambda: yc.classify_audio_event_windowed(voiced), stages["yamnet"])
        query = _time(lambda: fx.embed_waveform_chunks(voiced, 16000, None, drop_short_tails=True), stages["embedding"])
        _time(lambda: svc._score_against_candidates(gallery, query), stages["scoring"])
        stage = _time(lambda: svc.evaluate_window(None, "bench", w), stages["evaluate_window"])
        outcomes[stage["decision"]] = outcomes.get(stage["decision"], 0) + 1

    mode = "LEGACY (energy VAD, per-fragment embeddings, 1 thread)" if args.legacy else "CURRENT"
    print(f"\n{mode} | torch threads={torch.get_num_threads()} | windows={len(windows)} | "
          f"gallery={args.speakers}x{args.refs}")
    print(f"{'stage':<16} {'p50 ms':>8} {'p95 ms':>8} {'max ms':>8}")
    for name, runs in stages.items():
        a = np.asarray(runs)
        print(f"{name:<16} {np.percentile(a, 50):8.1f} {np.percentile(a, 95):8.1f} {a.max():8.1f}")
    e2e = np.asarray(stages["evaluate_window"])
    print(f"\nreal-time factor at 1 window/s: {np.percentile(e2e, 95) / 1000.0:.3f} (p95)")
    print("window outcomes:", outcomes)
    print("(The per-stage rows always use the current method; evaluate_window reflects the selected mode.)")


if __name__ == "__main__":
    main()
