"""Component ablations for the detection cascade (results feed the paper / design notes).

A1  Gate study.  1.5 s windows of speech (manifest groups) and of environmental
    sound (ESC-50, by its five major categories) go through Tier 1 with Silero VAD
    and with the energy VAD, then through Tier 2 (speech gate + YAMNet).  Reports
    how much speech each configuration keeps and how much non-speech reaches the
    speaker model.
A2  Enrollment method.  Per speaker, three ~10 s "enrollment recordings" are
    embedded with the current method (voiced speech concatenated, short tails
    dropped) and the previous one (one padded embedding per VAD segment).
    Reports the enrollment-quality score the app shows (0.65 median + 0.35 Q1 of
    pairwise cosine) and its good/fair/poor label.
A3  Inference method.  Window EER (1.5 s) for each enrollment x inference method.

    python -m scripts.eval.ablations --manifest m.csv --groups LibriSpeech Hindi \
        --esc50 <ESC-50-master> --out ablations.json
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.utils import vad  # noqa: E402
from app.utils.audio_preprocess import normalize_waveform  # noqa: E402
from app.utils.calibration import find_eer_threshold  # noqa: E402
from app.utils.feature_extractor import embed_waveform_chunks  # noqa: E402
from app.utils.prd_services_db import _score_against_candidates  # noqa: E402
from app.utils.speech_gate import assess_speech_likeness  # noqa: E402
from app.utils.yamnet_classifier import classify_audio_event_windowed  # noqa: E402
from scripts.eval.datasets import SR, load_audio, read_manifests  # noqa: E402

WIN = int(1.5 * SR)
ESC_MAJOR = ["Animals", "Natural soundscapes", "Human non-speech", "Domestic", "Urban noise"]


# ------------------------------------------------------------------ A1 gates

def segments(w: np.ndarray, backend: str) -> List[Tuple[int, int]]:
    os.environ["SAFEEAR_VAD_BACKEND"] = backend
    try:
        return vad.get_speech_segments(w, sr=SR, use_silero=True)
    finally:
        os.environ["SAFEEAR_VAD_BACKEND"] = "silero"


def gate(window: np.ndarray, backend: str) -> Dict[str, object]:
    """Tier 1 and Tier 2 exactly as evaluate_window applies them."""
    w = normalize_waveform(window)
    if float(np.sqrt(np.mean(w ** 2) + 1e-12)) < 0.003:
        return {"tier1": False, "tier2": False, "category": "silence"}
    segs = segments(w, backend)
    if sum(e - s for s, e in segs) <= int(0.5 * SR):
        return {"tier1": False, "tier2": False, "category": "no_speech"}
    voiced = np.concatenate([w[s:e] for s, e in segs]).astype(np.float32)
    speech_ok, _ = assess_speech_likeness(voiced, SR)
    y = classify_audio_event_windowed(voiced, SR)
    cat, conf = y.get("category", "uncertain"), float(y.get("confidence", 0.0))
    hard_reject = (not speech_ok) or cat == "reject" or (cat == "vocal_non_speech" and conf >= 0.70)
    # Only these windows can add to a stranger streak (decision_engine.allow_stranger_increment).
    stranger_evidence = (not hard_reject) and ((cat == "human_speech" and conf >= 0.70) or
                                               (cat == "uncertain" and conf >= 0.88))
    return {"tier1": True, "tier2": not hard_reject, "category": cat, "stranger_evidence": stranger_evidence}


def windows_from(wave: np.ndarray, n: int, rng: random.Random) -> List[np.ndarray]:
    if len(wave) < WIN:
        return []
    starts = list(range(0, len(wave) - WIN + 1, WIN))
    rng.shuffle(starts)
    return [wave[s:s + WIN] for s in starts[:n]]


def run_gates(speech_files: List[str], esc_root: str, rng: random.Random, n_speech: int, n_per_cat: int) -> Dict:
    pools: Dict[str, List[np.ndarray]] = defaultdict(list)
    for f in speech_files:
        if len(pools["Speech"]) >= n_speech:
            break
        pools["Speech"] += windows_from(load_audio(f), 2, rng)
    meta = os.path.join(esc_root, "meta", "esc50.csv")
    by_cat: Dict[str, List[str]] = defaultdict(list)
    with open(meta, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            by_cat[ESC_MAJOR[int(row["target"]) // 10]].append(os.path.join(esc_root, "audio", row["filename"]))
    for cat, files in by_cat.items():
        rng.shuffle(files)
        for f in files:
            if len(pools[cat]) >= n_per_cat:
                break
            pools[cat] += windows_from(load_audio(f), 1, rng)

    out: Dict[str, Dict] = {}
    for cat, wins in pools.items():
        row: Dict[str, object] = {"n": len(wins)}
        for backend in ("energy", "silero"):
            res = [gate(w, backend) for w in wins]
            row[f"{backend}_tier1"] = float(np.mean([r["tier1"] for r in res]))
            row[f"{backend}_tier2"] = float(np.mean([r["tier2"] for r in res]))
            row[f"{backend}_stranger_evidence"] = float(np.mean([bool(r.get("stranger_evidence")) for r in res]))
            if backend == "silero":
                row["silero_categories"] = dict(Counter(r["category"] for r in res))
        out[cat] = row
        print(f"  {cat:<20} n={row['n']:>4}  energy T1 {row['energy_tier1']:.0%}  silero T1 {row['silero_tier1']:.0%}"
              f"  silero+YAMNet {row['silero_tier2']:.0%}  stranger-evidence {row['silero_stranger_evidence']:.0%}",
              flush=True)
    return out


# ------------------------------------------------------- A2 / A3 embeddings

def embed(wave: np.ndarray, method: str, enrollment: bool) -> List[np.ndarray]:
    w = normalize_waveform(wave)
    segs = vad.get_speech_segments(w, sr=SR, use_silero=True)
    if method == "concat":
        if enrollment:
            return embed_waveform_chunks(w, SR, segments=segs, drop_short_tails=True, concat_segments=True)
        if sum(e - s for s, e in segs) <= SR // 2:
            return []
        voiced = np.concatenate([w[s:e] for s, e in segs]).astype(np.float32)
        return embed_waveform_chunks(voiced, SR, None, drop_short_tails=True)
    # Previous method: one embedding per VAD segment, short segments reflect-padded.
    if not enrollment and sum(e - s for s, e in segs) <= SR // 2:
        return []
    return embed_waveform_chunks(w, SR, segments=segs)


def quality(embs: List[np.ndarray]) -> Tuple[Optional[float], str]:
    """Same formula and labels as compute_and_store_enrollment_quality()."""
    if len(embs) < 2:
        return None, "pending"
    X = np.stack([e / (np.linalg.norm(e) + 1e-9) for e in embs])
    sims = (X @ X.T)[np.triu_indices(len(X), 1)]
    med, q1 = float(np.median(sims)), float(np.percentile(sims, 25))
    robust = 0.65 * med + 0.35 * q1
    label = "good" if robust >= 0.74 and q1 >= 0.62 else "fair" if robust >= 0.52 else "poor"
    return robust, label


def run_embedding_ablation(by_spk: Dict[str, List[str]], rng: random.Random, n_speakers: int,
                           windows_per_spk: int) -> Dict:
    speakers = [s for s in sorted(by_spk) if len(by_spk[s]) >= 6]
    rng.shuffle(speakers)
    speakers = speakers[:n_speakers]
    galleries: Dict[str, Dict[str, List[np.ndarray]]] = {"concat": {}, "segments": {}}
    qual: Dict[str, List[Tuple[Optional[float], str]]] = {"concat": [], "segments": []}
    tests: Dict[str, List[np.ndarray]] = {}
    for spk in speakers:
        files = sorted(by_spk[spk])
        rng.shuffle(files)
        recordings = [load_audio(f)[: 10 * SR] for f in files[:3]]
        for method in ("concat", "segments"):
            embs = [e for r in recordings for e in embed(r, method, enrollment=True)]
            galleries[method][spk] = embs
            qual[method].append(quality(embs))
        wins: List[np.ndarray] = []
        for f in files[3:9]:
            wins += windows_from(load_audio(f), 1, rng)
        tests[spk] = wins[:windows_per_spk]

    out: Dict[str, object] = {"speakers": len(speakers)}
    for method in ("concat", "segments"):
        vals = [q for q, _ in qual[method] if q is not None]
        labels = Counter(lbl for _, lbl in qual[method])
        out[f"enroll_{method}"] = {"quality_mean": float(np.mean(vals)) if vals else None,
                                   "quality_median": float(np.median(vals)) if vals else None,
                                   "labels": dict(labels)}
    for e_method in ("concat", "segments"):
        for i_method in ("concat", "segments"):
            target: List[float] = []
            impostor: List[float] = []
            for spk in speakers:
                for w in tests[spk]:
                    q = embed(w, i_method, enrollment=False)
                    if not q:
                        continue
                    for enrolled in speakers:
                        g = galleries[e_method][enrolled]
                        if not g:
                            continue
                        s, _, _ = _score_against_candidates({"g": {"embs": g, "role": "familiar_adult"}}, q)
                        (target if enrolled == spk else impostor).append(float(s))
            thr, eer = find_eer_threshold(target, impostor)
            out[f"eer_enroll_{e_method}_infer_{i_method}"] = {"eer": eer, "thr": thr,
                                                               "n_target": len(target), "n_impostor": len(impostor)}
            print(f"  enroll={e_method:<8} infer={i_method:<8} window EER {eer:.1%} "
                  f"({len(target)} target / {len(impostor)} impostor)", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", action="append", required=True)
    ap.add_argument("--groups", nargs="+", required=True, help="Speech groups used for the ablations")
    ap.add_argument("--esc50", required=True, help="ESC-50-master folder")
    ap.add_argument("--speech-windows", type=int, default=400)
    ap.add_argument("--noise-windows", type=int, default=200, help="Per ESC-50 major category")
    ap.add_argument("--speakers", type=int, default=20, help="Speakers per group for A2/A3")
    ap.add_argument("--test-windows", type=int, default=6)
    ap.add_argument("--exclude-speakers", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from scripts.eval.public_benchmark import read_exclusions

    excluded = read_exclusions(args.exclude_speakers)
    rows = [r for r in read_manifests(args.manifest) if r.speaker not in excluded]
    vad.get_speech_segments(np.zeros(SR, dtype=np.float32), sr=SR, use_silero=True)
    vad.restore_torch_threads()
    rng = random.Random(args.seed)

    result: Dict[str, object] = {}
    speech_files = [r.path for r in rows if any(g.lower() in r.group.lower() for g in args.groups)]
    rng.shuffle(speech_files)
    print("A1 gates", flush=True)
    result["gates"] = run_gates(speech_files, args.esc50, rng, args.speech_windows, args.noise_windows)

    for g in args.groups:
        by_spk: Dict[str, List[str]] = defaultdict(list)
        for r in rows:
            if g.lower() in r.group.lower() and "noisy" not in r.group.lower():
                by_spk[r.speaker].append(r.path)
        print(f"A2/A3 {g}: {len(by_spk)} speakers", flush=True)
        result[f"embedding_{g}"] = run_embedding_ablation(by_spk, random.Random(f"{args.seed}:{g}"),
                                                          args.speakers, args.test_windows)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
