"""Speaker-verification benchmark on public speech, reported per language and accent.

What this measures, and what it does not
  It runs SafeEar's own enrollment, VAD, embedding and scoring code on public
  corpora (build them with scripts/eval/build_manifest.py).  It shows how well the
  embedding model separates speakers across languages, accents and noise, and lets
  two pipeline versions be compared on identical trials.  The corpora are adults on
  phones and studio mics; the product hears children and families through a worn
  mic, so confirm thresholds on your own recordings (scripts/eval/replay.py).

Protocols
  pairs    Standard EER: one utterance against another from the same group.
           Comparable with published numbers.  With --kathbath-meta, Kathbath groups
           use the official IndicSUPERB trial lists.
  safeear  What the product does: enroll each speaker from a few utterances, then
           score 1.5 s windows (clean, an overlapping talker, noise) against them.
  Impostors always come from the same group: a stranger near a child usually shares
  the family's language and accent, which is the hard case.  Speakers recorded in
  more than one language also get a "cross-language" group (enroll in one, test in
  another).

Usage (from speaker-backend):
    python -m scripts.eval.public_benchmark --manifest eval_manifest.csv --out eval_results
    python -m scripts.eval.public_benchmark --manifest m.csv --protocol pairs --kathbath-meta <meta_data>
    python -m scripts.eval.public_benchmark --librispeech <path>/LibriSpeech/test-clean
"""

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.utils import vad  # noqa: E402
from app.utils.audio_preprocess import normalize_waveform  # noqa: E402
from app.utils.calibration import find_eer_threshold  # noqa: E402
from app.utils.feature_extractor import embed_waveform_chunks  # noqa: E402
from app.utils.prd_services_db import T_HIGH, T_LOW, _score_against_candidates  # noqa: E402
from scripts.eval.datasets import SR, Utterance, librispeech, load_audio, read_manifests  # noqa: E402

Conditions = List[Tuple[str, float, Optional[float], str]]


class Embedder:
    """Caches audio and enrollment-style embeddings per file for one group.  With
    `cache_dir` set, enrollment embeddings are also kept on disk across runs."""

    cache_dir: Optional[str] = None

    def __init__(self) -> None:
        self.audio: Dict[str, Optional[np.ndarray]] = {}
        self.embs: Dict[str, List[np.ndarray]] = {}

    def load(self, path: str) -> Optional[np.ndarray]:
        if path not in self.audio:
            try:
                self.audio[path] = load_audio(path)
            except Exception as exc:  # unreadable file: skip it, keep the run going
                print(f"    skip {os.path.basename(path)}: {exc}")
                self.audio[path] = None
        return self.audio[path]

    @classmethod
    def cache_path(cls, path: str) -> str:
        # Keyed by file name (unique within these corpora) so the cache survives moving the data.
        key = hashlib.sha1(f"enroll-v2|{os.path.basename(path)}".encode("utf-8")).hexdigest()
        return os.path.join(cls.cache_dir or "", key[:2], key + ".npy")

    def enrollment(self, path: str) -> List[np.ndarray]:
        if path in self.embs:
            return self.embs[path]
        disk = None
        if self.cache_dir:
            disk = self.cache_path(path)
            if os.path.exists(disk):
                arr = np.load(disk)
                self.embs[path] = [row for row in arr]
                return self.embs[path]
        wave = self.load(path)
        embs = [] if wave is None or len(wave) < SR // 2 else embed_like_enrollment(wave)
        if disk and wave is not None:  # never cache a file that failed to load
            os.makedirs(os.path.dirname(disk), exist_ok=True)
            np.save(disk, np.stack(embs) if embs else np.zeros((0, 0), dtype=np.float32))
        self.embs[path] = embs
        return embs


def embed_like_enrollment(wave: np.ndarray) -> List[np.ndarray]:
    w = normalize_waveform(wave)
    segs = vad.get_speech_segments(w, sr=SR, use_silero=True)
    return embed_waveform_chunks(w, SR, segments=segs, drop_short_tails=True, concat_segments=True)


def embed_like_inference(window: np.ndarray) -> List[np.ndarray]:
    """As evaluate_window does: windows with <= 0.5 s of speech are rejected (empty list),
    otherwise the voiced audio is joined and embedded."""
    w = normalize_waveform(window)
    segs = vad.get_speech_segments(w, sr=SR, use_silero=True)
    if sum(e - s for s, e in segs) <= SR // 2:
        return []
    voiced = np.concatenate([w[s:e] for s, e in segs]).astype(np.float32, copy=False)
    return embed_waveform_chunks(voiced, SR, None, drop_short_tails=True)


def score(gallery_embs: List[np.ndarray], query_embs: List[np.ndarray]) -> Optional[float]:
    if not gallery_embs or not query_embs:
        return None
    s, _, _ = _score_against_candidates({"g": {"embs": gallery_embs, "role": "familiar_adult"}}, query_embs)
    return float(s)


def mix(clean: np.ndarray, other: np.ndarray, snr_db: float, rng: random.Random) -> np.ndarray:
    if len(other) < len(clean):
        other = np.tile(other, int(np.ceil(len(clean) / max(1, len(other)))))
    start = rng.randrange(0, len(other) - len(clean) + 1)
    other = other[start:start + len(clean)]
    p_clean = float(np.mean(clean ** 2)) + 1e-12
    p_other = float(np.mean(other ** 2)) + 1e-12
    return (clean + np.sqrt(p_clean / (p_other * 10 ** (snr_db / 10.0))) * other).astype(np.float32)


def window_of(wave: np.ndarray, seconds: float, rng: random.Random) -> Optional[np.ndarray]:
    n = int(seconds * SR)
    if len(wave) < n:
        return None
    start = rng.randrange(0, len(wave) - n + 1)
    return wave[start:start + n]


def summarize(protocol: str, group: str, condition: str, target: Sequence[float],
              impostor: Sequence[float]) -> Optional[Dict]:
    if len(target) < 3 or len(impostor) < 3:
        return None
    t, i = np.asarray(target), np.asarray(impostor)
    thr, eer = find_eer_threshold(list(target), list(impostor))
    return {
        "protocol": protocol, "group": group, "condition": condition,
        "n_target": len(t), "n_impostor": len(i), "eer": eer, "thr_at_eer": thr,
        "target_median": float(np.median(t)), "impostor_median": float(np.median(i)),
        "target_ge_t_high": float(np.mean(t >= T_HIGH)), "target_le_t_low": float(np.mean(t <= T_LOW)),
        "impostor_ge_t_high": float(np.mean(i >= T_HIGH)), "impostor_le_t_low": float(np.mean(i <= T_LOW)),
    }


# ---------------------------------------------------------------- pairs protocol

def official_kathbath_trials(meta_root: str, rows: List[Utterance]) -> List[Tuple[int, str, str]]:
    """IndicSUPERB trial list for this group's language and split, mapped onto local files.
    Only trials between files in `rows` are kept (callers pass a random subset of files
    so embeddings are reused across many trials)."""
    by_stem = {os.path.splitext(os.path.basename(r.path))[0]: r.path for r in rows}
    noisy = rows[0].dataset == "kathbath_noisy"
    name = "test_noisy_data.txt" if noisy else "test_data.txt"
    path = os.path.join(meta_root, rows[0].language, name)
    trials: List[Tuple[int, str, str]] = []
    if not os.path.exists(path):
        return trials
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            bits = line.split()
            if len(bits) != 3:
                continue
            a = by_stem.get(os.path.splitext(os.path.basename(bits[1]))[0])
            b = by_stem.get(os.path.splitext(os.path.basename(bits[2]))[0])
            if a and b and a != b:
                trials.append((int(bits[0]), a, b))
    return trials


def sampled_trials(by_spk: Dict[str, List[str]], n: int, rng: random.Random) -> List[Tuple[int, str, str]]:
    """Every same-speaker pair, plus as many different-speaker pairs (all within the files given)."""
    target = [(1, a, b) for files in by_spk.values() for i, a in enumerate(files) for b in files[i + 1:]]
    rng.shuffle(target)
    target = target[: n // 2]
    spks = [s for s in by_spk if by_spk[s]]
    impostor = []
    if len(spks) >= 2:
        for _ in range(len(target)):
            s1, s2 = rng.sample(spks, 2)
            impostor.append((0, rng.choice(by_spk[s1]), rng.choice(by_spk[s2])))
    return target + impostor


def run_pairs(group: str, rows: List[Utterance], args, rng: random.Random, scores_out: List) -> None:
    by_spk: Dict[str, List[str]] = defaultdict(list)
    for r in rows:
        by_spk[r.speaker].append(r.path)
    trials: List[Tuple[int, str, str]] = []
    source = "sampled"
    if args.prefer_cached and Embedder.cache_dir:
        # Reuse files whose embeddings already exist (e.g. from audit_speakers.py, which
        # samples files at random): the subset stays random, the compute is saved.
        cached = [r for r in rows if os.path.exists(Embedder.cache_path(r.path))]
        if len(cached) >= min(len(rows), 100):
            rows = cached
            by_spk = defaultdict(list)
            for r in rows:
                by_spk[r.speaker].append(r.path)
    if args.kathbath_meta and rows[0].dataset.startswith("kathbath"):
        # A random subset of files, then every official trial between them.
        subset = rng.sample(rows, min(len(rows), args.pair_files))
        trials = official_kathbath_trials(args.kathbath_meta, subset)
        if trials:
            source = "official"
            rng.shuffle(trials)
            trials = trials[: args.pairs]
    if not trials:
        keep = rng.sample(sorted(by_spk), min(len(by_spk), args.pair_speakers))
        per_spk = max(2, args.pair_files // max(1, len(keep)))
        trials = sampled_trials({s: rng.sample(by_spk[s], min(len(by_spk[s]), per_spk)) for s in keep},
                                args.pairs, rng)

    spk_of = {r.path: r.speaker for r in rows}
    emb = Embedder()
    for label, a, b in trials:
        s = score(emb.enrollment(a), emb.enrollment(b))
        if s is not None:
            scores_out.append(("pairs", group, f"utterance pairs ({source})", label, s, spk_of[a], spk_of[b]))


# ---------------------------------------------------------------- safeear protocol

def conditions_for(noise_files: List[str]) -> Conditions:
    conds: Conditions = [
        ("clean 1.0s", 1.0, None, ""),
        ("clean 1.5s", 1.5, None, ""),
        ("clean 2.0s", 2.0, None, ""),
        ("clean 3.0s", 3.0, None, ""),
        ("talker 10dB 1.5s", 1.5, 10.0, "talker"),
        ("talker 5dB 1.5s", 1.5, 5.0, "talker"),
        ("talker 0dB 1.5s", 1.5, 0.0, "talker"),
    ]
    if noise_files:
        conds += [(f"noise {snr:g}dB 1.5s", 1.5, snr, "noise") for snr in (10.0, 5.0, 0.0)]
    return conds


def run_safeear(group: str, enroll_files: Dict[str, List[str]], test_files: Dict[str, List[str]],
                args, rng: random.Random, noise_files: List[str], scores_out: List) -> None:
    emb = Embedder()
    gallery: Dict[str, List[np.ndarray]] = {}
    test_audio: Dict[str, List[np.ndarray]] = {}
    for spk in enroll_files:
        embs = [e for f in enroll_files[spk] for e in emb.enrollment(f)]
        waves = [w for w in (emb.load(f) for f in test_files[spk]) if w is not None]
        if embs and waves:
            gallery[spk], test_audio[spk] = embs, waves
    speakers = sorted(gallery)
    if len(speakers) < 3:
        return

    for name, seconds, snr, kind in conditions_for(noise_files):
        for spk in speakers:
            for _ in range(args.trials):
                window = window_of(rng.choice(test_audio[spk]), seconds, rng)
                if window is None:
                    continue
                if kind == "talker":
                    other = rng.choice([s for s in speakers if s != spk])
                    window = mix(window, rng.choice(test_audio[other]), snr, rng)
                elif kind == "noise":
                    noise = emb.load(rng.choice(noise_files))
                    if noise is None or not len(noise):
                        continue
                    window = mix(window, noise, snr, rng)
                q = embed_like_inference(window)
                for enrolled in speakers:
                    s = score(gallery[enrolled], q)
                    if s is not None:
                        scores_out.append(("safeear", group, name, int(enrolled == spk), s, enrolled, spk))


def safeear_splits(rows: List[Utterance], args, rng: random.Random):
    by_spk: Dict[str, List[str]] = defaultdict(list)
    for r in rows:
        by_spk[r.speaker].append(r.path)
    eligible = sorted(s for s, f in by_spk.items() if len(f) >= args.enroll_utts + 2)
    chosen = rng.sample(eligible, min(len(eligible), args.speakers))
    enroll, test = {}, {}
    for spk in chosen:
        files = sorted(by_spk[spk])
        rng.shuffle(files)
        enroll[spk] = files[: args.enroll_utts]
        test[spk] = files[args.enroll_utts: args.enroll_utts + 6]
    return enroll, test


def cross_language_splits(rows: List[Utterance], args, rng: random.Random):
    """Speakers with recordings in 2+ languages: enroll in one, test in another."""
    by_spk_lang: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.language:
            by_spk_lang[r.speaker][r.language].append(r.path)
    enroll, test = {}, {}
    for spk, langs in by_spk_lang.items():
        ok = [l for l, f in langs.items() if len(f) >= 2]
        if len(ok) < 2:
            continue
        a, b = rng.sample(ok, 2)
        enroll[spk] = langs[a][: args.enroll_utts]
        test[spk] = langs[b][:6]
    keep = rng.sample(sorted(enroll), min(len(enroll), args.speakers))
    return {s: enroll[s] for s in keep}, {s: test[s] for s in keep}


# ---------------------------------------------------------------- reporting

COLS = [("eer", "EER", "{:>6.1%}"), ("thr_at_eer", "thr@EER", "{:>8.3f}"),
        ("target_median", "tgt med", "{:>8.3f}"), ("impostor_median", "imp med", "{:>8.3f}"),
        ("target_le_t_low", "tgt<=T_LOW", "{:>10.1%}"), ("impostor_ge_t_high", "imp>=T_HIGH", "{:>11.1%}")]


def print_rows(rows: List[Dict]) -> None:
    head = f"{'group':<34} {'condition':<28} {'n':>6} " + " ".join(f"{h:>{len(f.format(0.0))}}" for _, h, f in COLS)
    print(head)
    for r in rows:
        cells = " ".join(f.format(r[k]) for k, _, f in COLS)
        print(f"{r['group'][:34]:<34} {r['condition'][:28]:<28} {r['n_target'] + r['n_impostor']:>6} {cells}")




def summaries(scores: List[Tuple]) -> List[Dict]:
    """One row per (protocol, group, condition), then pooled rows across groups."""
    split: Dict[Tuple[str, str, str], Tuple[List[float], List[float]]] = defaultdict(lambda: ([], []))
    for p, g, cond, label, s, *_ in scores:
        split[(p, g, cond)][0 if int(label) else 1].append(float(s))
        split[(p, "ALL GROUPS (pooled)", cond.split(" (")[0])][0 if int(label) else 1].append(float(s))
    rows = [summarize(p, g, c, t, i) for (p, g, c), (t, i) in split.items()]
    order = {"ALL GROUPS (pooled)": 1}
    return sorted((r for r in rows if r), key=lambda r: (r["protocol"], order.get(r["group"], 0), r["group"]))


def threshold_guidance(scores: List[Tuple]) -> Optional[Dict]:
    t = np.asarray([float(r[4]) for r in scores if r[0] == "safeear" and r[2] == "clean 1.5s" and int(r[3]) == 1])
    i = np.asarray([float(r[4]) for r in scores if r[0] == "safeear" and r[2] == "clean 1.5s" and int(r[3]) == 0])
    if len(t) < 20 or len(i) < 20:
        return None
    return {
        "t_high_for_1pct_strangers_as_familiar": float(np.quantile(i, 0.99)),
        "t_low_for_2pct_family_windows_as_stranger": float(np.quantile(t, 0.02)),
        "current_t_high": T_HIGH, "current_t_low": T_LOW,
    }


def report(scores: List[Tuple], out: Optional[str], args) -> None:
    results = summaries(scores)
    for protocol in ("pairs", "safeear"):
        part = [r for r in results if r["protocol"] == protocol]
        if not part:
            continue
        print(f"\n=== {protocol} ===")
        print_rows(part)
        main_cond = "utterance pairs" if protocol == "pairs" else "clean 1.5s"
        per_group = [r["eer"] for r in part if not r["group"].startswith("ALL")
                     and r["condition"].startswith(main_cond)]
        if per_group:
            print(f"Mean EER across {len(per_group)} groups ({main_cond}): {np.mean(per_group):.1%}"
                  f"  (best {min(per_group):.1%}, worst {max(per_group):.1%})")

    guide = threshold_guidance(scores)
    if guide:
        print(f"\nThresholds from safeear 'clean 1.5s', pooled (current T_HIGH={T_HIGH}, T_LOW={T_LOW}):")
        print(f"  T_HIGH >= {guide['t_high_for_1pct_strangers_as_familiar']:.3f} keeps strangers scored as "
              "family to 1% of windows")
        print(f"  T_LOW  <= {guide['t_low_for_2pct_family_windows_as_stranger']:.3f} keeps family windows "
              "scored as stranger to 2%")
        print("  Alerts also need 3 stranger windows in a row, so per-alert error rates are lower than these.")
    print("\nAdults on phones and studio mics; confirm on your own recordings before changing thresholds.")

    if out:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "results.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(results[0].keys()) if results else ["group"])
            w.writeheader()
            w.writerows(results)
        with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as fh:
            json.dump({"args": vars(args), "results": results, "thresholds": guide}, fh, indent=2)
        print(f"Saved results to {out}")


def read_scores(paths: List[str]) -> List[Tuple]:
    scores: List[Tuple] = []
    for p in paths:
        with open(p, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                scores.append((r["protocol"], r["group"], r["condition"], int(r["label"]), float(r["score"]),
                               r.get("speaker_a", ""), r.get("speaker_b", "")))
    return scores


def read_exclusions(path: Optional[str]) -> set:
    """Speaker ids to leave out: a CSV with a `speaker` column (audit_speakers.py writes one;
    only rows with mixed=1 are used when that column exists) or one id per line."""
    if not path:
        return set()
    with open(path, encoding="utf-8", newline="") as fh:
        text = fh.read()
    lines = text.splitlines()
    if lines and "speaker" in lines[0].split(","):
        rows = list(csv.DictReader(lines))
        return {r["speaker"] for r in rows if r.get("mixed", "1") in ("1", "True", "true")}
    return {ln.strip() for ln in lines if ln.strip()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", action="append", default=[], help="Manifest CSV (repeatable)")
    ap.add_argument("--librispeech", default=None, help="Shortcut: LibriSpeech/test-clean without a manifest")
    ap.add_argument("--from-scores", nargs="*", default=None,
                    help="Skip scoring; report from scores.csv files (e.g. parallel runs)")
    ap.add_argument("--protocol", choices=["pairs", "safeear", "both"], default="both")
    ap.add_argument("--groups", nargs="*", default=None, help="Only groups whose name contains one of these")
    ap.add_argument("--speakers", type=int, default=16, help="Speakers per group (safeear)")
    ap.add_argument("--pair-speakers", type=int, default=40, help="Speakers per group (sampled pairs)")
    ap.add_argument("--pair-files", type=int, default=300, help="Files embedded per group (pairs)")
    ap.add_argument("--pairs", type=int, default=3000, help="Max trials per group (pairs)")
    ap.add_argument("--enroll-utts", type=int, default=3, help="Utterances per speaker for enrollment")
    ap.add_argument("--trials", type=int, default=6, help="Test windows per speaker per condition")
    ap.add_argument("--noise-dir", default=None)
    ap.add_argument("--kathbath-meta", default=None, help="IndicSUPERB meta_data folder (official trials)")
    ap.add_argument("--no-cross-language", action="store_true")
    ap.add_argument("--exclude-speakers", default=None,
                    help="Speakers to leave out (audit_speakers.py output: IDs that hold two voices)")
    ap.add_argument("--embedding-cache", default=None, help="Folder to keep enrollment embeddings across runs")
    ap.add_argument("--prefer-cached", action="store_true",
                    help="Pairs: draw the file subset from files already in the embedding cache")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="Folder for results.csv, scores.csv, summary.json")
    args = ap.parse_args()

    excluded = read_exclusions(args.exclude_speakers)
    if args.from_scores:
        scores = read_scores(args.from_scores)
        if excluded:
            before = len(scores)
            scores = [r for r in scores if r[5] not in excluded and r[6] not in excluded]
            print(f"Excluded {before - len(scores)} scores involving {len(excluded)} flagged speakers")
        report(scores, args.out, args)
        return

    Embedder.cache_dir = args.embedding_cache
    rows: List[Utterance] = read_manifests(args.manifest) if args.manifest else []
    if args.librispeech:
        rows += list(librispeech(args.librispeech))
    rows = [r for r in rows if r.speaker not in excluded]
    if not rows:
        raise SystemExit("Give --manifest (see build_manifest.py), --librispeech or --from-scores")

    by_group: Dict[str, List[Utterance]] = defaultdict(list)
    for r in rows:
        by_group[r.group].append(r)
    groups = sorted(g for g in by_group if not args.groups or any(k.lower() in g.lower() for k in args.groups))

    noise_files: List[str] = []
    if args.noise_dir:
        for ext in ("wav", "flac"):
            noise_files += [os.path.join(dp, f) for dp, _, fs in os.walk(args.noise_dir) for f in fs if f.endswith(ext)]

    # Silero switches torch to one thread when it loads; load it, then restore.
    vad.get_speech_segments(np.zeros(SR, dtype=np.float32), sr=SR, use_silero=True)
    vad.restore_torch_threads()

    # Scores are written after every group, so a long run can be stopped and still reported.
    score_fh = None
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        score_fh = open(os.path.join(args.out, "scores.csv"), "w", newline="", encoding="utf-8")
        csv.writer(score_fh).writerow(["protocol", "group", "condition", "label", "score", "speaker_a", "speaker_b"])

    scores: List[Tuple] = []

    def flush(start: int) -> None:
        if score_fh:
            csv.writer(score_fh).writerows(scores[start:])
            score_fh.flush()

    for g in groups:
        rng = random.Random(f"{args.seed}:{g}")
        n_spk = len({r.speaker for r in by_group[g]})
        t0, start = time.time(), len(scores)
        print(f"[{g}] {n_spk} speakers, {len(by_group[g])} utterances", flush=True)
        if args.protocol in ("pairs", "both"):
            run_pairs(g, by_group[g], args, rng, scores)
        if args.protocol in ("safeear", "both"):
            enroll, test = safeear_splits(by_group[g], args, rng)
            run_safeear(g, enroll, test, args, rng, noise_files, scores)
        flush(start)
        print(f"    {len(scores) - start} scores in {time.time() - t0:.0f}s", flush=True)

    if args.protocol in ("safeear", "both") and not args.no_cross_language:
        selected = [r for g in groups for r in by_group[g]]
        enroll, test = cross_language_splits(selected, args, random.Random(args.seed))
        if len(enroll) >= 3:
            start = len(scores)
            print(f"[cross-language] {len(enroll)} speakers", flush=True)
            run_safeear("Cross-language (enroll A, test B)", enroll, test, args,
                        random.Random(args.seed), noise_files, scores)
            flush(start)

    if score_fh:
        score_fh.close()
    report(scores, args.out, args)


if __name__ == "__main__":
    main()
