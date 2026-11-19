"""Fit adaptive score normalisation (AS-norm) on development speakers only.

Speakers come from corpora splits that never appear in the test protocols
(Kathbath *valid*, LibriSpeech *dev-clean*).  Per group, the speakers are split
once, deterministically, into
    cohort speakers  -> utterance embeddings that form the AS-norm cohort
    dev speakers     -> enrolled from 3 utterances; 1.5 s windows are cut from
                        other utterances and scored exactly like evaluate_window()

    embed  (run in parallel on group subsets; uses the shared embedding cache)
        python -m scripts.eval.asnorm_dev embed --manifest dev.csv --embedding-cache C --out d/w0.pkl --groups ...
    fit    (selects top-N on the dev windows, calibrates to the cosine scale)
        python -m scripts.eval.asnorm_dev fit --inputs d/w*.pkl --out app/data/asnorm --report d/fit.json
"""

import argparse
import glob
import json
import os
import pickle
import random
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.utils import score_normaliser as sn  # noqa: E402
from app.utils import vad  # noqa: E402
from scripts.eval.datasets import read_manifests  # noqa: E402
from scripts.eval.public_benchmark import Embedder, embed_like_inference, score, window_of  # noqa: E402

T_HIGH, T_LOW = 0.47, 0.31


def split_speakers(by_spk: Dict[str, List[str]], group: str, n_cohort: int, n_dev: int, need: int):
    eligible = sorted(s for s, f in by_spk.items() if len(f) >= need)
    random.Random(f"asnorm-split:{group}").shuffle(eligible)
    return eligible[:n_cohort], eligible[n_cohort:n_cohort + n_dev]


def cmd_embed(args) -> None:
    Embedder.cache_dir = args.embedding_cache
    rows = read_manifests(args.manifest)
    by_group: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if not args.groups or any(g.lower() in r.group.lower() for g in args.groups):
            by_group[r.group][r.speaker].append(r.path)
    vad.get_speech_segments(np.zeros(16000, dtype=np.float32), sr=16000, use_silero=True)
    vad.restore_torch_threads()
    emb = Embedder()
    out = {"cohort": [], "dev": []}
    for group in sorted(by_group):
        by_spk = by_group[group]
        need = max(args.cohort_utts, args.enroll_utts + 2)
        cohort_spk, dev_spk = split_speakers(by_spk, group, args.cohort_speakers, args.dev_speakers, need)
        print(f"[{group}] cohort {len(cohort_spk)} speakers, dev {len(dev_spk)} speakers", flush=True)
        for spk in cohort_spk:
            files = sorted(by_spk[spk])
            random.Random(f"cohort:{spk}").shuffle(files)
            for f in files[: args.cohort_utts]:
                e = emb.enrollment(f)
                if e:
                    v = np.mean(np.stack(e), axis=0)
                    out["cohort"].append((group, spk, v / (np.linalg.norm(v) + 1e-9)))
        for spk in dev_spk:
            files = sorted(by_spk[spk])
            rng = random.Random(f"dev:{spk}")
            rng.shuffle(files)
            gallery = [e for f in files[: args.enroll_utts] for e in emb.enrollment(f)]
            waves = [w for w in (emb.load(f) for f in files[args.enroll_utts: args.enroll_utts + 2]) if w is not None]
            windows = []
            for _ in range(args.windows):
                if not waves:
                    break
                w = window_of(rng.choice(waves), 1.5, rng)
                if w is None:
                    continue
                q = embed_like_inference(w)
                if q:
                    windows.append(np.stack(q))
            if gallery and windows:
                out["dev"].append({"group": group, "speaker": spk, "gallery": np.stack(gallery), "windows": windows})
        print(f"    cohort entries so far {len(out['cohort'])}, dev speakers {len(out['dev'])}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(out, fh)


def eer(t: np.ndarray, i: np.ndarray) -> float:
    thr = np.unique(np.concatenate([t, i]))
    frr = np.searchsorted(np.sort(t), thr, side="left") / len(t)
    far = 1 - np.searchsorted(np.sort(i), thr, side="left") / len(i)
    k = int(np.argmin(np.abs(far - frr)))
    return float((far[k] + frr[k]) / 2)


def dev_scores(dev: List[Dict]) -> Dict[str, Dict[str, List[float]]]:
    """Window-vs-gallery scores within each group, via the production scorer."""
    out: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"t": [], "i": []})
    by_group = defaultdict(list)
    for d in dev:
        by_group[d["group"]].append(d)
    for group, spks in by_group.items():
        for d in spks:
            for q in d["windows"]:
                for e in spks:
                    s = score(list(e["gallery"]), list(q))
                    if s is not None:
                        out[group]["t" if e is d else "i"].append(s)
    return out


def summarize(sc: Dict[str, Dict[str, List[float]]]) -> Dict:
    t = np.concatenate([np.asarray(v["t"]) for v in sc.values()])
    i = np.concatenate([np.asarray(v["i"]) for v in sc.values()])
    per = {}
    for g, v in sorted(sc.items()):
        tt, ii = np.asarray(v["t"]), np.asarray(v["i"])
        if len(tt) == 0 or len(ii) == 0:  # e.g. a group with a single dev speaker has no impostor trials
            continue
        per[g] = {"eer": eer(tt, ii), "imp_ge_th": float(np.mean(ii >= T_HIGH)), "tgt_le_tl": float(np.mean(tt <= T_LOW)),
                  "imp_le_tl": float(np.mean(ii <= T_LOW)), "imp_median": float(np.median(ii)),
                  "tgt_median": float(np.median(tt))}
    return {"pooled_eer": eer(t, i), "n_t": int(len(t)), "n_i": int(len(i)),
            "t_mean": float(t.mean()), "i_mean": float(i.mean()),
            "pooled_imp_ge_th": float(np.mean(i >= T_HIGH)), "pooled_tgt_le_tl": float(np.mean(t <= T_LOW)),
            "pooled_imp_le_tl": float(np.mean(i <= T_LOW)),
            "mean_group_eer": float(np.mean([p["eer"] for p in per.values()])), "per_group": per}


def cmd_fit(args) -> None:
    data = {"cohort": [], "dev": []}
    for p in sorted(sum((glob.glob(x) for x in args.inputs), [])):
        with open(p, "rb") as fh:
            d = pickle.load(fh)
        data["cohort"] += d["cohort"]
        data["dev"] += d["dev"]
    cohort = np.stack([c[2] for c in data["cohort"]])
    dev_spk = {d["speaker"] for d in data["dev"]}
    assert not dev_spk & {c[1] for c in data["cohort"]}, "cohort and dev speakers overlap"
    print(f"cohort {len(cohort)} utterances / {len({c[1] for c in data['cohort']})} speakers; "
          f"dev {len(data['dev'])} speakers")

    sn.set_asnorm(None)
    base = summarize(dev_scores(data["dev"]))
    print(f"baseline dev: pooled EER {100 * base['pooled_eer']:.2f}%  mean group EER {100 * base['mean_group_eer']:.2f}%")
    report = {"baseline": base, "candidates": {}}
    best = None
    for top_n in args.top_n:
        sn.set_asnorm(sn.AsNorm(cohort, top_n))
        z = summarize(dev_scores(data["dev"]))
        report["candidates"][str(top_n)] = {"pooled_eer": z["pooled_eer"], "mean_group_eer": z["mean_group_eer"]}
        print(f"top-N {top_n:4d}: pooled EER {100 * z['pooled_eer']:.2f}%  mean group EER {100 * z['mean_group_eer']:.2f}%")
        if best is None or z["pooled_eer"] < best[1]["pooled_eer"] - 1e-12:
            best = (top_n, z)
    top_n, z = best
    # Map z onto the cosine scale: pooled dev target / impostor means land where the raw ones were.
    scale = (base["t_mean"] - base["i_mean"]) / (z["t_mean"] - z["i_mean"])
    offset = base["t_mean"] - scale * z["t_mean"]
    norm = sn.AsNorm(cohort, top_n, scale, offset)
    sn.set_asnorm(norm)
    cal = summarize(dev_scores(data["dev"]))
    report.update({"chosen_top_n": top_n, "scale": scale, "offset": offset, "calibrated": cal,
                   "cohort_size": int(len(cohort))})
    print(f"chosen top-N {top_n}, scale {scale:.4f}, offset {offset:.4f}")
    print(f"calibrated dev: pooled EER {100 * cal['pooled_eer']:.2f}%  imp>=T_H {100 * cal['pooled_imp_ge_th']:.2f}% "
          f"(baseline {100 * base['pooled_imp_ge_th']:.2f}%)  tgt<=T_L {100 * cal['pooled_tgt_le_tl']:.2f}% "
          f"(baseline {100 * base['pooled_tgt_le_tl']:.2f}%)  imp<=T_L {100 * cal['pooled_imp_le_tl']:.2f}% "
          f"(baseline {100 * base['pooled_imp_le_tl']:.2f}%)")
    norm.save(args.out)
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed")
    e.add_argument("--manifest", action="append", required=True)
    e.add_argument("--embedding-cache", default=None)
    e.add_argument("--out", required=True)
    e.add_argument("--groups", nargs="*", default=None)
    e.add_argument("--cohort-speakers", type=int, default=12)
    e.add_argument("--dev-speakers", type=int, default=8)
    e.add_argument("--cohort-utts", type=int, default=4)
    e.add_argument("--enroll-utts", type=int, default=3)
    e.add_argument("--windows", type=int, default=6)
    f = sub.add_parser("fit")
    f.add_argument("--inputs", nargs="+", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--report", required=True)
    f.add_argument("--top-n", type=int, nargs="+", default=[25, 50, 100, 200, 400])
    args = ap.parse_args()
    {"embed": cmd_embed, "fit": cmd_fit}[args.cmd](args)


if __name__ == "__main__":
    main()
