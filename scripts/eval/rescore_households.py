"""Re-score captured household windows with a different scoring configuration.

The household simulator stores, for every window, the query embedding ("centroid")
and the Tier-1/2 outcome.  This tool rebuilds each household's gallery exactly
(the simulator's random choices are replayed with scoring switched off; no audio
is embedded, enrollment embeddings come from the cache), verifies that the
production scorer reproduces the stored scores, and writes new capture files in
which `score` / `closest_speaker_id` come from the production scorer with the
requested configuration (e.g. AS-norm).  The replay tool then runs unchanged.

    python -m scripts.eval.rescore_households --manifest m.csv --exclude-speakers audit.csv \
        --embedding-cache C --noise-dir ESC/audio --capture "eval/house_*/capture" \
        --asnorm app/data/asnorm --out eval/house_asnorm
"""

import argparse
import glob
import json
import os
import random
import sys
import tempfile
from collections import defaultdict
from types import SimpleNamespace
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.utils import prd_services_db as svc  # noqa: E402
from app.utils import score_normaliser as sn  # noqa: E402
from scripts.eval import simulate_households as sim  # noqa: E402
from scripts.eval.datasets import load_audio, read_manifests  # noqa: E402
from scripts.eval.public_benchmark import Embedder, read_exclusions  # noqa: E402


def rebuild_galleries(args) -> Dict[str, Dict]:
    Embedder.cache_dir = args.embedding_cache
    excluded = read_exclusions(args.exclude_speakers)
    rows = [r for r in read_manifests(args.manifest) if r.speaker not in excluded]
    by_group = defaultdict(list)
    for r in rows:
        by_group[r.group].append(r)
    files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(args.noise_dir) for f in fs if f.endswith((".wav", ".flac")))
    random.Random(args.seed).shuffle(files)
    noise = [n for n in (load_audio(f) for f in files[: args.noise_clips]) if np.sqrt(np.mean(n ** 2)) > 1e-3]
    sim_args = SimpleNamespace(households=args.households, family=args.family, enroll_utts=args.enroll_utts,
                               familiar_sessions=args.familiar_sessions, familiar_sec=args.familiar_sec,
                               stranger_sessions=args.stranger_sessions, stranger_sec=args.stranger_sec)
    sim.score_stream = lambda stream, household, device, t0, fh: (len(stream) - sim.WINDOW) // sim.WINDOW + 1 \
        if len(stream) >= sim.WINDOW else 0
    tmp = tempfile.mkdtemp(prefix="rescore_")
    for g in args.groups:
        sim.simulate_group(g, by_group[g], sim_args, random.Random(f"{args.seed}:{g}"), noise, tmp, [])
        print(f"  rebuilt {g}", flush=True)
    return dict(sim._GALLERIES)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", action="append", required=True)
    ap.add_argument("--exclude-speakers", default=None)
    ap.add_argument("--embedding-cache", required=True)
    ap.add_argument("--noise-dir", required=True)
    ap.add_argument("--noise-clips", type=int, default=300)
    ap.add_argument("--capture", required=True, help="glob of capture folders")
    ap.add_argument("--asnorm", default=None, help="AS-norm folder (cohort.npy + asnorm.json); omit for raw cosine")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gallery-cache", default=None, help="pickle of rebuilt galleries (written once, then reused)")
    ap.add_argument("--households", type=int, default=3)
    ap.add_argument("--family", type=int, default=3)
    ap.add_argument("--enroll-utts", type=int, default=3)
    ap.add_argument("--familiar-sessions", type=int, default=2)
    ap.add_argument("--familiar-sec", type=float, default=240.0)
    ap.add_argument("--stranger-sessions", type=int, default=2)
    ap.add_argument("--stranger-sec", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    records: List[dict] = []
    for d in sorted(glob.glob(args.capture)):
        for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            with open(p, encoding="utf-8") as fh:
                records += [json.loads(line) for line in fh if line.strip()]
    args.groups = sorted({r["parent_id"].rsplit("#", 1)[0] for r in records})
    cache = args.gallery_cache
    if cache and os.path.exists(cache):
        import pickle
        with open(cache, "rb") as fh:
            galleries = pickle.load(fh)
        print(f"{len(records)} windows; galleries loaded from {cache}", flush=True)
    else:
        print(f"{len(records)} windows from {len(args.groups)} groups; rebuilding galleries", flush=True)
        galleries = rebuild_galleries(args)

    # 1) Raw scorer must reproduce the stored scores (proves the galleries are identical).
    sn.set_asnorm(None)
    diffs = []
    for r in records:
        if r.get("tier3_pass") and r.get("centroid"):
            s, _, _ = svc._score_against_candidates(galleries[r["parent_id"]], [np.asarray(r["centroid"], dtype=np.float32)])
            diffs.append(abs(s - r["score"]))
    print(f"reproduction check: {len(diffs)} windows, max |diff| {max(diffs):.5f}, "
          f"share > 0.001: {np.mean(np.asarray(diffs) > 1e-3):.4f}", flush=True)
    if max(diffs) > 0.01:
        raise SystemExit("stored scores not reproduced: galleries differ, aborting")
    if cache and not os.path.exists(cache):
        import pickle
        with open(cache, "wb") as fh:
            pickle.dump({k: {s: {"embs": v["embs"], "role": v.get("role", "familiar_adult")} for s, v in g.items()}
                         for k, g in galleries.items()}, fh)

    # 2) Re-score with the requested configuration.
    sn.set_asnorm(sn.AsNorm.load(args.asnorm) if args.asnorm else None)
    by_device: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        r = dict(r)
        if r.get("tier3_pass") and r.get("centroid"):
            out: Dict[str, float] = {}
            s, sid, role = svc._score_against_candidates(galleries[r["parent_id"]],
                                                         [np.asarray(r["centroid"], dtype=np.float32)], scores_out=out)
            r.update(score=float(s), closest_speaker_id=sid, closest_speaker_role=role, speaker_scores=out)
        by_device[r["device_id"]].append(r)
    cap = os.path.join(args.out, "capture")
    os.makedirs(cap, exist_ok=True)
    for dev, rs in by_device.items():
        with open(os.path.join(cap, f"{dev}.jsonl"), "w", encoding="utf-8") as fh:
            for r in rs:
                fh.write(json.dumps(r, separators=(",", ":")) + "\n")
    print(f"wrote {sum(len(v) for v in by_device.values())} windows to {cap}")


if __name__ == "__main__":
    main()
