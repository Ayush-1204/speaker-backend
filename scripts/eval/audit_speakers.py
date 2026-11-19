"""Find speaker IDs that hold more than one voice (label noise in crowdsourced corpora).

Crowdsourced corpora such as Kathbath sometimes file two people under one
contributor ID (a shared phone or account).  Their same-speaker trials then look
like recognition errors.  For each speaker this embeds a sample of files, splits
them into two clusters and flags the ID when both clusters are tight but unlike
each other, i.e. two voices.  Feed the output to public_benchmark.py
--exclude-speakers and report both the raw and the cleaned numbers.

    python -m scripts.eval.audit_speakers --manifest m.csv --out flagged.csv --embedding-cache emb_cache
"""

import argparse
import csv
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.utils import vad  # noqa: E402
from scripts.eval.datasets import SR, read_manifests  # noqa: E402
from scripts.eval.public_benchmark import Embedder  # noqa: E402


def two_voice_split(vectors: np.ndarray) -> Tuple[np.ndarray, Optional[float], Optional[float], Optional[float]]:
    """2-means on unit vectors, seeded by the least similar pair.
    Returns (labels, within_a, within_b, between)."""
    sim = vectors @ vectors.T
    i, j = np.unravel_index(np.argmin(sim), sim.shape)
    labels = (sim[:, j] > sim[:, i]).astype(int)
    for _ in range(20):
        if labels.min() == labels.max():
            break
        centres = [vectors[labels == k].mean(0) for k in (0, 1)]
        new = (vectors @ centres[1] > vectors @ centres[0]).astype(int)
        if np.array_equal(new, labels):
            break
        labels = new
    a, b = vectors[labels == 0], vectors[labels == 1]

    def within(m: np.ndarray) -> Optional[float]:
        return float(np.mean((m @ m.T)[np.triu_indices(len(m), 1)])) if len(m) > 1 else None

    between = float(np.mean(a @ b.T)) if len(a) and len(b) else None
    return labels, within(a), within(b), between


def is_mixed(n_a: int, n_b: int, wa: Optional[float], wb: Optional[float], between: Optional[float],
             min_files: int = 3, max_between: float = 0.45, min_within: float = 0.55) -> bool:
    if wa is None or wb is None or between is None or min(n_a, n_b) < min_files:
        return False
    return between <= max_between and min(wa, wb) >= min_within and min(wa, wb) - between >= 0.2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", nargs="*", default=None)
    ap.add_argument("--files", type=int, default=14, help="Files embedded per speaker")
    ap.add_argument("--embedding-cache", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    Embedder.cache_dir = args.embedding_cache
    rows = read_manifests(args.manifest)
    by_spk: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for r in rows:
        if not args.groups or any(k.lower() in r.group.lower() for k in args.groups):
            by_spk[(r.group, r.speaker)].append(r.path)

    vad.get_speech_segments(np.zeros(SR, dtype=np.float32), sr=SR, use_silero=True)
    vad.restore_torch_threads()

    rng = random.Random(args.seed)
    out_rows = []
    for (group, spk), files in sorted(by_spk.items()):
        emb = Embedder()
        vecs = []
        for f in rng.sample(files, min(args.files, len(files))):
            e = emb.enrollment(f)
            if e:
                v = np.mean(e, axis=0)
                vecs.append(v / (np.linalg.norm(v) + 1e-9))
        if len(vecs) < 6:
            continue
        labels, wa, wb, between = two_voice_split(np.stack(vecs))
        n_b = int(labels.sum())
        n_a = len(labels) - n_b
        mixed = is_mixed(n_a, n_b, wa, wb, between)
        out_rows.append({"speaker": spk, "group": group, "files": len(vecs), "cluster_sizes": f"{n_a}+{n_b}",
                         "within_a": round(wa, 3) if wa is not None else "",
                         "within_b": round(wb, 3) if wb is not None else "",
                         "between": round(between, 3) if between is not None else "", "mixed": int(mixed)})
        if mixed:
            print(f"  two voices: {spk} ({group}) {n_a}+{n_b} files, within {wa:.2f}/{wb:.2f}, between {between:.2f}",
                  flush=True)

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()) if out_rows else ["speaker", "mixed"])
        w.writeheader()
        w.writerows(out_rows)
    flagged = sum(r["mixed"] for r in out_rows)
    print(f"Checked {len(out_rows)} speakers; {flagged} hold two voices. Wrote {args.out}")


if __name__ == "__main__":
    main()
