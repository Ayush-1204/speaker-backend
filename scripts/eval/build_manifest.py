"""Write (or extend) a benchmark manifest from a downloaded corpus.

Usage (from speaker-backend):
    python -m scripts.eval.build_manifest librispeech <path>/LibriSpeech/test-clean --out eval_manifest.csv
    python -m scripts.eval.build_manifest kathbath <path>/kb_data_clean_m4a --out eval_manifest.csv --append
    python -m scripts.eval.build_manifest crowdsourced <path>/slr83 --out eval_manifest.csv --append
    python -m scripts.eval.build_manifest commonvoice <path>/cv-corpus-xx/hi --out eval_manifest.csv --append
    python -m scripts.eval.build_manifest voxceleb1 <path>/vox1_test/wav --meta vox1_meta.csv --out ... --append
    python -m scripts.eval.build_manifest folder <path>/my_family --group "Our family" --out ... --append

See scripts/eval/datasets.py for the layouts and where each corpus comes from.
"""

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.eval.datasets import ADAPTERS, write_manifest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("adapter", choices=sorted(ADAPTERS))
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--group", default=None, help="Override the group name for every row")
    ap.add_argument("--language", default="", help="Language code, where the layout does not carry one")
    ap.add_argument("--meta", default=None, help="VoxCeleb1 vox1_meta.csv (nationality)")
    args = ap.parse_args()

    rows = list(ADAPTERS[args.adapter](args.root, group=args.group, language=args.language, meta=args.meta))
    if not rows:
        raise SystemExit(f"No audio found for '{args.adapter}' under {args.root}")
    n = write_manifest(rows, args.out, append=args.append)

    speakers = Counter()
    for r in {(r.group, r.speaker) for r in rows}:
        speakers[r[0]] += 1
    print(f"Wrote {n} utterances to {args.out}")
    for group, count in sorted(speakers.items()):
        print(f"  {group:<40} {count:>5} speakers")


if __name__ == "__main__":
    main()
