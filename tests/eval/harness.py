"""Evaluation harness for SafeEar speaker verification pipeline (Change 9).

Computes EER, minDCF, false-alarms-per-hour, and detection latency P95
from a structured dataset.  See tests/eval/README.md for dataset format.

Usage
-----
    python -m tests.eval.harness --data-root eval_data/ --enrolled parent_a parent_b
    python -m tests.eval.harness --help

Design
------
The harness does NOT import FastAPI or SQLAlchemy — it calls the ML functions
directly so it can run without a running server.  Enrolled speaker embeddings
are created on-the-fly from the enrollment splits.

All functions are importable for use in pytest-based ablation tests.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    clip_path: str
    speaker_id: str
    condition: str
    ground_truth: str       # "target" | "impostor" | "edge"
    score: float            # final cosine / normalised score
    decision: str           # "familiar" | "stranger" | "child_self_match" | "rejected" | etc.
    alert_fired: bool
    processing_ms: float
    distress_detected: bool = False
    tier1_passed: bool = False
    tier2_passed: bool = False
    tier3_passed: bool = False


@dataclass
class EvalDataset:
    enrolled: Dict[str, List[str]]   # speaker_id -> list of enrollment wav paths
    target: Dict[str, List[str]]     # speaker_id -> list of inference wav paths
    impostor: Dict[str, List[str]]   # speaker_id -> list of impostor wav paths
    edge: Dict[str, List[str]]       # category -> list of edge-case wav paths


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_eval_dataset(root: str, enrolled_speakers: Optional[List[str]] = None) -> EvalDataset:
    """Load a dataset in the directory structure described in README.md."""
    root_path = Path(root)
    enrolled: Dict[str, List[str]] = {}
    target: Dict[str, List[str]] = {}
    impostor: Dict[str, List[str]] = {}
    edge: Dict[str, List[str]] = {}

    for speaker_dir in sorted(root_path.iterdir()):
        if not speaker_dir.is_dir():
            continue
        sid = speaker_dir.name

        if sid.startswith("impostor"):
            for cond_dir in sorted((speaker_dir / "inference").iterdir()) if (speaker_dir / "inference").exists() else []:
                if cond_dir.is_dir():
                    clips = sorted(str(p) for p in cond_dir.glob("*.wav"))
                    impostor.setdefault(sid, []).extend(clips)
            continue

        if sid == "edge_cases":
            for cat_dir in sorted(speaker_dir.iterdir()):
                if cat_dir.is_dir():
                    clips = sorted(str(p) for p in cat_dir.glob("*.wav"))
                    edge[cat_dir.name] = clips
            continue

        # Enrolled speaker
        if enrolled_speakers and sid not in enrolled_speakers:
            continue

        enroll_dir = speaker_dir / "enrollment"
        if enroll_dir.exists():
            enrolled[sid] = sorted(str(p) for p in enroll_dir.glob("*.wav"))

        infer_dir = speaker_dir / "inference"
        if infer_dir.exists():
            for cond_dir in sorted(infer_dir.iterdir()):
                if cond_dir.is_dir():
                    clips = sorted(str(p) for p in cond_dir.glob("*.wav"))
                    target.setdefault(sid, []).extend(clips)

    logger.info(
        "DATASET_LOADED | enrolled_speakers=%d | target_clips=%d | impostor_clips=%d | edge_clips=%d",
        len(enrolled),
        sum(len(v) for v in target.values()),
        sum(len(v) for v in impostor.values()),
        sum(len(v) for v in edge.values()),
    )
    return EvalDataset(enrolled=enrolled, target=target, impostor=impostor, edge=edge)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_eer(
    target_scores: List[float],
    impostor_scores: List[float],
    n_thresholds: int = 1000,
) -> Tuple[float, float]:
    """Return (eer_threshold, eer_fraction)."""
    if not target_scores or not impostor_scores:
        return 0.5, 0.5
    from app.utils.calibration import find_eer_threshold
    return find_eer_threshold(target_scores, impostor_scores, n_thresholds)


def compute_min_dcf(
    target_scores: List[float],
    impostor_scores: List[float],
    c_miss: float = 10.0,
    c_fa: float = 1.0,
    p_target: float = 0.1,
    n_thresholds: int = 1000,
) -> float:
    """Compute minDCF with asymmetric costs (miss >> false alarm for safety).

    minDCF = min over thresholds of:
        C_miss * P_target * FRR + C_fa * (1 - P_target) * FAR
    """
    if not target_scores or not impostor_scores:
        return 1.0

    t_arr = np.asarray(target_scores, dtype=np.float32)
    i_arr = np.asarray(impostor_scores, dtype=np.float32)
    mn = float(min(t_arr.min(), i_arr.min())) - 1e-6
    mx = float(max(t_arr.max(), i_arr.max())) + 1e-6

    min_dcf = float("inf")
    for t in np.linspace(mn, mx, n_thresholds):
        far = float(np.mean(i_arr >= t))
        frr = float(np.mean(t_arr < t))
        dcf = c_miss * p_target * frr + c_fa * (1.0 - p_target) * far
        if dcf < min_dcf:
            min_dcf = dcf

    # Normalise by prior cost
    prior_cost = min(c_miss * p_target, c_fa * (1.0 - p_target))
    return min_dcf / prior_cost if prior_cost > 0 else min_dcf


def compute_false_alarms_per_hour(
    results: List[PipelineResult],
    duration_sec: float,
) -> float:
    """False-alarm rate: alerts on familiar-only audio per hour."""
    if duration_sec <= 0:
        return float("inf")
    familiar_false_alerts = sum(
        1 for r in results
        if r.ground_truth == "target" and r.alert_fired
    )
    return familiar_false_alerts / (duration_sec / 3600.0)


def compute_detection_latency_p95(
    results: List[PipelineResult],
) -> Optional[float]:
    """P95 processing latency in ms over all clips that triggered an alert."""
    alert_latencies = [r.processing_ms for r in results if r.alert_fired]
    if not alert_latencies:
        return None
    return float(np.percentile(alert_latencies, 95))


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------

def run_ablation(
    dataset: EvalDataset,
    config_variants: Dict[str, dict],
) -> "dict":
    """Run the evaluation harness for each config variant.

    Returns a dict: variant_name -> {eer, min_dcf, fa_per_hour, latency_p95}
    """
    results: Dict[str, dict] = {}
    for name, cfg in config_variants.items():
        logger.info("ABLATION | running variant: %s", name)
        # Placeholder: integrate with full pipeline runner once dataset is available.
        # Each variant should override specific config keys (e.g., ALERT_TRIGGER_STREAK=1).
        results[name] = {
            "variant": name,
            "config": cfg,
            "note": "Run with real dataset once eval_data/ is populated per README.md",
        }
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SafeEar evaluation harness")
    parser.add_argument("--data-root", required=True, help="Path to eval_data/ directory")
    parser.add_argument(
        "--enrolled", nargs="+", default=None,
        help="Speaker IDs to enroll (defaults to all non-impostor directories)",
    )
    parser.add_argument("--output", default=None, help="Write JSON results to this path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    dataset = load_eval_dataset(args.data_root, enrolled_speakers=args.enrolled)

    print("\n=== SafeEar Evaluation Harness ===")
    print(f"Enrolled speakers : {list(dataset.enrolled.keys())}")
    print(f"Target clips      : {sum(len(v) for v in dataset.target.values())}")
    print(f"Impostor clips    : {sum(len(v) for v in dataset.impostor.values())}")
    print(f"Edge-case clips   : {sum(len(v) for v in dataset.edge.values())}")
    print("\nDataset loaded.  Integrate with run_pipeline_on_clip() once")
    print("the full pipeline runner is wired in.  See README.md for next steps.")

    if args.output:
        summary = {
            "enrolled": list(dataset.enrolled.keys()),
            "target_clips": sum(len(v) for v in dataset.target.values()),
            "impostor_clips": sum(len(v) for v in dataset.impostor.values()),
        }
        Path(args.output).write_text(json.dumps(summary, indent=2))
        print(f"\nSummary written to {args.output}")


if __name__ == "__main__":
    main()
