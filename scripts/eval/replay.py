"""Replay captured detection windows through the decision engine and report metrics.

Inputs
  --capture   directory written by SAFEEAR_EVAL_CAPTURE_DIR (searched recursively
              for *.jsonl) or individual .jsonl files.
  --labels    sessions.csv with columns: start,end,label[,device_id][,notes]
              start/end: local time "YYYY-MM-DD HH:MM:SS" (or epoch ms).
              label: familiar_only | stranger_present | media | silence
              (see tests/eval/RECORDING_PROTOCOL.md).

Outputs
  - false alarms per hour on familiar_only / media / silence sessions
  - detection rate and time-to-first-alert on stranger_present sessions
  - window score distributions and an approximate EER from session labels
  - --sweep: grid search over T_HIGH / T_LOW / streak length, Pareto table
  - --check-live: confirms the replay reproduces the alerts the server fired

Run from the speaker-backend directory:
    python -m scripts.eval.replay --capture eval_capture --labels sessions.csv
    python -m scripts.eval.replay --capture eval_capture --labels sessions.csv --sweep
"""

import argparse
import csv
import dataclasses
import glob
import json
import os
import statistics
import sys
from datetime import datetime
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.services.decision_engine import DecisionParams, decide, mark_identity_alert_fired  # noqa: E402
from app.utils import prd_services_db as svc  # noqa: E402

WINDOW_MS = 1500  # one uploaded chunk
STREAM_GAP_MS = int(os.environ.get("SAFEEAR_STREAM_GAP_MS", "3000"))  # same default as app.main
from app.utils.calibration import find_eer_threshold  # noqa: E402

FALSE_ALARM_LABELS = {"familiar_only", "media", "silence"}
STRANGER_LABEL = "stranger_present"
KNOWN_LABELS = FALSE_ALARM_LABELS | {STRANGER_LABEL}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_windows(paths: List[str]) -> List[dict]:
    files: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(glob.glob(os.path.join(p, "**", "*.jsonl"), recursive=True))
        else:
            files.append(p)
    rows: List[dict] = []
    for f in sorted(files):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    rows.sort(key=lambda r: (r["device_id"], r["ts_ms"]))
    return rows


def _parse_time(value: str) -> int:
    value = value.strip()
    if value.isdigit():
        return int(value)
    return int(datetime.fromisoformat(value).timestamp() * 1000)


@dataclasses.dataclass
class Session:
    start_ms: int
    end_ms: int
    label: str
    device_id: Optional[str]
    notes: str

    @property
    def hours(self) -> float:
        return max(0.0, (self.end_ms - self.start_ms) / 3_600_000.0)

    def contains(self, row: dict) -> bool:
        if self.device_id and row["device_id"] != self.device_id:
            return False
        return self.start_ms <= row["ts_ms"] <= self.end_ms


def load_sessions(path: str) -> List[Session]:
    sessions: List[Session] = []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for rec in csv.DictReader(fh):
            label = (rec.get("label") or "").strip()
            if label not in KNOWN_LABELS:
                raise SystemExit(f"Unknown label '{label}' in {path}; expected one of {sorted(KNOWN_LABELS)}")
            sessions.append(Session(
                start_ms=_parse_time(rec["start"]),
                end_ms=_parse_time(rec["end"]),
                label=label,
                device_id=(rec.get("device_id") or "").strip() or None,
                notes=(rec.get("notes") or "").strip(),
            ))
    return sessions


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #

def simulate(rows: List[dict], params: DecisionParams) -> List[Tuple[str, int]]:
    """Run decide() over the windows; return (device_id, ts_ms) of every alert."""
    sessions: Dict[str, svc.SessionState] = {}
    last_ts: Dict[str, int] = {}
    alerts: List[Tuple[str, int]] = []
    for r in rows:
        dev = r["device_id"]
        ts = int(r["ts_ms"])
        st = sessions.get(dev)
        if st is None:
            st = svc.SessionState(parent_id=r.get("parent_id", ""), device_id=dev)
            sessions[dev] = st
        # Mirror the server: a break in the audio stream (more than STREAM_GAP_MS
        # beyond one window) resets the stranger streak, as in /detect/chunk.
        prev = last_ts.get(dev)
        if prev is not None and ts - prev > WINDOW_MS + STREAM_GAP_MS:
            svc.reset_stream(st)
        last_ts[dev] = ts
        if not r.get("tier3_pass"):
            continue
        centroid = np.asarray(r["centroid"], dtype=np.float32) if r.get("centroid") else None
        result = decide(
            st,
            score=float(r["score"]),
            closest_speaker_id=r.get("closest_speaker_id"),
            closest_speaker_role=r.get("closest_speaker_role"),
            tier2_category=r.get("tier2_category", "uncertain"),
            tier2_confidence=float(r.get("tier2_confidence", 0.0)),
            current_stranger_centroid=centroid,
            now_ms=int(r["ts_ms"]),
            params=params,
        )
        if result.should_alert:
            mark_identity_alert_fired(st, int(r["ts_ms"]), centroid)
            alerts.append((dev, int(r["ts_ms"])))
    return alerts


@dataclasses.dataclass
class Metrics:
    fa_per_hour: float
    fa_hours: float
    fa_alerts: int
    detected: int
    stranger_sessions: int
    latencies_s: List[float]

    @property
    def detection_rate(self) -> float:
        return self.detected / self.stranger_sessions if self.stranger_sessions else float("nan")

    @property
    def median_latency_s(self) -> float:
        return statistics.median(self.latencies_s) if self.latencies_s else float("nan")


def score_alerts(alerts: List[Tuple[str, int]], sessions: List[Session]) -> Metrics:
    fa_hours = sum(s.hours for s in sessions if s.label in FALSE_ALARM_LABELS)
    fa_alerts = 0
    detected = 0
    latencies: List[float] = []
    for s in sessions:
        hits = sorted(ts for dev, ts in alerts if s.contains({"device_id": dev, "ts_ms": ts}))
        if s.label in FALSE_ALARM_LABELS:
            fa_alerts += len(hits)
        elif s.label == STRANGER_LABEL and hits:
            detected += 1
            latencies.append((hits[0] - s.start_ms) / 1000.0)
    return Metrics(
        fa_per_hour=fa_alerts / fa_hours if fa_hours > 0 else float("nan"),
        fa_hours=fa_hours,
        fa_alerts=fa_alerts,
        detected=detected,
        stranger_sessions=sum(1 for s in sessions if s.label == STRANGER_LABEL),
        latencies_s=latencies,
    )


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #

def _label_of(row: dict, sessions: List[Session]) -> Optional[str]:
    for s in sessions:
        if s.contains(row):
            return s.label
    return None


def report_scores(rows: List[dict], sessions: List[Session], params: DecisionParams) -> None:
    familiar = [r["score"] for r in rows if r.get("tier3_pass") and _label_of(r, sessions) == "familiar_only"]
    stranger = [r["score"] for r in rows if r.get("tier3_pass") and _label_of(r, sessions) == STRANGER_LABEL]
    media = [r["score"] for r in rows if r.get("tier3_pass") and _label_of(r, sessions) == "media"]

    def desc(name: str, xs: List[float]) -> None:
        if not xs:
            print(f"  {name:<18} no scored windows")
            return
        q = np.percentile(xs, [5, 25, 50, 75, 95])
        print(f"  {name:<18} n={len(xs):<6} p5={q[0]:.3f} p25={q[1]:.3f} median={q[2]:.3f} p75={q[3]:.3f} p95={q[4]:.3f}")

    print("\nWindow score distribution (tier-3 scored windows only)")
    desc("familiar_only", familiar)
    desc("stranger_present", stranger)
    desc("media", media)
    if familiar:
        print(f"  familiar windows >= T_HIGH ({params.t_high:.2f}): {np.mean(np.asarray(familiar) >= params.t_high):.1%}"
              f" | <= T_LOW ({params.t_low:.2f}): {np.mean(np.asarray(familiar) <= params.t_low):.1%}")
    if familiar and stranger:
        thr, eer = find_eer_threshold(familiar, stranger)
        print(f"  approx. window EER {eer:.1%} at score {thr:.3f}")
        print("  (approximate: stranger sessions also contain the wearer's and family voices,")
        print("   so this understates separability. Use it to compare runs, not as an absolute.)")


def report_alerts(m: Metrics, title: str) -> None:
    print(f"\n{title}")
    print(f"  false alarms: {m.fa_alerts} in {m.fa_hours:.2f} h -> {m.fa_per_hour:.2f} per hour")
    print(f"  stranger sessions detected: {m.detected}/{m.stranger_sessions} ({m.detection_rate:.0%})")
    if m.latencies_s:
        print(f"  time to first alert: median {m.median_latency_s:.1f} s, max {max(m.latencies_s):.1f} s")


def check_live(rows: List[dict], alerts: List[Tuple[str, int]]) -> None:
    live = {(r["device_id"], r["ts_ms"]) for r in rows if r.get("alert_fired") and r.get("decision") != "distress_alert"}
    replayed = set(alerts)
    print("\nLive vs replay (default parameters)")
    print(f"  live identity alerts: {len(live)} | replayed: {len(replayed)} | in both: {len(live & replayed)}")
    if live != replayed:
        print("  MISMATCH: differences usually mean the server ran with different env overrides,")
        print("  context gates were active, or capture started mid-episode.")


def sweep(rows: List[dict], sessions: List[Session], base: DecisionParams, max_fa_per_hour: float) -> None:
    t_highs = np.round(np.arange(0.35, 0.625, 0.025), 3)
    t_lows = np.round(np.arange(0.15, 0.45, 0.025), 3)
    streaks = [2, 3, 4, 5]
    results = []
    for th, tl, k in product(t_highs, t_lows, streaks):
        if tl >= th - 0.05:
            continue
        p = dataclasses.replace(base, t_high=float(th), t_low=float(tl), alert_trigger_streak=k)
        m = score_alerts(simulate(rows, p), sessions)
        results.append((th, tl, k, m))

    def key(item):
        m = item[3]
        rate = -1 if np.isnan(m.detection_rate) else m.detection_rate
        lat = m.median_latency_s if not np.isnan(m.median_latency_s) else 1e9
        return (-rate, m.fa_per_hour, lat)

    ok = [r for r in results if not np.isnan(r[3].fa_per_hour) and r[3].fa_per_hour <= max_fa_per_hour]
    pool = sorted(ok or results, key=key)
    pareto = []
    best_fa = float("inf")
    for item in sorted(results, key=lambda r: (-(r[3].detection_rate if not np.isnan(r[3].detection_rate) else -1), r[3].fa_per_hour)):
        if item[3].fa_per_hour < best_fa:
            pareto.append(item)
            best_fa = item[3].fa_per_hour

    print(f"\nSweep: {len(results)} settings. Best with <= {max_fa_per_hour} false alarms/hour:" if ok else
          f"\nSweep: no setting met <= {max_fa_per_hour} false alarms/hour; best overall:")
    print(f"  {'T_HIGH':>6} {'T_LOW':>6} {'streak':>6} {'FA/h':>6} {'detect':>7} {'median s':>9}")
    for th, tl, k, m in pool[:10]:
        print(f"  {th:>6.3f} {tl:>6.3f} {k:>6d} {m.fa_per_hour:>6.2f} {m.detection_rate:>7.0%} {m.median_latency_s:>9.1f}")
    print("\nPareto front (no other setting has both higher detection and fewer false alarms)")
    for th, tl, k, m in pareto[:15]:
        print(f"  {th:>6.3f} {tl:>6.3f} {k:>6d} {m.fa_per_hour:>6.2f} {m.detection_rate:>7.0%} {m.median_latency_s:>9.1f}")
    print("\nWith only a few sessions these numbers are noisy. Prefer settings that are good across")
    print("neighbouring values, and confirm any change with a fresh recording session before deploying.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", nargs="+", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--t-high", type=float, default=None)
    ap.add_argument("--t-low", type=float, default=None)
    ap.add_argument("--streak", type=int, default=None)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--max-fa-per-hour", type=float, default=1.0)
    ap.add_argument("--check-live", action="store_true")
    args = ap.parse_args()

    rows = load_windows(args.capture)
    sessions = load_sessions(args.labels)
    if not rows:
        raise SystemExit("No captured windows found.")
    print(f"Loaded {len(rows)} windows ({sum(1 for r in rows if r.get('tier3_pass'))} scored) "
          f"and {len(sessions)} labelled sessions.")
    unlabelled = sum(1 for r in rows if _label_of(r, sessions) is None)
    if unlabelled:
        print(f"  {unlabelled} windows fall outside every labelled session and are ignored in metrics.")

    params = DecisionParams.from_env()
    overrides = {}
    if args.t_high is not None:
        overrides["t_high"] = args.t_high
    if args.t_low is not None:
        overrides["t_low"] = args.t_low
    if args.streak is not None:
        overrides["alert_trigger_streak"] = args.streak
    params = dataclasses.replace(params, **overrides)

    alerts = simulate(rows, params)
    report_alerts(score_alerts(alerts, sessions),
                  f"Alerts with T_HIGH={params.t_high:.3f} T_LOW={params.t_low:.3f} streak={params.alert_trigger_streak}")
    report_scores(rows, sessions, params)
    if args.check_live:
        check_live(rows, simulate(rows, DecisionParams.from_env()))
    if args.sweep:
        sweep(rows, sessions, params, args.max_fa_per_hour)


if __name__ == "__main__":
    main()
