"""Simulate a monitored child device: stream WAV files to the backend like the
phone (or a future hardware locket) does, and print each decision.

Each file is cut into contiguous chunks (default 1.0 s) sent with seq,
t_start_ms and sent_at_ms, authenticated with a device token obtained by
registering a child device under the given parent account.

Examples (from speaker-backend, server running locally):
    # enroll two familiar voices, then stream a familiar clip and a stranger clip
    python -m scripts.device_sim --base-url http://127.0.0.1:8000 \
        --email me@example.com --password secret123 --register \
        --enroll "Mom=mom1.wav,mom2.wav" \
        --stream mom_talking.wav stranger_talking.wav --realtime
"""

import argparse
import io
import sys
import time
import uuid
from typing import Dict, List

import numpy as np
import requests
import soundfile as sf


def _load_16k_mono(path: str) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    if sr != 16000:
        import librosa

        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
    return data.astype(np.float32)


def _wav_bytes(samples: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, 16000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _check(resp: requests.Response, what: str) -> Dict:
    if resp.status_code >= 400:
        sys.exit(f"{what} failed: HTTP {resp.status_code} {resp.text[:300]}")
    return resp.json()


def parent_login(base: str, email: str, password: str, register: bool) -> str:
    if register:
        r = requests.post(f"{base}/auth/register-email",
                          json={"email": email, "password": password, "display_name": "Device Sim"}, timeout=30)
        if r.status_code == 200:
            return r.json()["access_token"]
    return _check(requests.post(f"{base}/auth/login-email", json={"email": email, "password": password}, timeout=30),
                  "login")["access_token"]


def enroll(base: str, parent_token: str, spec: str) -> None:
    name, files = spec.split("=", 1)
    speaker_id = None
    for path in [f for f in files.split(",") if f]:
        data = {"display_name": name}
        if speaker_id:
            data["speaker_id"] = speaker_id
        with open(path, "rb") as fh:
            body = _check(requests.post(
                f"{base}/enroll/speaker", headers={"Authorization": f"Bearer {parent_token}"},
                data=data, files={"audio": (path, fh, "audio/wav")}, timeout=120,
            ), f"enroll {name} ({path})")
        speaker_id = body["speaker_id"]
        print(f"enrolled {name}: +{body['samples_saved']} embeddings, quality={body.get('quality_label')}")


def register_child(base: str, parent_token: str, name: str) -> Dict:
    body = _check(requests.post(
        f"{base}/devices", headers={"Authorization": f"Bearer {parent_token}"},
        data={"device_name": name, "role": "child_device", "installation_id": f"device-sim-{name}"}, timeout=30,
    ), "register child device")
    return body["device"]


def stream(base: str, device: Dict, path: str, chunk_sec: float, realtime: bool, seq_start: int) -> int:
    audio = _load_16k_mono(path)
    n = int(chunk_sec * 16000)
    headers = {"Authorization": f"Bearer {device['device_access_token']}"}
    t0 = int(time.time() * 1000)
    seq = seq_start
    stream_id = uuid.uuid4().hex
    alerts: List[str] = []
    print(f"\n--- streaming {path} ({len(audio) / 16000:.1f} s) ---")
    silent_skipped = 0
    for i in range(0, len(audio) - n + 1, n):
        chunk = audio[i:i + n]
        t_start = t0 + int(i * 1000 / 16000)
        # Same silence gate as the Android app: chunks below -60 dBFS are not sent.
        if float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2))) < 0.001:
            silent_skipped += 1
            print(f"t={i / 16000:5.1f}s           (silent, not uploaded)")
            if realtime:
                time.sleep(chunk_sec)
            continue
        # Realtime: real send time.  Fast mode: pretend each chunk is sent the moment
        # it finishes recording, as a live device would.
        sent_at = int(time.time() * 1000) if realtime else t_start + int(chunk_sec * 1000)
        started = time.perf_counter()
        for _attempt in range(5):
            resp = requests.post(
                f"{base}/detect/chunk", headers=headers,
                data={"device_id": device["id"], "chunk_id": uuid.uuid4().hex, "seq": str(seq),
                      "t_start_ms": str(t_start), "sent_at_ms": str(sent_at), "stream_id": stream_id,
                      "silent_skipped": str(silent_skipped)},
                files={"file": ("chunk.wav", _wav_bytes(chunk), "audio/wav")}, timeout=60,
            )
            if resp.status_code != 429:
                break
            time.sleep(0.5)
        body = _check(resp, f"chunk {seq}")
        silent_skipped = 0
        took = (time.perf_counter() - started) * 1000
        score = body.get("score")
        line = f"t={i / 16000:5.1f}s seq={seq:<4} {body.get('status'):<14} {str(body.get('decision') or ''):<24}"
        if score is not None:
            line += f" score={score:6.3f}"
        line += f" streak={body.get('stranger_streak')} rtt={took:5.0f}ms"
        if body.get("alert_fired"):
            line += "  <-- ALERT"
            alerts.append(body.get("alert_id"))
        print(line)
        seq += 1
        if realtime:
            time.sleep(max(0.0, chunk_sec - (time.perf_counter() - started)))
    print(f"alerts fired: {len(alerts)}")
    return seq


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--email", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--register", action="store_true", help="Create the parent account if it does not exist")
    ap.add_argument("--enroll", action="append", default=[], help='"Name=file1.wav,file2.wav" (repeatable)')
    ap.add_argument("--device-name", default="sim-child")
    ap.add_argument("--stream", nargs="*", default=[], help="WAV files to stream in order")
    ap.add_argument("--chunk-sec", type=float, default=1.5, help="1.5 s matches the Android app")
    ap.add_argument("--realtime", action="store_true", help="Pace chunks at real time (needed for debounce timing)")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    parent_token = parent_login(base, args.email, args.password, args.register)
    for spec in args.enroll:
        enroll(base, parent_token, spec)
    device = register_child(base, parent_token, args.device_name)
    seq = 1
    for path in args.stream:
        seq = stream(base, device, path, args.chunk_sec, args.realtime, seq) + 5  # gap between files


if __name__ == "__main__":
    main()
