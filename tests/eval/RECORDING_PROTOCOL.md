# Recording protocol: evaluating SafeEar without a dataset

You don't need a public dataset to tune SafeEar. The numbers that decide whether it works are:

- false alarms per hour
- the share of strangers it catches
- how fast it catches them

They depend on your device, the people around it, and where it's worn, so they can only come from your own recordings. This protocol takes about 2–3 hours in total. You label whole **sessions**, not individual clips.

## 0. Consent and storage

- Tell everyone who will be recorded what the recordings are for and how long you'll keep them. Get their consent; a signed note is best. Children need a parent's consent.
- Keep captures on your laptop only. `eval_capture/` must never be committed (add it to `.gitignore`).
- Delete the `wav/` folders once you've finished analysing. The `.jsonl` files hold scores only, no audio.

## 1. Run the backend locally with capture on

Render's disk is wiped on redeploy, so run the backend on your laptop and point the app at it. Your phone and laptop must be on the same Wi-Fi.

```powershell
cd speaker-backend
$env:SAFEEAR_EVAL_CAPTURE_DIR = "eval_capture"
$env:SAFEEAR_EVAL_CAPTURE_AUDIO = "true"      # optional: keeps each 1.5 s window as a wav (~170 MB/hour)
.\venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Each scored window is appended to `eval_capture/<device_id>/<date>.jsonl`.

## 2. Enroll as you normally would

- Enroll 2–3 family members in the app, two 10-second recordings each, in a quiet room.
- Don't change the enrollments during the evaluation. If you do, start a new set of sessions.

## 3. Record sessions

Keep a `sessions.csv` open and write the start and end time of each session as you go. Use the clock on the laptop running the backend.

```csv
start,end,label,notes
2026-10-05 18:00:00,2026-10-05 18:20:00,familiar_only,living room quiet, mom+dad talking to child
2026-10-05 18:30:00,2026-10-05 18:33:00,stranger_present,friend Rahul talks to wearer, quiet
```

The child (or you, as a stand-in) wears the device as it will really be worn, under clothing if that's the plan.

| # | label | what happens | duration | measures |
|---|---|---|---|---|
| A1 | `familiar_only` | family talks normally, quiet room | 20 min | false alarms, quiet |
| A2 | `familiar_only` | family talks, TV or music on at normal volume | 20 min | false alarms, noise |
| A3 | `familiar_only` | walking outdoors / traffic / market with family | 20 min | false alarms, outdoors |
| A4 | `familiar_only` | wearer alone or silent, moving around (clothing rustle, footsteps) | 15 min | false alarms from non-speech |
| B1–B5 | `stranger_present` | each unenrolled person (friend, classmate, neighbour) talks **to the wearer** for 2–3 min in a quiet room | 5 × 3 min | detection rate, time to alert |
| B6–B8 | `stranger_present` | the same with background noise, or the stranger 2–3 m away | 3 × 3 min | detection in harder conditions |
| C1 | `stranger_present` | a family member and a stranger in the same conversation | 5 min | detection with familiar voices nearby |
| D1 | `media` | YouTube/TV voices only, nobody else talking | 15 min | false alarms from recorded voices |

Guidelines:

- **Use different strangers.** Five people for 3 minutes each tells you far more than one person for 15 minutes.
- **Leave 2–3 minutes of gap between stranger sessions.** The debounce is 60 s, and you want each session to start fresh.
- **Keep sessions clean.** If a stranger walks into a `familiar_only` session, write down the time and split the session.

## 4. Analyse

```powershell
.\venv\Scripts\python.exe -m scripts.eval.replay --capture eval_capture --labels sessions.csv --check-live
.\venv\Scripts\python.exe -m scripts.eval.replay --capture eval_capture --labels sessions.csv --sweep --max-fa-per-hour 0.5
```

What you get:

- **False alarms per hour** on `familiar_only` + `media`. This is the number parents feel. A starting target is 0.5 or fewer per hour; tighten it once you have more data.
- **Detection rate and time to first alert** on `stranger_present`.
- **Window score percentiles** for familiar vs stranger sessions. If the familiar p5 sits below T_LOW (0.31), family members are being treated as strangers; re-enroll or move the thresholds.
- **`--check-live`** confirms the replay produces the same alerts the server actually fired. If it doesn't, something differs between the server and the replay (env overrides, context gates). Fix that before trusting the sweep.
- **`--sweep`** shows which T_HIGH / T_LOW / streak settings would have done best on your sessions.

## 5. Change settings safely

1. Pick a setting from the sweep that is good **and** whose neighbouring settings are also good. A lone best setting is probably overfitting a few sessions.
2. Set it as env vars on the server (`SAFEEAR_T_HIGH`, `SAFEEAR_T_LOW`, `SAFEEAR_ALERT_TRIGGER_STREAK`).
3. Record a **new** set of sessions and run `replay.py` again. Only keep the change if it still holds on data it wasn't tuned on.

## Optional: model sanity check on public speech

`scripts/eval/public_benchmark.py` uses LibriSpeech test-clean (OpenSLR resource 12) to show how scores shift with noise, a competing talker, and window length. It is adult read speech on good microphones, so use it only to compare pipeline versions. Never set thresholds from it.
