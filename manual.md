# SafeEar Backend Manual (Windows, No PostgreSQL Required)

This guide helps you run your project successfully on Windows using **SQLite** (so you do not need PostgreSQL installed).

## 1) What This Project Is

- FastAPI backend for SafeEar speaker recognition
- Audio pipeline with VAD/speech checks/embedding matching
- JWT auth + refresh token flow
- Device roles (`child_device`, `parent_device`)
- Alert creation + alert audio clip serving

## 2) Prerequisites

- Windows 10/11
- Python 3.10 (recommended for your current setup)
- Git (optional but recommended)
- Internet on first run (for dependency install and model fetching)

## 3) Project Folder

Open terminal in:

`C:\Users\AYUSH VERMA\Documents\sem6\speaker-backend`

## 4) Create and Activate Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate again:

```powershell
.\venv\Scripts\Activate.ps1
```

## 5) Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 6) Configure for SQLite (No PostgreSQL)

Your code defaults to PostgreSQL if `DATABASE_URL` is not set. Since you do not have PostgreSQL, set SQLite URL in PowerShell **before starting server**:

```powershell
$env:DATABASE_URL = "sqlite:///app/database/safeear_dev.db"
```

Also enable dev login token flow:

```powershell
$env:SAFEEAR_ALLOW_DEV_GOOGLE_TOKEN = "true"
```

Optional (for real Google login only):

```powershell
$env:GOOGLE_CLIENT_ID = "your-google-client-id"
```

## 7) Start Server

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Expected startup behavior:
- DB tables created automatically on startup (`init_db()`)
- Server available at `http://127.0.0.1:8000`

## 8) Quick Health Check

Open another terminal (activate venv if needed) and run:

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/health"
```

Expected:

```json
{ "status": "ok" }
```

## 9) Full Basic API Smoke Test (PowerShell)

### 9.1 Login (Dev Token)

```powershell
$base = "http://127.0.0.1:8000"
$loginBody = @{ id_token = "dev:test-parent-1" } | ConvertTo-Json
$login = Invoke-RestMethod -Method Post -Uri "$base/auth/google" -ContentType "application/json" -Body $loginBody
$access = $login.access_token
$refresh = $login.refresh_token
$headers = @{ Authorization = "Bearer $access" }
$login | ConvertTo-Json -Depth 6
```

### 9.2 Create Child Device

```powershell
$deviceResp = Invoke-RestMethod -Method Post -Uri "$base/devices" -Headers $headers -Form @{
  device_name = "child-phone"
  role = "child_device"
}
$deviceResp | ConvertTo-Json -Depth 6
$deviceId = $deviceResp.device.id
```

### 9.3 Update Location

```powershell
$loc = @{ device_id = $deviceId; latitude = 28.6139; longitude = 77.2090 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "$base/detect/location" -Headers $headers -ContentType "application/json" -Body $loc | ConvertTo-Json
```

### 9.4 Enroll Speaker (Using Existing WAV)

If you already have `app/data/temp_audio/voice.wav`, use:

```powershell
Invoke-RestMethod -Method Post -Uri "$base/enroll/speaker" -Headers $headers -Form @{
  display_name = "Ayush"
  file = Get-Item "app/data/temp_audio/voice.wav"
} | ConvertTo-Json -Depth 10
```

### 9.5 List Speakers

```powershell
Invoke-RestMethod -Method Get -Uri "$base/enroll/speakers" -Headers $headers | ConvertTo-Json -Depth 8
```

### 9.6 Detect Chunk (Child Device)

```powershell
Invoke-RestMethod -Method Post -Uri "$base/detect/chunk" -Headers $headers -Form @{
  device_id = $deviceId
  file = Get-Item "app/data/temp_audio/voice.wav"
  latitude = "28.6139"
  longitude = "77.2090"
} | ConvertTo-Json -Depth 12
```

### 9.7 Alerts

```powershell
$alerts = Invoke-RestMethod -Method Get -Uri "$base/alerts?limit=20&offset=0" -Headers $headers
$alerts | ConvertTo-Json -Depth 10
```

If an alert exists:

```powershell
$alertId = $alerts.items[0].id
Invoke-RestMethod -Method Post -Uri "$base/alerts/$alertId/ack" -Headers $headers | ConvertTo-Json -Depth 8
```

## 10) Refresh + Logout

### Refresh

```powershell
$rbody = @{ refresh_token = $refresh } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "$base/auth/refresh" -ContentType "application/json" -Body $rbody | ConvertTo-Json
```

### Logout

```powershell
$lobody = @{ refresh_token = $refresh } | ConvertTo-Json
Invoke-RestMethod -Method Delete -Uri "$base/auth/logout" -ContentType "application/json" -Body $lobody | ConvertTo-Json
```

## 11) Common Errors and Fixes

### Error: `sqlalchemy.exc.OperationalError ... localhost:5432 connection refused`

Cause:
- `DATABASE_URL` not set, so app tries PostgreSQL by default.

Fix:

```powershell
$env:DATABASE_URL = "sqlite:///app/database/safeear_dev.db"
```

Then restart uvicorn.

### Error: `audio_chunk_must_be_16khz`

Cause:
- Uploaded WAV is not 16kHz.

Fix:
- Use a 16kHz WAV file for enroll/detect.

### Error: `only_child_devices_can_stream_audio`

Cause:
- Device role is `parent_device`, but calling `/detect/chunk`.

Fix:
- Create/use device with role `child_device`.

### Error: `missing_authorization_header` or `invalid_token_subject`

Cause:
- Missing/invalid bearer token.

Fix:
- Ensure headers contain `Authorization: Bearer <access_token>` from `/auth/google`.

### Error importing `firebase_admin`, `twilio`, `sendgrid`, `sqlalchemy`

Cause:
- Packages not installed in active venv.

Fix:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 12) Daily Run Checklist

1. Open terminal in project folder
2. Activate venv
3. Set env vars:
   - `DATABASE_URL=sqlite:///app/database/safeear_dev.db`
   - `SAFEEAR_ALLOW_DEV_GOOGLE_TOKEN=true`
4. Start server: `python -m uvicorn app.main:app --host 0.0.0.0 --port 8000`
5. Test `/health`

## 13) Optional: If You Later Want PostgreSQL

Set:

```powershell
$env:DATABASE_URL = "postgresql://safeear:safeear@localhost:5432/safeear"
```

and ensure PostgreSQL server is running on port 5432.

---

If you follow this manual exactly, your project runs successfully without PostgreSQL using SQLite.
