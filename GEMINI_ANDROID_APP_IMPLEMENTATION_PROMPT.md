# Gemini Handoff Prompt: Build SafeEar Android App With Zero API Drift

Use this prompt as-is with Gemini.

---

You are a senior Android architect and Kotlin engineer. Build a production-grade Android app for the SafeEar backend described below.

## Critical Rule
Do not invent, rename, or change backend endpoints, request shapes, field names, auth headers, response fields, or role values. Match the contracts exactly.

If you think an API is odd, keep client compatibility and propose optional improvements separately.

## Goal
Create one Android app that supports:
- Parent role workflows
- Child role workflows
- Speaker enrollment
- Location updates
- Streaming audio chunk detection
- Alerts list, acknowledge, and audio clip playback

The backend is already running and must remain source of truth.

## Backend Contract (Exact)
Base URL example:
- http://10.0.2.2:8000 for emulator
- http://192.168.137.1:8000 for locally run server
- configurable via BuildConfig or environment

Auth:
- Bearer token in Authorization header for protected routes
- Header format: Bearer <access_token>

### 1) POST /auth/google
Content-Type: application/json
Body:
{
  "id_token": "string"
}

Dev mode supported by backend:
- id_token can be: dev:<parent_alias>

Success response:
{
  "access_token": "string",
  "refresh_token": "string",
  "token_type": "bearer",
  "expires_in": 604800,
  "parent": {
    "id": "uuid-string",
    "email": "string-or-null",
    "display_name": "string-or-null"
  }
}

### 2) POST /auth/refresh
Content-Type: application/json
Body:
{
  "refresh_token": "string"
}

Success response:
{
  "access_token": "string",
  "refresh_token": "string",
  "token_type": "bearer",
  "expires_in": 604800
}

### 3) DELETE /auth/logout
Content-Type: application/json
Body:
{
  "refresh_token": "string"
}

Success response:
{
  "status": "ok"
}

### 4) POST /devices
Auth required
Content-Type: multipart/form-data
Fields:
- device_name: string (required)
- role: string (required, must be exactly one of: child_device, parent_device)
- device_token: string (optional)

Success response:
{
  "status": "created",
  "device": {
    "id": "uuid-string",
    "parent_id": "uuid-string",
    "device_name": "string",
    "role": "child_device|parent_device",
    "device_token": "string-or-null"
  }
}

### 5) POST /enroll/speaker
Auth required
Content-Type: multipart/form-data
Fields:
- display_name: string (required)
- speaker_id: string (optional)
- file OR audio: WAV file upload (one required)

Success response:
{
  "status": "enrolled",
  "speaker_id": "uuid-string",
  "display_name": "string",
  "samples_saved": 1,
  "embedding_dim": 192,
  "stages": {
    "vad": { "voiced_ms": 1234.5, "num_segments": 4 },
    "speech_quality": { "passed": true }
  }
}

### 6) GET /enroll/speakers
Auth required
Success response:
{
  "items": [
    {
      "id": "uuid",
      "parent_id": "uuid",
      "display_name": "string",
      "sample_count": 3,
      "created_at": "datetime",
      "updated_at": "datetime"
    }
  ]
}

### 7) PATCH /enroll/speakers/{speaker_id}
Auth required
Content-Type: application/json
Body:
{
  "display_name": "new name"
}

### 8) DELETE /enroll/speakers/{speaker_id}
Auth required

### 9) POST /detect/location
Auth required
Content-Type: application/json
Body:
{
  "device_id": "uuid-string",
  "latitude": 12.34,
  "longitude": 56.78
}

Backend also accepts lat/lng aliases, but app should send latitude/longitude.

### 10) POST /detect/chunk
Auth required
Content-Type: multipart/form-data
Fields:
- device_id: string (required)
- file OR audio: WAV chunk file (required)
- latitude: float (optional)
- longitude: float (optional)

Audio constraints:
- Must be 16kHz or backend returns 400 with detail audio_chunk_must_be_16khz
- Mono preferred
- WAV expected

Possible responses:
- warming_up
- no_hop
- ok

Typical ok response:
{
  "status": "ok",
  "decision": "familiar|stranger_candidate|uncertain|hold",
  "stranger_streak": 0,
  "score": 0.12,
  "thresholds": { "t_high": 0.35, "t_low": 0.15 },
  "stage": { ... tier info ... },
  "alert_fired": false,
  "alert_id": "uuid-or-null"
}

Important role rule:
- Only child_device can call detect/chunk
- parent_device gets 403 detail only_child_devices_can_stream_audio

### 11) DELETE /detect/session?device_id=<id>
Auth required

### 12) GET /alerts?limit=50&offset=0
Auth required
Success response:
{
  "items": [
    {
      "id": "uuid",
      "parent_id": "uuid",
      "device_id": "uuid",
      "timestamp": "datetime",
      "confidence_score": 0.12,
      "audio_clip_path": "server-path",
      "latitude": 12.34,
      "longitude": 56.78,
      "acknowledged_at": "datetime-or-null",
      "created_at": "datetime",
      "updated_at": "datetime"
    }
  ]
}

### 13) POST /alerts/{alert_id}/ack
Auth required
Returns acknowledged alert object.

### 14) GET /alerts/{alert_id}/clip
Auth required
Returns audio/wav bytes.

### 15) GET /health
No auth

### 16) GET /
No auth

## Android App Requirements

### A. Tech Stack (Required)
- Kotlin
- Jetpack Compose UI
- MVVM + Clean-ish layering
- Hilt DI
- Retrofit + OkHttp + Moshi or Kotlinx serialization
- DataStore for tokens and user session
- WorkManager for periodic jobs (location sync, optional background detect loop)
- Media3/ExoPlayer for alert clip playback
- Coroutines + Flow

### B. App Architecture
Use package structure:
- app/core/network
- app/core/auth
- app/core/storage
- app/features/auth
- app/features/devices
- app/features/enrollment
- app/features/detection
- app/features/alerts
- app/features/settings
- app/navigation

Layering:
- UI: Composables + ViewModels
- Domain: UseCases
- Data: Repositories + Remote data source + DTO mappers

### C. Authentication Flow
- Login screen supports:
  - Google Sign-In token path
  - Dev token quick path input (dev:<name>)
- On login success:
  - persist access and refresh token
  - persist parent id and profile
- Add interceptor:
  - inject Bearer access token
- Add authenticator or repository retry logic:
  - on 401, call /auth/refresh once, retry original request
  - if refresh fails, force logout

### D. Device Registration and Role Handling
- First-run after login:
  - ask user to choose mode: Parent Device or Child Device
  - collect device name
  - call POST /devices
  - persist returned device.id and device.role
- Enforce role in UI:
  - child_device sees streaming detection controls
  - parent_device sees alerts dashboard and speaker management

### E. Enrollment Feature
- Parent flow to add familiar speakers
- Record or pick WAV file
- Ensure file sent as multipart file field named file
- Show stage feedback from response
- Speaker list screen:
  - list
  - rename
  - delete

### F. Child Detection Streaming Feature
Implement robust recorder pipeline:
- AudioRecord capture from mic
- Convert/resample to mono 16kHz PCM
- Package periodic WAV chunks and upload to /detect/chunk
- Include device_id every request
- Include latest latitude/longitude optionally

Suggested chunk strategy:
- 1.0 to 1.5 second chunks for stable server warmup
- request cadence roughly near hop interval with network tolerance

State machine in client:
- Idle
- Recording
- Uploading
- Server warming_up
- Server no_hop
- Active scoring
- Alert fired
- Error/permission denied

On stop:
- call DELETE /detect/session with device_id

### G. Location Updates
- Child mode sends location updates via /detect/location
- Use fused location provider
- Send updates:
  - on start
  - periodically while detecting
  - on meaningful movement threshold

### H. Alerts Feature (Parent)
- Alerts list with pagination limit/offset
- Polling strategy (or pull-to-refresh):
  - every 10 to 20 seconds while screen visible
- Card fields:
  - created time
  - confidence
  - map coordinates
  - ack status
- Actions:
  - Acknowledge via /alerts/{id}/ack
  - Play clip via /alerts/{id}/clip stream
  - Open map intent using latitude/longitude if available

### I. Error Handling Standards
Map backend details to user-friendly text but preserve technical detail in logs:
- missing_authorization_header
- invalid_authorization_header
- invalid_token_subject
- parent_not_found
- missing_audio_chunk
- invalid_audio_chunk
- audio_chunk_must_be_16khz
- only_child_devices_can_stream_audio
- speaker_not_found
- alert_not_found

### J. Permissions
Handle runtime permissions with clear rationale:
- RECORD_AUDIO
- ACCESS_FINE_LOCATION
- POST_NOTIFICATIONS (Android 13+)

Graceful degraded behavior when denied:
- no recording if mic denied
- no location attach if location denied

### K. Non-Functional Requirements
- Strong lifecycle safety (no leaked recorder or jobs)
- Retry with exponential backoff for transient network failures
- Cancel in-flight jobs when leaving detect screen
- Foreground service option for continuous child monitoring (if needed)
- Structured logging tags per feature

### L. UI/UX Requirements
Create polished, clear UI (not generic placeholder app):
- Role-first onboarding
- Large, obvious Child Monitoring toggle
- Parent alert cards with severity accent and quick actions
- Enrollment flow with progress and quality feedback
- Offline and reconnect banners

### M. Exact DTO Mapping Guidance
Use exact JSON keys as backend returns.
Do not auto-rename fields.
Where timestamps are datetime strings, parse safely with fallback.
UUID fields remain strings in app models.

### N. Deliverables Gemini Must Produce
1. Complete Android project scaffold and module/package layout
2. API interface definitions for every endpoint above
3. DTOs and domain models
4. Repositories and use cases
5. ViewModels and Compose screens for all critical flows
6. Audio recorder and WAV encoder utility compatible with 16kHz requirement
7. Token refresh handling in networking stack
8. Basic instrumentation/unit tests for:
   - auth refresh
   - device role gating
   - detect chunk upload
   - alerts ack
9. Setup instructions to run against local backend

### O. Acceptance Criteria
The app is accepted only if:
- Login works with dev token format
- Device registration works for both roles
- Parent can enroll, list, rename, delete speakers
- Child can stream chunks and receives valid server states
- Parent can view alerts, ack alerts, and play clips
- No API contract mismatch errors in normal flow

### P. What To Output
Output in this strict order:
1. Architecture summary
2. File tree
3. Full code files section by section
4. Run instructions
5. Test checklist
6. Known limitations and optional enhancements

Do not provide only pseudo-code. Provide concrete Kotlin code and exact implementations.

---

Additionally, while generating code, annotate any assumption that could cause integration mismatch, and provide a no-assumption fallback implementation path.
