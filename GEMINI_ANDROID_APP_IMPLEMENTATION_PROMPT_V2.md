# Gemini Handoff Prompt V2: SafeEar Android App (Alert-First, Record-First, Real Google Login)

Use this prompt as-is with Gemini.

---

You are a senior Android architect and Kotlin engineer. Build a production-grade Android app for SafeEar with an **alert-first familiarization flow**.

## Hard Product Requirements (From Owner)

1. Start with **0 familiar speakers** by default.
2. Parent should **NOT be forced to manually upload files** to add familiar voices.
3. Parent must be able to:
   - Receive stranger alerts
   - Play alert audio
   - Tap **Flag as Familiar** on an alert to add that person as familiar
4. Enrollment must support **record voice directly in app** (in-app recording UX), not only file picker.
5. Alerts UI must include at least:
   - Time
   - Location link/map action
   - Play audio action
   - Flag as Familiar action
6. Login must support **real Google account sign-in page** (Google account chooser / OAuth flow like Gmail/Drive).
7. New user onboarding/sign-up must be supported:
   - With Google auth, first login should auto-create user.
   - If separate email/password sign-up is needed, mark as backend gap and provide optional plan (do not fake unsupported APIs).

## Critical Rule
Do not invent, rename, or break backend APIs. Match contracts exactly.

## Backend API Contracts (Current)

Base URL examples:
- Emulator: http://10.0.2.2:8000
- Physical device: http://<LAN-IP>:8000

### Auth
- Protected endpoints require: Authorization: Bearer <access_token>

### POST /auth/google
Body JSON:
{
  "id_token": "string"
}
Returns access + refresh token + parent object.

Notes:
- Real flow: app should obtain Google ID token using Google Sign-In SDK and send it here.
- Dev mode also supports id_token like dev:test-parent-1 (for testing only).

### POST /auth/refresh
Body JSON: { "refresh_token": "string" }

### DELETE /auth/logout
Body JSON: { "refresh_token": "string" }

### POST /devices (multipart/form-data)
Fields:
- device_name (required)
- role (required): child_device or parent_device
- device_token (optional)

### POST /enroll/speaker (multipart/form-data)
Fields:
- display_name (required)
- speaker_id (optional)
- file or audio (WAV file)

### GET /enroll/speakers
### PATCH /enroll/speakers/{speaker_id}
### DELETE /enroll/speakers/{speaker_id}

### POST /detect/location (JSON)
Body: { device_id, latitude, longitude }

### POST /detect/chunk (multipart/form-data)
Fields:
- device_id
- file or audio (16kHz WAV)
- latitude (optional)
- longitude (optional)

Role rule:
- only child_device can stream detect chunks

### DELETE /detect/session?device_id=<id>

### GET /alerts?limit=&offset=

### POST /alerts/{alert_id}/ack

### GET /alerts/{alert_id}/clip
Returns audio/wav bytes.

### NEW: POST /alerts/{alert_id}/flag-familiar
Auth required
Body JSON:
{
  "display_name": "string",
  "speaker_id": "optional-uuid"
}
Behavior:
- Uses that alert's audio clip to run enrollment
- Creates/updates familiar speaker embeddings

Success response:
{
  "status": "flagged_familiar",
  "source_alert_id": "uuid",
  "speaker_id": "uuid",
  "display_name": "string",
  "samples_saved": 1,
  "stages": { ... }
}

### GET /health
### GET /

## Required App UX Flows

## 1) Auth Flow

Implement **real Google Sign-In**:
- Use Google Identity / Credential Manager / GoogleSignIn APIs for Android
- Show Google account picker and auth consent UI
- Obtain Google ID token
- Call /auth/google
- Store tokens in DataStore

On first successful Google login:
- Treat as sign-up + login automatically (backend upsert behavior)
- Show onboarding to register this phone as parent or child device

## 2) Device Onboarding
- Ask for device name
- Ask role (parent_device / child_device)
- Call /devices
- Persist returned device.id and role

## 3) Parent Home (Alert-First)
- Show alerts feed prominently
- Each alert card includes:
  - detection time
  - location action (map)
  - play audio action
  - "Flag as Familiar" CTA
- On flag familiar:
  - ask display name (required)
  - call /alerts/{alert_id}/flag-familiar
  - on success update familiar list

Do not force "upload file from storage" as primary flow.

## 4) Enrollment Screen
- Primary action: **Record Voice** in-app and submit
- Secondary optional action: pick file from storage (optional fallback)
- Upload recorded WAV via /enroll/speaker
- Show stage feedback returned by API

## 5) Child Monitoring Flow
- Child role screen with Start Monitoring / Stop Monitoring
- Mic capture + chunking + resample to 16kHz WAV
- Send chunks to /detect/chunk with device_id
- Send location via /detect/location periodically
- Handle statuses: warming_up, no_hop, ok

## 6) Familiar Voices Screen
- Initially empty for new users (0 speakers)
- Populate only after enrollment or flag-familiar
- Supports rename/delete

## Technical Requirements

- Kotlin + Jetpack Compose
- MVVM + repository/usecase layering
- Hilt
- Retrofit + OkHttp
- DataStore
- WorkManager
- ExoPlayer/Media3 for alert clip playback
- Coroutines + Flow

## Networking Rules

- Token interceptor for Bearer auth
- 401 refresh retry with /auth/refresh once
- Fail to login if refresh fails
- Exact field names in DTOs (no renaming)

## DataStore & Auth State Management (CRITICAL)

### On Login Success
```kotlin
// Save to DataStore
dataStore.edit { prefs ->
    prefs[ACCESS_TOKEN_KEY] = response.access_token
    prefs[REFRESH_TOKEN_KEY] = response.refresh_token
    prefs[PARENT_ID_KEY] = response.parent.id
    prefs[DEVICE_ID_KEY] = response.device.id  // if returned
}
```

### On Logout (MUST DO ALL STEPS)
```kotlin
// Step 1: Call backend logout endpoint FIRST
val logoutResult = apiService.logout(DeleteLogoutRequest(refreshToken = storedRefreshToken))

// Step 2: Clear ALL DataStore keys (do this even if logout call fails or throws)
dataStore.edit { prefs ->
    prefs.remove(ACCESS_TOKEN_KEY)
    prefs.remove(REFRESH_TOKEN_KEY)
    prefs.remove(PARENT_ID_KEY)
    prefs.remove(DEVICE_ID_KEY)
    prefs.remove(DEVICE_ROLE_KEY)
    // Clear ANY other auth-related cached data
}

// Step 3: Clear OkHttp cache (if using cache interceptor)
okHttpClient.cache?.evictAll()

// Step 4: Navigate to login screen WITHOUT caching/remembering state
navController.navigate(LOGIN_ROUTE) {
    popUpTo(0) // Clear entire nav backstack
}
```

### On App Startup (Check Auth State)
```kotlin
// DO NOT auto-login from cached tokens
// ONLY proceed if ALL of these are present:
// - ACCESS_TOKEN non-empty
// - REFRESH_TOKEN non-empty  
// - PARENT_ID non-empty
// - DEVICE_ID non-empty

// If any are missing → show login screen
```

If logout silently crashes the app, ensure:
1. You're NOT throwing uncaught exceptions in logout handler
2. You're clearing DataStore in a try/catch block
3. You're clearing cache before navigation


## Multipart Rules (CRITICAL)

### Rule 1: Text + File Combined
When uploading a file + text fields together (e.g., display_name + audio file):
```kotlin
@Multipart
@POST("/enroll/speaker")
suspend fun enrollSpeaker(
    @Part("display_name") displayName: RequestBody,
    @Part("speaker_id") speakerId: RequestBody?,  // optional
    @Part audio: MultipartBody.Part  // NO parameter name here; name embedded in Part
): EnrollSpeakerResponse

// Caller usage:
val displayNameBody = RequestBody.create("text/plain".toMediaType(), name)
val partId = RequestBody.create("text/plain".toMediaType(), speakerId ?: "")
val audioPart = MultipartBody.Part.createFormData("audio", filename, audioBody)
enrollSpeaker(displayNameBody, partId, audioPart)
```

### Rule 2: File-Only (No Text Fields)
```kotlin
@Multipart
@POST("/detect/chunk")
suspend fun detectChunk(
    @Part("device_id") deviceId: RequestBody,
    @Part("latitude") latitude: RequestBody?,
    @Part("longitude") longitude: RequestBody?,
    @Part audio: MultipartBody.Part  // NO param name for MultipartBody.Part
): DetectResponse
```

### Rule 3: NEVER Do This
```kotlin
// WRONG - Causes "must not include a part name" error:
@Part("audio") audio: MultipartBody.Part  // ❌ WRONG

// CORRECT:
@Part audio: MultipartBody.Part  // ✓ Correct
```

Key: MultipartBody.Part parameters must NOT have @Part("name") annotation. The name is embedded when you call `MultipartBody.Part.createFormData("name", filename, body)`.

## Error Handling Requirements

Map backend detail messages to user-friendly text while logging raw details:
- missing_authorization_header
- invalid_authorization_header
- parent_not_found
- missing_audio_chunk
- invalid_audio_chunk
- audio_chunk_must_be_16khz
- only_child_devices_can_stream_audio
- alert_not_found
- audio_clip_not_found
- invalid_parent_id / invalid_device_id / invalid_alert_id

## Debugging Child Monitoring (If "Nothing Happens")

If child monitoring doesn't produce alerts, verify in this order:

1. **Device Role Check**
   - After onboarding, verify persisted role = "child_device" in DataStore
   - If role is "parent_device", monitoring won't work (endpoint rejects non-child roles)

2. **Chunk Submission**
   - Add logging to see if detectChunk() is being called
   - Log device_id, audio size, sample rate
   - Check for 16kHz WAV requirement (if SampleRate != 16000 → error)

3. **Backend Response Status**
   - Log the DetectResponse.status field (should be "warming_up", "no_hop", or "ok")
   - "ok" means stranger detected → alert should be created
   - Still no alert? Check: Is familiar speaker list empty? If empty, all audio = stranger alert

4. **Alerts List**
   - Call GET /alerts after detection
   - Should show new Alert with confidence_score > 0.5
   - Location lat/lon should match submitted coordinates

5. **Common Failures**
   - 403 Forbidden + "only_child_devices_can_stream_audio" → fix device role
   - 422 "audio_chunk_must_be_16khz" → resample audio to 16kHz before upload
   - 200 response but status="no_hop" → normal operation, no stranger detected
   - 200 response but status="warming_up" → model still warming up, try again

## Acceptance Criteria

App is accepted only if:
1. Google account sign-in UI appears and works.
2. New user can onboard without manual backend setup.
3. Familiar list starts at zero for new account.
4. Parent can flag alert audio as familiar from alert card.
5. Parent can enroll by recording in-app voice.
6. Child monitoring streams chunks and handles response states.
7. Alert card includes time, location, play audio, flag familiar.
8. No API contract drift.

## Output Format Required from Gemini

1. Architecture summary
2. File tree
3. Full Kotlin code by file
4. Setup config (Google Sign-In + backend URL)
5. Test plan (manual + unit/integration)
6. Known backend gaps and optional future API additions

Do not output pseudo-code only. Provide concrete compilable Kotlin code.

---

If you detect an unsupported feature in backend, implement graceful fallback and clearly mark it as backend gap instead of inventing APIs.
