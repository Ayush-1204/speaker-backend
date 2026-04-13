# Render Persistent Disk Mount Guide

## Overview
To persist embeddings and alert audio files across service restarts, mount a persistent disk on Render.

## Mount Location
- **Disk Mount Path**: `/var/data` (or `/var/data/safear` for sub-folder isolation)
- **Backend Usage**: Embedding files and alert audio stored here survive service restarts

## Render Dashboard Steps

1. Go to your service dashboard: **speaker-recognition-backend**
2. Navigate to **Settings** → **Disks**
3. Click **+ New Disk**
   - **Name**: `safear-persistent-disk` (or any name)
   - **Mount Path**: `/var/data`
   - **Size**: 10GB (adjust as needed for your audio storage)
4. Click **Create**
5. Service will automatically restart with disk mounted

## Environment Variables (Optional)

If you want to override the data root, set these on the Render dashboard under **Environment**:

```
SAFEEAR_DATA_ROOT=/var/data/safear
```

Default is `app/data` which is ephemeral. With the persistent disk mounted at `/var/data`, the backend will store:
- Embeddings: `/var/data/safear/tenants/<parent_id>/embeddings/<speaker_id>/`
- Alerts: `/var/data/safear/tenants/<parent_id>/alerts/<alert_id>/`

## Verification

After disk is mounted and service restarts:
1. Go to **Logs** tab
2. Filter for `Persistent disk mounted` or check for `DATA_ROOT` in startup logs
3. Upload some audio → enroll speaker → verify embeddings survive a restart

## Performance Considerations

- **Disk I/O**: Reading/writing embeddings and alert clips uses disk I/O
- **Audio Quality**: 16kHz WAV files (~1.6 MB per minute)
- **Alert Storage**: Each alert stores full audio (~5-10 MB per alert depending on buffer size)
- **Embedding Storage**: Each speaker gets ~50 KB per embedding (192-D float32 vector)

## Rolling Buffer Changes

As of this update:
- **Ring Buffer Size**: 10 seconds (increased from 6)
- **Preroll Duration**: 5 seconds (increased from 2)
- **Impact**: Alert audio now captures up to 5 seconds of pre-event context
