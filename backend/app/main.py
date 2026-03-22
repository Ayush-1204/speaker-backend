# app/main.py

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import shutil
import time
import json

from app.utils.feature_extractor import get_ecapa_embedding_from_file
from app.utils.storage import save_embedding, load_all_embeddings
from app.utils.compare import match_embedding, THRESHOLD

app = FastAPI()

# --- DIRECTORIES ---
TEMP_DIR = "app/data/temp_audio"
AUDIO_STORAGE_DIR = "app/data/alert_audio"
DATA_DIR = "app/data"
LOCATION_FILE = os.path.join(DATA_DIR, "latest_location.json")

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(AUDIO_STORAGE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# --- IN-MEMORY STORAGE (for demonstration) ---
# In a real app, these should be managed in a persistent database.

# Holds the current state for mutual exclusivity
app_state = {
    "mode": "none",  # Can be "none", "child_active", or "parent_active"
    "active_device_id": None
}

# Stores alerts received from child devices
alerts_storage = []


class LocationData(BaseModel):
    latitude: float
    longitude: float

# ------------------ DEVICE MODE LOCKING ------------------

@app.post("/start_child_mode")
async def start_child_mode(device_id: str = Form(...)):
    '''Acquires a lock for a device to enter 'Child Mode'.'''
    if app_state["mode"] == "child_active" and app_state["active_device_id"] != device_id:
        raise HTTPException(status_code=423, detail="Child mode is already active on another device.")
    
    app_state["mode"] = "child_active"
    app_state["active_device_id"] = device_id
    print(f"✅ Child mode started for device: {device_id}")
    return {"status": "success", "mode": "child_active", "device_id": device_id}

@app.post("/start_parent_mode")
async def start_parent_mode(device_id: str = Form(...)):
    '''Acquires a lock for a device to enter 'Parent Mode'.'''
    if app_state["mode"] == "parent_active" and app_state["active_device_id"] != device_id:
        raise HTTPException(status_code=423, detail="Parent mode is already active on another device.")
        
    app_state["mode"] = "parent_active"
    app_state["active_device_id"] = device_id
    print(f"✅ Parent mode started for device: {device_id}")
    return {"status": "success", "mode": "parent_active", "device_id": device_id}

@app.post("/release_mode")
async def release_mode(device_id: str = Form(...)):
    '''Releases the lock held by a device.'''
    if app_state["active_device_id"] == device_id:
        print(f"ℹ️ Releasing lock for device: {device_id}. Current mode: {app_state['mode']}")
        app_state["mode"] = "none"
        app_state["active_device_id"] = None
        return {"status": "success", "message": "Mode released."}
    
    print(f"⚠️ Received release request from non-active device: {device_id}. Current active: {app_state['active_device_id']}")
    return {"status": "success", "message": "No active lock found for this device, but request was processed."}

@app.get("/get_status")
async def get_status():
    '''Returns the current status of the device mode lock.'''
    return app_state


# ------------------ ALERT HANDLING ------------------

@app.post("/alert")
async def receive_alert(
    latitude: str = Form(...),
    longitude: str = Form(...),
    file: UploadFile = File(...)
):
    '''Receives an alert from a child device, stores it, and prepares for parent pickup.'''
    timestamp = int(time.time() * 1000)
    audio_filename = f"{timestamp}.wav"
    audio_path = os.path.join(AUDIO_STORAGE_DIR, audio_filename)

    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    map_link = f"http://maps.google.com/maps?q={latitude},{longitude}"
    audio_url = f"/audio/{audio_filename}"

    new_alert = {
        "timestamp": timestamp,
        "location": map_link,
        "audio_url": audio_url
    }
    
    alerts_storage.append(new_alert)
    print(f"🚨 New Alert Received: {new_alert}")
    
    return {"status": "success", "alert_received": new_alert}

@app.get("/get_alerts")
async def get_alerts(since: int = 0):
    """
    Endpoint for the parent app to poll for new alerts.
    Filters alerts based on a 'since' timestamp (in milliseconds).
    The parent device is responsible for tracking the timestamp of the last alert it received.
    """
    # Filter alerts that are newer than the provided timestamp
    alerts_to_send = [alert for alert in alerts_storage if alert["timestamp"] > since]
    
    if alerts_to_send:
        print(f"📦 Found {len(alerts_to_send)} new alerts for a request with 'since={since}'.")
        
    return alerts_to_send

@app.get("/audio/{file_name}")
async def get_audio_file(file_name: str):
    '''Serves the stored audio file for an alert.'''
    file_path = os.path.join(AUDIO_STORAGE_DIR, file_name)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/wav")
    raise HTTPException(status_code=404, detail="Audio file not found")


# ------------------ UPDATE LOCATION ------------------
@app.post("/update_location")
async def update_location(location: LocationData):
    try:
        with open(LOCATION_FILE, "w") as f:
            json.dump(location.dict(), f)
        print(f"📍 Location updated and saved to file: {location.dict()}")
        return {"status": "success", "received_location": location.dict()}
    except Exception as e:
        print(f"🔥 ERROR: Failed to write location to file: {e}")
        raise HTTPException(status_code=500, detail="Could not save location data.")

# ------------------ TEST CONNECTION ------------------
@app.get("/test_connection")
def test_connection():
    return {"status": "ok"}


# ------------------ LIST SPEAKERS ------------------
@app.get("/list_speakers")
def list_speakers():
    users = load_all_embeddings()
    return {"speakers": list(users.keys())}

#---------------------DELETE----------------------
@app.delete("/delete_speaker/{name}")
def delete_speaker(name: str):
    speaker_dir = os.path.join("app", "data", "familiar_embeddings", name)
    if not os.path.exists(speaker_dir):
        raise HTTPException(status_code=404, detail="Speaker not found")
    shutil.rmtree(speaker_dir)
    return {"status": "deleted", "speaker": name}


# ------------------ CLEAR TEMP AUDIO ------------------
@app.post("/clear_temp_audio")
def clear_temp_audio():
    try:
        shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR, exist_ok=True)
        return {"status": "success", "message": "Temporary audio files cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear temporary audio files: {e}")


# ------------------ ENROLL ------------------
@app.post("/enroll")
async def enroll(
    name: str = Form(...),
    file: UploadFile = File(...)
):
    temp_path = os.path.join(TEMP_DIR, file.filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    emb = get_ecapa_embedding_from_file(temp_path)
    save_embedding(name, emb)
    return {
        "status": "enrolled",
        "name": name,
        "embedding_dim": len(emb)
    }


# ------------------ VERIFY (FAMILIARITY CHECK) ------------------
@app.post("/verify")
async def verify(file: UploadFile = File(...)):
    temp_path = os.path.join(TEMP_DIR, file.filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    query_emb = get_ecapa_embedding_from_file(temp_path)
    users = load_all_embeddings()

    # If there are no enrolled users, any voice is a stranger.
    if not users:
        print("🤷 No familiar voices enrolled. Automatically classifying as stranger.")
        best_score = 0.0 # Or some other indicator of no match
    else:
        best_match = None
        best_score = 0.0
        for user, stored_emb in users.items():
            score = match_embedding(query_emb, stored_emb)
            if score > best_score:
                best_score = score
                best_match = user
        
        if best_score >= THRESHOLD:
            return {
                "result": "familiar",
                "name": best_match,
                "similarity": float(best_score)
            }

    # Stranger detected. Create an alert.
    print("🕵️ Stranger detected. Checking for location data...")
    latest_location = None
    if os.path.exists(LOCATION_FILE):
        try:
            with open(LOCATION_FILE, "r") as f:
                latest_location = json.load(f)
            print(f"✅ Found location data: {latest_location}")
        except Exception as e:
            print(f"🔥 ERROR: Failed to read or parse location file: {e}")
    else:
        print("⚠️ Location file not found.")

    if latest_location:
        timestamp = int(time.time() * 1000)
        audio_filename = f"{timestamp}.wav"
        audio_path = os.path.join(AUDIO_STORAGE_DIR, audio_filename)
        
        # Copy the received audio file to the alert audio storage
        shutil.copyfile(temp_path, audio_path)
        
        map_link = f"http://maps.google.com/maps?q={latest_location['latitude']},{latest_location['longitude']}"
        audio_url = f"/audio/{audio_filename}"

        new_alert = {
            "timestamp": timestamp,
            "location": map_link,
            "audio_url": audio_url
        }
        
        alerts_storage.append(new_alert)
        print(f"🚨 New Alert generated from /recognize (stranger): {new_alert}")
    else:
        print("⚠️ Stranger detected via /recognize, but no location data available to create an alert.")
        
    return {
        "result": "stranger",
        "similarity": float(best_score)
    }


# ------------------ ANDROID USES /recognize ------------------
@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    return await verify(file)


# ------------------ ROOT ROUTE ------------------
@app.get("/")
def home():
    return {"message": "Speaker Recognition Backend Running"}

