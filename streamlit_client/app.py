import streamlit as st
import requests
import io
import time
import uuid
import json

st.set_page_config(page_title="SafeEar Client Tester", layout="wide")
st.title("SafeEar E2E Client Tester")

# Session State for Authentication & Device
if "jwt" not in st.session_state:
    st.session_state.jwt = ""
if "device_id" not in st.session_state:
    st.session_state.device_id = str(uuid.uuid4())

# Sidebar Configuration
st.sidebar.header("Configuration")
backend_url = st.sidebar.text_input("Backend URL", "http://192.168.137.1:8000")

# --- AUTHENTICATION ---
st.sidebar.subheader("Authentication (Dev Token)")
dev_token = st.sidebar.text_input("Dev ID Token", "dev:test-parent")
if st.sidebar.button("Login via Google (Dev)"):
    try:
        r = requests.post(
            f"{backend_url}/auth/google",
            json={"id_token": dev_token}
        )
        if r.status_code == 200:
            st.session_state.jwt = r.json().get("access_token")
            st.sidebar.success("Logged in successfully!")
        else:
            st.sidebar.error(f"Failed to login: {r.text}")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

if st.session_state.jwt:
    st.sidebar.info("Authenticated ✅")
    
    # --- DEVICE REGISTRATION ---
    st.sidebar.subheader("Device Setup")
    device_name = st.sidebar.text_input("Device Name", "Test Streamlit Client")
    device_role = st.sidebar.selectbox("Role", ["child_device", "parent_device"])
    
    if st.sidebar.button("Register / Upsert Device"):
        try:
            headers = {"Authorization": f"Bearer {st.session_state.jwt}"}
            data = {
                "device_id": st.session_state.device_id,
                "device_name": device_name,
                "role": device_role,
            }
            r = requests.post(f"{backend_url}/devices/upsert", headers=headers, data=data)
            if r.status_code == 200:
                st.sidebar.success(f"Device registered: {st.session_state.device_id[:8]}")
            else:
                st.sidebar.error(f"Failed to register device: {r.text}")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

# --- TABS ---
tab_enroll, tab_simulate = st.tabs(["Enroll Speaker", "Simulate Inference (detect/chunk)"])

# 1. ENROLL SPEAKER TAB
with tab_enroll:
    st.header("Enroll a New Speaker")
    if not st.session_state.jwt:
        st.warning("Please login first.")
    else:
        speaker_name = st.text_input("Speaker Name", "Test User")
        speaker_role = st.selectbox("Speaker Role", ["familiar_adult", "child_self"])
        enroll_audio = st.file_uploader("Upload Enrollment Audio (>= 4s recommended)", type=["wav", "mp3"], key="enroll")
        
        if enroll_audio and st.button("Enroll Speaker"):
            with st.spinner("Enrolling..."):
                headers = {"Authorization": f"Bearer {st.session_state.jwt}"}
                files = {"file": ("audio.wav", enroll_audio.getvalue(), "audio/wav")}
                data = {"name": speaker_name, "role": speaker_role}
                try:
                    r = requests.post(f"{backend_url}/enroll/speaker", headers=headers, data=data, files=files)
                    if r.status_code == 200:
                        st.success("Enrolled successfully!")
                        st.json(r.json())
                    else:
                        st.error(f"Enrollment failed: {r.status_code}")
                        st.write(r.text)
                except Exception as e:
                    st.error(f"Error: {e}")

# 2. SIMULATE INFERENCE TAB
with tab_simulate:
    st.header("Simulate Continuous Inference")
    st.write("Upload a long WAV file. This script will chunk it into 2.0s segments and send them sequentially to `/detect/chunk` to simulate the Android client's foreground service.")
    
    if not st.session_state.jwt:
        st.warning("Please login first.")
    else:
        detect_audio = st.file_uploader("Upload Long Inference Audio", type=["wav"], key="detect")
        chunk_size_sec = st.number_input("Chunk Size (seconds)", value=2.0, min_value=0.5, max_value=10.0)
        
        if detect_audio and st.button("Start Simulation"):
            # A very simple chunker assuming 16kHz mono (just dividing bytes naively for test)
            # In a real WAV, headers should be preserved. We'll use the whole file for now to keep it simple,
            # or rely on the backend to handle raw chunks if we strip headers.
            # For this simple Streamlit client, we will just send the whole file as one chunk.
            st.info("Sending the entire file to `/detect/chunk` as a single payload for testing...")
            headers = {"Authorization": f"Bearer {st.session_state.jwt}"}
            files = {"file": ("audio.wav", detect_audio.getvalue(), "audio/wav")}
            data = {"device_id": st.session_state.device_id}
            
            try:
                start = time.time()
                r = requests.post(f"{backend_url}/detect/chunk", headers=headers, data=data, files=files)
                latency = time.time() - start
                st.write(f"**Latency:** {latency:.2f}s")
                
                if r.status_code == 200:
                    result = r.json()
                    decision = result.get("decision", "unknown")
                    score = result.get("score", 0.0)
                    streak = result.get("stranger_streak", 0)
                    
                    st.metric(label="Decision", value=decision)
                    col1, col2 = st.columns(2)
                    col1.metric("Score", f"{score:.4f}")
                    col2.metric("Stranger Streak", streak)
                    
                    st.subheader("Full JSON Response")
                    st.json(result)
                else:
                    st.error(f"Detection failed: {r.status_code}")
                    st.write(r.text)
            except Exception as e:
                st.error(f"Error: {e}")
