# streamlit_client/app.py - minimal UI to test backend connectivity
import streamlit as st
import requests

st.title("Speaker Recognition — Client (test)")
backend = st.text_input("Backend verify URL", "http://127.0.0.1:8000/verify")
uploaded = st.file_uploader("Upload WAV", type=["wav","mp3","m4a"])

if uploaded is not None:
    st.audio(uploaded)
    if st.button("Send to backend"):
        files = {"file": ("audio.wav", uploaded.getvalue(), "audio/wav")}
        try:
            r = requests.post(backend, files=files, timeout=30)
            st.write("Status:", r.status_code)
            try:
                st.json(r.json())
            except:
                st.write(r.text)
        except Exception as e:
            st.error(str(e))
