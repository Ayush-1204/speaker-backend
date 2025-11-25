# /models/download_models.py

import os
from speechbrain.pretrained import SpeakerRecognition
import torch

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# Create model directory if missing
os.makedirs(MODEL_DIR, exist_ok=True)

# Download SpeechBrain ECAPA model
_ = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=os.path.join(MODEL_DIR, "ecapa_model")
)

print("Model downloaded successfully to", MODEL_DIR)
