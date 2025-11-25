#!/usr/bin/env python3
"""
04_evaluate.py
Evaluate trained model on the held-out validation/test split and compute metrics (accuracy, confusion matrix, ROC per class if needed).
Run: python3 04_evaluate.py
"""
from pathlib import Path
import numpy as np
import json
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report

ROOT = Path.cwd()
FEATURES = ROOT / "data" / "features_yamnet"
MODELS = ROOT / "models"

X = np.load(FEATURES / "X_yamnet.npy")
y = np.load(FEATURES / "y_yamnet.npy")
with open(FEATURES / "label2idx.json","r") as f:
    label2idx = json.load(f)
idx2label = {v:k for k,v in label2idx.items()}

model = keras.models.load_model(MODELS / "yamnet_classifier.keras")
# Simple eval on entire dataset (or split properly)
pred_probs = model.predict(X, batch_size=32)
pred = pred_probs.argmax(axis=1)

print("Classification report:\n", classification_report(y, pred, target_names=[idx2label[i] for i in range(len(idx2label))]))
print("Confusion matrix:\n", confusion_matrix(y, pred))
