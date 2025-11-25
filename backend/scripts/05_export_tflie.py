#!/usr/bin/env python3
"""
05_export_tflite.py
Convert the trained Keras classifier to TFLite and save under exports/.
Run: python3 05_export_tflite.py
"""
from pathlib import Path
import tensorflow as tf

ROOT = Path.cwd()
MODELS = ROOT / "models"
EXPORTS = ROOT / "exports"
EXPORTS.mkdir(parents=True, exist_ok=True)

keras_model = tf.keras.models.load_model(MODELS / "yamnet_classifier.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
# If you want quantization, modify below. For now, float32:
tflite_model = converter.convert()
tflite_path = EXPORTS / "yamnet_classifier.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)
print("Saved TFLite:", tflite_path)
