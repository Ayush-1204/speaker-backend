#!/usr/bin/env python3
"""
03_train.py
Train a Keras classifier on the YAMNet embeddings saved in data/features_yamnet.
Saves model to models/yamnet_classifier.keras and models/yamnet_classifier.h5
Run: python3 03_train.py
"""
from pathlib import Path
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import json

ROOT = Path.cwd()
FEATURES = ROOT / "data" / "features_yamnet"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

# Load data
X = np.load(FEATURES / "X_yamnet.npy")
y = np.load(FEATURES / "y_yamnet.npy")
with open(FEATURES / "label2idx.json","r") as f:
    label2idx = json.load(f)
idx2label = {v:k for k,v in label2idx.items()}
num_classes = len(label2idx)

print("Data shapes:", X.shape, y.shape, "classes:", num_classes)

# Build model
inp = layers.Input(shape=(1024,), name="yamnet_embedding")
x = layers.Dense(512, activation="relu")(inp)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
out = layers.Dense(num_classes, activation="softmax")(x)
model = keras.Model(inputs=inp, outputs=out)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Train
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint(str(MODELS / "yamnet_classifier.keras"), save_best_only=True, save_format="keras"),
]
history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=200, batch_size=32, callbacks=callbacks)

# Save final models
model.save(str(MODELS / "yamnet_classifier.keras"))
model.save(str(MODELS / "yamnet_classifier.h5"))
# Write classes in order of indices
with open(MODELS / "classes.json", "w") as f:
    json.dump([idx2label[i] for i in range(num_classes)], f, indent=2)

print("Saved model to", MODELS)
