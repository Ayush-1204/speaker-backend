import os

DEVICE = os.environ.get("SPEAKER_DEVICE", "cpu")
# Official ReDimNet2 torch.hub source.
REDIMNET_HUB_REPO = os.environ.get("REDIMNET_HUB_REPO", "PalabraAI/redimnet2")
REDIMNET_HUB_ENTRY = os.environ.get("REDIMNET_HUB_ENTRY", "redimnet2")
REDIMNET_MODEL_NAME = os.environ.get("REDIMNET_MODEL_NAME", "b6")
REDIMNET_TRAIN_TYPE = os.environ.get("REDIMNET_TRAIN_TYPE", "lm")
REDIMNET_DATASET = os.environ.get("REDIMNET_DATASET", "vox2")

EMBEDDING_DIM = 192

MIN_CHUNK_SAMPLES = int(os.environ.get("MIN_CHUNK_SAMPLES", "16000"))
MAX_CHUNK_SAMPLES = int(os.environ.get("MAX_CHUNK_SAMPLES", "48000"))
MAX_VERIFY_CHUNKS = int(os.environ.get("MAX_VERIFY_CHUNKS", "6"))
MIN_VOICED_MS = int(os.environ.get("MIN_VOICED_MS", "250"))

# Slightly relaxed defaults for real-world phone captures to avoid false rejects.
SPEECH_GATE_MAX_FLATNESS = float(os.environ.get("SPEECH_GATE_MAX_FLATNESS", "0.60"))
SPEECH_GATE_MIN_RMS = float(os.environ.get("SPEECH_GATE_MIN_RMS", "1e-6"))
SPEECH_GATE_CENTROID_MIN_HZ = float(os.environ.get("SPEECH_GATE_CENTROID_MIN_HZ", "150"))
SPEECH_GATE_CENTROID_MAX_HZ = float(os.environ.get("SPEECH_GATE_CENTROID_MAX_HZ", "8000"))
