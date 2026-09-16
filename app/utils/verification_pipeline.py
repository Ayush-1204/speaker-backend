"""Three-stage verify: VAD → speech-likeness → ReDimNet multi-chunk speaker match."""

from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf
try:
    import librosa
except Exception:
    librosa = None

from . import config
from .audio_preprocess import normalize_waveform
from .compare import THRESHOLD, match_embedding
from .feature_extractor import embed_waveform_chunks
from .speech_gate import assess_speech_likeness
from .storage import load_all_embeddings
from .vad import get_speech_segments

# Change 2: stricter voiced-duration floor for enrollment only.
# Inference still uses MIN_VOICED_MS (250 ms) so it is not affected.
MIN_VOICED_MS_ENROLL = int(config.__dict__.get("MIN_VOICED_MS_ENROLL",
    __import__("os").environ.get("MIN_VOICED_MS_ENROLL", "3000")))
MIN_SNR_DB_ENROLL = float(__import__("os").environ.get("SAFEEAR_MIN_SNR_DB_ENROLL", "8.0"))
# Fraction of total voiced duration that the longest segment must cover
# (proxy for single-speaker presence).
MIN_DOMINANT_SEGMENT_FRACTION = float(
    __import__("os").environ.get("SAFEEAR_MIN_DOMINANT_SEGMENT_FRACTION", "0.15")
)



def _load_mono_16k(path: str) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32")
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != 16000:
        if librosa is None:
            raise ImportError("librosa is not available")
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
    return normalize_waveform(data.astype(np.float32))


def _voiced_duration_ms(segments: List[Tuple[int, int]], sr: int) -> float:
    return sum(max(0, e - s) for s, e in segments) * 1000.0 / float(sr)


def run_verify_stages(audio_path: str) -> Dict[str, Any]:
    waveform = _load_mono_16k(audio_path)
    sr = 16000
    stages: Dict[str, Any] = {}

    try:
        segments = get_speech_segments(waveform, sr=sr, use_silero=True)
    except Exception:
        segments = get_speech_segments(waveform, sr=sr, use_silero=False)

    voiced_ms = _voiced_duration_ms(segments, sr)
    vad_ok = voiced_ms >= config.MIN_VOICED_MS and len(segments) > 0
    stages["vad"] = {
        "passed": vad_ok,
        "voiced_ms": round(voiced_ms, 1),
        "num_segments": len(segments),
    }

    if not vad_ok:
        return {
            "stages": stages,
            "result": "rejected",
            "reject_reason": "no_voice_activity",
            "similarity": 0.0,
            "name": None,
            "closest_speaker": None,
            "alert": True,
        }

    longest = max(segments, key=lambda ab: ab[1] - ab[0])
    gate_audio = waveform[longest[0] : longest[1]]
    speech_ok, speech_metrics = assess_speech_likeness(gate_audio, sr)
    stages["speech_quality"] = {"passed": speech_ok, **speech_metrics}

    if not speech_ok:
        return {
            "stages": stages,
            "result": "rejected",
            "reject_reason": "not_speech_like",
            "similarity": 0.0,
            "name": None,
            "closest_speaker": None,
            "alert": True,
        }

    query_embs = embed_waveform_chunks(waveform, sr=sr, segments=segments)
    users = load_all_embeddings()

    best_match = None
    best_score = 0.0
    for user, stored_emb in users.items():
        chunk_scores = [match_embedding(q, stored_emb) for q in query_embs]
        user_score = max(chunk_scores) if chunk_scores else 0.0
        if user_score > best_score:
            best_score = user_score
            best_match = user

    familiar = best_score >= THRESHOLD
    stages["speaker"] = {
        "passed": familiar,
        "similarity": float(best_score),
        "chunks_used": len(query_embs),
        "threshold": float(THRESHOLD),
    }

    if familiar:
        return {
            "stages": stages,
            "result": "familiar",
            "name": best_match,
            "closest_speaker": best_match,
            "similarity": float(best_score),
            "alert": False,
        }

    return {
        "stages": stages,
        "result": "stranger",
        "name": None,
        "closest_speaker": best_match,
        "similarity": float(best_score),
        "alert": True,
    }


def _estimate_snr_db(waveform: np.ndarray, vad_segments: List[Tuple[int, int]]) -> float:
    """Rough segmental SNR estimate (Change 2).

    Treats non-voiced frames as the noise floor.  Underestimates SNR in
    reverberant conditions (reverberation falls inside voiced frames), so it
    is conservative — appropriate for an enrollment gate.

    Returns float('inf') when no noise samples exist (perfectly clean audio).
    """
    voiced_set = set()
    for s, e in vad_segments:
        voiced_set.update(range(s, e))

    all_indices = set(range(len(waveform)))
    noise_indices = sorted(all_indices - voiced_set)

    if not noise_indices:
        return float('inf')

    noise_waveform = waveform[noise_indices]
    noise_energy = float(np.mean(noise_waveform ** 2))
    if noise_energy < 1e-12:
        return float('inf')

    voiced_indices = sorted(voiced_set)
    if not voiced_indices:
        return 0.0

    voiced_waveform = waveform[voiced_indices]
    voiced_energy = float(np.mean(voiced_waveform ** 2))

    return 10.0 * np.log10(max(1e-12, voiced_energy) / noise_energy)


def run_enroll_embedding(audio_path: str) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Extract enrollment embeddings with strict quality gates (Change 2).

    Quality gates applied:
        1. voiced_ms >= MIN_VOICED_MS_ENROLL (default 3000 ms)
        2. Speech-likeness gate (spectral flatness / centroid)
        3. Dominant segment fraction >= MIN_DOMINANT_SEGMENT_FRACTION (default 0.60)
           — rough proxy for single-speaker presence
        4. SNR >= MIN_SNR_DB_ENROLL (default 15 dB)
           — rejects clips recorded in loud environments

    Raises ValueError with a specific rejection code on every gate failure so
    the API can return a descriptive error to the client.
    """
    waveform = _load_mono_16k(audio_path)
    sr = 16000
    try:
        segments = get_speech_segments(waveform, sr=sr, use_silero=True)
    except Exception:
        segments = get_speech_segments(waveform, sr=sr, use_silero=False)

    voiced_ms = _voiced_duration_ms(segments, sr)
    info: Dict[str, Any] = {
        "vad": {"voiced_ms": round(voiced_ms, 1), "num_segments": len(segments)},
    }

    # Gate 1: minimum voiced duration
    if voiced_ms < MIN_VOICED_MS_ENROLL:
        info["reject_reason"] = "voiced_too_short"
        info["voiced_ms"] = round(voiced_ms, 1)
        info["min_voiced_ms_required"] = MIN_VOICED_MS_ENROLL
        raise ValueError(
            f"enroll_rejected_voiced_too_short | voiced_ms={voiced_ms:.0f} | min={MIN_VOICED_MS_ENROLL}"
        )

    # Gate 2: speech-likeness
    longest = max(segments, key=lambda ab: ab[1] - ab[0])
    speech_ok, speech_metrics = assess_speech_likeness(waveform[longest[0] : longest[1]], sr)
    info["speech_quality"] = {"passed": speech_ok, **speech_metrics}
    if not speech_ok:
        raise ValueError("enroll_rejected_not_speech_like")

    # Gate 3: dominant-segment fraction (overlap / multi-speaker proxy)
    longest_duration = max(0, longest[1] - longest[0])
    total_voiced_samples = sum(max(0, e - s) for s, e in segments)
    if total_voiced_samples > 0:
        dominant_fraction = longest_duration / total_voiced_samples
    else:
        dominant_fraction = 0.0

    info["dominant_fraction"] = round(dominant_fraction, 3)
    if dominant_fraction < MIN_DOMINANT_SEGMENT_FRACTION:
        raise ValueError(
            f"enroll_rejected_possible_overlap | dominant_fraction={dominant_fraction:.2f} "
            f"| min={MIN_DOMINANT_SEGMENT_FRACTION:.2f}"
        )

    # Gate 4: SNR estimate
    snr_db = _estimate_snr_db(waveform, segments)
    info["snr_db"] = round(snr_db, 1) if snr_db != float('inf') else None
    if snr_db < MIN_SNR_DB_ENROLL:
        raise ValueError(
            f"enroll_rejected_snr_too_low | snr_db={snr_db:.1f} | min={MIN_SNR_DB_ENROLL:.1f}"
        )

    embs = embed_waveform_chunks(waveform, sr=sr, segments=segments)
    if not embs:
        raise ValueError("enroll_rejected_no_chunks")
    return embs, info
