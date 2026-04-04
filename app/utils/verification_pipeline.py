"""Three-stage verify: VAD → speech-likeness → ReDimNet multi-chunk speaker match."""

from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf
import librosa

from . import config
from .audio_preprocess import normalize_waveform
from .compare import THRESHOLD, match_embedding
from .feature_extractor import embed_waveform_chunks
from .speech_gate import assess_speech_likeness
from .storage import load_all_embeddings
from .vad import get_speech_segments


def _load_mono_16k(path: str) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32")
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != 16000:
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


def run_enroll_embedding(audio_path: str) -> Tuple[List[np.ndarray], Dict[str, Any]]:
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

    if voiced_ms < config.MIN_VOICED_MS:
        raise ValueError("enroll_rejected_no_voice_activity")

    longest = max(segments, key=lambda ab: ab[1] - ab[0])
    speech_ok, speech_metrics = assess_speech_likeness(waveform[longest[0] : longest[1]], sr)
    info["speech_quality"] = {"passed": speech_ok, **speech_metrics}
    if not speech_ok:
        raise ValueError("enroll_rejected_not_speech_like")

    embs = embed_waveform_chunks(waveform, sr=sr, segments=segments)
    if not embs:
        raise ValueError("enroll_rejected_no_chunks")
    return embs, info
