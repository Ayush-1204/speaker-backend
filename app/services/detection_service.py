import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from app.utils import prd_services_db as svc


def _normalized_embedding(vec: Optional[Any]) -> Optional[np.ndarray]:
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    n = float(np.linalg.norm(arr))
    if n <= 1e-12:
        return None
    return (arr / n).astype(np.float32)


class DetectionService:
    @staticmethod
    def process_audio_chunk(
        db,
        parent_id: str,
        device_id: str,
        arr: np.ndarray,
        sr: int,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Process decoded audio array and run evaluation window.

        Returns a dict containing: session, stage, score, closest_speaker_id,
        current_stranger_centroid, perf_latency_ms, inference_buffer, window_samples
        """
        session = svc.get_or_create_session(parent_id, device_id)
        if latitude is not None and longitude is not None:
            session.lat = latitude
            session.lon = longitude

        inference_buffer = svc.append_frame(session, arr)

        window_samples = int(svc.WINDOW_SEC * session.sr)
        if inference_buffer.shape[0] < window_samples:
            return {
                "status": "no_hop",
                "reason": "insufficient_samples",
                "chunk_samples": inference_buffer.shape[0],
                "session": session,
            }

        if not svc.should_hop(len(arr), session.sr):
            return {"status": "no_hop", "reason": "should_not_hop", "session": session}

        window = inference_buffer[-window_samples:]

        stage = svc.evaluate_window(db, parent_id, window, session.sr)
        current_stranger_centroid = _normalized_embedding(stage.pop("_query_centroid", None))
        score = float(stage.get("tier3", {}).get("score", 0.0))
        closest_speaker_id = stage.get("tier3", {}).get("closest_speaker_id")
        perf_latency_ms = float(stage.get("perf", {}).get("processing_ms", 0.0))

        return {
            "status": "ok",
            "session": session,
            "stage": stage,
            "current_stranger_centroid": current_stranger_centroid,
            "score": score,
            "closest_speaker_id": closest_speaker_id,
            "perf_latency_ms": perf_latency_ms,
            "inference_buffer": inference_buffer,
            "window_samples": window_samples,
        }
