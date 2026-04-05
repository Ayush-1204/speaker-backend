"""ReDimNet2 B6 speaker embeddings via torch.hub. 192-D, L2-normalized."""

from typing import List, Optional, Tuple
import logging
import os
import time
import tempfile
import tarfile
import zipfile

import numpy as np
import soundfile as sf
import librosa
import torch
import requests

from . import config
from .audio_preprocess import normalize_waveform
from .vad import apply_vad

_redimnet = None
_redimnet_retry_after_ts = 0.0
_redimnet_load_failures = 0
logger = logging.getLogger(__name__)


def _bootstrap_local_repo_if_needed(configured_dir: str) -> str:
    """Return an existing local repo directory, optionally bootstrapping from an archive URL."""
    if configured_dir and os.path.isdir(configured_dir):
        return configured_dir

    archive_url = config.REDIMNET_LOCAL_REPO_ARCHIVE_URL
    extract_dir = config.REDIMNET_LOCAL_REPO_EXTRACT_DIR
    if not archive_url:
        return configured_dir

    try:
        os.makedirs(extract_dir, exist_ok=True)
        marker = os.path.join(extract_dir, ".ready")
        if not os.path.exists(marker):
            fd, archive_path = tempfile.mkstemp(prefix="redimnet2_", suffix=".archive")
            os.close(fd)
            try:
                with requests.get(archive_url, stream=True, timeout=60) as resp:
                    resp.raise_for_status()
                    with open(archive_path, "wb") as handle:
                        for chunk in resp.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                handle.write(chunk)

                extracted = False
                try:
                    with zipfile.ZipFile(archive_path) as zf:
                        zf.extractall(extract_dir)
                        extracted = True
                except zipfile.BadZipFile:
                    pass

                if not extracted:
                    with tarfile.open(archive_path, "r:*") as tf:
                        tf.extractall(extract_dir)

                with open(marker, "w", encoding="ascii") as handle:
                    handle.write("ok")
            finally:
                if os.path.exists(archive_path):
                    os.remove(archive_path)

        if os.path.isdir(extract_dir):
            logger.info("REDIMNET_LOCAL_REPO_BOOTSTRAPPED | path=%s", extract_dir)
            return extract_dir
    except Exception as exc:
        logger.warning("REDIMNET_LOCAL_REPO_BOOTSTRAP_FAILED | reason=%s", str(exc))

    return configured_dir


def _get_redimnet():
    global _redimnet
    global _redimnet_retry_after_ts
    global _redimnet_load_failures
    if _redimnet is None:
        now = time.time()
        if _redimnet_retry_after_ts > now:
            remaining = int(_redimnet_retry_after_ts - now)
            raise RuntimeError(f"ReDimNet temporarily unavailable; retry after {remaining}s")

        # Prefer official ReDimNet2 source; fallback to legacy ReDimNet for compatibility.
        last_exc: Optional[Exception] = None

        v2_train_type = config.REDIMNET_TRAIN_TYPE
        if v2_train_type == "ft_lm":
            v2_train_type = "lm"

        v1_train_type = config.REDIMNET_TRAIN_TYPE
        if v1_train_type == "lm":
            v1_train_type = "ft_lm"

        load_attempts = []
        local_repo = _bootstrap_local_repo_if_needed(config.REDIMNET_LOCAL_REPO_DIR)
        if local_repo and os.path.isdir(local_repo):
            load_attempts.append(
                (
                    local_repo,
                    config.REDIMNET_HUB_ENTRY,
                    {
                        "model_name": config.REDIMNET_MODEL_NAME,
                        "train_type": v2_train_type,
                        "dataset": config.REDIMNET_DATASET,
                        "pretrained": True,
                    },
                    "local",
                )
            )
        elif local_repo:
            logger.warning("REDIMNET_LOCAL_REPO_MISSING | path=%s", local_repo)

        allow_hub_network = not config.REDIMNET_DISABLE_HUB_NETWORK
        if config.REDIMNET_DISABLE_HUB_NETWORK and not (local_repo and os.path.isdir(local_repo)):
            if config.REDIMNET_STRICT_LOCAL:
                logger.warning("REDIMNET_STRICT_LOCAL_ENABLED | hub_network_fallback=disabled")
            else:
                logger.warning("REDIMNET_LOCAL_MISSING_FALLBACK_TO_HUB | strict_local=false")
                allow_hub_network = True

        if allow_hub_network:
            load_attempts.extend(
                [
                    (
                        config.REDIMNET_HUB_REPO,
                        config.REDIMNET_HUB_ENTRY,
                        {
                            "model_name": config.REDIMNET_MODEL_NAME,
                            "train_type": v2_train_type,
                            "dataset": config.REDIMNET_DATASET,
                            "pretrained": True,
                        },
                        None,
                    ),
                    (
                        "PalabraAI/redimnet2",
                        "redimnet2",
                        {
                            "model_name": config.REDIMNET_MODEL_NAME,
                            "train_type": v2_train_type,
                            "dataset": config.REDIMNET_DATASET,
                            "pretrained": True,
                        },
                        None,
                    ),
                    (
                        "IDRnD/ReDimNet",
                        "ReDimNet",
                        {
                            "model_name": config.REDIMNET_MODEL_NAME,
                            "train_type": v1_train_type,
                            "dataset": config.REDIMNET_DATASET,
                        },
                        None,
                    ),
                ]
            )

        seen = set()
        for hub_repo, hub_entry, load_kwargs, load_source in load_attempts:
            key = (hub_repo, hub_entry, load_source)
            if key in seen:
                continue
            seen.add(key)
            try:
                hub_kwargs = {
                    "repo_or_dir": hub_repo,
                    "model": hub_entry,
                    "trust_repo": True,
                    **load_kwargs,
                }
                if load_source is not None:
                    hub_kwargs["source"] = load_source
                _redimnet = torch.hub.load(
                    **hub_kwargs,
                )
                logger.info(
                    "REDIMNET_LOADED | hub_repo=%s | hub_entry=%s | source=%s | model_name=%s | train_type=%s | dataset=%s",
                    hub_repo,
                    hub_entry,
                    load_source or "github",
                    load_kwargs.get("model_name"),
                    load_kwargs.get("train_type"),
                    load_kwargs.get("dataset"),
                )
                break
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "REDIMNET_LOAD_FAILED | hub_repo=%s | hub_entry=%s | reason=%s",
                    hub_repo,
                    hub_entry,
                    str(exc),
                )
        if _redimnet is None:
            # Avoid hammering torch.hub/GitHub when unavailable, but recover sooner than fixed 5m.
            _redimnet_load_failures += 1
            backoff = min(
                max(1, config.REDIMNET_RETRY_MAX_SEC),
                max(1, config.REDIMNET_RETRY_BASE_SEC) * (2 ** (_redimnet_load_failures - 1)),
            )
            _redimnet_retry_after_ts = time.time() + float(backoff)
            raise RuntimeError("Failed to load ReDimNet model via torch.hub") from last_exc

        _redimnet_load_failures = 0
        _redimnet.eval()
        _redimnet.to(config.DEVICE)
    return _redimnet


def _load_audio(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    data, sr = sf.read(path, dtype="float32")
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
    return normalize_waveform(data.astype(np.float32)), target_sr


def _embed_waveform_array(waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
    w = normalize_waveform(np.asarray(waveform, dtype=np.float32).reshape(-1))
    min_s = max(16000, config.MIN_CHUNK_SAMPLES)

    if len(w) < 160:
        w = np.pad(w, (0, max(0, min_s - len(w))), mode="reflect")
    if len(w) < min_s:
        w = np.pad(w, (0, min_s - len(w)), mode="reflect")

    t = torch.tensor(w, dtype=torch.float32, device=config.DEVICE).unsqueeze(0)
    model = _get_redimnet()
    with torch.no_grad():
        emb_t = model(t)
    emb = emb_t.squeeze().detach().cpu().numpy().astype(np.float32).reshape(-1)
    if emb.shape[0] != config.EMBEDDING_DIM:
        raise RuntimeError(f"Embedding dim {emb.shape}; expected ({config.EMBEDDING_DIM},)")
    n = np.linalg.norm(emb)
    if n > 0:
        emb = emb / n
    return emb


def get_speaker_embedding_from_file(path: str) -> np.ndarray:
    """Enroll path: load → VAD trim → pad → one ReDimNet embedding."""
    waveform, sr = _load_audio(path, target_sr=16000)
    try:
        w = apply_vad(waveform, sr=sr, use_silero=True)
    except Exception:
        w = apply_vad(waveform, sr=sr, use_silero=False)
    if len(w) < 160:
        w = waveform
    return _embed_waveform_array(w, sr)


def embed_waveform_chunks(
    waveform: np.ndarray,
    sr: int = 16000,
    segments: Optional[List[Tuple[int, int]]] = None,
) -> List[np.ndarray]:
    """Bounded chunks from VAD segments; each chunk → one embedding (verify path)."""
    w = np.asarray(waveform, dtype=np.float32).reshape(-1)
    min_s = config.MIN_CHUNK_SAMPLES
    max_s = config.MAX_CHUNK_SAMPLES
    chunks: List[np.ndarray] = []

    seg_list = [(0, len(w))] if not segments else [(max(0, a), min(len(w), b)) for a, b in segments if b > a]

    for s, e in seg_list:
        piece = w[s:e]
        if len(piece) == 0:
            continue
        pos = 0
        while pos < len(piece):
            end = min(pos + max_s, len(piece))
            chunk = piece[pos:end]
            if len(chunk) < min_s:
                chunk = np.pad(chunk, (0, min_s - len(chunk)), mode="reflect")
            chunks.append(chunk.astype(np.float32, copy=False))
            pos = end
            if len(chunks) >= config.MAX_VERIFY_CHUNKS:
                break
        if len(chunks) >= config.MAX_VERIFY_CHUNKS:
            break

    if not chunks:
        chunk = w
        if len(chunk) < min_s:
            chunk = np.pad(chunk, (0, min_s - len(chunk)), mode="reflect")
        chunks = [chunk]

    out: List[np.ndarray] = []
    for c in chunks[: config.MAX_VERIFY_CHUNKS]:
        out.append(_embed_waveform_array(c, sr))
    return out
