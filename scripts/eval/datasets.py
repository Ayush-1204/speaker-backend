"""Turn public speech corpora into one manifest format for the speaker benchmark.

A manifest is a CSV with columns  speaker,path,group,language,gender,dataset.
`group` is what results are reported by (an accent, a language, a nationality);
impostor trials are drawn from the same group, because a stranger near a child
usually shares the family's language and accent.

Supported layouts (all free; some need you to accept terms on the provider's site):
  librispeech  LibriSpeech/test-clean                 openslr.org/12
  kathbath     Kathbath audio, 12 Indian languages    github.com/AI4Bharat/IndicSUPERB
  crowdsourced Google crowdsourced sets (OpenSLR 63-66, 70, 78, 79, 83 ...)
  commonvoice  Mozilla Common Voice locale folder     commonvoice.mozilla.org
  voxceleb1    VoxCeleb1 wav folder + vox1_meta.csv   robots.ox.ac.uk/~vgg/data/voxceleb
  vctk         VCTK 0.92 (speaker-info.txt accents)   datashare.ed.ac.uk/handle/10283/3443
  folder       <root>/<speaker>/**/*.wav (NISP, your own recordings, anything else)
"""

import csv
import glob
import os
import subprocess
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np

SR = 16000
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus")
FIELDS = ["speaker", "path", "group", "language", "gender", "dataset"]

KATHBATH_LANGUAGES = [
    "bengali", "gujarati", "hindi", "kannada", "malayalam", "marathi",
    "odia", "punjabi", "sanskrit", "tamil", "telugu", "urdu",
]

# Google crowdsourced corpora name speakers "<prefix>_<number>"; the prefix's first
# letters give the dialect or language and the last letter the gender.
CROWDSOURCED_GROUPS = {
    "ir": "English, Irish", "mi": "English, Midlands", "no": "English, Northern",
    "sc": "English, Scottish", "so": "English, Southern", "we": "English, Welsh",
    "ng": "English, Nigerian", "ml": "Malayalam", "mr": "Marathi", "ta": "Tamil",
    "te": "Telugu", "gu": "Gujarati", "kn": "Kannada", "bn": "Bengali",
}


@dataclass
class Utterance:
    speaker: str
    path: str
    group: str
    language: str = ""
    gender: str = ""
    dataset: str = ""


def _audio_files(root: str) -> List[str]:
    files: List[str] = []
    for ext in AUDIO_EXTS:
        files.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
    return sorted(files)


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def librispeech(root: str, group: Optional[str] = None, **_) -> Iterable[Utterance]:
    for path in glob.glob(os.path.join(root, "*", "*", "*.flac")):
        spk = _stem(path).split("-")[0]
        yield Utterance(f"ls-{spk}", path, group or "English, US (LibriSpeech)", "en", "", "librispeech")


def kathbath(root: str, group: Optional[str] = None, **_) -> Iterable[Utterance]:
    """Files are <utterance>-<speaker>-<m|f>.m4a under a per-language folder."""
    for path in _audio_files(root):
        parts = [p.lower() for p in os.path.normpath(path).split(os.sep)]
        lang = next((p for p in parts if p in KATHBATH_LANGUAGES), None)
        bits = _stem(path).split("-")
        if lang is None or len(bits) < 3:
            continue
        # The noisy split re-records the same speakers in noisy rooms; report it separately.
        noisy = any("noisy" in p for p in parts)
        name = group or f"{lang.title()} (Kathbath{' noisy' if noisy else ''})"
        yield Utterance(f"kb-{lang}-{bits[1]}", path, name, lang, bits[2][:1],
                        "kathbath_noisy" if noisy else "kathbath")


def crowdsourced(root: str, group: Optional[str] = None, language: str = "", **_) -> Iterable[Utterance]:
    for path in _audio_files(root):
        bits = _stem(path).split("_")
        if len(bits) < 3:
            continue
        prefix = bits[0].lower()
        spk = f"{prefix}_{bits[1]}"
        name = group or CROWDSOURCED_GROUPS.get(prefix[:2], prefix[:-1])
        yield Utterance(f"gc-{spk}", path, name, language, prefix[-1:], "crowdsourced")


def commonvoice(root: str, group: Optional[str] = None, **_) -> Iterable[Utterance]:
    """<root> is one locale folder holding validated.tsv and clips/."""
    tsv = os.path.join(root, "validated.tsv")
    with open(tsv, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            locale = row.get("locale") or os.path.basename(os.path.normpath(root))
            accent = (row.get("accents") or row.get("accent") or "").strip()
            name = group or (f"{locale}: {accent}" if accent else locale)
            yield Utterance(f"cv-{row['client_id'][:16]}", os.path.join(root, "clips", row["path"]),
                            name, locale, (row.get("gender") or "")[:1], "commonvoice")


def voxceleb1(root: str, group: Optional[str] = None, meta: Optional[str] = None, **_) -> Iterable[Utterance]:
    nationality: Dict[str, str] = {}
    if meta:
        with open(meta, encoding="utf-8") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                row = {k.strip(): (v or "").strip() for k, v in row.items()}
                nationality[row.get("VoxCeleb1 ID", "")] = row.get("Nationality", "")
    for path in glob.glob(os.path.join(root, "*", "*", "*.wav")):
        spk = os.path.basename(os.path.dirname(os.path.dirname(path)))
        name = group or f"{nationality.get(spk) or 'unknown'} (VoxCeleb1)"
        yield Utterance(f"vc-{spk}", path, name, "en", "", "voxceleb1")


def vctk(root: str, group: Optional[str] = None, **_) -> Iterable[Utterance]:
    accents: Dict[str, str] = {}
    info = os.path.join(root, "speaker-info.txt")
    if os.path.exists(info):
        with open(info, encoding="utf-8") as fh:
            next(fh, None)
            for line in fh:
                bits = line.split()
                if len(bits) >= 4:
                    accents[bits[0]] = bits[3]
    for path in _audio_files(root):
        spk = os.path.basename(os.path.dirname(path))
        if not spk.startswith("p") and not spk.startswith("s"):
            continue
        # VCTK 0.92 has two mics per utterance; keep mic1 only.
        if "_mic2" in path:
            continue
        name = group or f"English, {accents.get(spk, 'unknown')} (VCTK)"
        yield Utterance(f"vk-{spk}", path, name, "en", "", "vctk")


def folder(root: str, group: Optional[str] = None, language: str = "", **_) -> Iterable[Utterance]:
    """<root>/<speaker>/... ; a second folder level named like a language is kept as language."""
    for path in _audio_files(root):
        rel = os.path.relpath(path, root).split(os.sep)
        if len(rel) < 2:
            continue
        lang = language or (rel[1] if len(rel) > 2 else "")
        yield Utterance(f"fd-{rel[0]}", path, group or os.path.basename(os.path.normpath(root)),
                        lang, "", "folder")


ADAPTERS: Dict[str, Callable[..., Iterable[Utterance]]] = {
    "librispeech": librispeech, "kathbath": kathbath, "crowdsourced": crowdsourced,
    "commonvoice": commonvoice, "voxceleb1": voxceleb1, "vctk": vctk, "folder": folder,
}


def write_manifest(rows: Iterable[Utterance], out: str, append: bool = False) -> int:
    exists = append and os.path.exists(out)
    n = 0
    with open(out, "a" if exists else "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        for row in rows:
            w.writerow(asdict(row))
            n += 1
    return n


def read_manifests(paths: List[str]) -> List[Utterance]:
    rows: List[Utterance] = []
    for p in paths:
        with open(p, encoding="utf-8", newline="") as fh:
            rows.extend(Utterance(**{k: r.get(k, "") or "" for k in FIELDS}) for r in csv.DictReader(fh))
    return rows


def load_audio(path: str) -> np.ndarray:
    """Mono float32 at 16 kHz.  wav/flac/ogg via soundfile; anything else via ffmpeg."""
    if path.lower().endswith((".wav", ".flac", ".ogg")):
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32", always_2d=True)
        data = data.mean(axis=1)
        if sr != SR:
            import librosa

            data = librosa.resample(data, orig_sr=sr, target_sr=SR)
        return data.astype(np.float32)
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
