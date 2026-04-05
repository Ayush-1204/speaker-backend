"""Bootstrap a local ReDimNet source tree during Render builds.

The production app loads ReDimNet in strict local mode, so the repo must
exist inside the deployed image before the service starts.
"""

from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_ARCHIVE_URLS = (
    "https://github.com/PalabraAI/redimnet2/archive/refs/heads/main.zip",
    "https://github.com/PalabraAI/redimnet2/archive/refs/heads/master.zip",
)


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


def _download(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url, timeout=120) as response:
        destination.write_bytes(response.read())


def _extract_archive(archive_path: Path, extract_dir: Path) -> None:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extract_dir)
            return
    except zipfile.BadZipFile:
        pass

    with tarfile.open(archive_path, mode="r:*") as archive:
        archive.extractall(extract_dir)


def _pick_source_root(stage_dir: Path) -> Path:
    entries = [entry for entry in stage_dir.iterdir() if entry.name != ".ready"]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return stage_dir


def _ensure_target_from_source(source_dir: Path, target_dir: Path) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_dir), str(target_dir))


def main() -> int:
    target_dir = _env_path("REDIMNET_LOCAL_REPO_DIR", "vendor/redimnet2")
    if target_dir.is_dir() and any(target_dir.iterdir()):
        print(f"[bootstrap_redimnet] existing repo found at {target_dir}")
        return 0

    archive_override = os.environ.get("REDIMNET_LOCAL_REPO_ARCHIVE_URL", "").strip()
    candidate_urls = [archive_override] if archive_override else []
    candidate_urls.extend(url for url in DEFAULT_ARCHIVE_URLS if url not in candidate_urls)

    if not candidate_urls:
        print("[bootstrap_redimnet] no archive URL configured; skipping bootstrap")
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        target_dir.mkdir(parents=True, exist_ok=True)
        return 0

    with tempfile.TemporaryDirectory(prefix="redimnet_build_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        archive_path = tmp_path / "redimnet_source.archive"
        stage_dir = tmp_path / "stage"
        stage_dir.mkdir(parents=True, exist_ok=True)

        last_error: Exception | None = None
        for url in candidate_urls:
            try:
                print(f"[bootstrap_redimnet] downloading {url}")
                _download(url, archive_path)
                print("[bootstrap_redimnet] extracting archive")
                _extract_archive(archive_path, stage_dir)
                source_dir = _pick_source_root(stage_dir)
                _ensure_target_from_source(source_dir, target_dir)
                marker = target_dir / ".ready"
                marker.write_text("ok", encoding="ascii")
                print(f"[bootstrap_redimnet] ready at {target_dir}")
                return 0
            except (urllib.error.URLError, OSError, tarfile.TarError, zipfile.BadZipFile, shutil.Error) as exc:
                last_error = exc
                print(f"[bootstrap_redimnet] failed for {url}: {exc}")
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                if archive_path.exists():
                    archive_path.unlink()
                for child in list(stage_dir.iterdir()):
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()

        if last_error is not None:
            raise last_error

    return 0


if __name__ == "__main__":
    raise SystemExit(main())