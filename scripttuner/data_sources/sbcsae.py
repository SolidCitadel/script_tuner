"""Santa Barbara Corpus of Spoken American English (SBCSAE) downloader.

Source: UCSB Department of Linguistics
URL:    https://www.linguistics.ucsb.edu/research/santa-barbara-corpus-spoken-american-english
License: CC BY-ND 3.0 US

Citation:
    Du Bois, J. W., Chafe, W. L., Meyer, C., Thompson, S. A., Englebretson, R.,
    & Martey, N. (2000-2005). Santa Barbara Corpus of Spoken American English,
    Parts 1-4. Philadelphia: Linguistic Data Consortium.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

ZIP_URL = (
    "https://www.linguistics.ucsb.edu/sites/default/files/"
    "sitefiles/research/SBC/SBCSAE_chat.zip"
)
EXPECTED_FILES = 60
SOURCE_NAME = "SBCSAE"
"""IR `source` field value 및 디스크 산출물 디렉토리 이름 (e.g. data/parsed/SBCSAE/)."""


def _default_fetcher(url: str) -> bytes:
    with urllib.request.urlopen(url) as resp:
        data: bytes = resp.read()
    return data


def download(
    dest_dir: Path,
    *,
    force: bool = False,
    fetcher: Callable[[str], bytes] = _default_fetcher,
) -> list[Path]:
    """Download SBCSAE CHA transcripts into dest_dir.

    Returns a sorted list of .cha file paths.

    If force is False and 60 .cha files already exist in dest_dir, skip download.
    Raises RuntimeError if the extracted file count differs from EXPECTED_FILES.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(dest_dir.glob("SBC*.cha"))
    if not force and len(existing) == EXPECTED_FILES:
        return existing

    zip_bytes = fetcher(ZIP_URL)
    extracted = _extract_cha_files(zip_bytes, dest_dir)
    if len(extracted) != EXPECTED_FILES:
        raise RuntimeError(
            f"Expected {EXPECTED_FILES} .cha files in SBCSAE archive, got {len(extracted)}"
        )
    return extracted


def _extract_cha_files(zip_bytes: bytes, dest_dir: Path) -> list[Path]:
    extracted: list[Path] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            name = info.filename
            if name.startswith("__MACOSX/"):
                continue
            if not name.endswith(".cha"):
                continue
            basename = Path(name).name
            target = dest_dir / basename
            with zf.open(info) as src, target.open("wb") as dst:
                dst.write(src.read())
            extracted.append(target)
    return sorted(extracted)
