"""Switchboard (MSU/ISIP transcripts) downloader.

Source: OpenSLR #5 — a mirror of the Mississippi State manually-corrected
        transcripts and lexicon for the Switchboard-1 corpus.
URL:    https://www.openslr.org/5/
License: Unrestricted — text transcripts only, no LDC audio (cf. ADR-0010).

We fetch only the turn-level transcripts (``*-trans.text``); the word-level
alignments (``*-word.text``), the README, and the lexicon are not needed for the
(spoken, formal) pair pipeline. Files are flattened into ``dest_dir`` (the
archive nests them under ``swb_ms98_transcriptions/<NN>/<conv>/``) because the
basenames (``sw2005A-ms98-a-trans.text``) are already globally unique.

Citation:
    Godfrey, J., Holliman, E., & McDaniel, J. (1992). SWITCHBOARD: Telephone
    speech corpus for research and development. ICASSP. Transcripts:
    ISIP/Mississippi State University manually corrected release.
"""

from __future__ import annotations

import io
import tarfile
import urllib.request
from collections.abc import Callable
from pathlib import Path

TARBALL_URL = "https://openslr.trmal.net/resources/5/switchboard_word_alignments.tar.gz"
TRANS_SUFFIX = "-trans.text"
EXPECTED_FILES = 4876
"""Turn-level transcript files in the MSU release (2438 conversations x A/B sides)."""
SOURCE_NAME = "Switchboard"
"""IR `source` field value 및 디스크 산출물 디렉토리 이름 (e.g. data/parsed/Switchboard/)."""


def _default_fetcher(url: str) -> bytes:
    with urllib.request.urlopen(url) as resp:
        data: bytes = resp.read()
    return data


def download(
    dest_dir: Path,
    *,
    force: bool = False,
    fetcher: Callable[[str], bytes] = _default_fetcher,
    expected: int = EXPECTED_FILES,
) -> list[Path]:
    """Download Switchboard MSU turn-level transcripts into dest_dir (flattened).

    Returns a sorted list of ``*-trans.text`` file paths.

    If force is False and `expected` transcript files already exist in dest_dir,
    skip download. Raises RuntimeError if the extracted file count differs from
    `expected`.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(dest_dir.glob(f"sw*{TRANS_SUFFIX}"))
    if not force and len(existing) == expected:
        return existing

    tar_bytes = fetcher(TARBALL_URL)
    extracted = _extract_trans_files(tar_bytes, dest_dir)
    if len(extracted) != expected:
        raise RuntimeError(
            f"Expected {expected} {TRANS_SUFFIX} files in Switchboard archive, "
            f"got {len(extracted)}"
        )
    return extracted


def _extract_trans_files(tar_bytes: bytes, dest_dir: Path) -> list[Path]:
    extracted: list[Path] = []
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            if not member.name.endswith(TRANS_SUFFIX):
                continue
            src = tf.extractfile(member)
            if src is None:
                continue
            target = dest_dir / Path(member.name).name
            with src, target.open("wb") as dst:
                dst.write(src.read())
            extracted.append(target)
    return sorted(extracted)
