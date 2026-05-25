from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from scripttuner.data_sources import switchboard


def _make_fake_tar(conv_ids: list[str] | None = None) -> bytes:
    """Build a fake MSU tarball: nested dirs, A/B trans + word files, plus noise."""
    if conv_ids is None:
        conv_ids = ["sw2005", "sw2006"]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:

        def _add(name: str, content: str) -> None:
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

        root = "swb_ms98_transcriptions"
        _add(f"{root}/AAREADME.text", "readme noise")
        _add(f"{root}/sw-ms98-dict.text", "lexicon noise")
        for conv in conv_ids:
            num = conv[2:]
            prefix = num[:2]
            for side in ("A", "B"):
                base = f"{conv}{side}-ms98-a"
                d = f"{root}/{prefix}/{num}"
                _add(f"{d}/{base}-trans.text", f"{base}-0001 0.0 1.0 hello\n")
                # word-level alignment must be excluded
                _add(f"{d}/{base}-word.text", f"{base}-0001 0.0 1.0 hello\n")
    return buf.getvalue()


def test_download_extracts_only_trans_flattened(tmp_path: Path) -> None:
    fake = _make_fake_tar(["sw2005", "sw2006"])
    files = switchboard.download(tmp_path, fetcher=lambda _u: fake, expected=4)
    assert len(files) == 4
    assert all(f.name.endswith("-trans.text") for f in files)
    # flattened: every file sits directly in dest_dir
    assert all(f.parent == tmp_path for f in files)
    names = {f.name for f in files}
    assert names == {
        "sw2005A-ms98-a-trans.text",
        "sw2005B-ms98-a-trans.text",
        "sw2006A-ms98-a-trans.text",
        "sw2006B-ms98-a-trans.text",
    }


def test_download_excludes_word_readme_lexicon(tmp_path: Path) -> None:
    fake = _make_fake_tar(["sw2005"])
    files = switchboard.download(tmp_path, fetcher=lambda _u: fake, expected=2)
    assert not any("word" in f.name for f in files)
    assert not any(f.name == "AAREADME.text" for f in files)
    assert not any("dict" in f.name for f in files)


def test_download_is_idempotent(tmp_path: Path) -> None:
    fake = _make_fake_tar(["sw2005", "sw2006"])
    calls = {"n": 0}

    def fetcher(_u: str) -> bytes:
        calls["n"] += 1
        return fake

    switchboard.download(tmp_path, fetcher=fetcher, expected=4)
    switchboard.download(tmp_path, fetcher=fetcher, expected=4)
    assert calls["n"] == 1


def test_download_force_redownloads(tmp_path: Path) -> None:
    fake = _make_fake_tar(["sw2005", "sw2006"])
    calls = {"n": 0}

    def fetcher(_u: str) -> bytes:
        calls["n"] += 1
        return fake

    switchboard.download(tmp_path, fetcher=fetcher, expected=4)
    switchboard.download(tmp_path, force=True, fetcher=fetcher, expected=4)
    assert calls["n"] == 2


def test_download_raises_on_wrong_count(tmp_path: Path) -> None:
    fake = _make_fake_tar(["sw2005"])  # yields 2 trans files
    with pytest.raises(RuntimeError, match="Expected 4"):
        switchboard.download(tmp_path, fetcher=lambda _u: fake, expected=4)
