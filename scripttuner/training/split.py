"""Speaker-aware split creation for Pair fine-tuning data."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scripttuner.preprocessing.ir import Pair


SplitMap = dict[str, list[Pair]]


def split_by_speaker(
    pairs: list[Pair],
    *,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> SplitMap:
    """Split pairs by speaker so one speaker appears in only one split."""

    _validate_ratios(train_ratio, validation_ratio, test_ratio)
    grouped: dict[str, list[Pair]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.speaker].append(pair)

    speakers = list(grouped)
    rng = random.Random(seed)
    rng.shuffle(speakers)

    total = len(pairs)
    targets = {
        "train": total * train_ratio,
        "validation": total * validation_ratio,
        "test": total * test_ratio,
    }
    splits: SplitMap = {"train": [], "validation": [], "test": []}

    # Greedy assignment keeps split sizes close while respecting speaker boundaries.
    for speaker in sorted(speakers, key=lambda s: len(grouped[s]), reverse=True):
        chosen = min(
            splits,
            key=lambda name: (len(splits[name]) / targets[name]) if targets[name] else 1.0,
        )
        splits[chosen].extend(grouped[speaker])

    return splits


def write_split_files(
    splits: SplitMap,
    *,
    output_dir: Path,
    source_path: Path,
    seed: int,
    ratios: tuple[float, float, float],
) -> dict[str, Any]:
    """Write split JSONL files and a manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    speakers: dict[str, list[str]] = {}
    for name, items in splits.items():
        counts[name] = _write_pairs(output_dir / f"{name}.jsonl", items)
        speakers[name] = sorted({item.speaker for item in items})

    manifest: dict[str, Any] = {
        "stage": "finetune_split",
        "source_file": str(source_path),
        "split_strategy": "speaker_aware_greedy",
        "seed": seed,
        "ratios": {
            "train": ratios[0],
            "validation": ratios[1],
            "test": ratios[2],
        },
        "counts": counts,
        "speakers": {name: len(values) for name, values in speakers.items()},
        "speaker_lists": speakers,
        "note": (
            "Speakers are assigned to exactly one split to reduce leakage from "
            "speaker-specific phrasing."
        ),
    }
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def _validate_ratios(train: float, validation: float, test: float) -> None:
    if min(train, validation, test) <= 0:
        raise ValueError("split ratios must be positive")
    total = train + validation + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")


def _write_pairs(path: Path, pairs: list[Pair]) -> int:
    with path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(asdict(pair), ensure_ascii=False))
            f.write("\n")
    return len(pairs)

