"""Model-family-specific fine-tuning formatters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from scripttuner.persistence.jsonl import read_jsonl
from scripttuner.preprocessing.ir import Pair
from scripttuner.training.style import STYLE_SPECS, get_style_spec

CHAT_MODEL_KEYS: frozenset[str] = frozenset(
    {
        "gemma4-e4b",
        "gemma4-e2b",
        "qwen3-4b",
        "qwen3-1.7b",
    }
)
SEQ2SEQ_MODEL_KEYS: frozenset[str] = frozenset({"t5gemma2"})
MODEL_KEYS: frozenset[str] = CHAT_MODEL_KEYS | SEQ2SEQ_MODEL_KEYS

FormatKind = Literal["chat", "seq2seq"]


def model_format_kind(model_key: str) -> FormatKind:
    """Return the fine-tuning data shape for a model key."""

    if model_key in CHAT_MODEL_KEYS:
        return "chat"
    if model_key in SEQ2SEQ_MODEL_KEYS:
        return "seq2seq"
    supported = ", ".join(sorted(MODEL_KEYS))
    raise ValueError(f"Unsupported model key {model_key!r}. Supported models: {supported}")


def format_split_folder(
    *,
    splits_dir: Path,
    output_dir: Path,
    model_key: str,
) -> dict[str, Any]:
    """Format train/validation/test split files for one target model family."""

    kind = model_format_kind(model_key)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    styles_seen: set[str] = set()

    for split_name in ("train", "validation", "test"):
        pairs = read_jsonl(splits_dir / f"{split_name}.jsonl", Pair)
        rows = [_format_pair(pair, kind=kind) for pair in pairs]
        styles_seen.update(pair.style for pair in pairs)
        counts[split_name] = _write_dict_jsonl(output_dir / f"{split_name}.jsonl", rows)

    manifest: dict[str, Any] = {
        "stage": "finetune_format",
        "model_key": model_key,
        "format": kind,
        "source_splits": str(splits_dir),
        "counts": counts,
        "style_tokens": {
            label: spec.control_token for label, spec in sorted(STYLE_SPECS.items())
        },
        "styles_present": sorted(styles_seen),
        "semi_formal_status": (
            "Reserved for future external corpus or teacher-LLM generated data; "
            "current SBCSAE pairs are expected to be casual unless additional "
            "semi_formal rows are added."
        ),
    }
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def _format_pair(pair: Pair, *, kind: FormatKind) -> dict[str, Any]:
    spec = get_style_spec(pair.style)
    input_text = _build_input(pair.formal_text, spec.control_token, spec.instruction)
    base_meta = {
        "pair_id": pair.pair_id,
        "source": pair.source,
        "style": pair.style,
        "speaker": pair.speaker,
        "monologue_id": pair.monologue_id,
        "style_token": spec.control_token,
        "source_metadata": pair.metadata,
    }
    if kind == "chat":
        return {
            "messages": [
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": pair.spoken_text},
            ],
            **base_meta,
        }
    return {
        "input": input_text,
        "target": pair.spoken_text,
        **base_meta,
    }


def _build_input(formal_text: str, control_token: str, instruction: str) -> str:
    return (
        f"{control_token}\n"
        f"{instruction}\n\n"
        "Input:\n"
        f"{formal_text}\n\n"
        "Output:"
    )


def _write_dict_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
    return len(rows)
