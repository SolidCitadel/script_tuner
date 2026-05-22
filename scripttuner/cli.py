"""ScriptTuner CLI 진입점.

사용 예:
    uv run scripttuner download sbcsae
    uv run scripttuner download sbcsae --force
    uv run scripttuner parse sbcsae datasets/sbcsae/SBC016.cha
    uv run scripttuner clean sbcsae SBC016
    uv run scripttuner monologue sbcsae SBC016
    uv run scripttuner pairs sbcsae SBC016 --model deepseek/deepseek-v4-flash
    uv run scripttuner --help

`pairs` 서브커맨드는 `.env`에서 `OPENAI_API_KEY` / `OPENAI_BASE_URL` 자동 인식.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import ModuleType

from dotenv import load_dotenv

from scripttuner.data_sources import sbcsae
from scripttuner.llm.openai_compatible import OpenAICompatibleClient
from scripttuner.persistence.jsonl import read_jsonl, write_jsonl
from scripttuner.preprocessing.chat import cleaner as chat_cleaner
from scripttuner.preprocessing.chat import parser as chat_parser
from scripttuner.preprocessing.ir import Monologue, Utterance
from scripttuner.preprocessing.monologue import DEFAULT_MIN_TOKENS, build_monologues
from scripttuner.preprocessing.pairs import (
    DEFAULT_PROMPT_VERSION,
    DEFAULT_STYLE,
    convert_to_formal,
)

DEFAULT_DATASETS_DIR = Path("datasets")
DEFAULT_DATA_DIR = Path("data")
CORPORA: dict[str, ModuleType] = {"sbcsae": sbcsae}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scripttuner", description="ScriptTuner CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dl = subparsers.add_parser("download", help="Download a dataset by name.")
    dl.add_argument("corpus", choices=sorted(CORPORA), help="Corpus to download.")
    dl.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DATASETS_DIR,
        help=f"Base destination directory (default: {DEFAULT_DATASETS_DIR}).",
    )
    dl.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist.",
    )

    pr = subparsers.add_parser(
        "parse", help="Parse a CHA file into parsed Utterance JSONL."
    )
    pr.add_argument("corpus", choices=sorted(CORPORA), help="Corpus name (selects adapter).")
    pr.add_argument("input_path", type=Path, help="Path to input .cha file.")
    pr.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Base data directory (default: {DEFAULT_DATA_DIR}).",
    )

    cl = subparsers.add_parser(
        "clean", help="Apply marker cleaning to parsed Utterance JSONL."
    )
    cl.add_argument("corpus", choices=sorted(CORPORA), help="Corpus name (selects cleaner).")
    cl.add_argument("stem", help="File stem (e.g. SBC016).")
    cl.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Base data directory (default: {DEFAULT_DATA_DIR}).",
    )

    mn = subparsers.add_parser(
        "monologue", help="Build Monologues from cleaned Utterance JSONL."
    )
    mn.add_argument("corpus", choices=sorted(CORPORA), help="Corpus name (resolves source dir).")
    mn.add_argument("stem", help="File stem (e.g. SBC016).")
    mn.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Base data directory (default: {DEFAULT_DATA_DIR}).",
    )
    mn.add_argument(
        "--min-tokens",
        type=int,
        default=DEFAULT_MIN_TOKENS,
        help=f"Minimum word tokens per monologue (default: {DEFAULT_MIN_TOKENS}).",
    )

    pa = subparsers.add_parser(
        "pairs",
        help="Convert Monologues to (formal, spoken) Pair JSONL via LLM.",
    )
    pa.add_argument(
        "corpus", choices=sorted(CORPORA), help="Corpus name (resolves source dir)."
    )
    pa.add_argument("stem", help="File stem (e.g. SBC016).")
    pa.add_argument(
        "--model",
        default=os.environ.get("LLM_MODEL"),
        help="LLM model slug (required unless LLM_MODEL env is set).",
    )
    pa.add_argument(
        "--style",
        default=DEFAULT_STYLE,
        help=f"Style label for produced pairs (default: {DEFAULT_STYLE}).",
    )
    pa.add_argument(
        "--prompt-version",
        default=DEFAULT_PROMPT_VERSION,
        help=f"Prompt version identifier (default: {DEFAULT_PROMPT_VERSION}).",
    )
    pa.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Base data directory (default: {DEFAULT_DATA_DIR}).",
    )
    pa.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Override cache directory (default: <data-dir>/cache/pairs).",
    )
    pa.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable disk cache.",
    )
    pa.add_argument(
        "--no-progress",
        action="store_true",
        help="Hide tqdm progress bar.",
    )
    pa.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="OpenAI SDK transient retry count (default: 3).",
    )
    pa.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N monologues (default: all).",
    )

    return parser


def _source_name(corpus: str) -> str:
    name: str = CORPORA[corpus].SOURCE_NAME
    return name


def _run_download(args: argparse.Namespace) -> int:
    module = CORPORA[args.corpus]
    dest_dir = args.dest / args.corpus
    files = module.download(dest_dir, force=args.force)
    print(f"OK: {len(files)} files in {dest_dir}")
    return 0


def _run_parse(args: argparse.Namespace) -> int:
    if args.corpus != "sbcsae":
        raise ValueError(f"parse adapter not registered for corpus: {args.corpus}")
    source = _source_name(args.corpus)
    _, utterances = chat_parser.parse(args.input_path, source=source)
    out_path = args.data_dir / "parsed" / source / f"{args.input_path.stem}.jsonl"
    n = write_jsonl(out_path, utterances)
    print(f"OK: wrote {n} utterances to {out_path}")
    return 0


def _run_clean(args: argparse.Namespace) -> int:
    if args.corpus != "sbcsae":
        raise ValueError(f"clean adapter not registered for corpus: {args.corpus}")
    source = _source_name(args.corpus)
    in_path = args.data_dir / "parsed" / source / f"{args.stem}.jsonl"
    utterances = read_jsonl(in_path, Utterance)
    cleaned = chat_cleaner.clean(utterances)
    out_path = args.data_dir / "cleaned" / source / f"{args.stem}.jsonl"
    n = write_jsonl(out_path, cleaned)
    print(f"OK: wrote {n} cleaned utterances to {out_path}")
    return 0


def _run_monologue(args: argparse.Namespace) -> int:
    source = _source_name(args.corpus)
    in_path = args.data_dir / "cleaned" / source / f"{args.stem}.jsonl"
    utterances = read_jsonl(in_path, Utterance)
    monologues = build_monologues(utterances, min_tokens=args.min_tokens)
    out_path = args.data_dir / "monologues" / source / f"{args.stem}.jsonl"
    n = write_jsonl(out_path, monologues)
    print(f"OK: wrote {n} monologues to {out_path}")
    return 0


def _run_pairs(args: argparse.Namespace) -> int:
    if not args.model:
        print(
            "error: --model is required (or set LLM_MODEL environment variable)",
            file=sys.stderr,
        )
        return 2
    source = _source_name(args.corpus)
    in_path = args.data_dir / "monologues" / source / f"{args.stem}.jsonl"
    monologues = read_jsonl(in_path, Monologue)
    if args.limit is not None:
        monologues = monologues[: args.limit]

    cache_dir: Path | None
    if args.no_cache:
        cache_dir = None
    else:
        cache_dir = args.cache_dir or (args.data_dir / "cache" / "pairs")

    client = OpenAICompatibleClient(model=args.model, max_retries=args.max_retries)
    pairs = convert_to_formal(
        monologues,
        client=client,
        model=args.model,
        cache_dir=cache_dir,
        prompt_version=args.prompt_version,
        style=args.style,
        progress=not args.no_progress,
    )
    out_path = args.data_dir / "pairs" / source / f"{args.stem}.jsonl"
    n = write_jsonl(out_path, pairs)
    skipped = len(monologues) - n
    print(f"OK: wrote {n} pairs to {out_path} ({skipped} skipped)")
    return 0


_COMMANDS = {
    "download": _run_download,
    "parse": _run_parse,
    "clean": _run_clean,
    "monologue": _run_monologue,
    "pairs": _run_pairs,
}


def main(argv: list[str] | None = None) -> int:
    load_dotenv()  # no-op when .env is missing
    parser = _build_parser()
    args = parser.parse_args(argv)
    runner = _COMMANDS.get(args.command)
    if runner is None:
        parser.error(f"Unknown command: {args.command}")
        return 2  # unreachable; parser.error exits
    return runner(args)


if __name__ == "__main__":
    sys.exit(main())
