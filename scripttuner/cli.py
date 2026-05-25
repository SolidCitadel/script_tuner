"""ScriptTuner CLI 진입점.

사용 예:
    uv run scripttuner download sbcsae
    uv run scripttuner download switchboard --force
    uv run scripttuner parse sbcsae SBC016
    uv run scripttuner clean sbcsae SBC016
    uv run scripttuner monologue sbcsae SBC016
    uv run scripttuner pairs sbcsae SBC016 --model deepseek/deepseek-v4-flash
    uv run scripttuner run switchboard --all --through monologue
    uv run scripttuner --help

`pairs` 서브커맨드는 `.env`에서 `OPENAI_API_KEY` / `OPENAI_BASE_URL` 자동 인식.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from scripttuner.corpora import REGISTRY
from scripttuner.llm.openai_compatible import OpenAICompatibleClient
from scripttuner.llm.openrouter import OpenRouterClient
from scripttuner.persistence.jsonl import read_jsonl, write_jsonl
from scripttuner.preprocessing.ir import Monologue, Pair, Utterance
from scripttuner.preprocessing.monologue import DEFAULT_MIN_TOKENS, build_monologues
from scripttuner.preprocessing.pairs import (
    DEFAULT_PROMPT_VERSION,
    DEFAULT_STYLE,
    convert_to_formal,
)
from scripttuner.preprocessing.stats import compute_stats

DEFAULT_DATASETS_DIR = Path("datasets")
DEFAULT_DATA_DIR = Path("data")
RUN_STAGES = ("parse", "clean", "monologue", "pairs", "stats")
"""Ordered pipeline stages the `run` orchestrator can execute."""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scripttuner", description="ScriptTuner CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dl = subparsers.add_parser("download", help="Download a dataset by name.")
    dl.add_argument("corpus", choices=sorted(REGISTRY), help="Corpus to download.")
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
        "parse", help="Parse a corpus stem into parsed Utterance JSONL."
    )
    pr.add_argument("corpus", choices=sorted(REGISTRY), help="Corpus name (selects adapter).")
    pr.add_argument("stem", help="File stem (e.g. SBC016 or sw2005).")
    pr.add_argument(
        "--datasets-dir",
        type=Path,
        default=DEFAULT_DATASETS_DIR,
        help=f"Source corpus directory base (default: {DEFAULT_DATASETS_DIR}).",
    )
    pr.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Base data directory (default: {DEFAULT_DATA_DIR}).",
    )

    cl = subparsers.add_parser(
        "clean", help="Apply marker cleaning to parsed Utterance JSONL."
    )
    cl.add_argument("corpus", choices=sorted(REGISTRY), help="Corpus name (selects cleaner).")
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
    mn.add_argument("corpus", choices=sorted(REGISTRY), help="Corpus name (resolves source dir).")
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
        "corpus", choices=sorted(REGISTRY), help="Corpus name (resolves source dir)."
    )
    pa.add_argument("stem", help="File stem (e.g. SBC016).")
    pa.add_argument(
        "--model",
        default=os.environ.get("LLM_MODEL"),
        help="LLM model slug (required unless LLM_MODEL env is set).",
    )
    pa.add_argument(
        "--model-alias",
        default=os.environ.get("LLM_MODEL_ALIAS"),
        help=(
            "Cache-key identity. Use a stable name shared across routing variants "
            "of the same weights (e.g. omit `:free`/`:nitro` suffixes) so the "
            "cache hits across routes. Defaults to LLM_MODEL_ALIAS env, then to "
            "--model when unset (legacy behavior)."
        ),
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

    st = subparsers.add_parser(
        "stats", help="Compute aggregate statistics over Pair JSONL."
    )
    st.add_argument("corpus", choices=sorted(REGISTRY), help="Corpus name (resolves source dir).")
    st.add_argument("stem", help="File stem (e.g. SBC016).")
    st.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Base data directory (default: {DEFAULT_DATA_DIR}).",
    )
    st.add_argument(
        "--no-pos",
        action="store_true",
        help="Skip POS-based stats (lexical density, phrasal verbs).",
    )

    ag = subparsers.add_parser(
        "aggregate",
        help="Concat per-stem Pair JSONLs into _all.jsonl + corpus-wide stats.",
    )
    ag.add_argument("corpus", choices=sorted(REGISTRY), help="Corpus name.")
    ag.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Base data directory (default: {DEFAULT_DATA_DIR}).",
    )
    ag.add_argument(
        "--no-pos",
        action="store_true",
        help="Skip POS-based stats in the aggregate.",
    )

    rn = subparsers.add_parser(
        "run",
        help="End-to-end pipeline: parse > clean > monologue > pairs > stats. "
        "Accepts one or more stems, or --all for every stem in the corpus dir. "
        "Use --through to stop early (e.g. before the LLM pairs stage).",
    )
    rn.add_argument("corpus", choices=sorted(REGISTRY), help="Corpus name.")
    rn.add_argument(
        "stems",
        nargs="*",
        help="One or more file stems (e.g. SBC016 SBC017). Omit when using --all.",
    )
    rn.add_argument(
        "--all",
        dest="all_stems",
        action="store_true",
        help="Process every stem under <datasets-dir>/<corpus>/ (per the adapter).",
    )
    rn.add_argument(
        "--through",
        choices=RUN_STAGES,
        default="stats",
        help="Run stages up to and including this one (default: stats). "
        "Use 'monologue' to stop before the LLM pairs stage.",
    )
    rn.add_argument(
        "--datasets-dir",
        type=Path,
        default=DEFAULT_DATASETS_DIR,
        help=f"Source corpus directory base (default: {DEFAULT_DATASETS_DIR}).",
    )
    rn.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Base data directory (default: {DEFAULT_DATA_DIR}).",
    )
    rn.add_argument(
        "--model",
        default=os.environ.get("LLM_MODEL"),
        help="LLM model slug (required unless LLM_MODEL env is set).",
    )
    rn.add_argument(
        "--model-alias",
        default=os.environ.get("LLM_MODEL_ALIAS"),
        help="Cache-key identity (env LLM_MODEL_ALIAS). See `pairs` subcommand for details.",
    )
    rn.add_argument("--min-tokens", type=int, default=DEFAULT_MIN_TOKENS)
    rn.add_argument("--limit", type=int, default=None, help="Limit pairs to first N monologues.")
    rn.add_argument("--no-progress", action="store_true")
    rn.add_argument("--no-cache", action="store_true")
    rn.add_argument("--max-retries", type=int, default=3)
    rn.add_argument("--no-pos", action="store_true", help="Skip POS-based stats.")

    return parser


def _source_name(corpus: str) -> str:
    return REGISTRY[corpus].source_name


def _run_download(args: argparse.Namespace) -> int:
    adapter = REGISTRY[args.corpus]
    dest_dir = args.dest / args.corpus
    files = adapter.download(dest_dir, force=args.force)
    print(f"OK: {len(files)} files in {dest_dir}")
    return 0


def _run_parse(args: argparse.Namespace) -> int:
    adapter = REGISTRY[args.corpus]
    corpus_dir = args.datasets_dir / args.corpus
    utterances = adapter.parse_stem(corpus_dir, args.stem)
    out_path = args.data_dir / "parsed" / adapter.source_name / f"{args.stem}.jsonl"
    n = write_jsonl(out_path, utterances)
    print(f"OK: wrote {n} utterances to {out_path}")
    return 0


def _run_clean(args: argparse.Namespace) -> int:
    adapter = REGISTRY[args.corpus]
    in_path = args.data_dir / "parsed" / adapter.source_name / f"{args.stem}.jsonl"
    utterances = read_jsonl(in_path, Utterance)
    cleaned = adapter.clean(utterances)
    out_path = args.data_dir / "cleaned" / adapter.source_name / f"{args.stem}.jsonl"
    n = write_jsonl(out_path, cleaned)
    print(f"OK: wrote {n} cleaned utterances to {out_path}")
    return 0


def _run_monologue(args: argparse.Namespace) -> int:
    adapter = REGISTRY[args.corpus]
    in_path = args.data_dir / "cleaned" / adapter.source_name / f"{args.stem}.jsonl"
    utterances = read_jsonl(in_path, Utterance)
    monologues = build_monologues(
        utterances,
        min_tokens=args.min_tokens,
        backchannel_words=adapter.backchannel_words,
    )
    out_path = args.data_dir / "monologues" / adapter.source_name / f"{args.stem}.jsonl"
    n = write_jsonl(out_path, monologues)
    print(f"OK: wrote {n} monologues to {out_path}")
    return 0


def _build_llm_client(
    model: str, max_retries: int
) -> OpenAICompatibleClient | OpenRouterClient:
    """Pick an LLM client based on the configured base URL.

    OpenRouter free-tier models need header-aware RPM-cap recovery that the SDK
    does not provide; we auto-route to `OpenRouterClient` when the configured
    base URL points at openrouter.ai. Any other endpoint (OpenAI, Together,
    Groq, local vLLM, etc.) keeps the plain `OpenAICompatibleClient` with the
    SDK's built-in retry semantics.
    """
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    if "openrouter.ai" in base_url:
        return OpenRouterClient(model=model)
    return OpenAICompatibleClient(model=model, max_retries=max_retries)


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

    client = _build_llm_client(args.model, args.max_retries)
    pairs = convert_to_formal(
        monologues,
        client=client,
        model=args.model,
        model_alias=args.model_alias,
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


def _run_stats(args: argparse.Namespace) -> int:
    source = _source_name(args.corpus)
    in_path = args.data_dir / "pairs" / source / f"{args.stem}.jsonl"
    pairs = read_jsonl(in_path, Pair)
    result = compute_stats(pairs, include_pos=not args.no_pos)
    out_path = args.data_dir / "stats" / source / f"{args.stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"OK: wrote stats for {result['n_pairs']} pairs to {out_path}")
    return 0


def _run_aggregate(args: argparse.Namespace) -> int:
    """Concat per-stem Pair JSONLs and compute corpus-wide stats.

    Writes two outputs:
      data/pairs/<SOURCE>/_all.jsonl     — all pairs concatenated
      data/stats/<SOURCE>/_aggregate.json — stats over _all.jsonl

    Files prefixed with ``_`` (incl. the previous _all.jsonl) are excluded from
    the input glob to keep aggregation idempotent.
    """
    source = _source_name(args.corpus)
    pairs_dir = args.data_dir / "pairs" / source
    if not pairs_dir.is_dir():
        print(f"error: pairs dir not found: {pairs_dir}", file=sys.stderr)
        return 1

    stem_paths = sorted(p for p in pairs_dir.glob("*.jsonl") if not p.name.startswith("_"))
    if not stem_paths:
        print(f"error: no per-stem pair jsonls under {pairs_dir}", file=sys.stderr)
        return 1

    pairs: list[Pair] = []
    for path in stem_paths:
        pairs.extend(read_jsonl(path, Pair))

    all_path = pairs_dir / "_all.jsonl"
    write_jsonl(all_path, pairs)
    print(f"OK: wrote {len(pairs)} pairs to {all_path}", file=sys.stderr)

    stats_dir = args.data_dir / "stats" / source
    stats_dir.mkdir(parents=True, exist_ok=True)
    result = compute_stats(pairs, include_pos=not args.no_pos)
    agg_path = stats_dir / "_aggregate.json"
    agg_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"OK: wrote aggregate stats for {result['n_pairs']} pairs to {agg_path}")
    return 0


def _resolve_run_stems(args: argparse.Namespace) -> tuple[list[str], int]:
    """Resolve the stem list for `run` from positional stems or --all.

    Returns (stems, rc). rc=0 on success; rc=2 on argument conflict; rc=1 when
    --all finds no stems in the corpus directory (per the adapter's enumerator).
    """
    if bool(args.stems) == bool(args.all_stems):
        print(
            "error: provide either one or more stems OR --all (not both, not neither).",
            file=sys.stderr,
        )
        return [], 2
    if args.all_stems:
        corpus_dir = args.datasets_dir / args.corpus
        stems = REGISTRY[args.corpus].enumerate_stems(corpus_dir)
        if not stems:
            print(f"error: no stems found under {corpus_dir}", file=sys.stderr)
            return [], 1
        return stems, 0
    return list(args.stems), 0


def _run_single_stem(stem: str, args: argparse.Namespace) -> int:
    """Process one stem through the pipeline up to and including args.through.

    Input presence is verified via the adapter's stem enumerator so the check is
    corpus-agnostic (CHAT = one .cha, Switchboard = a conversation's A/B files).
    """
    adapter = REGISTRY[args.corpus]
    source = adapter.source_name
    corpus_dir = args.datasets_dir / args.corpus
    if stem not in set(adapter.enumerate_stems(corpus_dir)):
        print(
            f"error: input not found for stem {stem!r} under {corpus_dir}\n"
            f"hint: run `scripttuner download {args.corpus}` first.",
            file=sys.stderr,
        )
        return 1

    base: dict[str, object] = {"corpus": args.corpus, "data_dir": args.data_dir}
    stage_kwargs: dict[str, dict[str, object]] = {
        "parse": {**base, "stem": stem, "datasets_dir": args.datasets_dir},
        "clean": {**base, "stem": stem},
        "monologue": {**base, "stem": stem, "min_tokens": args.min_tokens},
        "pairs": {
            **base,
            "stem": stem,
            "model": args.model,
            "model_alias": args.model_alias,
            "style": DEFAULT_STYLE,
            "prompt_version": DEFAULT_PROMPT_VERSION,
            "cache_dir": None,
            "no_cache": args.no_cache,
            "no_progress": args.no_progress,
            "max_retries": args.max_retries,
            "limit": args.limit,
        },
        "stats": {**base, "stem": stem, "no_pos": args.no_pos},
    }
    last = RUN_STAGES.index(args.through)
    for name in RUN_STAGES[: last + 1]:
        print(f"[run] {name} {args.corpus} {stem}", file=sys.stderr)
        rc = _COMMANDS[name](argparse.Namespace(**stage_kwargs[name]))
        if rc != 0:
            print(f"[run] {name} failed with rc={rc}", file=sys.stderr)
            return rc
    out_subdir = {"parse": "parsed", "clean": "cleaned", "monologue": "monologues",
                  "pairs": "pairs", "stats": "stats"}[args.through]
    ext = "json" if args.through == "stats" else "jsonl"
    print(
        f"[run] complete (through {args.through}): "
        f"data/{out_subdir}/{source}/{stem}.{ext}",
        file=sys.stderr,
    )
    return 0


def _run_run(args: argparse.Namespace) -> int:
    """End-to-end orchestrator. Multi-stem aware; per-stem failures are isolated."""
    stems, rc = _resolve_run_stems(args)
    if rc != 0:
        return rc

    succeeded: list[str] = []
    failed: list[str] = []
    for stem in stems:
        if _run_single_stem(stem, args) == 0:
            succeeded.append(stem)
        else:
            failed.append(stem)

    multi = len(stems) > 1 or args.all_stems
    if multi:
        summary = f"[run] summary: {len(succeeded)}/{len(stems)} succeeded"
        if failed:
            summary += f"; failed: {', '.join(failed)}"
        print(summary, file=sys.stderr)
    return 1 if failed else 0


_COMMANDS = {
    "download": _run_download,
    "parse": _run_parse,
    "clean": _run_clean,
    "monologue": _run_monologue,
    "pairs": _run_pairs,
    "stats": _run_stats,
    "aggregate": _run_aggregate,
    "run": _run_run,
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
