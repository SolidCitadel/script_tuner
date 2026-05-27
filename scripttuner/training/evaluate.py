"""예측 평가 — prediction vs reference의 구어성 피처 비교.

모듈 ⑤ stats의 피처 추출기(filler/pause/lexical density)를 재사용해, 예측이 reference만큼
"구어체다운지"를 같은 척도로 비교한다. 의존성 추가 없음.

표면·의미 유사도(BLEU/ROUGE/embedding)는 의도적으로 보류 — 정량 품질 메트릭은 본격 학습
단계 진입 시 도입 결정(docs/status.md `보류` 참조).

입력: `generate`가 만든 predictions.jsonl (prediction/reference 필드).
출력: metrics.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ⑤ stats의 per-text 피처 추출기 재사용(동일 패키지 내부 유틸).
from scripttuner.preprocessing.stats import (
    _PAUSE_LONG_RE,
    _PAUSE_SHORT_RE,
    _count_fillers,
    _distribution,
    _word_tokens,
)


def _text_features(texts: list[str]) -> dict[str, Any]:
    """텍스트 리스트의 구어성 피처 분포(길이/filler/pause)."""
    token_lists = [_word_tokens(t) for t in texts]
    return {
        "tokens": _distribution([len(ts) for ts in token_lists]),
        "fillers_per_item": _distribution([_count_fillers(ts) for ts in token_lists]),
        "pause_short_per_item": _distribution([len(_PAUSE_SHORT_RE.findall(t)) for t in texts]),
        "pause_long_per_item": _distribution([len(_PAUSE_LONG_RE.findall(t)) for t in texts]),
    }


def run_evaluate(
    *,
    predictions_path: Path,
    output_path: Path,
    include_pos: bool = True,
) -> dict[str, Any]:
    """predictions.jsonl을 읽어 구어성 메트릭을 계산하고 metrics.json을 쓴다."""

    rows: list[dict[str, Any]] = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"no predictions in {predictions_path}")

    preds = [(r.get("prediction") or "") for r in rows]
    refs = [(r.get("reference") or "") for r in rows]
    pred_lens = [len(_word_tokens(t)) for t in preds]
    ref_lens = [len(_word_tokens(t)) for t in refs]
    length_ratios = [
        (p / r) if r > 0 else 0.0 for p, r in zip(pred_lens, ref_lens, strict=True)
    ]

    prediction = _text_features(preds)
    reference = _text_features(refs)

    if include_pos:
        from scripttuner.preprocessing.stats import _load_spacy, _pos_stats

        nlp = _load_spacy()
        prediction["lexical_density"] = _distribution(
            [_pos_stats(t, nlp)["lexical_density"] for t in preds]
        )
        reference["lexical_density"] = _distribution(
            [_pos_stats(t, nlp)["lexical_density"] for t in refs]
        )

    metrics: dict[str, Any] = {
        "stage": "finetune_eval",
        "predictions": str(predictions_path),
        "n": len(rows),
        "n_empty_predictions": sum(1 for p in preds if not p.strip()),
        "length_ratio_pred_over_ref": _distribution(length_ratios),
        "prediction": prediction,
        "reference": reference,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metrics
