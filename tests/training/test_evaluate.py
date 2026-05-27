from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripttuner import cli
from scripttuner.training.evaluate import run_evaluate


def _write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def test_evaluate_computes_core_metrics(tmp_path: Path) -> None:
    pred_path = tmp_path / "predictions.jsonl"
    _write_predictions(
        pred_path,
        [
            {
                "prediction": "well I mean you know it was fine",
                "reference": "<pause:long> well it was fine you know",
            },
            {"prediction": "", "reference": "okay sure"},
        ],
    )
    out = tmp_path / "metrics.json"

    metrics = run_evaluate(predictions_path=pred_path, output_path=out, include_pos=False)

    assert metrics["n"] == 2
    assert metrics["n_empty_predictions"] == 1
    assert "tokens" in metrics["prediction"]
    assert "fillers_per_item" in metrics["prediction"]
    assert "pause_long_per_item" in metrics["reference"]
    assert "length_ratio_pred_over_ref" in metrics
    # lexical_density is POS-based, so absent when include_pos=False.
    assert "lexical_density" not in metrics["prediction"]
    assert out.exists()


def test_evaluate_cli_no_pos(tmp_path: Path) -> None:
    pred_path = tmp_path / "predictions.jsonl"
    _write_predictions(pred_path, [{"prediction": "hi there", "reference": "hello"}])

    rc = cli.main(["evaluate", "--predictions", str(pred_path), "--no-pos"])

    assert rc == 0
    assert (tmp_path / "metrics.json").exists()
