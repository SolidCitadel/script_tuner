from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripttuner.training.plot import _split_log_history, plot_training_curves


def _fake_log_history() -> list[dict[str, object]]:
    # HF Trainer 로그를 모사한 구조: train 항목은 'loss' 키, eval 항목은 'eval_loss' 키.
    # 맨 마지막 train summary 항목(step 없이)은 무시되어야 한다.
    return [
        {"loss": 1.5, "epoch": 0.1, "step": 10},
        {"loss": 1.2, "epoch": 0.5, "step": 50},
        {"eval_loss": 1.1, "epoch": 1.0, "step": 88},
        {"loss": 0.9, "epoch": 1.5, "step": 132},
        {"eval_loss": 0.8, "epoch": 2.0, "step": 176},
        # train 종료 summary — step이 없으면 곡선에 포함하지 않는다.
        {"train_runtime": 1000.0, "train_loss": 0.95, "epoch": 2.0},
    ]


def test_split_log_history_separates_train_and_eval() -> None:
    train_steps, train_losses, eval_steps, eval_losses = _split_log_history(_fake_log_history())

    assert train_steps == [10, 50, 132]
    assert train_losses == [1.5, 1.2, 0.9]
    assert eval_steps == [88, 176]
    assert eval_losses == [1.1, 0.8]


def test_plot_training_curves_writes_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")

    log_path = tmp_path / "log_history.json"
    log_path.write_text(json.dumps(_fake_log_history()), encoding="utf-8")
    out_path = tmp_path / "training_curves.png"

    summary = plot_training_curves(log_history_path=log_path, output_path=out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert summary["n_train_points"] == 3
    assert summary["n_eval_points"] == 2


def test_plot_training_curves_handles_empty_log(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")

    log_path = tmp_path / "log_history.json"
    log_path.write_text("[]", encoding="utf-8")
    out_path = tmp_path / "empty.png"

    summary = plot_training_curves(log_history_path=log_path, output_path=out_path)

    assert out_path.exists()
    assert summary["n_train_points"] == 0
    assert summary["n_eval_points"] == 0
