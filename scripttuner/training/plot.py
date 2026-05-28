"""학습 로그 시각화 — log_history.json → PNG.

HuggingFace `Trainer.state.log_history`는 학습 step마다의 train loss 항목과
eval epoch마다의 eval loss 항목을 한 리스트에 섞어 담는다. 이 모듈은 그 둘을
분리해 step 축에 동시에 표시한다. matplotlib는 train 의존성 그룹에 들어있다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def plot_training_curves(
    *,
    log_history_path: Path,
    output_path: Path,
    title: str | None = None,
) -> dict[str, Any]:
    """log_history.json을 읽어 train/eval loss 곡선을 그리고 PNG로 저장한다.

    반환: {n_train_points, n_eval_points, output_path}.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    entries = json.loads(log_history_path.read_text(encoding="utf-8"))
    train_steps, train_losses, eval_steps, eval_losses = _split_log_history(entries)

    fig, ax = plt.subplots(figsize=(10, 6))
    if train_losses:
        ax.plot(
            train_steps,
            train_losses,
            label="train_loss",
            color="#1f77b4",
            alpha=0.5,
            linewidth=1,
        )
    if eval_losses:
        # HF Trainer는 validation set 평가 결과를 "eval_loss"로 키잉하지만, 보고서 관행상
        # 표시 라벨은 "val_loss"로 둔다.
        ax.plot(
            eval_steps,
            eval_losses,
            label="val_loss",
            color="#d62728",
            marker="o",
            linewidth=2,
        )
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(title or f"Training curves: {log_history_path.parent.name}")
    ax.grid(True, alpha=0.3)
    if train_losses or eval_losses:
        ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return {
        "n_train_points": len(train_losses),
        "n_eval_points": len(eval_losses),
        "output_path": str(output_path),
    }


def _split_log_history(
    entries: list[dict[str, Any]],
) -> tuple[list[int], list[float], list[int], list[float]]:
    """log_history 리스트를 (train_steps, train_losses, eval_steps, eval_losses)로 분리."""

    train_steps: list[int] = []
    train_losses: list[float] = []
    eval_steps: list[int] = []
    eval_losses: list[float] = []
    for entry in entries:
        step = entry.get("step")
        if step is None:
            continue
        if "eval_loss" in entry:
            eval_steps.append(int(step))
            eval_losses.append(float(entry["eval_loss"]))
        elif "loss" in entry:
            train_steps.append(int(step))
            train_losses.append(float(entry["loss"]))
    return train_steps, train_losses, eval_steps, eval_losses
