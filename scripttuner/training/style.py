"""Style labels and control-token prompts for fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StyleSpec:
    """Operational definition for one generation style."""

    label: str
    control_token: str
    instruction: str
    description: str


STYLE_SPECS: dict[str, StyleSpec] = {
    "casual": StyleSpec(
        label="casual",
        control_token="<STYLE=casual>",
        instruction=(
            "Rewrite the input into natural casual spoken English. Preserve the "
            "meaning, use conversational phrasing, and avoid sounding like a "
            "formal written essay."
        ),
        description=(
            "Everyday spoken style with natural discourse markers, contractions, "
            "shorter sentence chunks, and mild disfluency when appropriate."
        ),
    ),
    "semi_formal": StyleSpec(
        label="semi_formal",
        control_token="<STYLE=semi_formal>",
        instruction=(
            "Rewrite the input into clear semi-formal spoken English suitable for "
            "an English speaking test. Preserve the meaning and make it sound "
            "spoken, but keep it organized and not overly casual."
        ),
        description=(
            "Speaking-test style with clear organization, natural spoken wording, "
            "restrained discourse markers, and fewer fillers than casual speech."
        ),
    ),
}


def get_style_spec(label: str) -> StyleSpec:
    """Return the style definition for a normalized label."""

    try:
        return STYLE_SPECS[label]
    except KeyError as e:
        supported = ", ".join(sorted(STYLE_SPECS))
        raise ValueError(f"Unsupported style {label!r}. Supported styles: {supported}") from e

