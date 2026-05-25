from __future__ import annotations

import pytest

from scripttuner.preprocessing.ir import Utterance
from scripttuner.preprocessing.switchboard import cleaner


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # non-verbal events removed
        ("[silence]", ""),
        ("[noise] i um belong to a club", "i um belong to a club"),
        ("well [vocalized-noise] i think so", "well i think so"),
        # laughter-word keeps the word
        ("[laughter-yeah]", "yeah"),
        (
            "fat free and as [laughter-cholesterol] [laughter-free] all that",
            "fat free and as cholesterol free all that",
        ),
        # partial-word restart stubs removed
        ("to re[new]- to re[new]- renew they wanted", "to to renew they wanted"),
        ("h[ow]- how about you", "how about you"),
        # mispronunciation -> intended (after slash)
        ("that is just [bidness/business]", "that is just business"),
        ("they speak [what'n/wasn't] sure", "they speak wasn't sure"),
        # coinage braces stripped
        ("{alrighty} then we go", "alrighty then we go"),
        # aside markers removed, inner text kept
        ("<b_aside> hush now <e_aside> okay sorry", "hush now okay sorry"),
        # disambiguation index suffix stripped
        ("because_1 i think them_1 are ready", "because i think them are ready"),
        # legit ampersand preserved
        ("i worked at AT&T and went to A&M", "i worked at AT&T and went to A&M"),
        # combo: stub "fr[eeze]-" removed; the speaker's restarted "the" stays
        # (faithful disfluency on the spoken side)
        (
            "[silence] um [laughter] the fr[eeze]- the freeze just got them_1",
            "um the the freeze just got them",
        ),
    ],
)
def test_clean_text_rules(raw: str, expected: str) -> None:
    assert cleaner.clean_text(raw) == expected


def _utt(uid: str, text: str) -> Utterance:
    return Utterance(source="Switchboard", utterance_id=uid, speaker="A", text=text)


def test_clean_drops_empty_utterances_and_preserves_fields() -> None:
    utts = [
        _utt("sw1#A_0001", "[silence]"),
        _utt("sw1#A_0002", "well i think that is [bidness/business]"),
        _utt("sw1#A_0003", "[noise]"),
    ]
    out = cleaner.clean(utts)
    # the two marker-only lines are dropped; one real utterance remains
    assert len(out) == 1
    assert out[0].utterance_id == "sw1#A_0002"
    assert out[0].text == "well i think that is business"
    assert out[0].source == "Switchboard"
    assert out[0].speaker == "A"
