"""
Parsing helpers for short LLM answers, kept free of torch/transformers so
they can be smoke-tested on any machine (solver_check.py itself needs a
GPU just to import).

Both replace substring checks in solver_check.py that could misfire
(review 2026-10-21):
  - the answer letter used to be the FIRST capital letter anywhere after
    "정답:", so an English word or an echoed "A:" option line in the reply
    could be read as the pick;
  - yes/no used to be '"예" in text and "아니" not in text', so "예시"
    ("example") or "예를 들어" ("for example") read as YES.
Both now look only at the first token of the answer, and return None
rather than guess when that token isn't a recognised answer.
"""
from __future__ import annotations

import re

_YES = ("예", "네")
_NO = ("아니요", "아니오", "아니")
_TOKEN_RE = re.compile(r"[^\s.,!?:;\"'()\[\]]+")


def _first_token(text: str) -> str | None:
    match = _TOKEN_RE.search(text)
    return match.group(0) if match else None


def parse_answer_letter(answer_part: str, labels) -> str | None:
    """The option label the reply starts with ("A", "B: 에서", "(C)" ...),
    or None if the first token isn't one of `labels`.
    """
    token = _first_token(answer_part)
    if token is None:
        return None
    letter = token[0]
    if letter not in labels:
        return None
    # "A" or "A:" style only -- not a word that merely starts with a capital
    # (e.g. "Actually"), which the old regex accepted as a pick of "A".
    return letter if len(token) == 1 else None


def parse_yes_no(answer_part: str) -> bool | None:
    """True for 예/네, False for 아니요/아니오/아니, None otherwise -- judged
    on the first token only, so 예시/예를 들어 no longer count as yes.
    """
    token = _first_token(answer_part)
    if token is None:
        return None
    if token in _YES:
        return True
    if any(token.startswith(no) for no in _NO):
        return False
    return None


if __name__ == "__main__":
    labels = ("A", "B", "C", "D")
    assert parse_answer_letter(" A", labels) == "A"
    assert parse_answer_letter("B: 에서", labels) == "B"
    assert parse_answer_letter("(C)", labels) == "C"
    assert parse_answer_letter("Actually the answer is D", labels) is None, "a word is not a pick"
    assert parse_answer_letter("E", labels) is None, "not an offered label"
    assert parse_answer_letter("", labels) is None

    assert parse_yes_no(" 예") is True
    assert parse_yes_no("네, 자연스러워요.") is True
    assert parse_yes_no("아니요.") is False
    assert parse_yes_no("아니오") is False
    assert parse_yes_no("예시: 저는 학교에서 공부해요.") is None, "예시 is not 예"
    assert parse_yes_no("예를 들어 ...") is None
    assert parse_yes_no("자연스러운 문장입니다") is None, "unrecognised -> None, not a guess"
    print("all llm_parsing smoke tests passed")
