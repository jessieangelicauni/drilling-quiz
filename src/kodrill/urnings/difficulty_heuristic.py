from __future__ import annotations

from kiwipiepy import Kiwi

_KIWI = Kiwi()

_ADVANCED_TAGS = {"EP", "ETM", "EC", "NNB", "VA", "XSN"}

_TYPICAL_MORPHEME_COUNT = 8.0
_LENGTH_WEIGHT = 0.15
_ADVANCED_TAG_WEIGHT = 0.4
_MU_CLIP = 3.0
_FIXED_SIGMA2 = 1.5


def estimate_difficulty(sentence: str) -> tuple[float, float]:
    tokens = _KIWI.analyze(sentence)[0][0]
    n_morphemes = len(tokens)
    n_advanced = sum(1 for t in tokens if t.tag in _ADVANCED_TAGS)

    raw = (n_morphemes - _TYPICAL_MORPHEME_COUNT) * _LENGTH_WEIGHT + n_advanced * _ADVANCED_TAG_WEIGHT
    mu_hat = max(-_MU_CLIP, min(_MU_CLIP, raw))
    return mu_hat, _FIXED_SIGMA2


if __name__ == "__main__":
    simple = "저는 밥을 먹어요."
    honorific = "선생님께서 교실에 들어오셨습니다."
    complex_ = "이 문제가 학생들에 의해 풀릴 수 있었던 것으로 보인다."

    mu_simple, sig_simple = estimate_difficulty(simple)
    mu_honorific, sig_honorific = estimate_difficulty(honorific)
    mu_complex, sig_complex = estimate_difficulty(complex_)

    print(f"simple:    mu={mu_simple:.2f} sigma2={sig_simple}")
    print(f"honorific: mu={mu_honorific:.2f} sigma2={sig_honorific}")
    print(f"complex:   mu={mu_complex:.2f} sigma2={sig_complex}")

    assert mu_simple < mu_honorific < mu_complex
    assert sig_simple == sig_honorific == sig_complex == _FIXED_SIGMA2
    assert -3.0 <= mu_simple <= 3.0 and -3.0 <= mu_complex <= 3.0

    from .urnings_tracker import seed_item_from_encoder
    item_state = seed_item_from_encoder(mu_complex, sig_complex)
    assert 0 <= item_state.k <= item_state.n

    print("all smoke tests passed")
