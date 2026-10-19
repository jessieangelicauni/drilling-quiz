"""
Deterministic difficulty proxy (PRD §2.3, §3.1): stand-in for the
KLUE-RoBERTa regression head, which is genuinely blocked -- it needs a
calibrated difficulty-labeled Korean item dataset that does not exist, and
no amount of engineering effort fixes a missing dataset. Rather than leave
Novelty 1's prior-injection path unimplemented and untested, this uses
morpheme count and grammatical complexity (kiwipiepy, deterministic, no
training) as a weak but real difficulty signal with the SAME (mu_hat,
sigma_hat2) interface urnings_tracker.seed_item_from_encoder() already
expects -- swap in the trained encoder later with zero changes downstream.

This is intentionally a weaker proxy than Li et al. 2025's fine-tuned
encoder result (PRD §2.3 citation) -- a rule-based heuristic, not a
substitute claim of equivalent accuracy. sigma_hat2 is fixed and
deliberately high: this estimator should never look more confident than a
trained model would be, so new items get a weak prior that real response
data overwrites quickly (see urn-size mapping in seed_item_from_encoder).
"""
from __future__ import annotations

from kiwipiepy import Kiwi

_KIWI = Kiwi()

# Tags marking subordination, modification, honorific/tense prefinal
# morphemes, bound nouns -- structures that make a sentence harder to
# parse, found by direct inspection of kiwipiepy output across sentences
# of increasing complexity (see docs/PRD.md).
_ADVANCED_TAGS = {"EP", "ETM", "EC", "NNB", "VA", "XSN"}

_TYPICAL_MORPHEME_COUNT = 8.0   # a plain declarative sentence, e.g. "저는 학교에서 공부해요."
_LENGTH_WEIGHT = 0.15
_ADVANCED_TAG_WEIGHT = 0.4
_MU_CLIP = 3.0
_FIXED_SIGMA2 = 1.5   # moderate-high on purpose -- see module docstring


def estimate_difficulty(sentence: str) -> tuple[float, float]:
    """Returns (mu_hat, sigma_hat2) on the same logit scale as theta/beta
    elsewhere in this codebase. Monotonic in length/complexity, not
    calibrated to any real difficulty scale -- ordering is the only thing
    this should be trusted for until real data exists.
    """
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

    assert mu_simple < mu_honorific < mu_complex, "difficulty must increase with structural complexity"
    assert sig_simple == sig_honorific == sig_complex == _FIXED_SIGMA2, "uncertainty is fixed by design, not inferred"
    assert -3.0 <= mu_simple <= 3.0 and -3.0 <= mu_complex <= 3.0

    from urnings_tracker import seed_item_from_encoder
    item_state = seed_item_from_encoder(mu_complex, sig_complex)
    assert 0 <= item_state.k <= item_state.n, "must plug directly into the existing Novelty 1 seeding function"

    print("all smoke tests passed")
