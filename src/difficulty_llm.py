"""
Item difficulty prior from the key check's own scores (PRD §9.87, §9.91), to
replace the morpheme count of difficulty_heuristic.py as the (mu_hat,
sigma_hat2) that urnings_tracker.seed_item_from_encoder() takes.

The key check already scores the stem filled with the key and with each
distractor under Qwen3-4B. The log-odds that the model picks the key over
any distractor,
    key_logit = logprob(key) - logsumexp(logprob(distractors)),
is high for an item whose distractors the model finds clearly wrong and low
for one whose distractors it finds almost as natural as the key. No model
call of its own: it is a by-product of a check every item passes anyway.

Used directly, -key_logit saturates: Qwen3 is far surer than a learner, and
v2.3 had 9 of 11 accepted items at the -3 clip. The scale is the model's,
so only the ORDER is kept: key_logit is standardised against the item bank
(mean 0, SD 1 on the tracker's logit scale), and sigma_hat2 is wide, so the
item gets a small urn that real answers overwrite quickly. Once Urnings has
calibrated items from real learners, the KLUE-RoBERTa encoder of the paper,
trained on those calibrated difficulties, replaces it.
"""
from __future__ import annotations

import math

PRIOR_VARIANCE = 1.0  # the model's scale is not yet linked to the learners'
_MU_CLIP = 3.0


def key_logit(key_logprob: float, distractor_logprobs: list[float]) -> float:
    """Log-odds of the key against the distractors under the language model."""
    top = max(distractor_logprobs)
    return key_logprob - top - math.log(sum(math.exp(s - top) for s in distractor_logprobs))


def difficulty_priors(logits: list[float], bank: list[float] | None = None) -> list[tuple[float, float]]:
    """(mu_hat, sigma_hat2) for each item, from its key_logit standardised
    against the bank's key_logits (the new items themselves when the bank
    is empty)."""
    ref = bank if bank and len(bank) > 1 else logits
    mean = sum(ref) / len(ref)
    sd = math.sqrt(sum((x - mean) ** 2 for x in ref) / len(ref)) if len(ref) > 1 else 0.0
    sd = sd or 1.0
    return [(max(-_MU_CLIP, min(_MU_CLIP, -(x - mean) / sd)), PRIOR_VARIANCE) for x in logits]


if __name__ == "__main__":
    easy = key_logit(-10.0, [-18.0, -19.0, -20.0])  # distractors far less likely than the key
    hard = key_logit(-10.0, [-10.5, -11.0, -14.0])  # one distractor almost as likely
    mid = key_logit(-10.0, [-14.0, -15.0, -16.0])
    priors = difficulty_priors([easy, mid, hard])
    print("mu:", [f"{m:+.2f}" for m, _ in priors])
    assert priors[0][0] < priors[1][0] < priors[2][0]
    assert abs(sum(m for m, _ in priors)) < 1e-9

    from urnings_tracker import seed_item_from_encoder
    state = seed_item_from_encoder(*priors[2])
    assert 0 <= state.k <= state.n
    print("all smoke tests passed")
