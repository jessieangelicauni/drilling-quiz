"""
Item difficulty prior from the key check's own scores (PRD §9.87), to
replace the morpheme count of difficulty_heuristic.py as the (mu_hat,
sigma_hat2) that urnings_tracker.seed_item_from_encoder() takes.

The key check already scores the stem filled with the key and with each
distractor under Qwen3-4B. The softmax over those log-probabilities is the
probability p that the model picks the key; an item whose distractors the
model itself finds almost as plausible as the key is a hard item, so
    mu_hat = -logit(p)
on the logit scale of the tracker. No model call of its own: the prior is a
by-product of a check every item passes anyway.

The scale is the model's, not the learners', so sigma_hat2 is wide: the item
gets a small urn that real answers overwrite quickly. Once Urnings has
calibrated items from real learners, the KLUE-RoBERTa encoder of the paper,
trained on those calibrated difficulties, replaces it.
"""
from __future__ import annotations

import math

PRIOR_VARIANCE = 1.0  # the model's scale is not yet linked to the learners'
_MU_CLIP = 3.0


def difficulty_prior(key_logprob: float, distractor_logprobs: list[float]) -> tuple[float, float]:
    """(mu_hat, sigma_hat2) from the log-probabilities of the stem filled
    with the key and with each distractor."""
    top = max(key_logprob, *distractor_logprobs)
    weights = [math.exp(s - top) for s in [key_logprob, *distractor_logprobs]]
    p = min(max(weights[0] / sum(weights), 1e-6), 1 - 1e-6)
    mu = -math.log(p / (1 - p))
    return max(-_MU_CLIP, min(_MU_CLIP, mu)), PRIOR_VARIANCE


if __name__ == "__main__":
    easy = difficulty_prior(-10.0, [-18.0, -19.0, -20.0])  # distractors far less likely than the key
    hard = difficulty_prior(-10.0, [-10.5, -11.0, -14.0])  # one distractor almost as likely
    print(f"easy: mu={easy[0]:+.2f}  hard: mu={hard[0]:+.2f}")
    assert easy[0] < hard[0]
    assert -_MU_CLIP <= easy[0] <= _MU_CLIP

    from urnings_tracker import seed_item_from_encoder
    state = seed_item_from_encoder(*hard)
    assert 0 <= state.k <= state.n
    print("all smoke tests passed")
