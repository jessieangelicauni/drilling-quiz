"""
Item difficulty from LLM respondents (PRD §9.83), to replace the morpheme
count of difficulty_heuristic.py as the (mu_hat, sigma_hat2) prior that
urnings_tracker.seed_item_from_encoder() takes.

Language models of increasing size stand in for learners of increasing
ability (Liu et al., 2025, "Leveraging LLM respondents for item
evaluation", Brit. J. Educ. Tech.). Each one answers every item the way the
key check does: the stem filled with each option is scored by its
log-probability, and the softmax over the four options is the probability
p that this respondent picks the key. A Rasch model,
    logit p[m, i] = theta[m] - beta[i],
is then fitted to the respondents x items matrix by least squares
(identified by mean beta = 0): beta[i] is the item's difficulty and the
spread of its residuals across respondents its standard error. An item
only the large models answer gets a high beta.

The scale is the respondents', not the learners': until learner responses
exist, betas are standardised and sigma_hat2 includes LINK_VARIANCE for
that unknown link, so seed_item_from_encoder() gives the item a small urn
that real answers overwrite quickly. Once Urnings has calibrated items from
real learners, a linear link (or the KLUE-RoBERTa encoder of the paper,
trained on those calibrated difficulties) replaces it.

Usage (needs a GPU):
  python3 src/difficulty_llm.py      # scores the curated items of log_likelihood_results.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESPONDENTS = ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B")
LINK_VARIANCE = 0.5  # the respondents' scale is not yet linked to the learners'
_CLIP = 0.01


def respond(items: list[dict], model_name: str) -> np.ndarray:
    """Probability that `model_name` picks the key of each item
    ({"stem", "answer", "distractors": [str, ...]})."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to("cuda").eval()

    @torch.no_grad()
    def logprob(sentence: str) -> float:
        ids = tok(sentence, return_tensors="pt")["input_ids"].to("cuda")
        return -model(ids, labels=ids).loss.item() * (ids.shape[1] - 1)

    probs = []
    for it in items:
        scores = np.array([logprob(it["stem"].replace("___", o, 1)) for o in [it["answer"], *it["distractors"]]])
        e = np.exp(scores - scores.max())
        probs.append(e[0] / e.sum())
    del model
    torch.cuda.empty_cache()
    return np.array(probs)


def fit_rasch(p: np.ndarray) -> dict:
    """Least-squares Rasch fit of logit p[m, i] = theta[m] - beta[i]."""
    L = np.log(np.clip(p, _CLIP, 1 - _CLIP) / (1 - np.clip(p, _CLIP, 1 - _CLIP)))
    theta = L.mean(axis=1)
    beta = theta.mean() - L.mean(axis=0)
    resid = L - (theta[:, None] - beta[None, :])
    se2 = (resid ** 2).sum(axis=0) / max(len(theta) - 1, 1) / len(theta)
    return {"theta": theta, "beta": beta, "se2": se2}


def difficulty_priors(items: list[dict], respondents: tuple[str, ...] = RESPONDENTS) -> dict:
    """(mu_hat, sigma_hat2) per item on a standardised scale, plus the fit."""
    p = np.stack([respond(items, m) for m in respondents])
    fit = fit_rasch(p)
    sd = fit["beta"].std() or 1.0
    mu = fit["beta"] / sd
    sigma2 = fit["se2"] / sd ** 2 + LINK_VARIANCE
    return {"respondents": list(respondents), "p_key": p.tolist(),
            "respondent_accuracy": (p > 0.5).mean(axis=1).tolist(), "theta": fit["theta"].tolist(),
            "mu_hat": mu.tolist(), "sigma_hat2": sigma2.tolist()}


if __name__ == "__main__":
    from difficulty_heuristic import estimate_difficulty

    data = json.loads((Path(__file__).resolve().parent.parent / "log_likelihood_results.json").read_text(encoding="utf-8"))
    items = [{"skill": s, "stem": r["stem"], "answer": r["correct"], "distractors": [r["distractor"]]}
             for s, block in data["per_skill"].items() for r in block["rows"]]
    out = difficulty_priors(items)
    heur = [estimate_difficulty(it["stem"].replace("___", it["answer"], 1))[0] for it in items]
    print("respondent accuracy:", dict(zip(out["respondents"], np.round(out["respondent_accuracy"], 2))))
    print(f"corr(LLM difficulty, morpheme heuristic) = {np.corrcoef(out['mu_hat'], heur)[0, 1]:.2f}")
    for skill in data["per_skill"]:
        mus = [m for it, m in zip(items, out["mu_hat"]) if it["skill"] == skill]
        print(f"{skill:16s} mean difficulty {np.mean(mus):+.2f}")
