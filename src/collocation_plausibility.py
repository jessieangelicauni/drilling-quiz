"""
Noun-verb collocation plausibility gate (PRD audit, 2026-10-20 -- user
request: "you are llm engineer senior expert, investigate what can we try to
replace it" re: SKILL_SPECS's ~55 hand-curated verb_lookup/pairs tuples in
exaone_provider.py).

Those tables exist because free LLM generation picked semantically odd
pairings (exaone_provider.py's module docstring, Round 3: "커피 -> 차다"
= "kick coffee"). This module does NOT replace the curated tables
themselves -- those stay exactly as they are, zero regression risk. It
gates the OTHER path: `_pick_verb_lemma`'s LLM fallback, used only for a
slot that is not yet in `verb_lookup` (exaone_provider.py line ~943:
`spec["verb_lookup"].get(slot) or self._pick_verb_lemma(...)`). Before this
module, that fallback path had a grammatical-form check
(`_extract_verb_lemma`) but NO semantic-plausibility check at all -- exactly
the gap that produced "커피 -> 차다" in the first place.

Method: Korean Sentence-BERT (`jhgan/ko-sroberta-multitask`, a RoBERTa
variant fine-tuned for sentence embeddings) cosine similarity between the
bare noun and the bare verb. Empirically validated (not assumed) on the
user's GPU machine before being implemented here: 10 known-good pairs
(drawn from this project's own curated verb_lookup) vs 10 known-bad pairs
(including the actual documented "커피/차다" failure) --
GOOD mean=0.5575, BAD mean=0.3180, only 7/100 pairwise comparisons had a
bad pair scoring above a good pair. Could not be run in the cloud sandbox
this module was written in (huggingface.co is blocked by that session's
network policy -- a policy denial, not a transient failure). The
calibration script (investigate_sbert_plausibility.py) was removed in a
later codebase cleanup (PRD §9.60) once this was implemented and its
result recorded here -- recreate a similar script (10 good/10 bad pairs,
jhgan/ko-sroberta-multitask cosine similarity) if this needs
re-validating from scratch.

Threshold (_PLAUSIBILITY_THRESHOLD = 0.43) is a conservative split of the
calibration data above, not a value exactly between the means: chosen to
minimize accepting a bad pair (the actual problem this gate exists to
prevent) at the cost of occasionally rejecting a borderline-good one (cheap
-- the caller already has a retry/critic-loop/skip path for "no good
answer found", see _pick_verb_lemma). One known residual failure mode,
disclosed rather than hidden: the calibration data's single worst case was
("공원", "마시다") = 0.4689, almost indistinguishable from the weakest good
pair -- a real collocation ("공원에서 마시다", drink AT a park) that this
embedding signal cannot reliably separate from the genuinely nonsensical
("공원을 마시다", drink THE park itself), because the two nouns are
topically related (parks and drinking are associated activities) even
though the verb-argument relationship differs. This is the general,
documented weakness of cosine similarity for collocation plausibility
(topical relatedness vs. true argument-structure fit) -- not a bug to
"fix" with a better threshold, a property of the method, scoped and
accepted rather than papered over.

If sentence-transformers or the model is unavailable (e.g. no internet,
not installed), this gate fails CLOSED (rejects every pair) by default.
It used to fail open, but a validator that silently stops validating when
the network drops is the wrong default for a gate whose whole job is to
block bad items -- a rejected pair is cheap (the caller already has a
retry/critic/skip path), a nonsensical item reaching a learner is not.
Set KO_DRILL_ALLOW_PLAUSIBILITY_BYPASS=1 to opt back into fail-open, e.g.
in a sandbox without access to huggingface.co. Either way a one-time
warning is printed so the state is visible, not silent.
"""
from __future__ import annotations

import os

import numpy as np

_MODEL_NAME = "jhgan/ko-sroberta-multitask"
_PLAUSIBILITY_THRESHOLD = 0.43

_model = None
_load_failed = False


def _bypass_allowed() -> bool:
    return os.environ.get("KO_DRILL_ALLOW_PLAUSIBILITY_BYPASS") == "1"


def _get_model():
    global _model, _load_failed
    if _model is not None or _load_failed:
        return _model
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_NAME)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, see module docstring
        _load_failed = True
        if _bypass_allowed():
            print(f"[collocation_plausibility] could not load {_MODEL_NAME} ({exc!r}); "
                  "KO_DRILL_ALLOW_PLAUSIBILITY_BYPASS is set, so the gate is BYPASSED "
                  "(treating every pair as plausible).")
        else:
            print(f"[collocation_plausibility] could not load {_MODEL_NAME} ({exc!r}); "
                  "gate fails CLOSED (rejecting every pair) until this is fixed. "
                  "Set KO_DRILL_ALLOW_PLAUSIBILITY_BYPASS=1 to bypass instead.")
    return _model


def collocation_similarity(noun: str, verb: str) -> float | None:
    """Cosine similarity between `noun` and `verb`'s own embeddings.
    Returns None if the model is unavailable (see module docstring).
    """
    model = _get_model()
    if model is None:
        return None
    emb = model.encode([noun, verb])
    return float(np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1])))


def is_plausible_collocation(noun: str, verb: str) -> bool:
    """True if `verb` is a semantically plausible action/state for `noun`,
    per the calibrated threshold. If the model is unavailable: False
    (fail-closed), or True when KO_DRILL_ALLOW_PLAUSIBILITY_BYPASS=1 --
    see module docstring.
    """
    sim = collocation_similarity(noun, verb)
    if sim is None:
        return _bypass_allowed()
    return sim >= _PLAUSIBILITY_THRESHOLD


if __name__ == "__main__":
    # Smoke test only -- the real calibration run's results are recorded
    # in this module's docstring (PRD §9.57), not reproduced here.
    if _get_model() is None:
        os.environ.pop("KO_DRILL_ALLOW_PLAUSIBILITY_BYPASS", None)
        assert is_plausible_collocation("아무개", "아무거나") is False, (
            "must fail CLOSED (block) by default when the model can't be reached")
        os.environ["KO_DRILL_ALLOW_PLAUSIBILITY_BYPASS"] = "1"
        assert is_plausible_collocation("아무개", "아무거나") is True, (
            "must fail OPEN only when the bypass is explicitly opted into")
        print("model unavailable here: fail-closed default and opt-in bypass both confirmed. "
              "Real plausibility-separation behavior needs a machine with internet access -- "
              "see this module's docstring (PRD §9.57) for the calibration results instead.")
    else:
        assert is_plausible_collocation("커피", "마시다")
        assert not is_plausible_collocation("커피", "차다"), "the documented 'kick coffee' failure"
        print("model loaded: known-good pair accepted, documented bad pair rejected.")
