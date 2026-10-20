"""
Investigation script (not production code) for the "replace verb_lookup with
a learned plausibility filter" idea (PRD audit, 2026-10-20, user request: "you
are llm engineer senior expert, investigate what can we try to replace it").

Tests whether Korean Sentence-BERT embedding similarity can separate
plausible noun-verb collocations (학교 가다, "go to school") from implausible
ones (커피 차다, "kick coffee" -- the actual documented failure case from
exaone_provider.py's module docstring, Round 3) -- BEFORE committing to
implement it as a replacement for the ~55 hand-curated pairs in SKILL_SPECS.

Could NOT be run in the cloud sandbox this was written in: huggingface.co is
blocked by that session's network policy (confirmed via the agent-proxy
status endpoint, not a transient failure -- a policy denial). Run this on a
machine with real internet access; no GPU needed, CPU inference is fine for
a base-sized SBERT model.

Usage: python3 src/investigate_sbert_plausibility.py
(needs: pip install sentence-transformers)
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

# Good pairs: drawn from exaone_provider.py's existing curated verb_lookup
# (known-plausible collocations this project already relies on).
GOOD_PAIRS = [
    ("커피", "마시다"), ("밥", "먹다"), ("학교", "가다"), ("책", "읽다"),
    ("영화", "보다"), ("음악", "듣다"), ("편지", "쓰다"), ("사진", "찍다"),
    ("공원", "걷다"), ("도서관", "가다"),
]
# Bad pairs: includes the ACTUAL documented failure (커피->차다, module
# docstring Round 3) plus matched analogues (same nouns/verbs, swapped wrong).
BAD_PAIRS = [
    ("커피", "차다"), ("밥", "차다"), ("책", "마시다"), ("음악", "먹다"),
    ("영화", "차다"), ("사진", "듣다"), ("학교", "마시다"), ("편지", "차다"),
    ("공원", "마시다"), ("도서관", "먹다"),
]


def collocation_sim(model: SentenceTransformer, noun: str, verb: str) -> float:
    emb = model.encode([noun, verb])
    return float(np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1])))


def main() -> None:
    model = SentenceTransformer("jhgan/ko-sroberta-multitask")

    print("=== GOOD pairs (expect higher similarity) ===")
    good_scores = [collocation_sim(model, n, v) for n, v in GOOD_PAIRS]
    for (n, v), s in zip(GOOD_PAIRS, good_scores):
        print(f"{n}+{v}: {s:.4f}")

    print("\n=== BAD pairs (expect lower similarity) ===")
    bad_scores = [collocation_sim(model, n, v) for n, v in BAD_PAIRS]
    for (n, v), s in zip(BAD_PAIRS, bad_scores):
        print(f"{n}+{v}: {s:.4f}")

    print(f"\nGOOD mean={np.mean(good_scores):.4f} std={np.std(good_scores):.4f} "
          f"min={min(good_scores):.4f} max={max(good_scores):.4f}")
    print(f"BAD  mean={np.mean(bad_scores):.4f} std={np.std(bad_scores):.4f} "
          f"min={min(bad_scores):.4f} max={max(bad_scores):.4f}")

    overlap = sum(1 for b in bad_scores for g in good_scores if b > g)
    total = len(bad_scores) * len(good_scores)
    print(f"\noverlap check: {overlap}/{total} (bad,good) pairs where BAD > GOOD "
          f"(0 = perfect separation, {total} = total failure, no signal at all)")
    print("\nDecision rule for this investigation: if overlap is large (say >20%"
          "of pairs), this signal does not reliably separate plausible from "
          "implausible collocations and should NOT replace the curated "
          "verb_lookup tables -- same conclusion PRD §9.50 reached for 6 other "
          "local-model plausibility signals on a related task.")


if __name__ == "__main__":
    main()
