"""
At-scale validation of distractor_validator.lm_plausibility_gap() (PRD
§9.34) -- the replacement for solver_check.py's check (b), which has a
confirmed, unfixable yes-bias (PRD §9.16) and two further framings tried
and also found unreliable (PRD §9.31). This is a structurally different
approach: kiwipiepy's own corpus-derived parse score, no LLM call at all.

Scoped deliberately to the 5 skills where distractor_validator.py's
LM_PLAUSIBILITY_RELIABLE_SKILLS was empirically confirmed reliable
(particle_loc, time_range, irregular_conj, adj_relativizer, verb_choice)
-- honorific, tense_past, and noun_counter are skipped here on purpose,
not omitted by oversight: correctness there depends on a social or
temporal cue external to local n-gram frequency (honorific register, a
time adverb, an arbitrary noun-class convention), which a frequency-based
parse score cannot see, and testing confirmed the signal points the WRONG
way for those skills more often than not. Running this script on them
would misrepresent, not strengthen, the evidence.

The plausibility check itself (distractor_validator.lm_plausibility_gap)
is pure CPU, no model call at all. GPU is still needed here only because
ExaoneGrammarProvider's constructor loads EXAONE unconditionally to
generate real items to check -- none of these 5 skills actually call the
LLM during generation (closed curated sets, see exaone_provider.py), so
this is solely a constructor cost, not a per-item one.
"""
from __future__ import annotations

import argparse
import json

from distractor_validator import LM_PLAUSIBILITY_RELIABLE_SKILLS, validate_lm_plausibility
from exaone_provider import ExaoneGrammarProvider


def run(n_per_skill: int, seed: int) -> dict:
    provider = ExaoneGrammarProvider(seed=seed)
    per_skill = {}

    for skill_id in sorted(LM_PLAUSIBILITY_RELIABLE_SKILLS):
        seen_stems = set()
        rows = []
        attempts = 0
        while len(rows) < n_per_skill and attempts < n_per_skill * 10:
            attempts += 1
            item_id = provider.next_new_grammar_item(skill_id)
            if item_id is None:
                continue
            item = provider.items[item_id]
            if item.stem in seen_stems:
                continue
            seen_stems.add(item.stem)

            correct_text = item.options[item.answer]
            distractor_text = item.options[item.primary_distractor_label]
            filled_correct = item.stem.replace(item.blank_marker, correct_text, 1)
            filled_distractor = item.stem.replace(item.blank_marker, distractor_text, 1)
            result = validate_lm_plausibility(skill_id, filled_correct, filled_distractor)
            rows.append({"stem": item.stem, "correct": correct_text, "distractor": distractor_text,
                         "valid": result.valid, "reason": result.reason})

        n = len(rows)
        n_valid = sum(r["valid"] for r in rows)
        per_skill[skill_id] = {"n": n, "n_valid": n_valid, "rows": rows,
                                "note": "unique curated stems may be fewer than n_per_skill" if n < n_per_skill else None}

    total = sum(v["n"] for v in per_skill.values())
    total_valid = sum(v["n_valid"] for v in per_skill.values())
    return {"n_per_skill": n_per_skill, "seed": seed, "per_skill": per_skill,
            "overall_pass_rate": total_valid / total if total else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-per-skill", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="lm_plausibility_results.json")
    args = parser.parse_args()

    overall = run(args.n_per_skill, args.seed)
    print(f"n_per_skill={overall['n_per_skill']} seed={overall['seed']} "
          f"(curated closed sets -- fewer unique stems than n_per_skill is expected, not an error)\n")
    print(f"{'skill':20s} {'valid':>10s}")
    for skill_id, v in overall["per_skill"].items():
        note = f"  ({v['note']})" if v["note"] else ""
        print(f"{skill_id:20s} {v['n_valid']}/{v['n']:>7d}{note}")
    print(f"\nOVERALL: {overall['overall_pass_rate']:.1%}")
    print("\n(scoped to the 5 skills confirmed reliable for this check, PRD §9.34 -- honorific/")
    print(" tense_past/noun_counter deliberately excluded, not silently passed or omitted by accident)")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)
    print(f"\nFull results written to {args.out}")
