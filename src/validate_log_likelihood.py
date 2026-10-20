"""
At-scale validation of SolverCheckProvider.log_likelihood_check() (PRD
§9.35) -- the working replacement for check (b)'s broken yes/no judgment,
found by changing the TASK (scoring, not judging) rather than the wording,
after three judgment framings all failed (PRD §9.16, §9.31). Broader
coverage than distractor_validator.py's kiwipiepy-based lm_plausibility_gap
(PRD §9.34, 5/8 same-tag skills): this reuses Qwen3-4B's own deep
contextual understanding instead of kiwipiepy's simpler statistical
parser, and reaches 10 of 11 skills (all except subj_topic, excluded on
purpose -- see module docstring of solver_check.py).

Needs GPU (loads Qwen3-4B and EXAONE-4.0-1.2B, same as validate_novelty1.py).
"""
from __future__ import annotations

import argparse
import json

from exaone_provider import ExaoneGrammarProvider
from solver_check import LOGPROB_RELIABLE_SKILLS, SolverCheckProvider


def run(n_per_skill: int, seed: int) -> dict:
    provider = ExaoneGrammarProvider(seed=seed)
    solver = SolverCheckProvider()
    per_skill = {}

    for skill_id in sorted(LOGPROB_RELIABLE_SKILLS):
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
            result = solver.log_likelihood_check(skill_id, item.stem, item.blank_marker,
                                                   correct_text, distractor_text)
            rows.append({"stem": item.stem, "correct": correct_text, "distractor": distractor_text,
                         "valid": result.valid, "gap": result.gap, "reason": result.reason})

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
    parser.add_argument("--out", default="log_likelihood_results.json")
    args = parser.parse_args()

    overall = run(args.n_per_skill, args.seed)
    print(f"n_per_skill={overall['n_per_skill']} seed={overall['seed']} "
          f"(curated closed sets -- fewer unique stems than n_per_skill is expected)\n")
    print(f"{'skill':20s} {'valid':>10s}")
    for skill_id, v in overall["per_skill"].items():
        note = f"  ({v['note']})" if v["note"] else ""
        print(f"{skill_id:20s} {v['n_valid']}/{v['n']:>7d}{note}")
    print(f"\nOVERALL: {overall['overall_pass_rate']:.1%}")
    print("\n(scoped to LOGPROB_RELIABLE_SKILLS, PRD §9.35 -- subj_topic deliberately excluded,")
    print(" its exclusion independently confirms PRD §9.11's 'not a hard rule' framing)")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)
    print(f"\nFull results written to {args.out}")
