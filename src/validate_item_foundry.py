"""
Novelty 1 offline validation of the item foundry (PRD §9.74). Needs a GPU:
EXAONE-3.5-7.8B (4-bit) generates, Qwen3-4B scores. The two fit together
on one 16GB GPU (7.3GB + ~8GB, PRD §9.53).

  1. Requests --per-skill items per skill: half untargeted, half asking for
     a misconception already seen in this skill's accepted items, drawn
     uniformly -- the personalised request the drill makes once a
     learner's dominant misconception is known.
  2. Reports, per skill, with Wilson 95% intervals (eval_harness.wilson):
       - acceptance rate of untargeted and of targeted requests (an
         accepted item has passed every check in item_foundry.verify);
       - revisions the critic needed, and why replies were rejected;
       - the distinct misconception codes the accepted items use, and how
         often each recurs (a code seen in one item only says little about
         a learner);
       - the first rejected replies with their reasons, to diagnose a low
         acceptance rate.
     The file is rewritten after every skill, so a crash keeps the skills
     already done.
  3. Writes every accepted item, with its option codes, to
     item_foundry_results.json, the sample instructors rate for key
     uniqueness, distractor plausibility and whether each code names a
     real learner mistake (docs/eval-protocol.md).

Usage: python3 src/validate_item_foundry.py [--per-skill 20] [--seed 0] [--skills particle_loc,tense_past]
                                         [--generator gemma-3-12b]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict

from eval_harness import rate
from item_foundry import SKILLS, ExaoneChat, MisconceptionItemGenerator
from solver_check import SolverCheckProvider

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-skill", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skills", default="all", help="comma-separated skill ids, for a quick check")
    parser.add_argument("--out", default="item_foundry_results.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    chat = ExaoneChat()
    solver = SolverCheckProvider()
    gen = MisconceptionItemGenerator(chat, solver.sentence_logprob, seed=args.seed)

    report = {"settings": vars(args), "per_skill": {}}
    accepted_items = []
    for skill_id in (list(SKILLS) if args.skills == "all" else args.skills.split(",")):
        counts = {"untargeted": [0, 0], "targeted": [0, 0]}
        codes_seen = Counter()
        for i in range(args.per_skill):
            target = rng.choice(sorted(codes_seen)) if (i % 2 and codes_seen) else None
            kind = "targeted" if target else "untargeted"
            counts[kind][1] += 1
            item = gen.generate(skill_id, target)
            if item is None:
                continue
            counts[kind][0] += 1
            codes_seen.update(c for c in item.option_codes.values() if c)
            accepted_items.append({**asdict(item), "requested_misconception": target})
        stats = gen.stats.get(skill_id)
        report["per_skill"][skill_id] = {
            "accepted_untargeted": rate(*counts["untargeted"]),
            "accepted_targeted": rate(*counts["targeted"]),
            "revisions_used": stats.revisions_used if stats else 0,
            "rejection_reasons": stats.reasons if stats else {},
            "codes": dict(codes_seen.most_common()),
            "codes_recurring": sum(n >= 2 for n in codes_seen.values()),
            "rejected_examples": stats.rejected if stats else [],
        }
        r = report["per_skill"][skill_id]
        print(f"{skill_id:16s} untargeted {r['accepted_untargeted']['k']}/{r['accepted_untargeted']['n']}  "
              f"targeted {r['accepted_targeted']['k']}/{r['accepted_targeted']['n']}  "
              f"codes {len(codes_seen)} ({r['codes_recurring']} recurring)  "
              f"rejections {r['rejection_reasons']}")
        report["items"] = accepted_items
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    total = [sum(r[k]["k"] for r in report["per_skill"].values() for k in ("accepted_untargeted", "accepted_targeted")),
             sum(r[k]["n"] for r in report["per_skill"].values() for k in ("accepted_untargeted", "accepted_targeted"))]
    report["overall_acceptance"] = rate(*total)
    report["items"] = accepted_items
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    o = report["overall_acceptance"]
    print(f"\nOVERALL accepted {o['k']}/{o['n']} = {o['rate']:.1%}  95% CI {o['ci95']}")
    print(f"written to {args.out}")
