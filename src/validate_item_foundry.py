"""
Offline validation of the item foundry (PRD §9.74). Needs a GPU:
EXAONE-3.5-7.8B (4-bit) generates, Qwen3-4B scores. The two fit together
on one 16GB GPU (7.3GB + ~8GB, PRD §9.53).

  1. Requests --per-skill items per skill.
  2. Reports, per skill, with Wilson 95% intervals (eval_harness.wilson):
     acceptance rate (an accepted item has passed every check in
     item_foundry.verify or the deterministic path's equivalent checks),
     revisions the critic needed, and why replies were rejected, plus the
     first rejected replies with their reasons, to diagnose a low
     acceptance rate. The file is rewritten after every skill, so a crash
     keeps the skills already done.
  3. Writes every accepted item, with its reasoning, to
     item_foundry_results.json, the sample instructors rate for key
     uniqueness, distractor plausibility and reasoning quality
     (docs/eval-protocol.md).

Usage: python3 src/validate_item_foundry.py [--per-skill 20] [--seed 0] [--skills particle_loc,tense_past]
"""
from __future__ import annotations

import argparse
import json
import random
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
        accepted, attempts = 0, args.per_skill
        for _ in range(attempts):
            item = gen.generate(skill_id)
            if item is None:
                continue
            accepted += 1
            accepted_items.append(asdict(item))
        stats = gen.stats.get(skill_id)
        report["per_skill"][skill_id] = {
            "accepted": rate(accepted, attempts),
            "revisions_used": stats.revisions_used if stats else 0,
            "rejection_reasons": stats.reasons if stats else {},
            "rejected_examples": stats.rejected if stats else [],
        }
        r = report["per_skill"][skill_id]
        print(f"{skill_id:16s} accepted {r['accepted']['k']}/{r['accepted']['n']}  "
              f"rejections {r['rejection_reasons']}")
        report["items"] = accepted_items
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    total = [sum(r["accepted"]["k"] for r in report["per_skill"].values()),
             sum(r["accepted"]["n"] for r in report["per_skill"].values())]
    report["overall_acceptance"] = rate(*total)
    report["items"] = accepted_items
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    o = report["overall_acceptance"]
    print(f"\nOVERALL accepted {o['k']}/{o['n']} = {o['rate']:.1%}  95% CI {o['ci95']}")
    print(f"written to {args.out}")
