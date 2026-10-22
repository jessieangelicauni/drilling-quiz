"""
Benchmark CLI for item_foundry.MisconceptionItemGenerator, the project's
only item generator. This script does not reimplement generation -- it
drives the generator in a loop, with the real EXAONE chat model and the
real Qwen3 log-probability scorer, and reports the acceptance rate per
skill with a 95% CI.

Usage:
  python3 src/benchmark_v2.py --per-skill 6 [--skills particle_loc,subj_topic]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import item_foundry
from eval_harness import rate
from item_foundry import SKILLS, ExaoneChat, MisconceptionItemGenerator


def run(skills: list[str], per_skill: int, seed: int, scorer_model: str | None) -> dict:
    sentence_logprob = None
    if scorer_model:
        from solver_check import SolverCheckProvider

        sentence_logprob = SolverCheckProvider(scorer_model, four_bit="8B" in scorer_model).sentence_logprob

    gen = MisconceptionItemGenerator(ExaoneChat(), sentence_logprob, seed=seed)
    report = {"settings": {"skills": skills, "per_skill": per_skill, "seed": seed, "scorer": scorer_model},
              "per_skill": {}, "items": []}
    total_k = total_n = 0
    for skill_id in skills:
        k = 0
        for _ in range(per_skill):
            item_id = gen.next_new_grammar_item(skill_id)
            if item_id is not None:
                k += 1
                report["items"].append(gen.items[item_id])
        report["per_skill"][skill_id] = rate(k, per_skill)
        total_k += k
        total_n += per_skill
        r = report["per_skill"][skill_id]
        print(f"{skill_id:16s} accepted {r['k']}/{r['n']} = {r['rate']:.0%}  CI {r['ci95']}")
    report["accepted"] = rate(total_k, total_n)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-skill", type=int, default=6)
    parser.add_argument("--skills", default=",".join(SKILLS), help="comma-separated skill ids to run")
    parser.add_argument("--scorer", default="Qwen/Qwen3-4B",
                        help="log-probability scoring model, used only where a skill's logprob_gate "
                             "is on; pass an empty string to disable scoring entirely")
    parser.add_argument("--no-deterministic", action="store_true",
                        help="force every skill through the free-generation path instead of its "
                             "deterministic builder, for comparing against the deterministic "
                             "pipeline head-to-head -- not for shipping.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="benchmark_v2_results.json")
    args = parser.parse_args()

    if args.no_deterministic:
        item_foundry.DETERMINISTIC_SKILLS = frozenset()

    report = run(args.skills.split(","), args.per_skill, args.seed, args.scorer or None)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({**report, "items": [asdict(it) for it in report["items"]]}, f, ensure_ascii=False, indent=2)
    a = report["accepted"]
    print(f"\nACCEPTED {a['k']}/{a['n']} = {a['rate']:.1%}  95% CI {a['ci95']}")
    print(f"written to {args.out}")
