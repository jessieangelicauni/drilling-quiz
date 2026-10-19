"""
Batch validation run for Novelty 1 (Misconception-Conditioned Diagnostic
Distractors, PRD §3.2) -- the automatable half of Section V-B's evidence
plan. The other half (independent Korean instructors rating items and
assigning their own KAGAS label) needs human raters and is deliberately
out of scope here; this script only reports what can be measured without
one.

Requires a CUDA GPU (loads EXAONE-4.0-1.2B and Qwen3-4B) -- not runnable
in a GPU-less environment. Run on the machine noted in README's "Content
generation findings" (2026-10-19), then bring the printed table and the
JSON file this writes back for Section V-B.

Three numbers this produces, all from mechanisms that already exist and
are already exercised by exaone_provider.py's and solver_check.py's own
smoke tests -- this script just runs them at a larger, paper-citable N
and aggregates the result instead of asserting a smoke-test pass/fail:

  1. Per-skill generation pass rate -- does distractor_validator's
     morphological tag check (check (a), PRD §3.2) accept the item.
  2. Critic-loop recovery rate -- of the generation attempts that
     needed the critic step at all, how many it rescued
     (ExaoneGrammarProvider.critic_attempts/critic_recoveries).
  3. Solver round-trip key-acceptance rate -- check (c) from
     solver_check.py: a model blind to the key still picks it.

Deliberately NOT reported: solver_check.py's check (b), "is this
distractor ALSO a correct answer" -- its docstring records a confirmed
yes-bias on Qwen3-4B (tested on 6 known cases, wrong in both reasoning
and non-reasoning mode), so including it here would make every item
look falsely unambiguous. Leave it out rather than report a number known
to be wrong.

Also deliberately not claimed: that any number here "beats" the 53%
teacher-rated-quality figure in [12] (Bitew et al., distractor
generation via predictive prompting). That figure is a human-judged
quality score; an automated pass rate is a different construct, and
comparing them directly is a category error. The legitimate claim this
script supports is about *when* a bad distractor is caught (automatically,
at generation time) rather than *how good* average distractors are
rated to be -- see docs/PRD.md's novelty validation plan (§5) for how
this fits alongside the (separately tracked, human-rater-dependent) half
of the evidence.
"""
from __future__ import annotations

import argparse
import json

from exaone_provider import ExaoneGrammarProvider
from solver_check import SolverCheckProvider

SKILLS = ["particle_loc", "particle_obj", "honorific", "time_range", "tense_past",
          "irregular_conj", "adj_relativizer", "subj_topic"]


def run(n_per_skill: int, seed: int) -> dict:
    provider = ExaoneGrammarProvider(seed=seed)
    solver = SolverCheckProvider()

    per_skill = {}
    for skill_id in SKILLS:
        attempts = successes = key_accepted = solved = 0
        critic_attempts_before, critic_recoveries_before = provider.critic_attempts, provider.critic_recoveries

        for _ in range(n_per_skill):
            attempts += 1
            item_id = provider.next_new_grammar_item(skill_id)
            if item_id is None:
                continue
            successes += 1
            item = provider.items[item_id]
            result = solver.check(item.stem, item.blank_marker, item.options, item.answer)
            solved += 1
            key_accepted += int(result.key_accepted)

        per_skill[skill_id] = {
            "attempts": attempts,
            "generation_pass": successes,
            "critic_attempts": provider.critic_attempts - critic_attempts_before,
            "critic_recoveries": provider.critic_recoveries - critic_recoveries_before,
            "solver_checked": solved,
            "solver_key_accepted": key_accepted,
        }

    total_attempts = sum(v["attempts"] for v in per_skill.values())
    total_pass = sum(v["generation_pass"] for v in per_skill.values())
    total_critic_attempts = sum(v["critic_attempts"] for v in per_skill.values())
    total_critic_recoveries = sum(v["critic_recoveries"] for v in per_skill.values())
    total_solved = sum(v["solver_checked"] for v in per_skill.values())
    total_key_accepted = sum(v["solver_key_accepted"] for v in per_skill.values())

    return {
        "n_per_skill": n_per_skill,
        "seed": seed,
        "per_skill": per_skill,
        "overall_generation_pass_rate": total_pass / total_attempts,
        "overall_critic_recovery_rate": (
            total_critic_recoveries / total_critic_attempts if total_critic_attempts else None
        ),
        "overall_solver_key_accept_rate": total_key_accepted / total_solved if total_solved else None,
    }


def print_report(overall: dict) -> None:
    print(f"n_per_skill={overall['n_per_skill']} seed={overall['seed']}\n")
    header = f"{'skill':20s} {'gen pass':>10s} {'critic rec':>12s} {'solver key-ok':>14s}"
    print(header)
    for skill_id, v in overall["per_skill"].items():
        gen = f"{v['generation_pass']}/{v['attempts']}"
        critic = f"{v['critic_recoveries']}/{v['critic_attempts']}" if v["critic_attempts"] else "-"
        solver = f"{v['solver_key_accepted']}/{v['solver_checked']}" if v["solver_checked"] else "-"
        print(f"{skill_id:20s} {gen:>10s} {critic:>12s} {solver:>14s}")

    print(f"\nOVERALL generation pass rate:  {overall['overall_generation_pass_rate']:.1%}")
    if overall["overall_critic_recovery_rate"] is not None:
        print(f"OVERALL critic recovery rate:  {overall['overall_critic_recovery_rate']:.1%}")
    if overall["overall_solver_key_accept_rate"] is not None:
        print(f"OVERALL solver key-accept rate: {overall['overall_solver_key_accept_rate']:.1%}")
    print("\n(check (b), 'is this distractor also correct', intentionally omitted -- confirmed")
    print(" yes-bias on this solver model, see solver_check.py's module docstring.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-per-skill", type=int, default=20,
                         help="generation attempts per skill (default: 20, i.e. 160 total)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="novelty1_validation_results.json",
                         help="where to write the full results as JSON")
    args = parser.parse_args()

    overall = run(args.n_per_skill, args.seed)
    print_report(overall)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)
    print(f"\nFull results written to {args.out}")
