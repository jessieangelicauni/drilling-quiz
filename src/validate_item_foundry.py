"""
Novelty 1 offline validation of the item foundry (PRD §9.69). Needs a GPU:
EXAONE-3.5-7.8B (4-bit) generates, Qwen3-4B scores. The two fit together
on one 16GB GPU (7.3GB + ~8GB, PRD §9.53).

  1. Builds the misconception codebook for every skill (or loads the
     committed one with --reuse-codebook) and saves it to
     data/misconception_codebook.json.
  2. Requests --per-skill items per skill: half untargeted, half asking for
     a specific codebook misconception, drawn uniformly -- the
     personalised request the drill makes once a learner's dominant
     misconception is known.
  3. Reports, per skill, with Wilson 95% intervals (eval_harness.wilson):
       - acceptance rate of untargeted and of targeted requests (an
         accepted item has passed every check in item_foundry.verify);
       - revisions the critic needed, and why replies were rejected;
       - how many distinct misconception codes the accepted items cover;
       - an independent solver check, reported, not gated: Qwen3-4B
         picking the key by letter under every rotation of the options
         (solver_check.pick_best_rotated). It judges, which this model does
         poorly (PRD §9.16), so it is a second view, not a gate.
  4. Writes every accepted item, with its option codes and the codebook
     entries, to item_foundry_results.json, the sample an instructor
     reviews to measure whether each distractor really realises the
     misconception it is labelled with (the human check of the labels;
     docs/eval-protocol.md).

Every codebook reply and every entry's verdict is saved in the output
(`codebook_log`), so an empty codebook can be diagnosed from the file.

Usage: python3 src/validate_item_foundry.py [--per-skill 20] [--seed 0] [--reuse-codebook]
       python3 src/validate_item_foundry.py --codebook-only   # step 1 only, quick check
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict
from datetime import date

from eval_harness import rate
from item_foundry import (CODEBOOK_PATH, SKILLS, ExaoneChat, MisconceptionItemGenerator, build_codebook,
                          load_codebook, save_codebook)
from solver_check import SolverCheckProvider

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-skill", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reuse-codebook", action="store_true")
    parser.add_argument("--codebook-only", action="store_true",
                        help="build and report the codebook, skip item generation and the solver")
    parser.add_argument("--out", default="item_foundry_results.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    chat = ExaoneChat()

    codebook_log: dict[str, list] = {}
    codebook = load_codebook() if args.reuse_codebook else {}
    if not any(codebook.values()):
        codebook = {}
        for skill_id in SKILLS:
            codebook_log[skill_id] = []
            codebook[skill_id] = build_codebook(skill_id, chat, log=codebook_log[skill_id])
    for skill_id, entries in codebook.items():
        print(f"  {skill_id:16s} {len(entries)} verified misconceptions: {[e.code for e in entries]}")
        for attempt in codebook_log.get(skill_id, []):
            verdicts = Counter(r["verdict"].split(":")[0] for r in attempt["rows"])
            print(f"      reply parsed={attempt['parsed']} rows={len(attempt['rows'])} {dict(verdicts)}")
            if not attempt["parsed"]:
                print(f"      raw reply starts: {attempt['reply'][:200]!r}")
    if codebook_log and any(codebook.values()):
        save_codebook(codebook, {"model": ExaoneChat.MODEL, "date": date.today().isoformat(),
                                 "seed": args.seed})
        print(f"codebook written to {CODEBOOK_PATH}")
    elif codebook_log:
        print("!! every codebook is empty -- not saved; see codebook_log in the output file")
    if args.codebook_only or not any(codebook.values()):
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"settings": vars(args), "codebook_log": codebook_log,
                       "codebook": {s: [asdict(e) for e in es] for s, es in codebook.items()}},
                      f, ensure_ascii=False, indent=2)
        print(f"written to {args.out}")
        raise SystemExit(0 if any(codebook.values()) else 1)

    solver = SolverCheckProvider()

    gen = MisconceptionItemGenerator(chat, solver.sentence_logprob, codebook, seed=args.seed)
    report = {"settings": vars(args), "per_skill": {}}
    accepted_items = []
    for skill_id in SKILLS:
        entries = codebook.get(skill_id, [])
        counts = {"untargeted": [0, 0], "targeted": [0, 0]}
        codes_seen, solver_ok, solver_n = set(), 0, 0
        for i in range(args.per_skill):
            target = rng.choice(entries).code if (i % 2 and entries) else None
            kind = "targeted" if target else "untargeted"
            counts[kind][1] += 1
            item = gen.generate(skill_id, target)
            if item is None:
                continue
            counts[kind][0] += 1
            codes_seen.update(c for c in item.option_codes.values() if c)
            picked, agreement = solver.pick_best_rotated(item.stem, item.options)
            solver_ok += picked == item.answer
            solver_n += 1
            accepted_items.append({**asdict(item), "requested_misconception": target,
                                   "solver_pick": picked, "solver_rotation_agreement": agreement})
        stats = gen.stats.get(skill_id)
        report["per_skill"][skill_id] = {
            "codebook_size": len(entries),
            "accepted_untargeted": rate(*counts["untargeted"]),
            "accepted_targeted": rate(*counts["targeted"]),
            "revisions_used": stats.revisions_used if stats else 0,
            "rejection_reasons": stats.reasons if stats else {},
            "distinct_codes_covered": len(codes_seen),
            "solver_rotated_key_pick": rate(solver_ok, solver_n),
        }
        r = report["per_skill"][skill_id]
        print(f"{skill_id:16s} untargeted {r['accepted_untargeted']['k']}/{r['accepted_untargeted']['n']}  "
              f"targeted {r['accepted_targeted']['k']}/{r['accepted_targeted']['n']}  "
              f"codes {len(codes_seen)}/{len(entries)}  solver {solver_ok}/{solver_n}  "
              f"rejections {r['rejection_reasons']}")

    total = [sum(r[k]["k"] for r in report["per_skill"].values() for k in ("accepted_untargeted", "accepted_targeted")),
             sum(r[k]["n"] for r in report["per_skill"].values() for k in ("accepted_untargeted", "accepted_targeted"))]
    report["overall_acceptance"] = rate(*total)
    report["items"] = accepted_items
    report["codebook"] = {s: [asdict(e) for e in entries] for s, entries in codebook.items()}
    report["codebook_log"] = codebook_log
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    o = report["overall_acceptance"]
    print(f"\nOVERALL accepted {o['k']}/{o['n']} = {o['rate']:.1%}  95% CI {o['ci95']}")
    print(f"written to {args.out}")
