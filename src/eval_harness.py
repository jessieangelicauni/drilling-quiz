"""
Statistics layer for the validation scripts (review 2026-10-21, fix plan
Prioritas 4). CPU-only, no model calls: it reads the result JSONs the GPU
scripts already write and reports what those headline rates do not.

  1. UNIQUE items. validate_novelty1.py / validate_log_likelihood.py draw
     "N=20/skill" with replacement from 3-7 curated slots or pairs, so the
     same stem is counted several times. Rates are recomputed per unique
     (skill, stem, correct form), each item counted once.
  2. Wilson 95% intervals on every rate (Wilson 1927), which behave at
     small n and at 0%/100%, where the normal approximation does not.
  3. Several seeds. Pass more than one result file of the same kind (one
     per seed) and items are pooled across them before deduplicating, and
     the per-seed rates are reported next to the pooled one.
  4. Solver position bias. From the recorded picks: how often the solver
     picked each label, and its accuracy by the position of the key. A
     solver that is right far more often when the key is "A" is reacting
     to position, not to the item (see solver_check.pick_best_rotated /
     pick_by_logprob for the debiased alternatives).
  5. A frozen dev/test split of the curated items, by a salted hash of
     (skill, stem) so it never depends on run order or seed. Tune prompts,
     pairs and thresholds on "dev" only; report paper numbers on "test"
     only. The salt is versioned -- changing it is a new split, and must
     be recorded as such (docs/eval-protocol.md).

Usage:
  python3 src/eval_harness.py                       # the committed result files
  python3 src/eval_harness.py --solver a.json b.json --loglik c.json d.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict

SPLIT_SALT = "ko-drill-eval-v1"
TEST_FRACTION = 0.3


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes out of n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": (k / n) if n else None, "ci95": [round(lo, 4), round(hi, 4)]}


def split_of(skill_id: str, stem: str, salt: str = SPLIT_SALT) -> str:
    """"dev" or "test", fixed per (skill, stem) for a given salt."""
    digest = hashlib.sha256(f"{salt}|{skill_id}|{stem}".encode("utf-8")).digest()
    return "test" if int.from_bytes(digest[:8], "big") / 2**64 < TEST_FRACTION else "dev"


def _unique_rate(items_by_skill: dict[str, list[tuple[tuple, bool]]]) -> dict:
    """items: (unique key, passed) pairs. An item seen several times counts
    once, as passed only if it passed MORE than half the times it was seen
    (ties count as failed -- the conservative side for a quality gate).
    """
    per_skill, all_k, all_n = {}, 0, 0
    splits = {"dev": [0, 0], "test": [0, 0]}
    for skill_id, items in sorted(items_by_skill.items()):
        seen: dict[tuple, list[bool]] = defaultdict(list)
        for key, passed in items:
            seen[key].append(passed)
        k = sum(sum(v) * 2 > len(v) for v in seen.values())
        n = len(seen)
        per_skill[skill_id] = {**rate(k, n), "draws": len(items)}
        all_k, all_n = all_k + k, all_n + n
        for key, v in seen.items():
            bucket = splits[split_of(skill_id, key[0])]
            bucket[0] += sum(v) * 2 > len(v)
            bucket[1] += 1
    return {"per_skill": per_skill, "overall_unique": rate(all_k, all_n),
            "by_split": {name: rate(k, n) for name, (k, n) in splits.items()}}


def summarize_solver(paths: list[str]) -> dict:
    """validate_novelty1.py output(s): check (c) key acceptance."""
    items_by_skill = defaultdict(list)
    per_seed = []
    picks, key_pos_total, key_pos_ok = Counter(), Counter(), Counter()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        seed_k = seed_n = 0
        for skill_id, block in data["per_skill"].items():
            for item in block.get("items", []):
                key = (item["stem"], item["correct_option_text"])
                ok = bool(item["solver_key_accepted"])
                items_by_skill[skill_id].append((key, ok))
                seed_k, seed_n = seed_k + ok, seed_n + 1
                picks[item["solver_picked"]] += 1
                key_pos_total[item["correct_answer"]] += 1
                key_pos_ok[item["correct_answer"]] += ok
        per_seed.append({"file": path, "seed": data.get("seed"), **rate(seed_k, seed_n)})
    out = _unique_rate(items_by_skill)
    out["per_seed_draws"] = per_seed
    out["position_bias"] = {
        "picked_label_counts": {str(k): v for k, v in sorted(picks.items(), key=lambda kv: str(kv[0]))},
        "accuracy_by_key_position": {label: rate(key_pos_ok[label], key_pos_total[label])
                                     for label in sorted(key_pos_total)},
    }
    return out


def summarize_loglik(paths: list[str]) -> dict:
    """validate_log_likelihood.py / validate_lm_plausibility.py output(s)."""
    items_by_skill = defaultdict(list)
    per_seed = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        seed_k = seed_n = 0
        for skill_id, block in data["per_skill"].items():
            for row in block.get("rows", []):
                key = (row["stem"], row["correct"], row["distractor"])
                items_by_skill[skill_id].append((key, bool(row["valid"])))
                seed_k, seed_n = seed_k + bool(row["valid"]), seed_n + 1
        per_seed.append({"file": path, "seed": data.get("seed"), **rate(seed_k, seed_n)})
    out = _unique_rate(items_by_skill)
    out["per_seed_rows"] = per_seed
    return out


def _print_block(title: str, block: dict) -> None:
    print(f"\n=== {title} ===")
    for skill_id, r in block["per_skill"].items():
        lo, hi = r["ci95"]
        print(f"  {skill_id:16s} {r['k']:3d}/{r['n']:<3d} unique  = {r['rate']:6.1%}  "
              f"95% CI [{lo:5.1%}, {hi:5.1%}]   ({r['draws']} draws)")
    o = block["overall_unique"]
    print(f"  {'OVERALL':16s} {o['k']:3d}/{o['n']:<3d} unique  = {o['rate']:6.1%}  "
          f"95% CI [{o['ci95'][0]:5.1%}, {o['ci95'][1]:5.1%}]")
    for name, r in block["by_split"].items():
        if r["n"]:
            print(f"  split={name:4s}        {r['k']:3d}/{r['n']:<3d} unique  = {r['rate']:6.1%}  "
                  f"95% CI [{r['ci95'][0]:5.1%}, {r['ci95'][1]:5.1%}]")


if __name__ == "__main__":
    # Built-in checks first, so a broken helper can't silently produce a report.
    lo, hi = wilson(49, 52)
    assert 0.83 < lo < 0.86 and 0.97 < hi < 0.99, (lo, hi)
    assert wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == 1.0
    assert wilson(0, 0) == (0.0, 1.0)
    assert split_of("particle_loc", "저는 학교___ 공부해요.") == split_of("particle_loc", "저는 학교___ 공부해요.")
    toy = _unique_rate({"s": [(("a",), True), (("a",), True), (("a",), False), (("b",), False)]})
    assert toy["per_skill"]["s"]["k"] == 1 and toy["per_skill"]["s"]["n"] == 2, "dedup by majority"
    tie = _unique_rate({"s": [(("a",), True), (("a",), False)]})
    assert tie["per_skill"]["s"]["k"] == 0, "a tie counts as failed"

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--solver", nargs="+", default=["1_validation_results.json"])
    parser.add_argument("--loglik", nargs="+", default=["log_likelihood_results.json"])
    parser.add_argument("--parse-score", nargs="+", default=["src/lm_plausibility_results.json"])
    parser.add_argument("--out", default="eval_report.json")
    args = parser.parse_args()

    report = {"split_salt": SPLIT_SALT, "test_fraction": TEST_FRACTION,
              "solver_key_acceptance": summarize_solver(args.solver),
              "loglik_correct_above_distractor": summarize_loglik(args.loglik),
              "kiwipiepy_parse_score": summarize_loglik(args.parse_score)}

    _print_block("check (c): solver picks the key (Qwen3-4B, letter prompt)", report["solver_key_acceptance"])
    pb = report["solver_key_acceptance"]["position_bias"]
    print(f"  picked labels: {pb['picked_label_counts']}")
    for label, r in pb["accuracy_by_key_position"].items():
        print(f"  key at {label}: {r['k']}/{r['n']} = {r['rate']:.1%}  95% CI [{r['ci95'][0]:.1%}, {r['ci95'][1]:.1%}]")
    _print_block("log-likelihood: correct form outscores distractor (Qwen3-4B)",
                 report["loglik_correct_above_distractor"])
    _print_block("kiwipiepy parse score: correct form outscores distractor", report["kiwipiepy_parse_score"])

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nwritten to {args.out}")
