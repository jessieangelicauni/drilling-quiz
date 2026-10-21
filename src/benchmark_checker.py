"""
Which model should be the key-uniqueness checker (item_foundry.verify)?
Needs a GPU. Every candidate is scored on the same minimal pairs with the
same rule the gate uses: the correct sentence must get a higher total
log-probability than the wrong one (solver_check.sentence_logprob).

Pair sets:
  - curated: the 52 curated (stem, key, distractor) pairs of
    log_likelihood_results.json, filled into their stems. These are the
    distractor types the drill produces; Qwen3-4B already scores 51/52, so
    they mostly catch a candidate that is WORSE.
  - kolla: real learner sentences from KoLLA v2.0 (Song et al. 2026) that
    carry exactly one annotated error, paired with their correction. Real
    errors are harder than generated ones, so this set separates the
    candidates. Needs kolla_corpus_download/ (investigate_kolla_corpus.py
    downloads it). KoLLA annotates particles only, so it tests the
    particle skills.
  - codebook: the correct / wrong examples of data/misconception_codebook.json.

Candidates differ in family, size and pre-training (instruct vs base);
the checker must not be EXAONE, the generator, since a model scoring its
own outputs shares their blind spots. Gemma-3 and Kanana are gated or need
a recent transformers; a candidate that fails to load is reported and
skipped, so the run continues.

Output: checker_benchmark_results.json, and a table with accuracy per set
(Wilson 95% CI), mean margin, VRAM and seconds per pair.

Usage: python3 src/benchmark_checker.py [--models all|NAME,...] [--kolla-max 400]
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import re
import time
from pathlib import Path

import torch

from eval_harness import rate
from solver_check import SolverCheckProvider

ROOT = Path(__file__).resolve().parent.parent

# (Hugging Face id, load in 4-bit). bf16 when it fits beside the 4-bit
# generator on 16GB (PRD §9.53), 4-bit otherwise.
CANDIDATES = {
    "qwen3-4b": ("Qwen/Qwen3-4B", False),            # current checker
    "qwen3-4b-base": ("Qwen/Qwen3-4B-Base", False),
    "qwen3-8b": ("Qwen/Qwen3-8B", True),
    "qwen3-8b-base": ("Qwen/Qwen3-8B-Base", True),
    "kanana-1.5-8b-base": ("kakaocorp/kanana-1.5-8b-base", True),
    "gemma-3-12b-pt": ("google/gemma-3-12b-pt", True),
}

_A_LINE = re.compile(r"^A (-?\d+) (-?\d+)\|\|\|([^|]*)\|\|\|([^|]*)\|\|\|.*\|\|\|(\d+)\s*$")


def curated_pairs() -> list[dict]:
    data = json.loads((ROOT / "log_likelihood_results.json").read_text(encoding="utf-8"))
    return [{"skill": skill, "correct": row["stem"].replace("___", row["correct"], 1),
             "wrong": row["stem"].replace("___", row["distractor"], 1)}
            for skill, block in data["per_skill"].items() for row in block["rows"]]


def kolla_pairs(limit: int, seed: int) -> list[dict]:
    """Sentences with exactly one edit by annotator 0, as (correction, original)."""
    files = sorted((ROOT / "kolla_corpus_download").rglob("*.m2"))
    pairs, tokens, edits = [], [], []

    def flush():
        if tokens and len(edits) == 1:
            start, end, etype, corr = edits[0]
            fixed = tokens[:start] + corr.split() + tokens[end:]
            if fixed != tokens:
                pairs.append({"skill": f"kolla/{etype}", "correct": " ".join(fixed), "wrong": " ".join(tokens)})

    for path in files:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("S "):
                flush()
                tokens, edits = line[2:].split(), []
                continue
            m = _A_LINE.match(line)
            if not m or m.group(5) != "0" or m.group(3).strip() in ("noop", "UNK"):
                continue
            start, end = int(m.group(1)), int(m.group(2))
            corr = m.group(4).strip()
            if start < 0 or end > len(tokens) or end < start:
                continue
            edits.append((start, end, m.group(3).strip(), "" if corr == "-NONE-" else corr))
        flush()
        tokens, edits = [], []
    random.Random(seed).shuffle(pairs)
    return pairs[:limit]


def codebook_pairs() -> list[dict]:
    path = ROOT / "data" / "misconception_codebook.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [{"skill": skill, "correct": e["example_correct"], "wrong": e["example_wrong"]}
            for skill, entries in data["skills"].items() for e in entries]


def score(checker: SolverCheckProvider, pairs: list[dict]) -> dict:
    rows, start = [], time.time()
    for p in pairs:
        margin = checker.sentence_logprob(p["correct"]) - checker.sentence_logprob(p["wrong"])
        rows.append({**p, "margin": margin, "ok": margin > 0})
    k = sum(r["ok"] for r in rows)
    per_skill = {}
    for r in rows:
        s = per_skill.setdefault(r["skill"], [0, 0])
        s[0] += r["ok"]
        s[1] += 1
    return {"accuracy": rate(k, len(rows)),
            "mean_margin": sum(r["margin"] for r in rows) / len(rows) if rows else None,
            "per_skill": {s: rate(*kn) for s, kn in sorted(per_skill.items())},
            "seconds_per_pair": (time.time() - start) / len(rows) if rows else None,
            "failures": [r for r in rows if not r["ok"]][:30]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="all", help=f"comma-separated from {list(CANDIDATES)}")
    parser.add_argument("--kolla-max", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="checker_benchmark_results.json")
    args = parser.parse_args()

    names = list(CANDIDATES) if args.models == "all" else args.models.split(",")
    sets = {"curated": curated_pairs(), "kolla": kolla_pairs(args.kolla_max, args.seed),
            "codebook": codebook_pairs()}
    for name, pairs in sets.items():
        print(f"{name}: {len(pairs)} pairs" + (f", e.g. {pairs[0]['wrong']!r} -> {pairs[0]['correct']!r}"
                                                if pairs else ""))
    if not sets["kolla"]:
        print("  (no KoLLA pairs: run src/investigate_kolla_corpus.py once to download the corpus)")

    report = {"settings": vars(args), "n_pairs": {k: len(v) for k, v in sets.items()}, "models": {}}
    for name in names:
        hf_id, four_bit = CANDIDATES[name]
        print(f"\n=== {name} ({hf_id}, {'4-bit' if four_bit else 'bf16'}) ===")
        torch.cuda.reset_peak_memory_stats()
        try:
            checker = SolverCheckProvider(hf_id, load_in_4bit=four_bit)
        except Exception as exc:  # gated repo, no access, unsupported architecture
            print(f"  load failed, skipped: {exc!r}"[:300])
            report["models"][name] = {"hf_id": hf_id, "error": repr(exc)[:500]}
            continue
        entry = {"hf_id": hf_id, "4bit": four_bit}
        for set_name, pairs in sets.items():
            if pairs:
                entry[set_name] = score(checker, pairs)
                a = entry[set_name]["accuracy"]
                print(f"  {set_name:9s} {a['k']}/{a['n']} = {a['rate']:.1%}  CI {a['ci95']}  "
                      f"margin {entry[set_name]['mean_margin']:.2f}")
        entry["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
        print(f"  peak VRAM {entry['peak_vram_gb']:.1f} GB")
        report["models"][name] = entry
        del checker
        gc.collect()
        torch.cuda.empty_cache()
        with open(args.out, "w", encoding="utf-8") as f:  # after every model, so a crash keeps the rest
            json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'model':20s} " + " ".join(f"{s:>16s}" for s in sets))
    for name, entry in report["models"].items():
        cells = [f"{entry[s]['accuracy']['rate']:>15.1%} " if s in entry else f"{'-':>16s}" for s in sets]
        print(f"{name:20s} " + " ".join(cells) + ("  (load failed)" if "error" in entry else ""))
    print(f"written to {args.out}")
