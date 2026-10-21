"""
Measures the three published components that would replace our own
verification heuristics (model-based pipeline proposal, PRD §9.79), on the
52 curated (correct, wrong) pairs in log_likelihood_results.json, before
any of them goes into the item foundry:

  1. GEC: the Korean GEC model released with KAGAS (Yoon et al., 2023),
     KoBART `Soyoung97/gec_kr`. A good item needs
       - key kept:    GEC(correct sentence) == correct sentence
       - wrong fixed: GEC(wrong sentence)   == correct sentence
  2. KAGAS: the official annotator (github.com/soyoung97/Standard_Korean_GEC,
     KAGAS/parallel_to_m2_korean.py), run on (wrong, correct). A good label
     is exactly one edit whose KAGAS type matches the skill
     (particle skills -> PARTICLE, ending skills -> ENDING, ...).
  3. Skill classifier: Qwen3-4B picks which of the 11 skills a question
     tests (closed-set choice, scored by the log-probability of each
     option letter, no free-text judgment). Also run on the items of the
     latest item_foundry_results.json, to see whether it catches the
     off-skill items found there.

Each component reports its accuracy with a Wilson 95% interval, and the
failures, to results file benchmark_components_results.json.

Setup for KAGAS (once): a Java JDK (KKma) and the Hunspell library
(apt install default-jdk libhunspell-dev), the repo, and its Python
dependencies -- `hunspell` (pyhunspell), not the pinned cyhunspell, which
no longer builds:
  git clone https://github.com/soyoung97/Standard_Korean_GEC ~/Standard_Korean_GEC
  pip install konlpy soynlp soylemma jamo tqdm spacy hunspell
Usage:
  python3 src/benchmark_components.py --kagas-dir ~/Standard_Korean_GEC/KAGAS
  python3 src/benchmark_components.py --skip kagas      # without KAGAS
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

from eval_harness import rate
from item_foundry import SKILLS

ROOT = Path(__file__).resolve().parent.parent

# Our skill categories in KAGAS's type names (KAGAS/pos_granularity.py).
KAGAS_TYPES = {"PART": {"PARTICLE"}, "END": {"ENDING"}, "CONJ": {"CONJUGATION"},
               "VERB": {"VERB"}, "NOUN": {"NOUN"}}


def curated_pairs() -> list[dict]:
    data = json.loads((ROOT / "log_likelihood_results.json").read_text(encoding="utf-8"))
    return [{"skill": skill, "stem": row["stem"], "key": row["correct"], "wrong_text": row["distractor"],
             "correct": row["stem"].replace("___", row["correct"], 1),
             "wrong": row["stem"].replace("___", row["distractor"], 1)}
            for skill, block in data["per_skill"].items() for row in block["rows"]]


def _same(a: str, b: str) -> bool:
    return re.sub(r"\s+", " ", a).strip(" .?!") == re.sub(r"\s+", " ", b).strip(" .?!")


# -- 1. GEC -----------------------------------------------------------------
def run_gec(pairs: list[dict]) -> dict:
    import torch
    from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast.from_pretrained("Soyoung97/gec_kr")
    model = BartForConditionalGeneration.from_pretrained("Soyoung97/gec_kr").to("cuda").eval()

    @torch.no_grad()
    def correct(text: str) -> str:  # the model card's own generation settings
        ids = [tok.bos_token_id] + tok.encode(text) + [tok.eos_token_id]
        out = model.generate(torch.tensor([ids], device="cuda"), max_length=128, eos_token_id=1,
                             num_beams=4, early_stopping=True, repetition_penalty=2.0)
        return tok.decode(out.squeeze().tolist(), skip_special_tokens=True)

    rows = []
    for p in pairs:
        fixed_key, fixed_wrong = correct(p["correct"]), correct(p["wrong"])
        rows.append({**p, "gec_correct": fixed_key, "gec_wrong": fixed_wrong,
                     "key_kept": _same(fixed_key, p["correct"]), "wrong_fixed": _same(fixed_wrong, p["correct"])})
    del model
    torch.cuda.empty_cache()
    k_kept, k_fixed = sum(r["key_kept"] for r in rows), sum(r["wrong_fixed"] for r in rows)
    return {"key_kept": rate(k_kept, len(rows)), "wrong_fixed": rate(k_fixed, len(rows)),
            "both": rate(sum(r["key_kept"] and r["wrong_fixed"] for r in rows), len(rows)),
            "failures": [{k: r[k] for k in ("skill", "correct", "wrong", "gec_correct", "gec_wrong")}
                         for r in rows if not (r["key_kept"] and r["wrong_fixed"])]}


# -- 2. KAGAS ---------------------------------------------------------------
def _tokenised(sentence: str) -> str:  # KAGAS assumes punctuation split off
    return re.sub(r"\s*([.?!,])", r" \1", sentence).strip()


def run_kagas(pairs: list[dict], kagas_dir: Path, python: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        orig, cor, out = Path(tmp, "orig.txt"), Path(tmp, "cor.txt"), Path(tmp, "out.m2")
        orig.write_text("\n".join(_tokenised(p["wrong"]) for p in pairs) + "\n", encoding="utf-8")
        cor.write_text("\n".join(_tokenised(p["correct"]) for p in pairs) + "\n", encoding="utf-8")
        proc = subprocess.run([python, "parallel_to_m2_korean.py", "-orig", str(orig), "-cor", str(cor),
                               "-out", str(out), "-noprint", "-hunspell", "./aff-dic"],
                              cwd=kagas_dir, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                              timeout=1800)
        if proc.returncode != 0 or not out.exists():
            return {"error": (proc.stderr or proc.stdout)[-2000:]}
        blocks = out.read_text(encoding="utf-8").strip().split("\n\n")
    rows = []
    for p, block in zip(pairs, blocks):
        edits = [line.split("|||") for line in block.splitlines() if line.startswith("A ")]
        types = [e[1] for e in edits if e[1] != "noop"]
        wanted = set().union(*(KAGAS_TYPES.get(c, {c}) for c in SKILLS[p["skill"]].categories))
        rows.append({"skill": p["skill"], "correct": p["correct"], "wrong": p["wrong"], "types": types,
                     "ok": len(types) == 1 and types[0] in wanted})
    return {"one_edit": rate(sum(len(r["types"]) == 1 for r in rows), len(rows)),
            "one_edit_in_skill_type": rate(sum(r["ok"] for r in rows), len(rows)),
            "n_blocks": len(blocks), "failures": [r for r in rows if not r["ok"]],
            "types_seen": sorted({t for r in rows for t in r["types"]})}


# -- 3. Skill classifier ----------------------------------------------------
_LETTERS = "ABCDEFGHIJK"
_CLASSIFY_PROMPT = """Which grammar point does this Korean exercise test?

Exercise: {stem}
Correct answer: {key}
Wrong answer: {wrong}

Grammar points:
{options}

Answer with one letter only."""


def run_classifier(pairs: list[dict], items: list[dict]) -> dict:
    import torch

    from solver_check import SolverCheckProvider

    solver = SolverCheckProvider()
    tok, model = solver.tok, solver.model
    skills = list(SKILLS)
    options = "\n".join(f"{_LETTERS[i]}. {SKILLS[s].description}" for i, s in enumerate(skills))
    letter_ids = [tok.encode(_LETTERS[i], add_special_tokens=False)[0] for i in range(len(skills))]

    @torch.no_grad()
    def classify(stem: str, key: str, wrong: str) -> str:
        messages = [{"role": "user", "content": _CLASSIFY_PROMPT.format(stem=stem, key=key, wrong=wrong,
                                                                        options=options)}]
        enc = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False,
                                      return_tensors="pt", return_dict=True).to("cuda")
        logits = model(**enc).logits[0, -1]
        return skills[int(torch.argmax(logits[letter_ids]))]

    rows = [{"skill": p["skill"], "stem": p["stem"], "predicted": classify(p["stem"], p["key"], p["wrong_text"])}
            for p in pairs]
    generated = []
    for it in items:
        key = it["options"][it["answer"]]
        wrong = next(v for k, v in it["options"].items() if k != it["answer"])
        generated.append({"skill": it["skill_id"], "stem": it["stem"], "key": key,
                          "predicted": classify(it["stem"], key, wrong)})
    return {"curated_accuracy": rate(sum(r["predicted"] == r["skill"] for r in rows), len(rows)),
            "curated_failures": [r for r in rows if r["predicted"] != r["skill"]],
            "generated_items": generated,
            "generated_kept": rate(sum(g["predicted"] == g["skill"] for g in generated), len(generated))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kagas-dir", type=Path, help="the KAGAS folder of Standard_Korean_GEC")
    parser.add_argument("--kagas-python", default="python3",
                        help="the interpreter with KAGAS's dependencies, if not this one")
    parser.add_argument("--skip", default="", help="comma-separated: gec,kagas,classifier")
    parser.add_argument("--out", default="benchmark_components_results.json")
    args = parser.parse_args()
    skip = set(filter(None, args.skip.split(",")))

    pairs = curated_pairs()
    report = {"n_pairs": len(pairs)}
    if "gec" not in skip:
        report["gec"] = run_gec(pairs)
        g = report["gec"]
        print(f"GEC        key kept {g['key_kept']['k']}/{g['key_kept']['n']}  "
              f"wrong fixed {g['wrong_fixed']['k']}/{g['wrong_fixed']['n']}  both {g['both']['rate']:.1%}")
    if "kagas" not in skip:
        if args.kagas_dir is None:
            parser.error("--kagas-dir is required unless --skip kagas")
        report["kagas"] = run_kagas(pairs, args.kagas_dir.expanduser(), args.kagas_python)
        k = report["kagas"]
        print(f"KAGAS      error: {k['error'][-500:]}" if "error" in k else
              f"KAGAS      one edit {k['one_edit']['k']}/{k['one_edit']['n']}, of the skill's type "
              f"{k['one_edit_in_skill_type']['k']}/"
              f"{k['one_edit_in_skill_type']['n']}  types seen {k['types_seen']}")
    if "classifier" not in skip:
        results = ROOT / "item_foundry_results.json"
        items = json.loads(results.read_text(encoding="utf-8")).get("items", []) if results.exists() else []
        report["classifier"] = run_classifier(pairs, items)
        c = report["classifier"]
        print(f"Classifier curated {c['curated_accuracy']['k']}/{c['curated_accuracy']['n']}  "
              f"generated items kept {c['generated_kept']['k']}/{c['generated_kept']['n']}")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"written to {args.out}")
