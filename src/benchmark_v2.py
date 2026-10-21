"""
Measures the model-only item pipeline (Versi 2, PRD §9.80) before it
replaces item_foundry's verification code. Every step is a model or a
library; nothing is parsed, diffed or labelled by our own rules.

  1. Generate: EXAONE-3.5-7.8B through Outlines with a JSON schema, so the
     output is valid by construction: a Hangul stem with exactly one ___,
     a one-word answer, and three distractors, each with the learner
     mistake it stands for, in one English sentence.
  2. Key check: under Qwen3-4B the stem filled with the answer must have a
     higher log-probability than with any distractor (BLiMP-style minimal
     pairs; 51/52 on the curated pairs, PRD §9.68).
  3. Objective check: Qwen3-4B in reasoning mode picks which learning
     objective of the curriculum the item tests (43/52 on the curated
     pairs, PRD §9.79); the item passes when it picks the requested one.
  4. Misconception embeddings: every distractor's mistake sentence is
     embedded with a multilingual sentence encoder. A useful embedding puts
     the same mistake in different items close together. Measured against
     the KAGAS edit of each distractor -- used here ONLY as the answer key
     of this evaluation, not in the pipeline: nearest-neighbour agreement
     and mean cosine similarity of same-mistake vs different-mistake pairs,
     within each objective.

The curriculum is data: --curriculum names a JSON file
{"context": "...", "level": "...", "objectives": {"id": "description", ...}};
without it, the 11 grammar skills of item_foundry.SKILLS are used.
--context sets the domain the sentences come from, e.g.
"smart factory EV battery manufacturing".

Usage:
  python3 src/benchmark_v2.py --per-objective 6 [--context "..."] [--curriculum file.json]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from pydantic import BaseModel, Field

from eval_harness import rate
from item_foundry import SKILLS, ExaoneChat

HANGUL_WORD = r"^[가-힣]{1,10}$"
HANGUL_STEM = r"^[가-힣0-9 ,.?!]{1,60}___[가-힣0-9 ,.?!]{0,60}$"


class Distractor(BaseModel):
    text: str = Field(pattern=HANGUL_WORD)
    mistake: str = Field(min_length=10, max_length=160)


class Item(BaseModel):
    stem: str = Field(pattern=HANGUL_STEM)
    answer: str = Field(pattern=HANGUL_WORD)
    distractors: list[Distractor] = Field(min_length=3, max_length=3)


_PROMPT = """Write one multiple-choice Korean exercise for {level} learners whose first language is Indonesian.
Context of the course: {context}
Learning objective: {objective}

- "stem": one short, natural Korean sentence about the context that needs the learning objective,
  with ___ where the answer goes.
- "answer": the correct word or form for ___.
- "distractors": three wrong fillers for ___ that learners really choose, each a DIFFERENT mistake
  about the learning objective; "mistake" says in one English sentence what the learner got wrong.
Write a new sentence, different from: {avoid}"""

_CLASSIFY_PROMPT = """A learner wrote the wrong Korean sentence below. Which learning objective is the mistake about?

Correct: {correct}
Wrong:   {wrong}

Learning objectives:
{options}

Answer with one letter only."""


def load_curriculum(path: Path | None, context: str) -> dict:
    if path is not None:
        cur = json.loads(path.read_text(encoding="utf-8"))
    else:
        cur = {"level": "beginner (TOPIK I)", "context": "everyday life",
               "objectives": {sid: s.description for sid, s in SKILLS.items()}}
    if context:
        cur["context"] = context
    return cur


def generate_items(cur: dict, per_objective: int, seed: int) -> list[dict]:
    import outlines
    import torch

    chat = ExaoneChat()
    model = outlines.from_transformers(chat.model, chat.tok)
    generator = outlines.Generator(model, Item)
    rng = random.Random(seed)
    items = []
    for oid, objective in cur["objectives"].items():
        seen: list[str] = []
        for _ in range(per_objective):
            content = _PROMPT.format(level=cur["level"], context=cur["context"], objective=objective,
                                     avoid="; ".join(seen[-6:]) or "(none yet)")
            prompt = chat.tok.apply_chat_template([{"role": "user", "content": content}],
                                                  tokenize=False, add_generation_prompt=True)
            torch.manual_seed(rng.randrange(2**31))
            raw = generator(prompt, max_new_tokens=600, do_sample=True, temperature=0.6, top_p=0.95)
            try:
                item = Item.model_validate_json(raw).model_dump()
            except ValueError:  # cut off by max_new_tokens
                items.append({"objective": oid, "raw": raw, "format_ok": False})
                continue
            seen.append(item["stem"])
            items.append({"objective": oid, **item, "format_ok": True})
    del generator, model, chat
    torch.cuda.empty_cache()
    return items


def check_items(items: list[dict], cur: dict) -> None:
    import re

    import torch

    from solver_check import SolverCheckProvider

    solver = SolverCheckProvider()
    tok, model = solver.tok, solver.model
    oids = list(cur["objectives"])
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(oids)]
    options = "\n".join(f"{letters[i]}. {cur['objectives'][o]}" for i, o in enumerate(oids))

    @torch.no_grad()
    def classify(correct: str, wrong: str) -> str:
        messages = [{"role": "user", "content": _CLASSIFY_PROMPT.format(correct=correct, wrong=wrong,
                                                                        options=options)}]
        enc = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=True,
                                      return_tensors="pt", return_dict=True).to("cuda")
        out = model.generate(**enc, max_new_tokens=2048, do_sample=False)
        answer = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).split("</think>")[-1]
        found = re.findall(rf"\b([{letters}])\b", answer)
        return oids[letters.index(found[-1])] if found else "none"

    for it in items:
        if not it["format_ok"]:
            continue
        fill = lambda text: it["stem"].replace("___", text, 1)
        texts = [d["text"] for d in it["distractors"]]
        key_score = solver.sentence_logprob(fill(it["answer"]))
        it["distinct_options"] = len({it["answer"], *texts}) == 4
        it["key_ok"] = it["distinct_options"] and all(solver.sentence_logprob(fill(t)) < key_score for t in texts)
        it["predicted_objective"] = classify(fill(it["answer"]), fill(texts[0]))
        it["objective_ok"] = it["predicted_objective"] == it["objective"]
        it["accepted"] = it["key_ok"] and it["objective_ok"]


def embedding_agreement(items: list[dict], encoder: str) -> dict:
    """Nearest-neighbour agreement of mistake embeddings with the KAGAS edit,
    within each objective, over distractors whose edit recurs."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    from kagas_annotator import annotate  # evaluation key only

    rows = []
    for it in items:
        if not it.get("format_ok"):
            continue
        correct = it["stem"].replace("___", it["answer"], 1)
        for d in it["distractors"]:
            edits = annotate(correct, it["stem"].replace("___", d["text"], 1))
            if len(edits) == 1:
                rows.append({"objective": it["objective"], "mistake": d["mistake"], "code": edits[0].code})
    enc = SentenceTransformer(encoder)
    vecs = enc.encode([f"query: {r['mistake']}" for r in rows], normalize_embeddings=True)
    hits = n = 0
    same, diff = [], []
    for i, r in enumerate(rows):
        peers = [j for j, s in enumerate(rows) if j != i and s["objective"] == r["objective"]]
        if not peers or not any(rows[j]["code"] == r["code"] for j in peers):
            continue  # its mistake occurs once: nothing to find
        sims = {j: float(vecs[i] @ vecs[j]) for j in peers}
        n += 1
        hits += rows[max(sims, key=sims.get)]["code"] == r["code"]
        for j, s in sims.items():
            (same if rows[j]["code"] == r["code"] else diff).append(s)
    return {"distractors": len(rows), "nearest_neighbour_same_mistake": rate(hits, n),
            "mean_cos_same": float(np.mean(same)) if same else None,
            "mean_cos_different": float(np.mean(diff)) if diff else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-objective", type=int, default=6)
    parser.add_argument("--curriculum", type=Path)
    parser.add_argument("--context", default="")
    parser.add_argument("--encoder", default="intfloat/multilingual-e5-base")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="benchmark_v2_results.json")
    args = parser.parse_args()

    cur = load_curriculum(args.curriculum, args.context)
    items = generate_items(cur, args.per_objective, args.seed)
    check_items(items, cur)
    report = {"settings": {k: str(v) for k, v in vars(args).items()}, "per_objective": {}}
    for oid in cur["objectives"]:
        mine = [it for it in items if it["objective"] == oid]
        ok = [it for it in mine if it["format_ok"]]
        report["per_objective"][oid] = {
            "format_ok": rate(len(ok), len(mine)),
            "key_ok": rate(sum(it["key_ok"] for it in ok), len(ok)),
            "objective_ok": rate(sum(it["objective_ok"] for it in ok), len(ok)),
            "accepted": rate(sum(it["accepted"] for it in ok), len(mine))}
        r = report["per_objective"][oid]
        print(f"{oid:16s} format {r['format_ok']['k']}/{r['format_ok']['n']}  key {r['key_ok']['k']}  "
              f"objective {r['objective_ok']['k']}  accepted {r['accepted']['k']}/{r['accepted']['n']}")
    report["accepted"] = rate(sum(it.get("accepted", False) for it in items), len(items))
    report["embeddings"] = embedding_agreement(items, args.encoder)
    report["items"] = items
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    a, e = report["accepted"], report["embeddings"]
    print(f"\nACCEPTED {a['k']}/{a['n']} = {a['rate']:.1%}  95% CI {a['ci95']}")
    print(f"EMBEDDINGS nearest neighbour same mistake {e['nearest_neighbour_same_mistake']}  "
          f"cos same {e['mean_cos_same']}  different {e['mean_cos_different']}")
    print(f"written to {args.out}")
