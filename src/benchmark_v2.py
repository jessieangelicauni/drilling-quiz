"""
Measures the model-only item pipeline (Versi 2, PRD §9.80) before it
replaces item_foundry's verification code. Every step is a model or a
library; nothing is parsed, diffed or labelled by our own rules.

  1. Generate: EXAONE-3.5-7.8B through Outlines with a JSON schema, so the
     output is valid by construction: one short correct Korean sentence,
     the word in it that the objective is about (the answer), and three
     distractors, each with the learner mistake it stands for, in one
     English sentence. The stem is the sentence with the answer blanked;
     writing the whole sentence first and choosing the word after is the
     order an LLM writes well (asking it for a stem with ___ directly put
     the blank in the wrong place in most items of the first run, §9.82).
     One curated example of the objective, from log_likelihood_results.json,
     is shown in the prompt when there is one.
  2. Key check: under Qwen3-4B the stem filled with the answer must have a
     higher log-probability than with any distractor (BLiMP-style minimal
     pairs; 51/52 on the curated pairs, PRD §9.68), and Qwen3-4B in
     reasoning mode, shown the four options, must find only the answer
     correct -- a distractor that is merely a different correct word
     (빠르게 for 신중하게) fails the item.
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
{"context": "...", "level": "...", "objectives": {"id": "description", ...},
 "modules": {"module name": ["topic", ...], ...}} -- by default
data/curriculum_ev_smart_factory.json, the 14-week smart factory / EV
manufacturing course. The objectives are what the drill teaches (Korean
grammar); each item's sentence is set in one topic of the course, drawn at
random, so the sentences use the vocabulary learners meet in the course.

Usage:
  python3 src/benchmark_v2.py --per-objective 6 [--curriculum file.json]
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
HANGUL_SENTENCE = r"^[가-힣0-9 ,]{4,40}[.?!]$"


class Distractor(BaseModel):
    text: str = Field(pattern=HANGUL_WORD)
    mistake: str = Field(min_length=10, max_length=160)


class Item(BaseModel):
    sentence: str = Field(pattern=HANGUL_SENTENCE)
    answer: str = Field(pattern=HANGUL_WORD)
    distractors: list[Distractor] = Field(min_length=3, max_length=3)


_PROMPT = """Write one multiple-choice Korean exercise for {level} learners whose first language is Indonesian.
Course: {context}. Topic: {topic}
Learning objective: {objective}
{example}
- "sentence": one short, simple, correct Korean sentence (at most 8 words) that uses the learning
  objective. Use one everyday word from the topic; keep the grammar at the learners' level.
- "answer": the word or word ending in the sentence that the learning objective is about, copied exactly.
- "distractors": three wrong forms that replace the answer and make the sentence ungrammatical,
  each a DIFFERENT mistake learners really make with this learning objective (never another correct
  word); "mistake" says in one English sentence what the learner got wrong.
Write a new sentence, different from: {avoid}"""

_EXAMPLE = "Example of this objective: correct {correct} / learners write {wrong}\n"

_UNIQUE_PROMPT = """Fill the blank in this Korean sentence with each option.

{stem}

{options}

Which options give a grammatically correct, natural Korean sentence? Answer with the letters only, separated by commas."""

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
    cur.setdefault("modules", {"": [cur["context"]]})
    if context:
        cur["context"] = context
    return cur


def curated_examples() -> dict[str, tuple[str, str]]:
    """One (correct, wrong) sentence pair per objective from the curated pairs."""
    path = Path(__file__).resolve().parent.parent / "log_likelihood_results.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {oid: (row["stem"].replace("___", row["correct"], 1), row["stem"].replace("___", row["distractor"], 1))
            for oid, block in data["per_skill"].items() for row in block["rows"][:1]}


def generate_items(cur: dict, per_objective: int, seed: int) -> list[dict]:
    import outlines
    import torch

    chat = ExaoneChat()
    model = outlines.from_transformers(chat.model, chat.tok)
    generator = outlines.Generator(model, Item)
    rng = random.Random(seed)
    examples = curated_examples()
    items = []
    for oid, objective in cur["objectives"].items():
        seen: list[str] = []
        for _ in range(per_objective):
            module = rng.choice(sorted(cur["modules"]))
            topic = rng.choice(cur["modules"][module])
            example = _EXAMPLE.format(correct=examples[oid][0], wrong=examples[oid][1]) if oid in examples else ""
            content = _PROMPT.format(level=cur["level"], context=cur["context"], topic=f"{module}: {topic}" if module else topic,
                                     objective=objective, example=example, avoid="; ".join(seen[-6:]) or "(none yet)")
            prompt = chat.tok.apply_chat_template([{"role": "user", "content": content}],
                                                  tokenize=False, add_generation_prompt=True)
            torch.manual_seed(rng.randrange(2**31))
            raw = generator(prompt, max_new_tokens=600, do_sample=True, temperature=0.6, top_p=0.95)
            try:
                item = Item.model_validate_json(raw).model_dump()
            except ValueError:  # cut off by max_new_tokens
                items.append({"objective": oid, "topic": topic, "raw": raw, "format_ok": False})
                continue
            if item["sentence"].count(item["answer"]) != 1:  # the answer must be a part of the sentence
                items.append({"objective": oid, "topic": topic, **item, "format_ok": False})
                continue
            seen.append(item["sentence"])
            stem = item["sentence"].replace(item["answer"], "___", 1)
            items.append({"objective": oid, "topic": topic, "stem": stem, **item, "format_ok": True})
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
    def reason(content: str) -> str:  # Qwen3 thinks first; the reply after </think>
        enc = tok.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                      enable_thinking=True, return_tensors="pt", return_dict=True).to("cuda")
        out = model.generate(**enc, max_new_tokens=2048, do_sample=False)
        return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).split("</think>")[-1]

    def classify(correct: str, wrong: str) -> str:
        found = re.findall(rf"\b([{letters}])\b", reason(_CLASSIFY_PROMPT.format(correct=correct, wrong=wrong,
                                                                                options=options)))
        return oids[letters.index(found[-1])] if found else "none"

    def correct_options(stem: str, texts: list[str]) -> list[str]:
        shown = "\n".join(f"{'ABCD'[i]}. {t}" for i, t in enumerate(texts))
        found = set(re.findall(r"\b([ABCD])\b", reason(_UNIQUE_PROMPT.format(stem=stem, options=shown))))
        return [texts["ABCD".index(x)] for x in sorted(found)]

    for it in items:
        if not it["format_ok"]:
            continue
        fill = lambda text: it["stem"].replace("___", text, 1)
        texts = [d["text"] for d in it["distractors"]]
        key_score = solver.sentence_logprob(fill(it["answer"]))
        it["distinct_options"] = len({it["answer"], *texts}) == 4
        it["key_ok"] = it["distinct_options"] and all(solver.sentence_logprob(fill(t)) < key_score for t in texts)
        shown = random.Random(it["stem"]).sample([it["answer"], *texts], 4)
        it["judged_correct"] = correct_options(it["stem"], shown)
        it["unique_ok"] = it["judged_correct"] == [it["answer"]]
        it["predicted_objective"] = classify(fill(it["answer"]), fill(texts[0]))
        it["objective_ok"] = it["predicted_objective"] == it["objective"]
        it["accepted"] = it["key_ok"] and it["unique_ok"] and it["objective_ok"]


# How each encoder wants its input (from the model cards).
_ENCODER_PREFIX = {"intfloat/multilingual-e5-base": "query: ",
                   "Qwen/Qwen3-Embedding-0.6B": "Instruct: Name the kind of grammar mistake a learner made\nQuery: "}


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
    prefix = _ENCODER_PREFIX.get(encoder, "")
    vecs = enc.encode([prefix + r["mistake"] for r in rows], normalize_embeddings=True)
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
    parser.add_argument("--curriculum", type=Path,
                        default=Path(__file__).resolve().parent.parent / "data" / "curriculum_ev_smart_factory.json")
    parser.add_argument("--context", default="")
    parser.add_argument("--encoders", default="intfloat/multilingual-e5-base,Qwen/Qwen3-Embedding-0.6B",
                        help="comma-separated sentence encoders to compare")
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
            "unique_ok": rate(sum(it["unique_ok"] for it in ok), len(ok)),
            "objective_ok": rate(sum(it["objective_ok"] for it in ok), len(ok)),
            "accepted": rate(sum(it["accepted"] for it in ok), len(mine))}
        r = report["per_objective"][oid]
        print(f"{oid:16s} format {r['format_ok']['k']}/{r['format_ok']['n']}  key {r['key_ok']['k']}  "
              f"unique {r['unique_ok']['k']}  "
              f"objective {r['objective_ok']['k']}  accepted {r['accepted']['k']}/{r['accepted']['n']}")
    report["accepted"] = rate(sum(it.get("accepted", False) for it in items), len(items))
    report["embeddings"] = {e: embedding_agreement(items, e) for e in args.encoders.split(",")}
    report["items"] = items
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    a = report["accepted"]
    print(f"\nACCEPTED {a['k']}/{a['n']} = {a['rate']:.1%}  95% CI {a['ci95']}")
    for name, e in report["embeddings"].items():
        print(f"EMBEDDINGS {name}: nearest neighbour same mistake {e['nearest_neighbour_same_mistake']}  "
              f"cos same {e['mean_cos_same']}  different {e['mean_cos_different']}")
    print(f"written to {args.out}")
