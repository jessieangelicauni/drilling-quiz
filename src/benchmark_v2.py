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
  2. Distractors: overgenerate and filter (PRD §9.86). EXAONE writes six
     candidate distractors; Qwen3-4B in reasoning mode, shown the key and
     all six, names every option that makes a correct sentence. The key
     must be among them; a candidate is kept only if it is NOT, and if its
     filled stem has a lower log-probability than the key's (BLiMP-style
     minimal pairs; 51/52 on the curated pairs, PRD §9.68). The three kept
     candidates with the highest log-probability -- the most plausible
     wrong answers -- become the distractors.
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

  5. Difficulty: the prior for Urnings is -logit of Qwen3-4B's probability
     of the key among the four options, from the log-probabilities the key
     check already computed (difficulty_llm.py, PRD §9.87) -- no extra model
     call; reported next to the morpheme heuristic it replaces.

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
HANGUL_OPTION = r"^[가-힣]{1,8}$"  # a distractor replaces one word, not a sentence
N_CANDIDATES, N_DISTRACTORS = 6, 3
HANGUL_SENTENCE = r"^[가-힣0-9 ,]{4,40}[.?!]$"


class Distractor(BaseModel):
    text: str = Field(pattern=HANGUL_OPTION)
    mistake: str = Field(min_length=10, max_length=160)


class Item(BaseModel):
    sentence: str = Field(pattern=HANGUL_SENTENCE)
    answer: str = Field(pattern=HANGUL_WORD)
    distractors: list[Distractor] = Field(min_length=N_CANDIDATES, max_length=N_CANDIDATES)


_PROMPT = """Write one multiple-choice Korean exercise for {level} learners whose first language is Indonesian.
Course: {context}. Topic: {topic}
Learning objective: {objective}
{example}
- "sentence": one short, simple, correct Korean sentence (at most 8 words) that uses the learning
  objective. Use one everyday word from the topic; keep the grammar at the learners' level.
- "answer": the word or word ending in the sentence that the learning objective is about, copied exactly.
- "distractors": six wrong forms that replace the answer and make the sentence ungrammatical:
  each changes the answer the way learners really get this learning objective wrong (a wrong
  particle, ending or form of the same word), each a DIFFERENT mistake, never another correct word
  and never more than one word; "mistake" says in one English sentence what the learner got wrong.
Write a new sentence, different from: {avoid}"""

_EXAMPLE = "Example of this objective: correct {correct} / learners write {wrong}\n"

_UNIQUE_PROMPT = """Fill the blank in this Korean sentence with each option.

{stem}

{options}

Which options give a grammatically correct, natural Korean sentence? Answer with the letters only, separated by commas."""
_LETTERS = "ABCDEFG"

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
    import gc

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
            raw = generator(prompt, max_new_tokens=1200, do_sample=True, temperature=0.6, top_p=0.95)
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
    gc.collect()  # EXAONE has to leave the GPU before Qwen3 loads
    torch.cuda.empty_cache()
    return items


def check_items(items: list[dict], cur: dict) -> None:
    import gc
    import re

    import torch

    from difficulty_llm import difficulty_prior
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

    def correct_options(stem: str, texts: list[str]) -> set[str]:
        letters_ = _LETTERS[:len(texts)]
        shown = "\n".join(f"{letters_[i]}. {t}" for i, t in enumerate(texts))
        found = set(re.findall(rf"\b([{letters_}])\b", reason(_UNIQUE_PROMPT.format(stem=stem, options=shown))))
        return {texts[letters_.index(x)] for x in found}

    for it in items:
        if not it["format_ok"]:
            continue
        fill = lambda text: it["stem"].replace("___", text, 1)
        it["candidates"] = it.pop("distractors")
        candidates = {d["text"]: d for d in it["candidates"] if d["text"] != it["answer"]}  # distinct
        key_score = solver.sentence_logprob(fill(it["answer"]))
        shown = random.Random(it["stem"]).sample([it["answer"], *candidates], len(candidates) + 1)
        it["judged_correct"] = sorted(correct_options(it["stem"], shown))
        it["key_ok"] = it["answer"] in it["judged_correct"]
        scored = [(solver.sentence_logprob(fill(t)), t) for t in candidates if t not in it["judged_correct"]]
        kept = sorted((s, t) for s, t in scored if s < key_score)[::-1][:N_DISTRACTORS]
        it["distractors"] = [candidates[t] for _, t in kept]
        if len(kept) == N_DISTRACTORS:
            it["difficulty_mu"], it["difficulty_sigma2"] = difficulty_prior(key_score, [s for s, _ in kept])
        it["enough_distractors"] = len(it["distractors"]) == N_DISTRACTORS
        it["predicted_objective"] = (classify(fill(it["answer"]), fill(it["distractors"][0]["text"]))
                                     if it["distractors"] else "none")
        it["objective_ok"] = it["predicted_objective"] == it["objective"]
        it["accepted"] = it["key_ok"] and it["enough_distractors"] and it["objective_ok"]

    del solver, model, tok
    gc.collect()
    torch.cuda.empty_cache()


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
    enc = SentenceTransformer(encoder, device="cpu")  # 278M parameters: fast enough, and leaves the GPU free
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


def difficulty_report(items: list[dict]) -> dict:
    import numpy as np

    from difficulty_heuristic import estimate_difficulty

    ok = [it for it in items if "difficulty_mu" in it]
    if len(ok) < 2:
        return {"items": len(ok)}
    mus = [it["difficulty_mu"] for it in ok]
    heur = [estimate_difficulty(it["sentence"])[0] for it in ok]
    by_objective = {}
    for it in ok:
        by_objective.setdefault(it["objective"], []).append(it["difficulty_mu"])
    return {"items": len(ok), "mean": float(np.mean(mus)), "sd": float(np.std(mus)),
            "corr_with_morpheme_heuristic": float(np.corrcoef(mus, heur)[0, 1]),
            "mean_by_objective": {k: float(np.mean(v)) for k, v in by_objective.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-objective", type=int, default=6)
    parser.add_argument("--curriculum", type=Path,
                        default=Path(__file__).resolve().parent.parent / "data" / "curriculum_ev_smart_factory.json")
    parser.add_argument("--context", default="")
    parser.add_argument("--encoders", default="intfloat/multilingual-e5-base",
                        help="comma-separated sentence encoders to compare, e.g. add Qwen/Qwen3-Embedding-0.6B")
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
            "enough_distractors": rate(sum(it["enough_distractors"] for it in ok), len(ok)),
            "objective_ok": rate(sum(it["objective_ok"] for it in ok), len(ok)),
            "accepted": rate(sum(it["accepted"] for it in ok), len(mine))}
        r = report["per_objective"][oid]
        print(f"{oid:16s} format {r['format_ok']['k']}/{r['format_ok']['n']}  key {r['key_ok']['k']}  "
              f"3 distractors {r['enough_distractors']['k']}  "
              f"objective {r['objective_ok']['k']}  accepted {r['accepted']['k']}/{r['accepted']['n']}")
    report["accepted"] = rate(sum(it.get("accepted", False) for it in items), len(items))
    report["embeddings"] = {e: embedding_agreement(items, e) for e in args.encoders.split(",")}
    report["difficulty"] = difficulty_report(items)
    report["items"] = items
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    a = report["accepted"]
    print(f"\nACCEPTED {a['k']}/{a['n']} = {a['rate']:.1%}  95% CI {a['ci95']}")
    for name, e in report["embeddings"].items():
        print(f"EMBEDDINGS {name}: nearest neighbour same mistake {e['nearest_neighbour_same_mistake']}  "
              f"cos same {e['mean_cos_same']}  different {e['mean_cos_different']}")
    d = report["difficulty"]
    if "mean" in d:
        print(f"DIFFICULTY {d['items']} items, mean {d['mean']:+.2f} sd {d['sd']:.2f}  "
              f"corr with morpheme heuristic {d['corr_with_morpheme_heuristic']:.2f}")
    print(f"written to {args.out}")
