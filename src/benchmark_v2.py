"""
Measures the model-only item pipeline (Versi 2, PRD §9.80) before it
replaces item_foundry's verification code. Every step is a model or a
library; nothing is parsed, diffed or labelled by our own rules.

  1. Generate: EXAONE-3.5-7.8B through Outlines with JSON schemas, in three
     constrained decodes, so the output is valid by construction. First one
     short correct Korean sentence. Then the answer -- a schema enum of the
     sentence's own words, so it is always one whole word of it (asked for
     in one schema, it glued several words together in 31/66 items, §9.91)
     -- then three distractors, each a whole word that replaces it. The stem
     is the sentence with that word blanked; writing the sentence first and
     choosing the word after is the order an LLM writes well (§9.82).
     Each distractor states its learner mistake (one English sentence)
     BEFORE its wrong form, so the model decides on a mistake and then
     writes it, instead of writing a word and explaining it after (§9.90).
     All curated examples of the objective, from log_likelihood_results.json,
     are shown in both prompts.
  2. Judge: one Qwen3-8B (NF4) reasoning call per item (§9.88, §9.92) names the options
     that make a correct sentence, the learning objective the item tests, and
     the wrong options a real learner could choose. A distractor fails when
     it is named correct, when it is not named plausible (v2.2 kept glued
     fragments such as 공장에서는다섯개 because they are wrong, §9.90), or
     when its filled stem has a higher log-probability than the key's (BLiMP-style minimal pairs;
     51/52 on the curated pairs, PRD §9.68).
  3. Repair: EXAONE replaces ONLY the failed distractors of an item whose
     key and objective passed, told why each one failed; the item is judged
     again. One round (§9.90). The item keeps exactly three distractors.
  4. Misconception embeddings: every distractor's mistake sentence is
     embedded with a multilingual sentence encoder. A useful embedding puts
     the same mistake in different items close together. Measured against
     the KAGAS edit of each distractor -- used here ONLY as the answer key
     of this evaluation, not in the pipeline: nearest-neighbour agreement
     and mean cosine similarity of same-mistake vs different-mistake pairs,
     within each objective.
  5. Difficulty: the prior for Urnings is the log-odds of the judge model choosing
     the key over the distractors, from the log-probabilities the key check
     already computed, standardised across the accepted items (difficulty_llm.py,
     PRD §9.91) -- no extra model call; reported next to the morpheme
     heuristic it replaces.

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
import functools
import json
import random
from pathlib import Path

from typing import Literal

from pydantic import BaseModel, Field, create_model

from eval_harness import rate
from item_foundry import SKILLS, ExaoneChat

HANGUL_OPTION = r"^[가-힣]{1,8}$"  # a distractor replaces one word, not a sentence
HANGUL_SENTENCE = r"^[가-힣0-9 ,]{4,40}[.?!]$"
N_DISTRACTORS = 3


class Distractor(BaseModel):
    mistake: str = Field(min_length=10, max_length=160)  # first: the mistake decides the form
    text: str = Field(pattern=HANGUL_OPTION)


class Sentence(BaseModel):
    sentence: str = Field(pattern=HANGUL_SENTENCE)


def _answer_schema(words: list[str]) -> type[BaseModel]:
    """The answer can only be one of the sentence's own words (v2.3 glued
    several words into one answer in 31/66 items, PRD §9.91). A schema of
    its own, so the one that changes with every sentence is tiny to compile."""
    return create_model("Answer", answer=(Literal[tuple(words)], ...))


@functools.cache  # the same class each time, so _Exaone compiles it once
def _replacements(n: int) -> type[BaseModel]:
    return create_model(f"Replacements{n}", distractors=(list[Distractor], Field(min_length=n, max_length=n)))


_SENTENCE_PROMPT = """Write one short, correct Korean sentence (at most 8 words) for {level} learners
(first language: Indonesian) that uses this grammar: {objective}
Setting: {context}; {topic}. Use one word of the setting.
{examples}
Do not reuse these sentences: {avoid}"""

_ANSWER_PROMPT = """Objective: {objective}
Korean sentence: {sentence}
answer: the word of the sentence the objective is about."""

_DISTRACTOR_PROMPT = """Korean fill-in-the-blank item for {level} learners. Objective: {objective}
{examples}
Stem: {stem}   Answer: {answer}
Write 3 wrong replacements for the answer, each from a different learner mistake:
- mistake: one English sentence naming the mistake.
- text: the single wrong word it produces (wrong particle, ending or form). It must make the sentence wrong."""

_REPAIR_PROMPT = """Korean fill-in-the-blank item for {level} learners. Objective: {objective}
{examples}
Stem: {stem}   Answer: {answer}
Keep: {kept}
Rejected: {rejected}
Write {n} replacement distractor(s), each from a learner mistake not listed above:
- mistake: one English sentence naming the mistake.
- text: the single wrong word it produces. It must make the sentence wrong."""

_EXAMPLES = "Examples (correct / learner error):\n{pairs}\n"

_JUDGE_PROMPT = """Korean fill-in-the-blank item:
{stem}

Options:
{options}

Learning objectives:
{objectives}

Answer in exactly three lines:
Correct: <letters of the options that give a grammatical, natural sentence>
Objective: <number of the objective this item tests>
Plausible: <letters of the wrong options a real learner might choose, not broken or random text; or none>"""
_LETTERS = "ABCD"


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


def curated_examples() -> dict[str, str]:
    """All curated (correct, wrong) sentence pairs of each objective, as prompt text."""
    path = Path(__file__).resolve().parent.parent / "log_likelihood_results.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {oid: _EXAMPLES.format(pairs="\n".join(
                f"  {row['stem'].replace('___', row['correct'], 1)} / {row['stem'].replace('___', row['distractor'], 1)}"
                for row in block["rows"]))
            for oid, block in data["per_skill"].items()}


class _Exaone:
    """EXAONE with Outlines, loaded for one phase and freed after, so the GPU
    holds one model at a time."""

    def __enter__(self):
        import outlines

        self.chat = ExaoneChat()
        from outlines.backends.outlines_core import OutlinesCoreBackend

        self.model = outlines.from_transformers(self.chat.model, self.chat.tok)
        # outlines.Generator(model, schema) converts the whole vocabulary and
        # compiles the schema on every call; Outlines 1.3 caches neither. The
        # vocabulary is converted once here and each schema compiled once.
        self.backend = OutlinesCoreBackend(self.model)
        self.generators: dict[str, object] = {}
        return self

    def __call__(self, content: str, schema: type[BaseModel], seed: int) -> str:
        import outlines
        import torch

        key = json.dumps(schema.model_json_schema(), sort_keys=True)
        if key not in self.generators:
            self.generators[key] = outlines.Generator(
                self.model, processor=self.backend.get_json_schema_logits_processor(key))
        prompt = self.chat.tok.apply_chat_template([{"role": "user", "content": content}],
                                                   tokenize=False, add_generation_prompt=True)
        torch.manual_seed(seed)
        return self.generators[key](prompt, max_new_tokens=800, do_sample=True, temperature=0.6, top_p=0.95)

    def __exit__(self, *exc):
        import gc

        import torch

        del self.model, self.chat, self.backend, self.generators
        gc.collect()  # EXAONE has to leave the GPU before Qwen3 loads
        torch.cuda.empty_cache()


def generate_items(cur: dict, per_objective: int, seed: int) -> list[dict]:
    """Three constrained decodes with the same model: the sentence, the answer
    (one of its words, by schema; a few tokens) and the distractors."""
    rng = random.Random(seed)
    examples = curated_examples()
    items = []
    with _Exaone() as exaone:
        for oid, objective in cur["objectives"].items():
            seen: list[str] = []
            for _ in range(per_objective):
                module = rng.choice(sorted(cur["modules"]))
                topic = rng.choice(cur["modules"][module])
                base = {"objective": oid, "topic": topic}
                try:
                    sentence = Sentence.model_validate_json(exaone(_SENTENCE_PROMPT.format(
                        level=cur["level"], context=cur["context"], topic=f"{module}: {topic}" if module else topic,
                        objective=objective, examples=examples.get(oid, ""),
                        avoid="; ".join(seen[-6:]) or "(none yet)"), Sentence, rng.randrange(2**31))).sentence
                except ValueError:  # cut off by max_new_tokens
                    items.append({**base, "format_ok": False})
                    continue
                words = sentence.rstrip(".?!").split()
                unique = [w for w in dict.fromkeys(words) if words.count(w) == 1]
                if not unique:
                    items.append({**base, "sentence": sentence, "format_ok": False})
                    continue
                answer = _answer_schema(unique).model_validate_json(exaone(
                    _ANSWER_PROMPT.format(objective=objective, sentence=sentence), _answer_schema(unique),
                    rng.randrange(2**31))).answer
                stem = " ".join("___" if w == answer else w for w in sentence[:-1].split()) + sentence[-1]
                raw = exaone(_DISTRACTOR_PROMPT.format(level=cur["level"], objective=objective, stem=stem,
                                                       answer=answer, examples=examples.get(oid, "")),
                             _replacements(N_DISTRACTORS), rng.randrange(2**31))
                try:
                    distractors = _replacements(N_DISTRACTORS).model_validate_json(raw).distractors
                except ValueError:
                    items.append({**base, "sentence": sentence, "answer": answer, "raw": raw, "format_ok": False})
                    continue
                seen.append(sentence)
                items.append({**base, "stem": stem, "sentence": sentence, "answer": answer,
                              "distractors": [d.model_dump() for d in distractors], "format_ok": True})
    return items


def check_items(items: list[dict], cur: dict, judge_model: str = "Qwen/Qwen3-8B") -> None:
    """Judges every given item: one Qwen3 reasoning call plus four log-probabilities."""
    import gc
    import re

    import torch

    from difficulty_llm import key_logit
    from solver_check import SolverCheckProvider

    solver = SolverCheckProvider(judge_model, four_bit="8B" in judge_model)
    tok, model = solver.tok, solver.model
    oids = list(cur["objectives"])
    objectives = "\n".join(f"{i + 1}. {cur['objectives'][o]}" for i, o in enumerate(oids))

    @torch.no_grad()
    def reason(content: str) -> str:  # Qwen3 thinks first; the reply after </think>
        enc = tok.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                      enable_thinking=True, return_tensors="pt", return_dict=True).to("cuda")
        out = model.generate(**enc, max_new_tokens=2048, do_sample=False)
        return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).split("</think>")[-1]

    def judge(stem: str, texts: list[str]) -> tuple[set[str], str, set[str]]:
        """One reasoning call: the correct options, the objective tested, and
        the wrong options a learner could plausibly choose. Only the answer
        lines are read, not the explanation around them."""
        letters = _LETTERS[:len(texts)]
        shown = "\n".join(f"{letters[i]}. {t}" for i, t in enumerate(texts))
        reply = reason(_JUDGE_PROMPT.format(stem=stem, options=shown, objectives=objectives))

        def options_on(label: str) -> set[str]:
            line = re.search(rf"{label}:([^\n]*)", reply)
            return {texts[letters.index(x)] for x in re.findall(rf"\b([{letters}])\b", line.group(1))} if line else set()

        objective_line = re.search(r"Objective:\D*(\d+)", reply)
        number = int(objective_line.group(1)) if objective_line else 0
        return (options_on("Correct"), oids[number - 1] if 1 <= number <= len(oids) else "none",
                options_on("Plausible"))

    for it in items:
        fill = lambda text: it["stem"].replace("___", text, 1)
        texts = [d["text"] for d in it["distractors"]]
        key_score = solver.sentence_logprob(fill(it["answer"]))
        judged, it["predicted_objective"], plausible = judge(it["stem"], random.Random(it["stem"]).sample(
            [it["answer"], *texts], len(texts) + 1))
        it["judged_correct"] = sorted(judged)
        it["key_ok"] = it["answer"] in judged
        it["objective_ok"] = it["predicted_objective"] == it["objective"]
        scores, it["rejected"] = [], {}
        for t in texts:
            score = solver.sentence_logprob(fill(t))
            scores.append(score)
            if t == it["answer"] or texts.count(t) > 1:
                it["rejected"][t] = "it repeats the answer or another distractor"
            elif t in judged:
                it["rejected"][t] = "it also makes a correct sentence"
            elif score >= key_score:
                it["rejected"][t] = "a language model finds it more natural than the answer"
            elif t not in plausible:
                it["rejected"][t] = "it is not a mistake a real learner would make"
        it["distractors_ok"] = not it["rejected"]
        it["accepted"] = it["key_ok"] and it["objective_ok"] and it["distractors_ok"]
        it["key_logit"] = key_logit(key_score, scores)

    del solver, model, tok
    gc.collect()
    torch.cuda.empty_cache()


def repair_items(items: list[dict], cur: dict, seed: int) -> list[dict]:
    """Replaces the rejected distractors of items whose key and objective passed."""
    examples = curated_examples()
    rng = random.Random(seed + 1)
    todo = [it for it in items if it["key_ok"] and it["objective_ok"] and not it["distractors_ok"]]
    if not todo:
        return []
    with _Exaone() as exaone:
        for it in todo:
            kept = [d for d in it["distractors"] if d["text"] not in it["rejected"]]
            n = N_DISTRACTORS - len(kept)
            content = _REPAIR_PROMPT.format(
                level=cur["level"], objective=cur["objectives"][it["objective"]],
                examples=examples.get(it["objective"], ""), answer=it["answer"],
                stem=it["stem"], kept=", ".join(f"{d['text']} ({d['mistake']})" for d in kept) or "(none)",
                rejected="; ".join(f"{t}: {why}" for t, why in it["rejected"].items()), n=n)
            schema = _replacements(n)
            try:
                new = schema.model_validate_json(exaone(content, schema, rng.randrange(2**31))).model_dump()
            except ValueError:
                continue
            it["first_distractors"] = it["distractors"]
            it["distractors"] = kept + new["distractors"]
            it["repaired"] = True
    return [it for it in todo if it.get("repaired")]


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
    """Sets the Urnings prior of the accepted items (difficulty_llm.py, standardised
    within this batch, as the bank is empty) and reports it."""
    import numpy as np

    from difficulty_heuristic import estimate_difficulty
    from difficulty_llm import difficulty_priors

    ok = [it for it in items if it.get("accepted")]
    if len(ok) < 2:
        return {"items": len(ok)}
    for it, (mu, s2) in zip(ok, difficulty_priors([it["key_logit"] for it in ok])):
        it["difficulty_mu"], it["difficulty_sigma2"] = mu, s2
    mus = [it["difficulty_mu"] for it in ok]
    heur = [estimate_difficulty(it["sentence"])[0] for it in ok]
    by_objective = {}
    for it in ok:
        by_objective.setdefault(it["objective"], []).append(it["difficulty_mu"])
    return {"items": len(ok), "mean": float(np.mean(mus)), "sd": float(np.std(mus)),
            "at_clip": sum(abs(m) >= 3.0 for m in mus) / len(mus),
            "key_logit_range": [float(min(it["key_logit"] for it in ok)), float(max(it["key_logit"] for it in ok))],
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
    parser.add_argument("--judge", default="Qwen/Qwen3-8B",
                        help="judge and key-gate model; Qwen/Qwen3-4B was used up to v2.4 (PRD §9.92)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="benchmark_v2_results.json")
    args = parser.parse_args()

    cur = load_curriculum(args.curriculum, args.context)
    items = generate_items(cur, args.per_objective, args.seed)
    check_items([it for it in items if it["format_ok"]], cur, args.judge)
    first_pass = sum(it.get("accepted", False) for it in items)
    check_items(repair_items([it for it in items if it["format_ok"]], cur, args.seed), cur, args.judge)
    report = {"settings": {k: str(v) for k, v in vars(args).items()}, "per_objective": {}}
    for oid in cur["objectives"]:
        mine = [it for it in items if it["objective"] == oid]
        ok = [it for it in mine if it["format_ok"]]
        report["per_objective"][oid] = {
            "format_ok": rate(len(ok), len(mine)),
            "key_ok": rate(sum(it["key_ok"] for it in ok), len(ok)),
            "distractors_ok": rate(sum(it["distractors_ok"] for it in ok), len(ok)),
            "repaired": sum(it.get("repaired", False) for it in ok),
            "objective_ok": rate(sum(it["objective_ok"] for it in ok), len(ok)),
            "accepted": rate(sum(it["accepted"] for it in ok), len(mine))}
        r = report["per_objective"][oid]
        print(f"{oid:16s} format {r['format_ok']['k']}/{r['format_ok']['n']}  key {r['key_ok']['k']}  "
              f"distractors {r['distractors_ok']['k']} (repaired {r['repaired']})  "
              f"objective {r['objective_ok']['k']}  accepted {r['accepted']['k']}/{r['accepted']['n']}")
    report["accepted_before_repair"] = rate(first_pass, len(items))
    report["accepted"] = rate(sum(it.get("accepted", False) for it in items), len(items))
    report["embeddings"] = {e: embedding_agreement(items, e) for e in args.encoders.split(",")}
    report["difficulty"] = difficulty_report(items)
    report["items"] = items
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    a = report["accepted"]
    print(f"\nACCEPTED {a['k']}/{a['n']} = {a['rate']:.1%}  95% CI {a['ci95']}  "
          f"(before repair {report['accepted_before_repair']['k']})")
    for name, e in report["embeddings"].items():
        print(f"EMBEDDINGS {name}: nearest neighbour same mistake {e['nearest_neighbour_same_mistake']}  "
              f"cos same {e['mean_cos_same']}  different {e['mean_cos_different']}")
    d = report["difficulty"]
    if "mean" in d:
        print(f"DIFFICULTY {d['items']} items, mean {d['mean']:+.2f} sd {d['sd']:.2f}  "
              f"at clip {d['at_clip']:.0%}  corr with morpheme heuristic {d['corr_with_morpheme_heuristic']:.2f}")
    print(f"written to {args.out}")
