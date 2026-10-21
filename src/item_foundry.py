"""
Item foundry for Novelty 1 (PRD §9.69): Korean grammar drill items whose
every distractor realises one NAMED misconception, so that the option a
learner picks says which misconception they hold (misconception_tracker.py),
not only that they were wrong.

Replaces the template generator (exaone_provider.py before 2026-10-21),
whose items came from hand-curated slot lists, verb lookups, fixed
correct/distractor pairs and filler options that carried no misconception
at all. Here the LLM writes the content, and code only VERIFIES it:

  1. Codebook (once per skill, build_codebook): EXAONE lists the
     misconceptions learners commonly hold about the grammar point, each
     with a correct sentence and the learner's wrong version. Each entry is
     kept only if kagas_annotator finds that example to be exactly one edit
     in the grammatical category the skill tests. Saved to
     data/misconception_codebook.json, so codes stay stable across items.
  2. Item (MisconceptionItemGenerator.generate): EXAONE writes one natural
     sentence using the grammar point, marks the tested span, and writes
     three wrong versions of that span, each realising a DIFFERENT codebook
     misconception -- one of them, when asked, the misconception the
     learner currently holds (personalisation).
  3. Verification, none of it by asking a model to judge (Qwen3-4B could
     not judge acceptability reliably in any of three framings, PRD
     §9.16/§9.31):
       - each distractor is a minimal pair of the key: kagas_annotator finds
         exactly one edit, in the skill's category;
       - its misconception code: for a closed-class edit (particles,
         endings, counters) the edit itself (PART:에서>에), which must agree
         with the codebook entry the generator named when that entry has a
         code; for an open-class edit (one verb for another) the codebook
         entry the generator named, which must exist;
       - the three codes differ, so every pick is informative;
       - key uniqueness by SCORING: under Qwen3-4B (a different model
         family from the generator), the sentence with the key must have a
         higher log-probability than every sentence with a distractor
         (BLiMP-style minimal pairs; 51/52 on the curated pairs, PRD §9.68).
         Skipped for subj_topic, whose correct answer is a discourse
         convention log-probability does not reflect (PRD §9.35).
  4. Critic: a failed item is sent back to EXAONE with the exact reasons,
     for up to MAX_REVISIONS revisions, instead of being discarded.

Models are injected as plain callables (chat, sentence_logprob), so the
pipeline logic is tested on CPU below with stand-ins; ExaoneChat and
solver_check.SolverCheckProvider are the real ones.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from kagas_annotator import annotate

Chat = Callable[[list[dict]], str]
Scorer = Callable[[str], float]

CODEBOOK_PATH = Path(__file__).resolve().parent.parent / "data" / "misconception_codebook.json"
N_DISTRACTORS = 3
CODEBOOK_SIZE = 6
MAX_REVISIONS = 2
BLANK = "___"


@dataclass(frozen=True)
class Skill:
    description: str           # the grammar point, as the curriculum defines it
    categories: frozenset      # KAGAS categories an edit of this skill may touch
    level: str = "beginner (TOPIK I)"
    logprob_gate: bool = True  # key uniqueness checkable by log-probability (PRD §9.35)


SKILLS = {
    "particle_loc": Skill("location particles: 에서 marks where an action takes place, 에 marks a "
                          "destination or where something is or exists", frozenset({"PART"})),
    "particle_obj": Skill("the object particle 을/를 on the direct object of a transitive verb "
                          "(을 after a final consonant, 를 after a vowel)", frozenset({"PART"})),
    "honorific": Skill("the subject honorific -(으)시- on the verb when the subject is a person "
                       "owed respect", frozenset({"END"})),
    "time_range": Skill("부터 and 까지 marking the start and the end of a time span",
                        frozenset({"PART"})),
    "tense_past": Skill("the past tense -았/었- in the polite -어요 style", frozenset({"END"})),
    "irregular_conj": Skill("irregular verb and adjective stems (르, ㅂ, ㄷ, ㅅ, ㅎ irregulars) before "
                            "an ending that starts with a vowel", frozenset({"CONJ", "END"})),
    "adj_relativizer": Skill("an adjective modifying a following noun with -(으)ㄴ",
                             frozenset({"END"})),
    "subj_topic": Skill("the subject particle 이/가 versus the topic particle 은/는",
                        frozenset({"PART"}), logprob_gate=False),
    "verb_choice": Skill("choosing the verb that collocates with its object noun",
                         frozenset({"VERB"})),
    "noun_counter": Skill("native Korean counters after a number", frozenset({"NOUN"})),
    "mod_adverb": Skill("turning an adjective into an adverb with -게 to modify a verb",
                        frozenset({"END"})),
}


@dataclass
class GeneratedItem:
    skill_id: str
    stem: str
    blank_marker: str
    options: dict[str, str]
    answer: str
    # KAGAS category the skill tests (PRD §9.8), kept for the item-level
    # escalation trigger when an item has no per-option codes.
    kagas_code: str | None = None
    # Misconception realised by each option, keyed by option label: None for
    # the key, a codebook code for each distractor (Novelty 1, PRD §9.69).
    # What misconception_tracker learns from.
    option_codes: dict[str, str | None] | None = None


@dataclass
class MisconceptionEntry:
    code: str                  # stable id: the edit (PART:에서>에) or skill_id/name
    name: str
    explanation: str
    example_correct: str
    example_wrong: str
    category: str
    functional: bool


@dataclass
class FoundryStats:
    """Why items were rejected, for the validation report (validate_item_foundry.py)."""
    attempts: int = 0
    accepted: int = 0
    revisions_used: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        key = reason.split(":")[0]
        self.reasons[key] = self.reasons.get(key, 0) + 1


def _loads_lenient(text: str):
    """json.loads, then once more without trailing commas (`[1, 2,]`), which
    LLMs write often."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(re.sub(r",\s*([\]}])", r"\1", text))


def extract_json(text: str):
    """The first parseable JSON object or array in a model reply (models
    often wrap it in prose or a ```json fence, or leave trailing commas)."""
    text = re.sub(r"```(?:json)?", "", text)
    pairs = sorted((("{", "}"), ("[", "]")), key=lambda p: (text.find(p[0]) == -1, text.find(p[0])))
    for opener, closer in pairs:  # whichever bracket opens first
        start = text.find(opener)
        while start != -1:
            depth, in_string, escaped = 0, False, False
            for i in range(start, len(text)):
                ch = text[i]
                if in_string:  # brackets inside strings do not count
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == '"':
                        in_string = False
                elif ch == '"':
                    in_string = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return _loads_lenient(text[start:i + 1])
                        except json.JSONDecodeError:
                            break
            start = text.find(opener, start + 1)
    return None


def _slug(name: str) -> str:
    """A stable ASCII id for a misconception name (a hash if the name has no
    ASCII letters, e.g. one written in Korean)."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40]
    return slug or hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]


def single_edit(skill: Skill, correct: str, wrong: str):
    """The one edit turning `correct` into `wrong`, or (None, reason)."""
    edits = annotate(correct, wrong)
    if len(edits) != 1:
        return None, f"not-minimal: {len(edits)} edits"
    edit = edits[0]
    if edit.category not in skill.categories:
        return None, f"wrong-category: {edit.category}, skill tests {sorted(skill.categories)}"
    return edit, ""


# ------------------------------------------------------------------ codebook

_CODEBOOK_PROMPT = """You are an expert teacher of Korean as a foreign language to Indonesian speakers.
Grammar point: {description}. Learners: {level}.

List the {k} most common misconceptions learners hold about THIS grammar point -- the mistaken
rule in their head, not random slips. For each, give a short name, a one-sentence explanation,
and one example: a short natural correct Korean sentence, and the same sentence as a learner
holding the misconception would write it, changed ONLY where the misconception shows.

Answer with JSON only:
[{{"name": "...", "explanation": "...", "correct": "...", "wrong": "..."}}, ...]"""


def _codebook_rows(raw) -> list:
    """The list of entries in a parsed reply: the list itself, or the first
    list value of an object ({"misconceptions": [...]})."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        return next((v for v in raw.values() if isinstance(v, list)), [])
    return []


def build_codebook(skill_id: str, chat: Chat, log: list | None = None,
                   attempts: int = 3) -> list[MisconceptionEntry]:
    """Verified codebook for one skill. The generator is asked again (its
    replies are sampled) until at least N_DISTRACTORS entries survive or
    `attempts` runs out. Every reply and every entry's verdict is appended to
    `log` when given, so an empty codebook can be diagnosed."""
    skill = SKILLS[skill_id]
    entries, seen = [], set()
    for _ in range(attempts):
        reply = chat([{"role": "user", "content": _CODEBOOK_PROMPT.format(
            description=skill.description, level=skill.level, k=CODEBOOK_SIZE)}])
        raw = extract_json(reply)
        record = {"reply": reply, "parsed": raw is not None, "rows": []}
        for row in _codebook_rows(raw):
            if not isinstance(row, dict) or not all(isinstance(row.get(k), str) for k in
                                                    ("name", "explanation", "correct", "wrong")):
                record["rows"].append({"row": row, "verdict": "format: needs name, explanation, correct, wrong"})
                continue
            correct, wrong = row["correct"].strip(), row["wrong"].strip()
            edit, why = single_edit(skill, correct, wrong)
            if edit is None:
                record["rows"].append({"row": row, "verdict": why,
                                       "edits": [vars(e) for e in annotate(correct, wrong)]})
                continue
            code = edit.code if edit.functional else f"{skill_id}/{_slug(row['name'])}"
            if code in seen:
                record["rows"].append({"row": row, "verdict": f"duplicate: {code}"})
                continue
            seen.add(code)
            record["rows"].append({"row": row, "verdict": f"kept: {code}"})
            entries.append(MisconceptionEntry(code, row["name"].strip(), row["explanation"].strip(),
                                              correct, wrong, edit.category, edit.functional))
        if log is not None:
            log.append(record)
        if len(entries) >= N_DISTRACTORS:
            break
    return entries


def load_codebook(path: Path = CODEBOOK_PATH) -> dict[str, list[MisconceptionEntry]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {s: [MisconceptionEntry(**e) for e in entries] for s, entries in data["skills"].items()}


def save_codebook(codebook: dict[str, list[MisconceptionEntry]], provenance: dict,
                  path: Path = CODEBOOK_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"provenance": provenance,
            "skills": {s: [asdict(e) for e in entries] for s, entries in codebook.items()}}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------------------------------------------------ items

_ITEM_PROMPT = """You write Korean grammar drill questions for {level} learners whose first language is Indonesian.
Grammar point: {description}.

Misconceptions learners hold about it:
{codebook}

Write ONE new, natural, everyday Korean sentence that uses the grammar point. Then:
- "target": the exact part of the sentence that tests the grammar point (it must appear in the
  sentence exactly once; as short as possible, e.g. a particle or a word ending, or the one word
  that carries it);
- "distractors": {n} wrong replacements for "target", each what a learner holding a DIFFERENT
  misconception above would write, changing nothing else, and each clearly wrong in this sentence.
  Give the misconception's number for each.{target_line}
Do not reuse these sentences: {avoid}

Answer with JSON only:
{{"sentence": "...", "target": "...", "distractors": [{{"text": "...", "misconception": 1}}, ...]}}"""

_REVISION_PROMPT = """That question cannot be used, for these reasons:
{reasons}
Write a corrected version that fixes every reason. Answer with JSON only, in the same format."""


class MisconceptionItemGenerator:
    """Implements daily_drill.ContentProvider's grammar half."""

    def __init__(self, chat: Chat, sentence_logprob: Scorer | None,
                 codebook: dict[str, list[MisconceptionEntry]], seed: int | None = None):
        self.chat = chat
        self.sentence_logprob = sentence_logprob
        self.codebook = codebook
        self.rng = random.Random(seed)
        self.items: dict[str, GeneratedItem] = {}
        self.stats: dict[str, FoundryStats] = {}
        self._recent: dict[str, list[str]] = {}
        self._counter = 0

    # -- verification ------------------------------------------------------
    def verify(self, skill_id: str, reply: dict, target_code: str | None = None):
        """(item fields, []) if the reply passes every check, else (None, reasons)."""
        skill, entries = SKILLS[skill_id], self.codebook.get(skill_id, [])
        if not isinstance(reply, dict):
            return None, ["format: the reply was not a JSON object"]
        sentence, target = reply.get("sentence"), reply.get("target")
        distractors = reply.get("distractors")
        if not (isinstance(sentence, str) and isinstance(target, str) and isinstance(distractors, list)):
            return None, ["format: needs sentence, target and a list of distractors"]
        sentence, target = sentence.strip(), target.strip()
        if not target or sentence.count(target) != 1:
            return None, [f"target: '{target}' must appear in the sentence exactly once"]
        if len(distractors) != N_DISTRACTORS:
            return None, [f"format: exactly {N_DISTRACTORS} distractors are needed, got {len(distractors)}"]

        reasons, texts, codes = [], [], []
        for d in distractors:
            text = d.get("text", "").strip() if isinstance(d, dict) else ""
            number = d.get("misconception") if isinstance(d, dict) else None
            if not text or text == target or text in texts:
                reasons.append(f"duplicate: distractor '{text}' repeats the target or another distractor")
                continue
            wrong_sentence = sentence.replace(target, text, 1)
            edit, why = single_edit(skill, sentence, wrong_sentence)
            if edit is None:
                reasons.append(f"{why.split(':')[0]}: '{wrong_sentence}' {why}")
                continue
            entry = entries[number - 1] if isinstance(number, int) and 1 <= number <= len(entries) else None
            if edit.functional:
                code = edit.code
                if entry is not None and entry.functional and entry.code != code:
                    reasons.append(f"label: '{text}' realises {code}, not misconception {number} "
                                   f"({entry.name}, {entry.code})")
                    continue
            elif entry is None:
                reasons.append(f"label: '{text}' must name one of the listed misconceptions")
                continue
            else:
                code = entry.code
            texts.append(text)
            codes.append(code)
        if len(set(codes)) != len(codes):
            reasons.append("distinct: two distractors realise the same misconception")
        if target_code is not None and target_code not in codes:
            name = next((e.name for e in entries if e.code == target_code), target_code)
            reasons.append(f"target-misconception: no distractor realises '{name}'")
        if reasons:
            return None, reasons

        if skill.logprob_gate and self.sentence_logprob is not None:
            key_score = self.sentence_logprob(sentence)
            for text in texts:
                if self.sentence_logprob(sentence.replace(target, text, 1)) >= key_score:
                    reasons.append(f"ambiguous: '{text}' reads at least as naturally as the key "
                                   f"'{target}'; it may also be correct")
            if reasons:
                return None, reasons
        return (sentence, target, texts, codes), []

    # -- generation --------------------------------------------------------
    def _prompt(self, skill_id: str, target_code: str | None) -> str:
        skill, entries = SKILLS[skill_id], self.codebook.get(skill_id, [])
        listed = "\n".join(f"{i}. {e.name}: {e.explanation} (e.g. {e.example_correct} -> {e.example_wrong})"
                           for i, e in enumerate(entries, 1))
        target_line = ""
        if target_code is not None:
            number = next((i for i, e in enumerate(entries, 1) if e.code == target_code), None)
            if number is not None:
                target_line = f"\n  One distractor MUST realise misconception {number}."
        avoid = "; ".join(self._recent.get(skill_id, [])[-8:]) or "(none yet)"
        return _ITEM_PROMPT.format(level=skill.level, description=skill.description, codebook=listed,
                                   n=N_DISTRACTORS, target_line=target_line, avoid=avoid)

    def generate(self, skill_id: str, target_code: str | None = None) -> GeneratedItem | None:
        if skill_id not in SKILLS or not self.codebook.get(skill_id):
            return None
        stats = self.stats.setdefault(skill_id, FoundryStats())
        stats.attempts += 1
        messages = [{"role": "user", "content": self._prompt(skill_id, target_code)}]
        for revision in range(MAX_REVISIONS + 1):
            reply_text = self.chat(messages)
            fields, reasons = self.verify(skill_id, extract_json(reply_text), target_code)
            if fields is not None:
                stats.accepted += 1
                stats.revisions_used += revision
                return self._assemble(skill_id, *fields)
            for reason in reasons:
                stats.reject(reason)
            messages += [{"role": "assistant", "content": reply_text},
                         {"role": "user", "content": _REVISION_PROMPT.format(
                             reasons="\n".join(f"- {r}" for r in reasons))}]
        return None

    def _assemble(self, skill_id, sentence, target, texts, codes) -> GeneratedItem:
        labels = ["A", "B", "C", "D"]
        self.rng.shuffle(labels)
        options = {labels[0]: target, **dict(zip(labels[1:], texts))}
        option_codes = {labels[0]: None, **dict(zip(labels[1:], codes))}
        options = dict(sorted(options.items()))
        option_codes = dict(sorted(option_codes.items()))
        stem = sentence.replace(target, BLANK, 1)
        self._recent.setdefault(skill_id, []).append(sentence)
        categories = sorted(SKILLS[skill_id].categories)
        return GeneratedItem(skill_id, stem, BLANK, options, labels[0], categories[0], option_codes)

    def next_new_grammar_item(self, skill_id: str, learner_urn=None, exclude: set[str] = frozenset(),
                              target_misconception: str | None = None) -> str | None:
        item = self.generate(skill_id, target_misconception)
        if item is None:
            return None
        item_id = f"grammar-{skill_id}-{self._counter}"
        self._counter += 1
        self.items[item_id] = item
        return item_id


class ExaoneChat:
    """EXAONE-3.5-7.8B-Instruct, 4-bit (PRD §9.53), as a Chat callable.
    Sampled, not greedy, so repeated calls give different sentences."""

    MODEL = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"

    def __init__(self, model_name: str = MODEL, temperature: float = 0.8, max_new_tokens: int = 1500):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, device_map="cuda",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16))
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def __call__(self, messages: list[dict]) -> str:
        enc = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                           return_dict=True).to("cuda")
        with self._torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=True,
                                      temperature=self.temperature, top_p=0.95)
        return self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


if __name__ == "__main__":
    # CPU smoke test: the pipeline logic with stand-in models.
    assert extract_json('Sure! ```json\n{"a": [1, {"b": 2}]}\n``` done') == {"a": [1, {"b": 2}]}
    assert extract_json("[1, 2] trailing") == [1, 2]
    assert extract_json("no json") is None
    assert extract_json('```json\n[{"a": "x]y"}, {"b": 2},]\n```') == [{"a": "x]y"}, {"b": 2}]
    assert _codebook_rows({"misconceptions": [1]}) == [1]

    codebook_reply = json.dumps([
        {"name": "에 for every location", "explanation": "uses 에 for the place of an action",
         "correct": "저는 학교에서 공부해요.", "wrong": "저는 학교에 공부해요."},
        {"name": "object particle on places", "explanation": "marks the place like an object",
         "correct": "저는 학교에서 공부해요.", "wrong": "저는 학교를 공부해요."},
        {"name": "subject particle on places", "explanation": "marks the place as the subject",
         "correct": "저는 학교에서 공부해요.", "wrong": "저는 학교가 공부해요."},
        {"name": "not minimal", "explanation": "two changes, must be dropped",
         "correct": "저는 학교에서 공부해요.", "wrong": "제가 학교에 공부해요."},
        {"name": "duplicate edit", "explanation": "same edit as the first, must be dropped",
         "correct": "친구가 카페에서 일해요.", "wrong": "친구가 카페에 일해요."},
    ], ensure_ascii=False)
    log = []
    book = build_codebook("particle_loc", lambda m: "Here you go:\n" + codebook_reply, log=log)
    assert len(log) == 1 and sum(r["verdict"].startswith("kept") for r in log[0]["rows"]) == 3
    assert [e.code for e in book] == ["PART:에서>에", "PART:에서>을", "PART:에서>이"], [e.code for e in book]

    good = {"sentence": "동생이 공원에서 축구를 해요.", "target": "에서",
            "distractors": [{"text": "에", "misconception": 1}, {"text": "을", "misconception": 2},
                            {"text": "이", "misconception": 3}]}
    mislabelled = {**good, "distractors": [{"text": "에", "misconception": 2}, {"text": "을", "misconception": 1},
                                           {"text": "이", "misconception": 3}]}
    replies = iter([json.dumps(mislabelled, ensure_ascii=False), json.dumps(good, ensure_ascii=False)])
    seen_prompts = []

    def chat(messages):
        seen_prompts.append(messages[-1]["content"])
        return next(replies)

    # A scorer that prefers the key sentence: stands in for Qwen3-4B log-probs.
    gen = MisconceptionItemGenerator(chat, lambda s: 0.0 if "공원에서" in s else -5.0,
                                     {"particle_loc": book}, seed=0)
    item_id = gen.next_new_grammar_item("particle_loc", target_misconception="PART:에서>이")
    assert item_id is not None, gen.stats
    item = gen.items[item_id]
    assert item.stem == "동생이 공원___ 축구를 해요." and item.options[item.answer] == "에서"
    assert sorted(c for c in item.option_codes.values() if c) == ["PART:에서>에", "PART:에서>을", "PART:에서>이"]
    assert item.option_codes[item.answer] is None
    assert "MUST realise misconception 3" in seen_prompts[0]
    assert "label:" in seen_prompts[1], "the critic must be told exactly why the first reply failed"
    assert gen.stats["particle_loc"].revisions_used == 1

    # Ambiguity is caught by scoring: a distractor that reads as naturally as the key.
    flat = MisconceptionItemGenerator(lambda m: json.dumps(good, ensure_ascii=False), lambda s: 0.0,
                                      {"particle_loc": book}, seed=0)
    assert flat.generate("particle_loc") is None
    assert flat.stats["particle_loc"].reasons.get("ambiguous", 0) > 0

    # A distractor that edits something other than the tested category is rejected.
    _, reasons = gen.verify("particle_loc", {**good, "distractors": [
        {"text": "에", "misconception": 1}, {"text": "을", "misconception": 2},
        {"text": "에서 축구", "misconception": 3}]})
    assert any(r.startswith("wrong-category") for r in reasons), reasons

    # Open-class skills take their code from the named codebook entry.
    verb_book = build_codebook("verb_choice", lambda m: json.dumps([
        {"name": "eat for all consumption", "explanation": "uses 먹다 for drinks",
         "correct": "물을 마셔요.", "wrong": "물을 먹어요."}], ensure_ascii=False))
    assert verb_book and verb_book[0].code == "verb_choice/eat-for-all-consumption"
    assert _slug("먹다 과잉 일반화") and _slug("먹다 과잉 일반화") != _slug("마시다")
    print("all smoke tests passed")
