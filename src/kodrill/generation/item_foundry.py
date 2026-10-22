from __future__ import annotations

import itertools
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

Chat = Callable[[list[dict]], str]
Scorer = Callable[[str], float]

MAX_REVISIONS = 2
BLANK = "___"
_LABELS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class CurriculumItem:
    description: str
    level: str = "beginner (TOPIK I)"


@dataclass(frozen=True)
class Curriculum:
    items: dict[str, CurriculumItem]
    context: str = ""
    modules: dict[str, list[str]] = field(default_factory=dict)

    @staticmethod
    def from_json(path: str | Path) -> "Curriculum":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        items = {item_id: CurriculumItem(description=spec["description"],
                                         level=spec.get("level", "beginner (TOPIK I)"))
                 for item_id, spec in data["items"].items()}
        return Curriculum(items=items, context=data.get("context", ""), modules=data.get("modules", {}))

    def random_item_id(self, rng: random.Random) -> str:
        return rng.choice(sorted(self.items))

    def random_setting(self, rng: random.Random) -> str:
        if not self.modules:
            return ""
        module = rng.choice(sorted(self.modules))
        topic = rng.choice(self.modules[module])
        return f"Setting: {self.context}; {module}: {topic}. Use one word of the setting."


@dataclass
class GeneratedItem:
    skill_id: str
    stem: str
    blank_marker: str
    options: dict[str, str]
    answer: str
    reasoning: dict[str, str] = field(default_factory=dict)


@dataclass
class FoundryStats:
    attempts: int = 0
    accepted: int = 0
    revisions_used: int = 0
    reasons: dict[str, int] = field(default_factory=dict)
    rejected: list[dict] = field(default_factory=list)

    def reject(self, reason: str) -> None:
        key = reason.split(":")[0]
        self.reasons[key] = self.reasons.get(key, 0) + 1


def _loads_lenient(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        text = re.sub(r'(?m)(["\]}\d,])[ \t]*//[^\n]*$', r"\1", text)
        return json.loads(re.sub(r",\s*([\]}])", r"\1", text))


def extract_json(text: str):
    text = re.sub(r"```(?:json)?", "", text)
    pairs = sorted((("{", "}"), ("[", "]")), key=lambda p: (text.find(p[0]) == -1, text.find(p[0])))
    for opener, closer in pairs:
        start = text.find(opener)
        while start != -1:
            depth, in_string, escaped = 0, False, False
            for i in range(start, len(text)):
                ch = text[i]
                if in_string:
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


_SYSTEM_PROMPT = """You are a native Korean speaker and an experienced Korean language teacher.
You write Korean sentences for Indonesian learners at {level} level."""

_SENTENCE_PROMPT = """Task: write one Korean sentence that uses the grammar/vocabulary point below
correctly, and identify the exact word that carries it.

Grammar/vocabulary point: {description}
{setting}

Requirements:
1. One natural, everyday Korean sentence at {level} level. Hangul only -- no romanisation, no
   translation, no brackets.
2. "target": the exact word/particle/ending that IS the grammar/vocabulary point itself, not a
   different word that merely sits near it. Copied character-for-character from "sentence",
   occurring in it exactly once. Check: if "target" were blanked out, filling it back in
   correctly must require knowing THIS point, not some other one.
3. Output: one JSON object, nothing else -- no explanation, no comments, no markdown.

{{"sentence": "...", "target": "..."}}

{example_note}
{example}{avoid}"""

_FORMAT_EXAMPLE = '{"sentence": "동생이 공원에서 축구를 해요.", "target": "에서"}'

_DISTRACTOR_PROMPT = """Task: write three WRONG alternatives for the blank below -- real mistakes a
learner would make with this exact grammar/vocabulary point.

Grammar/vocabulary point: {description}
Sentence: {stem}
Correct answer: {target}

Requirements:
1. Each alternative: a MINIMAL variation of "{target}" -- comparable in length to "{target}"
   itself (swap the particle/ending, never a full phrase or a restated sentence), or (for a
   vocabulary point) a different word that still fits this exact slot. Never an unrelated word.
2. Each must still fit grammatically in the blank: wrong because of the grammar/vocabulary
   point, never because the sentence becomes nonsense.
3. Do not repeat any word that already appears in "Sentence" -- filling the blank must never
   duplicate a word that is already right next to it.
4. No notation characters such as "-" or "/" -- those are documentation shorthand, never real
   Korean text.
5. All three different from each other and from "{target}". Hangul only -- no romanisation, no
   translation, no brackets.
6. Output: one JSON object, nothing else -- no explanation, no comments, no markdown.

{{"wrong": ["...", "...", "..."]}}"""

_REVISION_PROMPT = """This was rejected:
{reasons}
Try again, following every rule. Output the JSON object only."""

_REASONING_PROMPT = """Exercise: {stem}
Options:
{options_list}
Correct answer: {answer}) {correct_text}

For EACH option, write ONE short sentence in English explaining why it is correct, or what
mistake it represents if it is wrong. Quote the option's own Korean form in your sentence.
Output one JSON object mapping each option's letter to its sentence, nothing else.

Output format:
{{"A": "...", "B": "...", ...}}"""


class MisconceptionItemGenerator:
    def __init__(self, chat: Chat, curriculum: Curriculum, sentence_logprob: Scorer | None = None,
                 seed: int | None = None):
        self.chat = chat
        self.sentence_logprob = sentence_logprob
        self.rng = random.Random(seed)
        self.curriculum = curriculum
        self.items: dict[str, GeneratedItem] = {}
        self.stats: dict[str, FoundryStats] = {}
        self._recent: dict[str, list[str]] = {}
        self._example: dict[str, str] = {}
        self._counter = 0

    def _sentence_prompt(self, skill_id: str) -> str:
        skill = self.curriculum.items[skill_id]
        recent = self._recent.get(skill_id, [])[-8:]
        avoid = f"\nUse a different sentence from these: {'; '.join(recent)}" if recent else ""
        example = self._example.get(skill_id)
        note = ("A good exercise for this grammar point:" if example else
                "Example of the output shape only (it tests a different grammar point):")
        setting = self.curriculum.random_setting(self.rng)
        return _SENTENCE_PROMPT.format(description=skill.description, level=skill.level, setting=setting,
                                       avoid=avoid, example_note=note, example=example or _FORMAT_EXAMPLE)

    def _system(self, skill_id: str) -> str:
        return _SYSTEM_PROMPT.format(level=self.curriculum.items[skill_id].level)

    def verify_sentence(self, reply: dict):
        if not isinstance(reply, dict):
            return None, ["format: the reply was not a JSON object"]
        sentence, target = reply.get("sentence"), reply.get("target")
        if not isinstance(sentence, str) or not sentence.strip():
            return None, ['format: needs a non-empty "sentence" string']
        if not isinstance(target, str) or not target:
            return None, ['format: needs a non-empty "target" string']
        count = sentence.count(target)
        if count != 1:
            return None, [f"target: \"{target}\" must occur exactly once in the sentence, found {count}"]
        return (sentence, target), []

    def verify_distractors(self, reply: dict, target: str):
        if not isinstance(reply, dict):
            return None, ["format: the reply was not a JSON object"]
        wrong = reply.get("wrong")
        if not isinstance(wrong, list):
            return None, ['format: needs a "wrong" list']
        texts = list(dict.fromkeys(w for w in wrong if isinstance(w, str) and w.strip() and w != target))
        if len(texts) < 3:
            return None, [f"format: 3 distinct wrong alternatives are needed, got {len(texts)}"]
        return texts[:3], []

    def _retry(self, build_prompt: list[dict], call_and_verify, stats: FoundryStats):
        messages = build_prompt
        for revision in range(MAX_REVISIONS + 1):
            reply_text = self.chat(messages)
            fields, reasons = call_and_verify(reply_text)
            if fields is not None:
                stats.revisions_used += revision
                return fields
            for reason in reasons:
                stats.reject(reason)
            if len(stats.rejected) < 10:
                stats.rejected.append({"reply": reply_text[:500], "reasons": reasons})
            messages = build_prompt + [{"role": "assistant", "content": reply_text},
                                       {"role": "user", "content": _REVISION_PROMPT.format(
                                           reasons="\n".join(f"- {r}" for r in reasons))}]
        return None

    def generate(self, skill_id: str | None = None) -> GeneratedItem | None:
        if skill_id is None:
            skill_id = self.curriculum.random_item_id(self.rng)
        elif skill_id not in self.curriculum.items:
            return None
        stats = self.stats.setdefault(skill_id, FoundryStats())
        stats.attempts += 1
        skill = self.curriculum.items[skill_id]

        sentence_prompt = [{"role": "system", "content": self._system(skill_id)},
                           {"role": "user", "content": self._sentence_prompt(skill_id)}]
        sentence_fields = self._retry(
            sentence_prompt, lambda text: self.verify_sentence(extract_json(text)), stats)
        if sentence_fields is None:
            return None
        sentence, target = sentence_fields
        stem = sentence.replace(target, BLANK, 1)

        distractor_prompt = [{"role": "system", "content": self._system(skill_id)},
                             {"role": "user", "content": _DISTRACTOR_PROMPT.format(
                                 level=skill.level, description=skill.description, stem=stem, target=target)}]
        wrongs = self._retry(
            distractor_prompt, lambda text: self.verify_distractors(extract_json(text), target), stats)
        if wrongs is None:
            return None

        stats.accepted += 1
        return self._assemble(skill_id, stem, target, wrongs, sentence)

    def _reasoning(self, skill_id: str, stem: str, options: dict[str, str], answer: str) -> dict[str, str]:
        options_list = "\n".join(f"{label}) {text}" for label, text in options.items())
        messages = [{"role": "system", "content": self._system(skill_id)},
                    {"role": "user", "content": _REASONING_PROMPT.format(
            stem=stem, options_list=options_list, answer=answer, correct_text=options[answer])}]
        reply = extract_json(self.chat(messages))
        if isinstance(reply, dict):
            return {label: str(reply.get(label, "")) for label in options}
        return {label: "" for label in options}

    def _assemble(self, skill_id: str, stem: str, target: str, wrongs: list[str],
                  sentence: str) -> GeneratedItem:
        labels = list(_LABELS)
        self.rng.shuffle(labels)
        options = dict(sorted({labels[0]: target, **dict(zip(labels[1:], wrongs))}.items()))
        answer = labels[0]
        self._recent.setdefault(skill_id, []).append(sentence)
        self._example[skill_id] = json.dumps({"sentence": sentence, "target": target}, ensure_ascii=False)
        reasoning = self._reasoning(skill_id, stem, options, answer)
        return GeneratedItem(skill_id, stem, BLANK, options, answer, reasoning)

    def next_new_grammar_item(self, skill_id: str | None = None, learner_urn=None,
                              exclude: set[str] = frozenset()) -> str | None:
        item = self.generate(skill_id)
        if item is None:
            return None
        item_id = f"grammar-{item.skill_id}-{self._counter}"
        self._counter += 1
        self.items[item_id] = item
        return item_id


class ExaoneChat:
    MODEL = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"

    def __init__(self, model_name: str = MODEL, temperature: float = 0.6, max_new_tokens: int = 400):
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
        text = self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        del enc, out
        self._torch.cuda.empty_cache()
        return text


def _echo_reasoning(messages: list[dict]) -> str:
    content = messages[-1]["content"]
    labels_texts = re.findall(r"^([A-D])\) (.+)$", content, re.MULTILINE)
    return json.dumps({label: f"'{text}' is the point being tested here." for label, text in labels_texts},
                      ensure_ascii=False)


def _make_chat(*replies):
    it = itertools.chain(replies, itertools.repeat(replies[-1]))
    seen: list[str] = []

    def chat(messages: list[dict]) -> str:
        content = messages[-1]["content"]
        if content.startswith("Exercise:"):
            return _echo_reasoning(messages)
        seen.append(content)
        return json.dumps(next(it), ensure_ascii=False)

    chat.seen = seen
    return chat


if __name__ == "__main__":
    assert extract_json('Sure! ```json\n{"a": [1, {"b": 2}]}\n``` done') == {"a": [1, {"b": 2}]}
    assert extract_json("[1, 2] trailing") == [1, 2]
    assert extract_json("no json") is None
    assert extract_json('```json\n[{"a": "x]y"}, {"b": 2},]\n```') == [{"a": "x]y"}, {"b": 2}]
    assert extract_json('{"a": "x", // note\n"b": 1}') == {"a": "x", "b": 1}

    good_sentence = {"sentence": "동생이 공원에서 축구를 해요.", "target": "에서"}
    bad_sentence = {"sentence": "동생이 공원에서 축구를 해요.", "target": "학교"}
    good_wrong = {"wrong": ["에", "을", "이"]}
    duplicate_wrong = {"wrong": ["에", "에", "이"]}

    one_item = Curriculum(items={"particle_loc": CurriculumItem(
        "location particles: 에서 marks where an action takes place, 에 marks a destination or "
        "where something is or exists")})

    chat = _make_chat(bad_sentence, good_sentence, duplicate_wrong, good_wrong)
    gen = MisconceptionItemGenerator(chat, one_item, seed=0)
    item_id = gen.next_new_grammar_item("particle_loc")
    assert item_id is not None, gen.stats
    item = gen.items[item_id]
    assert item.stem == "동생이 공원___ 축구를 해요." and item.options[item.answer] == "에서", item
    assert set(item.options.values()) == {"에서", "에", "을", "이"}, item
    assert set(item.reasoning) == set(item.options), item
    assert all(item.options[label] in item.reasoning[label] for label in item.options), item.reasoning
    assert "target:" in chat.seen[1]
    assert "format:" in chat.seen[3]
    stats = gen.stats["particle_loc"]
    assert stats.rejected and stats.rejected[0]["reasons"]
    assert stats.revisions_used == 2

    curriculum = Curriculum(items=one_item.items, context="a smart factory course", modules={
        "Module 1": ["predictive maintenance using sensor data"]})
    cur_chat = _make_chat(good_sentence, good_wrong)
    cur_gen = MisconceptionItemGenerator(cur_chat, curriculum, seed=0)
    assert cur_gen.next_new_grammar_item("particle_loc") is not None, cur_gen.stats
    assert any("a smart factory course" in p and "predictive maintenance" in p for p in cur_chat.seen)
    no_setting_prompt = MisconceptionItemGenerator(lambda m: "", one_item)._sentence_prompt("particle_loc")
    assert "Setting:" not in no_setting_prompt

    random_gen = MisconceptionItemGenerator(_make_chat(good_sentence, good_wrong), one_item, seed=0)
    picked_id = random_gen.next_new_grammar_item()
    assert picked_id is not None and random_gen.items[picked_id].skill_id == "particle_loc"
    assert random_gen.generate("not_in_this_curriculum") is None

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump({"context": "a bakery", "modules": {"M": ["bread"]},
                  "items": {"only_skill": {"description": "a made-up grammar point"}}}, f)
        custom_path = f.name
    custom_curriculum = Curriculum.from_json(custom_path)
    assert set(custom_curriculum.items) == {"only_skill"}
    no_items_path = custom_path.replace(".json", "2.json")
    with open(no_items_path, "w", encoding="utf-8") as f:
        json.dump({"context": "a bakery", "modules": {"M": ["bread"]}}, f)
    try:
        Curriculum.from_json(no_items_path)
        raise AssertionError('a curriculum JSON with no "items" key must fail, not fall back to anything')
    except KeyError:
        pass

    plain = MisconceptionItemGenerator(_make_chat(good_sentence, good_wrong), one_item)
    fields, reasons = plain.verify_sentence(good_sentence)
    assert fields == ("동생이 공원에서 축구를 해요.", "에서"), (fields, reasons)
    fields, reasons = plain.verify_sentence({**good_sentence, "target": "학교"})
    assert fields is None and reasons[0].startswith("target"), (fields, reasons)
    fields, reasons = plain.verify_sentence({"sentence": "에서 공원에서 놀아요.", "target": "에서"})
    assert fields is None and reasons[0].startswith("target"), (fields, reasons)
    fields, reasons = plain.verify_distractors({"wrong": ["에", "에", "에서"]}, "에서")
    assert fields is None, (fields, reasons)
    fields, reasons = plain.verify_distractors({"wrong": ["에", "을", "이"]}, "에서")
    assert fields == ["에", "을", "이"], (fields, reasons)
    fields, reasons = plain.verify_distractors({"wrong": ["사과", "학교", "강아지"]}, "에서")
    assert fields is not None, (fields, reasons)

    print("all smoke tests passed")
