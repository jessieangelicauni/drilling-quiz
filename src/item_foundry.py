"""
Item foundry for Novelty 1 (PRD §9.69, §9.74): Korean grammar drill items
whose every distractor realises one NAMED error, so that the option a
learner picks says which misconception they hold (misconception_tracker.py),
not only that they were wrong.

The LLM writes the content; code only VERIFIES it and LABELS it:

  1. Item (MisconceptionItemGenerator.generate): EXAONE writes one natural
     sentence using the grammar point and three wrong copies of it, each
     with a different error a learner typically makes -- one of them, when
     asked, the error the learner currently makes most (personalisation).
     The blank is where the wrong copies differ from the sentence (_blank):
     the model only writes sentences, the form it handles most reliably
     (PRD §9.78).
  2. Label: each distractor's misconception code is the KAGAS edit that
     turns the key sentence into the distractor sentence
     (kagas_annotator.annotate), e.g. PART:에서>에 or END:었>았. The code is
     computed, not chosen by the model, so it is the same for the same
     error in any item: the stable label the choice model needs, without a
     hand-written or generated list of misconceptions.
  3. Verification, none of it by asking a model to judge (Qwen3-4B could
     not judge acceptability reliably in any of three framings, PRD
     §9.16/§9.31):
       - each distractor is a minimal pair of the key: exactly one edit, in
         the skill's KAGAS category;
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

import json
import random
import re
from dataclasses import dataclass, field
from typing import Callable

from kagas_annotator import _KIWI, annotate

Chat = Callable[[list[dict]], str]
Scorer = Callable[[str], float]

N_DISTRACTORS = 3
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
    # the key, the KAGAS edit code for each distractor (Novelty 1, PRD §9.74).
    # What misconception_tracker learns from.
    option_codes: dict[str, str | None] | None = None


@dataclass
class FoundryStats:
    """Why items were rejected, for the validation report (validate_item_foundry.py)."""
    attempts: int = 0
    accepted: int = 0
    revisions_used: int = 0
    reasons: dict[str, int] = field(default_factory=dict)
    # The first few rejected replies with their reasons, so a low acceptance
    # rate can be diagnosed from the results file.
    rejected: list[dict] = field(default_factory=list)

    def reject(self, reason: str) -> None:
        key = reason.split(":")[0]
        self.reasons[key] = self.reasons.get(key, 0) + 1


def _loads_lenient(text: str):
    """json.loads, then once more without trailing commas (`[1, 2,]`) and
    `// comments` after a value, which LLMs write often."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        text = re.sub(r'(?m)(["\]}\d,])[ \t]*//[^\n]*$', r"\1", text)
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


_LATIN = re.compile(r"[A-Za-z]")


def korean_only(sentence: str) -> str | None:
    """The Korean sentence without the romanisation or translation the
    generator often appends -- "공원에서 산책해요. (Gongwon-eseo ...) - I walk
    in the park." -- or None when Latin text remains inside the sentence
    itself, which is then not a clean example."""
    s = re.sub(r"\s*\([^()]*[A-Za-z][^()]*\)", "", sentence)   # (romanisation), (translation)
    s = re.split(r"\s+[-\u2013\u2014:]\s+(?=.*[A-Za-z])", s, maxsplit=1)[0]  # " - I walk ..."
    s = s.strip()
    return None if not s or _LATIN.search(s) else s


# KAGAS categories in words, for the critic's feedback.
_CATEGORY_WORDS = {"PART": "particle", "END": "verb or adjective ending", "CONJ": "conjugation",
                   "VERB": "verb choice", "NOUN": "noun or counter", "ADJ": "adjective choice",
                   "SPELL": "spelling", "PUNCT": "punctuation", "MULTI": "mixed", "OTHER": "other"}


def _category_words(category: str) -> str:
    return _CATEGORY_WORDS.get(category, category)


def single_edit(skill: Skill, correct: str, wrong: str):
    """The one edit turning `correct` into `wrong`, or (None, reason)."""
    edits = annotate(correct, wrong)
    if len(edits) != 1:
        return None, (f"not-minimal: it differs from the correct sentence in {len(edits)} places; "
                      f"it must differ in exactly one")
    edit = edits[0]
    if edit.category not in skill.categories:
        wanted = " or ".join(_category_words(c) for c in sorted(skill.categories))
        if edit.category == "MULTI":
            return None, (f"wrong-category: it is not one {wanted} mistake; it mixes in other "
                          f"words or forms")
        return None, (f"wrong-category: it makes a {_category_words(edit.category)} mistake, "
                      f"but this question tests a {wanted} mistake")
    return edit, ""


# ------------------------------------------------------------------ items

_SYSTEM_PROMPT = """You write Korean grammar exercises for Indonesian learners at {level} level.

You get a grammar point. Write:
- "sentence": one natural, everyday Korean sentence that uses the grammar point correctly.
- "wrong_sentences": three copies of that same sentence, each with exactly ONE mistake in the
  grammar point, the kind of mistake learners really make. Change only that one particle, ending
  or word; keep every other word exactly the same. The three mistakes must be different.

Rules:
1. Hangul only: no romanisation, no translation, no brackets.
2. The sentence is short and correct; each wrong sentence is wrong only because of the one mistake.
3. Output one JSON object and nothing else: no explanation, no comments, no markdown.

Output format:
{{"sentence": "...", "wrong_sentences": ["...", "...", "..."]}}"""

_ITEM_PROMPT = """Grammar point: {description}.{target_line}
{example_note}
{example}
Write a new exercise.{avoid}"""

# The example until a skill has an accepted item of its own; after that, the
# skill's latest accepted item is the example (on-skill, not fixed).
_FORMAT_EXAMPLE = ('{"sentence": "동생이 공원에서 축구를 해요.", "wrong_sentences": '
                   '["동생이 공원에 축구를 해요.", "동생이 공원을 축구를 해요.", "동생이 공원이 축구를 해요."]}')

_REVISION_PROMPT = """This exercise was rejected:
{reasons}
Write a new exercise that follows every rule. Output the JSON object only."""


def describe_code(code: str) -> str:
    """A misconception code in words, for the prompt: "PART:에서>에" ->
    "'에' written where '에서' is needed"."""
    _, _, change = code.partition(":")
    correct, _, wrong = change.partition(">")
    correct, wrong = correct.replace("+", ""), wrong.replace("+", "")
    if not wrong:
        return f"'{correct}' left out"
    if not correct:
        return f"'{wrong}' added where nothing belongs"
    return f"'{wrong}' written where '{correct}' is needed"


def _blank(sentence: str, wrongs: list[str]) -> tuple[int, int] | None:
    """(start, end) of the blank in `sentence`: the span where the wrong
    sentences differ from it, widened to whole morphemes, or to the whole
    word when a wrong sentence only leaves something out (no empty option).
    None if the span crosses a space (more than one word)."""
    p = min(next((i for i, (a, b) in enumerate(zip(sentence, w)) if a != b), min(len(sentence), len(w)))
            for w in wrongs)
    q = min(next((i for i, (a, b) in enumerate(zip(sentence[::-1], w[::-1])) if a != b),
                 min(len(sentence), len(w))) for w in wrongs)
    q = min(q, len(sentence) - p, *(len(w) - p for w in wrongs))
    start, end = p, len(sentence) - q
    for t in _KIWI.tokenize(sentence):  # widen to whole morphemes
        if t.start <= start < t.start + t.len:
            start = min(start, t.start)
        if start < end and t.start < end <= t.start + t.len:
            end = max(end, t.start + t.len)
    if start == end or any(len(w) - (len(sentence) - end) == start for w in wrongs):
        start = sentence.rfind(" ", 0, start) + 1  # an option would be empty: blank the word
        end = (sentence + " ").find(" ", end)
    while end > start + 1 and sentence[end - 1] in ".?!," and all(
            w[len(w) - (len(sentence) - end) - 1] == sentence[end - 1] for w in wrongs):
        end -= 1  # keep sentence punctuation out of the options
    if " " in sentence[start:end].strip():
        return None
    return start, end


class MisconceptionItemGenerator:
    """Implements daily_drill.ContentProvider's grammar half."""

    def __init__(self, chat: Chat, sentence_logprob: Scorer | None, seed: int | None = None):
        self.chat = chat
        self.sentence_logprob = sentence_logprob
        self.rng = random.Random(seed)
        self.items: dict[str, GeneratedItem] = {}
        self.stats: dict[str, FoundryStats] = {}
        self._recent: dict[str, list[str]] = {}
        self._example: dict[str, str] = {}
        self._counter = 0

    # -- verification ------------------------------------------------------
    def verify(self, skill_id: str, reply: dict, target_code: str | None = None):
        """(item fields, []) if the reply passes every check, else (None, reasons)."""
        skill = SKILLS[skill_id]
        if not isinstance(reply, dict):
            return None, ["format: the reply was not a JSON object"]
        sentence, wrongs_in = reply.get("sentence"), reply.get("wrong_sentences")
        if not (isinstance(sentence, str) and isinstance(wrongs_in, list)):
            return None, ['format: needs a "sentence" and a list of "wrong_sentences"']
        clean = lambda t: korean_only(re.sub(r"[\[\]*]", "", t).strip()) if isinstance(t, str) else None
        sentence = clean(sentence)
        if not sentence:
            return None, ["format: the sentence must be Korean only, without romanisation or translation"]
        if len(wrongs_in) < N_DISTRACTORS:
            return None, [f"format: {N_DISTRACTORS} wrong sentences are needed, got {len(wrongs_in)}"]

        reasons, wrongs, codes = [], [], []
        for w in wrongs_in:
            wrong = clean(w)
            if not wrong or wrong == sentence or wrong in wrongs:
                reasons.append(f"duplicate: wrong sentence '{w}' is empty, the correct sentence, or a repeat")
                continue
            edit, why = single_edit(skill, sentence, wrong)
            if edit is None:
                tag, _, text = why.partition(": ")
                reasons.append(f"{tag}: wrong sentence '{wrong}' -- {text}")
                continue
            if edit.code in codes:
                reasons.append(f"distinct: '{wrong}' makes the same mistake as another wrong sentence "
                               f"({describe_code(edit.code)})")
                continue
            wrongs.append(wrong)
            codes.append(edit.code)
        if target_code is not None and target_code not in codes:
            reasons.append(f"target-misconception: no wrong sentence has {describe_code(target_code)}")
        if len(codes) < N_DISTRACTORS or (target_code is not None and target_code not in codes):
            return None, reasons
        # More valid wrong sentences than needed: keep the requested one and the first others.
        keep = sorted(range(len(codes)), key=lambda i: (codes[i] != target_code, i))[:N_DISTRACTORS]
        wrongs, codes = [wrongs[i] for i in sorted(keep)], [codes[i] for i in sorted(keep)]
        span = _blank(sentence, wrongs)
        if span is None:
            return None, ["target: the mistakes must all be in the same word of the sentence"]
        before, target, after = sentence[:span[0]], sentence[span[0]:span[1]], sentence[span[1]:]
        texts = [w[len(before):len(w) - len(after)] for w in wrongs]
        reasons = []

        if skill.logprob_gate and self.sentence_logprob is not None:
            key_score = self.sentence_logprob(sentence)
            for wrong, text in zip(wrongs, texts):
                if self.sentence_logprob(wrong) >= key_score:
                    reasons.append(f"ambiguous: '{wrong}' reads at least as naturally as the correct "
                                   f"sentence; it may also be correct")
            if reasons:
                return None, reasons
        return (before, target, after, texts, codes), []

    # -- generation --------------------------------------------------------
    def _prompt(self, skill_id: str, target_code: str | None) -> str:
        skill = SKILLS[skill_id]
        target_line = ""
        if target_code is not None:
            target_line = f"\nOne of the three wrong sentences MUST make this mistake: {describe_code(target_code)}."
        recent = self._recent.get(skill_id, [])[-8:]
        avoid = f" Use a different sentence from these: {'; '.join(recent)}" if recent else ""
        example = self._example.get(skill_id)
        note = ("A good exercise for this grammar point:" if example else
                "Example of the output shape only (it tests a different grammar point):")
        return _ITEM_PROMPT.format(description=skill.description, target_line=target_line,
                                   avoid=avoid, example_note=note, example=example or _FORMAT_EXAMPLE)

    def _system(self, skill_id: str) -> str:
        return _SYSTEM_PROMPT.format(level=SKILLS[skill_id].level)

    def generate(self, skill_id: str, target_code: str | None = None) -> GeneratedItem | None:
        if skill_id not in SKILLS:
            return None
        stats = self.stats.setdefault(skill_id, FoundryStats())
        stats.attempts += 1
        prompt = [{"role": "system", "content": self._system(skill_id)},
                  {"role": "user", "content": self._prompt(skill_id, target_code)}]
        messages = prompt
        for revision in range(MAX_REVISIONS + 1):
            reply_text = self.chat(messages)
            fields, reasons = self.verify(skill_id, extract_json(reply_text), target_code)
            if fields is not None:
                stats.accepted += 1
                stats.revisions_used += revision
                return self._assemble(skill_id, *fields)
            for reason in reasons:
                stats.reject(reason)
            if len(stats.rejected) < 10:
                stats.rejected.append({"reply": reply_text[:500], "reasons": reasons})
            # Only the latest reply is sent back, so the context stays short.
            messages = prompt + [{"role": "assistant", "content": reply_text},
                                 {"role": "user", "content": _REVISION_PROMPT.format(
                                     reasons="\n".join(f"- {r}" for r in reasons))}]
        return None

    def _assemble(self, skill_id, before, target, after, texts, codes) -> GeneratedItem:
        labels = ["A", "B", "C", "D"]
        self.rng.shuffle(labels)
        options = {labels[0]: target, **dict(zip(labels[1:], texts))}
        option_codes = {labels[0]: None, **dict(zip(labels[1:], codes))}
        options = dict(sorted(options.items()))
        option_codes = dict(sorted(option_codes.items()))
        stem = before + BLANK + after
        self._recent.setdefault(skill_id, []).append(before + target + after)
        self._example[skill_id] = json.dumps(
            {"sentence": before + target + after, "wrong_sentences": [before + t + after for t in texts]},
            ensure_ascii=False)
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
        self._torch.cuda.empty_cache()  # shares the GPU with the checker
        return text


if __name__ == "__main__":
    # CPU smoke test: the pipeline logic with stand-in models.
    assert extract_json('Sure! ```json\n{"a": [1, {"b": 2}]}\n``` done') == {"a": [1, {"b": 2}]}
    assert extract_json("[1, 2] trailing") == [1, 2]
    assert extract_json("no json") is None
    assert extract_json('```json\n[{"a": "x]y"}, {"b": 2},]\n```') == [{"a": "x]y"}, {"b": 2}]
    assert extract_json('{"a": "x", // note\n"b": 1}') == {"a": "x", "b": 1}
    assert korean_only("공원에서 산책했어요. (Gongwon-eseo ...) - I walked.") == "공원에서 산책했어요."
    assert korean_only("저는 저녁을 먹었어요, ate.") is None
    assert describe_code("PART:에서>에") == "'에' written where '에서' is needed"
    assert describe_code("END:었>") == "'었' left out"

    W = lambda *ps: ["동생이 공원" + p + " 축구를 해요." for p in ps]
    good = {"sentence": "동생이 공원에서 축구를 해요.", "wrong_sentences": W("에", "을", "이")}
    repeated = {**good, "wrong_sentences": W("에", "에", "이")}
    replies = iter([json.dumps(repeated, ensure_ascii=False), json.dumps(good, ensure_ascii=False)])
    seen_prompts = []

    def chat(messages):
        seen_prompts.append(messages[-1]["content"])
        return next(replies)

    # A scorer that prefers the key sentence: stands in for Qwen3-4B log-probs.
    gen = MisconceptionItemGenerator(chat, lambda s: 0.0 if "공원에서" in s else -5.0, seed=0)
    item_id = gen.next_new_grammar_item("particle_loc", target_misconception="PART:에서>이")
    assert item_id is not None, gen.stats
    item = gen.items[item_id]
    # The blank is found from where the wrong sentences differ, not marked by the generator.
    assert item.stem == "동생이 공원___ 축구를 해요." and item.options[item.answer] == "에서", item
    # Codes come from the annotator, so the same error gets the same code in any item.
    assert sorted(c for c in item.option_codes.values() if c) == ["PART:에서>에", "PART:에서>을", "PART:에서>이"]
    assert item.option_codes[item.answer] is None
    assert "MUST make this mistake: '이' written where '에서' is needed" in seen_prompts[0]
    assert "duplicate:" in seen_prompts[1], "the critic must be told exactly why the first reply failed"
    assert gen.stats["particle_loc"].rejected and gen.stats["particle_loc"].rejected[0]["reasons"]
    assert gen.stats["particle_loc"].revisions_used == 1

    plain = MisconceptionItemGenerator(chat, None)
    # All three start with 에 (에, 에게, 에다): the blank still covers the whole particle.
    fields, reasons = plain.verify("particle_loc", {**good, "wrong_sentences": W("에", "에게", "을")})
    assert fields is not None and fields[1] == "에서" and fields[3] == ["에", "에게", "을"], (fields, reasons)
    # A wrong sentence that only leaves something out blanks the whole word (no empty option).
    past = {"sentence": "어제 영화를 봤어요.", "wrong_sentences": ["어제 영화를 봐요.", "어제 영화를 보았어요요.",
                                                                "어제 영화를 볼 거예요."]}
    fields, reasons = plain.verify("tense_past", past)
    assert fields is None or all(fields[3]), (fields, reasons)
    # Mistakes in different words cannot share one blank.
    fields, reasons = plain.verify("particle_loc", {"sentence": "동생이 공원에서 축구를 해요.", "wrong_sentences": [
        "동생이 공원에 축구를 해요.", "동생이 공원을 축구를 해요.", "동생이 공원에서 축구가 해요."]})
    assert fields is None, fields
    # Brackets or markdown the generator adds anyway are ignored.
    fields, _ = plain.verify("particle_loc", {**good, "sentence": "동생이 공원[에서] 축구를 해요."})
    assert fields is not None and fields[1] == "에서", fields
    # A fourth valid wrong sentence is tolerated; the requested one is kept.
    fields, reasons = plain.verify("particle_loc", {**good, "wrong_sentences": W("에", "을", "도", "이")},
                                   target_code="PART:에서>이")
    assert fields is not None and "PART:에서>이" in fields[4] and len(fields[4]) == 3, (fields, reasons)
    # A wrong sentence that changes something else as well is rejected.
    _, reasons = plain.verify("particle_loc", {**good, "wrong_sentences": W("에", "을") + ["형이 공원에서 야구를 해요."]})
    assert any(r.startswith("not-minimal") for r in reasons), reasons

    # Two wrong sentences that make the same mistake (allomorphs 을/를) are one misconception.
    _, reasons = gen.verify("particle_loc", {**good, "wrong_sentences": W("을", "를", "이")})
    assert any(r.startswith("distinct") for r in reasons), reasons

    # The requested misconception must be among the wrong sentences.
    _, reasons = gen.verify("particle_loc", good, target_code="PART:에서>도")
    assert any(r.startswith("target-misconception") for r in reasons), reasons

    # Ambiguity is caught by scoring: a wrong sentence that reads as naturally as the key.
    flat = MisconceptionItemGenerator(lambda m: json.dumps(good, ensure_ascii=False), lambda s: 0.0, seed=0)
    assert flat.generate("particle_loc") is None
    assert flat.stats["particle_loc"].reasons.get("ambiguous", 0) > 0

    # A wrong sentence that edits something other than the tested category is rejected.
    _, reasons = gen.verify("particle_loc", {**good, "wrong_sentences": W("에", "을") + ["동생이 공원에서 축구를 해."]})
    assert any(r.startswith("wrong-category") for r in reasons), reasons

    # Open-class skills get the lexical edit as their code.
    verb = {"sentence": "저는 물을 마셔요.", "wrong_sentences": ["저는 물을 먹어요.", "저는 물을 봐요.", "저는 물을 입어요."]}
    fields, reasons = MisconceptionItemGenerator(chat, None).verify("verb_choice", verb)
    assert fields is not None and "VERB:마시>먹" in fields[4], (fields, reasons)
    print("all smoke tests passed")
