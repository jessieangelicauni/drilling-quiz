"""
Item foundry for Novelty 1 (PRD §9.69, §9.74): Korean grammar drill items
whose every distractor realises one NAMED error, so that the option a
learner picks says which misconception they hold (misconception_tracker.py),
not only that they were wrong.

The LLM writes the content; code only VERIFIES it and LABELS it:

  1. Item (MisconceptionItemGenerator.generate): EXAONE writes one natural
     sentence using the grammar point, marks the tested span, and writes
     three wrong versions of that span, each a different error a learner
     typically makes -- one of them, when asked, the error the learner
     currently makes most (personalisation).
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

from kagas_annotator import annotate

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


def single_edit(skill: Skill, correct: str, wrong: str):
    """The one edit turning `correct` into `wrong`, or (None, reason)."""
    edits = annotate(correct, wrong)
    if len(edits) != 1:
        return None, f"not-minimal: {len(edits)} edits"
    edit = edits[0]
    if edit.category not in skill.categories:
        return None, f"wrong-category: {edit.category}, skill tests {sorted(skill.categories)}"
    return edit, ""


# ------------------------------------------------------------------ items

_ITEM_PROMPT = """You write Korean grammar drill questions for {level} learners whose first language is Indonesian.
Grammar point: {description}.

Write ONE new, natural, everyday Korean sentence that uses the grammar point, in Hangul only, and put
square brackets around the ONE part that tests this grammar point -- as short as possible, e.g. a
particle or a word ending, or the one word that carries it.
Then give {n} "distractors": wrong replacements for the bracketed part, each a DIFFERENT mistake
learners typically make with this grammar point, changing nothing else, and each clearly wrong in
this sentence. Write only the replacement text, not the whole sentence.{target_line}
Do not reuse these sentences: {avoid}

Answer with the JSON only, no explanation. {example_note}
{example}"""

# The format example until a skill has an accepted item of its own; after
# that, the skill's latest accepted item is the example (on-skill, not fixed).
_FORMAT_EXAMPLE = '{"sentence": "동생이 공원[에서] 축구를 해요.", "distractors": ["에", "을", "이"]}'

_REVISION_PROMPT = """That question cannot be used, for these reasons:
{reasons}
Write a corrected version that fixes every reason. Answer with JSON only, in the same format."""


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


def _wrong_sentence(distractor: str, before: str, target: str, after: str) -> str | None:
    """The full wrong sentence a distractor stands for. Generators answer in
    three shapes, all accepted: the replacement for the bracketed part
    ("에서"), the whole word that contains it ("카페에서" for 카페[에]), or the
    whole wrong sentence. Whether the result is a minimal edit of the key is
    then checked by the annotator, as for any distractor."""
    d = korean_only(distractor.replace("[", "").replace("]", ""))
    if not d:
        return None
    sentence = before + target + after
    if len(d) >= len(sentence) / 2:
        return d
    prefix = before.rsplit(" ", 1)[-1]           # start of the bracketed word
    suffix = after.split(" ", 1)[0]               # its end
    if prefix and d.startswith(prefix):
        return before[:len(before) - len(prefix)] + d + after[len(suffix):]
    return before + d + after


def _shared_blank(sentence: str, wrongs: list[str], before: str, after: str):
    """(before, after) of the blank every distractor fills: the bracketed
    span if every wrong sentence differs only there, else the whole word
    around it; None if a distractor changes the sentence elsewhere."""
    word_before = before[:len(before) - len(before.rsplit(" ", 1)[-1])]
    word_after = after[len(after.split(" ", 1)[0]):]
    for b, a in ((before, after), (word_before, word_after)):
        if all(w.startswith(b) and w.endswith(a) and len(w) > len(b) + len(a) - 1 for w in wrongs):
            if len(sentence) >= len(b) + len(a):
                return b, a
    return None


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
        marked, distractors = reply.get("sentence"), reply.get("distractors")
        if not (isinstance(marked, str) and isinstance(distractors, list)):
            return None, ["format: needs a sentence and a list of distractors"]
        marked = korean_only(marked.replace("*", ""))   # **[께]** markdown
        if marked is None:
            return None, ["format: the sentence must be Korean only, without romanisation or translation"]
        spans = re.findall(r"\[([^\[\]]*)\]", marked)
        if len(spans) != 1 or not spans[0].strip():
            return None, ["target: put square brackets around exactly one part of the sentence"]
        target = spans[0]
        before, after = marked.split(f"[{target}]", 1)
        if not before.strip() or before.rstrip()[-1] in ".?!":
            return None, ["target: the brackets must be inside the sentence, not after it"]
        sentence = before + target + after
        if len(distractors) < N_DISTRACTORS:
            return None, [f"format: {N_DISTRACTORS} distractors are needed, got {len(distractors)}"]

        reasons, wrongs, codes = [], [], []
        for d in distractors:
            wrong = _wrong_sentence(d, before, target, after) if isinstance(d, str) else None
            if wrong is None or wrong == sentence or wrong in wrongs:
                reasons.append(f"duplicate: distractor '{d}' is empty, the key, or a repeat")
                continue
            edit, why = single_edit(skill, sentence, wrong)
            if edit is None:
                reasons.append(f"{why.split(':')[0]}: '{wrong}' {why}")
                continue
            if edit.code in codes:
                reasons.append(f"distinct: '{wrong}' makes the same mistake as another distractor "
                               f"({describe_code(edit.code)})")
                continue
            wrongs.append(wrong)
            codes.append(edit.code)
        if target_code is not None and target_code not in codes:
            reasons.append(f"target-misconception: no distractor has {describe_code(target_code)}")
        if len(codes) < N_DISTRACTORS or (target_code is not None and target_code not in codes):
            return None, reasons
        # More valid distractors than needed: keep the requested one and the first others.
        keep = sorted(range(len(codes)), key=lambda i: (codes[i] != target_code, i))[:N_DISTRACTORS]
        wrongs, codes = [wrongs[i] for i in sorted(keep)], [codes[i] for i in sorted(keep)]
        blank = _shared_blank(sentence, wrongs, before, after)
        if blank is None:
            return None, ["target: each distractor must change only the bracketed part"]
        before, after = blank
        target = sentence[len(before):len(sentence) - len(after)]
        texts = [w[len(before):len(w) - len(after)] for w in wrongs]
        if any(" " in t.strip() for t in [target, *texts]):
            return None, ["target: the tested part must be one word or less, not a phrase"]
        reasons = []

        if skill.logprob_gate and self.sentence_logprob is not None:
            key_score = self.sentence_logprob(sentence)
            for text in texts:
                if self.sentence_logprob(before + text + after) >= key_score:
                    reasons.append(f"ambiguous: '{text}' reads at least as naturally as the key "
                                   f"'{target}'; it may also be correct")
            if reasons:
                return None, reasons
        return (before, target, after, texts, codes), []

    # -- generation --------------------------------------------------------
    def _prompt(self, skill_id: str, target_code: str | None) -> str:
        skill = SKILLS[skill_id]
        target_line = ""
        if target_code is not None:
            target_line = f"\n  One distractor MUST be this mistake: {describe_code(target_code)}."
        avoid = "; ".join(self._recent.get(skill_id, [])[-8:]) or "(none yet)"
        example = self._example.get(skill_id)
        note = ("An earlier accepted answer for this grammar point:" if example else
                "Format example (a different grammar point may be tested; follow only its shape):")
        return _ITEM_PROMPT.format(level=skill.level, description=skill.description,
                                   n=N_DISTRACTORS, target_line=target_line, avoid=avoid,
                                   example_note=note, example=example or _FORMAT_EXAMPLE)

    def generate(self, skill_id: str, target_code: str | None = None) -> GeneratedItem | None:
        if skill_id not in SKILLS:
            return None
        stats = self.stats.setdefault(skill_id, FoundryStats())
        stats.attempts += 1
        prompt = [{"role": "user", "content": self._prompt(skill_id, target_code)}]
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
        self._example[skill_id] = json.dumps({"sentence": f"{before}[{target}]{after}", "distractors": texts},
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
    """An instruct model, 4-bit (PRD §9.53), as a Chat callable: by default
    EXAONE-3.5-7.8B-Instruct, or any GENERATORS entry. Sampled, not greedy,
    so repeated calls give different sentences."""

    MODEL = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
    GENERATORS = {"exaone-3.5-7.8b": MODEL, "gemma-3-12b": "google/gemma-3-12b-it"}

    def __init__(self, model_name: str = MODEL, temperature: float = 0.6, max_new_tokens: int = 400):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self._torch = torch
        self.model_name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, device_map="cuda",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16))
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def __call__(self, messages: list[dict]) -> str:
        if "exaone" in self.model_name.lower():  # EXAONE's recommended system prompt
            messages = [{"role": "system", "content": "You are EXAONE model from LG AI Research, "
                                                      "a helpful assistant."}] + messages
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

    good = {"sentence": "동생이 공원[에서] 축구를 해요.", "distractors": ["에", "을", "이"]}
    repeated = {**good, "distractors": ["에", "에", "이"]}
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
    assert item.stem == "동생이 공원___ 축구를 해요." and item.options[item.answer] == "에서"
    # Codes come from the annotator, so the same error gets the same code in any item.
    assert sorted(c for c in item.option_codes.values() if c) == ["PART:에서>에", "PART:에서>을", "PART:에서>이"]
    assert item.option_codes[item.answer] is None
    assert "MUST be this mistake: '이' written where '에서' is needed" in seen_prompts[0]
    assert "duplicate:" in seen_prompts[1], "the critic must be told exactly why the first reply failed"
    assert gen.stats["particle_loc"].rejected and gen.stats["particle_loc"].rejected[0]["reasons"]
    assert gen.stats["particle_loc"].revisions_used == 1

    # The bracket fixes the position even when the target also occurs elsewhere (에 in 에서).
    fields, _ = MisconceptionItemGenerator(chat, None).verify(
        "particle_loc", {"sentence": "학교에서 집[에] 가요.", "distractors": ["에서", "을", "이"]})
    assert fields is not None and fields[0] == "학교에서 집" and fields[1] == "에", fields
    _, reasons = gen.verify("particle_loc", {"sentence": "동생이 공원에서 축구를 해요.", "distractors": ["에", "을", "이"]})
    assert reasons and reasons[0].startswith("target"), reasons

    # The three reply shapes seen from EXAONE (PRD §9.76): replacement, whole word, whole sentence.
    plain = MisconceptionItemGenerator(chat, None)
    shapes = {"sentence": "친구가 카페[에서] 책을 읽어요.",
              "distractors": ["카페에", "친구가 카페[를] 책을 읽어요.", "가"]}
    fields, reasons = plain.verify("particle_loc", shapes)
    assert fields is not None, reasons
    before, target, after, texts, codes = fields
    # All three differ only in the particle, so the blank stays the bracketed span.
    assert (before, target, after) == ("친구가 카페", "에서", " 책을 읽어요."), fields
    assert texts == ["에", "를", "가"] and codes == ["PART:에서>에", "PART:에서>을", "PART:에서>이"]
    # A word-level distractor that changes the noun too widens the blank to the whole word.
    fields, _ = plain.verify("noun_counter", {"sentence": "사과 세 [개]를 샀어요.",
                                              "distractors": ["명", "권을", "마리"]})
    assert fields is None or fields[1] in ("개", "개를"), fields
    # A fourth valid distractor is tolerated; the requested one is kept.
    fields, reasons = plain.verify("particle_loc", {**good, "distractors": ["에", "을", "도", "이"]},
                                   target_code="PART:에서>이")
    assert fields is not None and "PART:에서>이" in fields[4] and len(fields[4]) == 3, (fields, reasons)
    # A whole sentence that changes something else is still rejected.
    _, reasons = plain.verify("particle_loc", {**good, "distractors": ["에", "을", "형이 공원에서 야구를 해요."]})
    assert any(r.startswith("not-minimal") for r in reasons), reasons

    # Two distractors that make the same mistake (allomorphs 을/를) are one misconception.
    _, reasons = gen.verify("particle_loc", {**good, "distractors": ["을", "를", "이"]})
    assert any(r.startswith("distinct") for r in reasons), reasons

    # The requested misconception must be among the distractors.
    _, reasons = gen.verify("particle_loc", good, target_code="PART:에서>도")
    assert any(r.startswith("target-misconception") for r in reasons), reasons

    # Ambiguity is caught by scoring: a distractor that reads as naturally as the key.
    flat = MisconceptionItemGenerator(lambda m: json.dumps(good, ensure_ascii=False), lambda s: 0.0, seed=0)
    assert flat.generate("particle_loc") is None
    assert flat.stats["particle_loc"].reasons.get("ambiguous", 0) > 0

    # A distractor that edits something other than the tested category is rejected.
    _, reasons = gen.verify("particle_loc", {**good, "distractors": ["에", "을", "에서 축구"]})
    assert any(r.startswith("wrong-category") for r in reasons), reasons

    # Open-class skills get the lexical edit as their code.
    verb = {"sentence": "저는 물을 [마셔요].", "distractors": ["먹어요", "봐요", "입어요"]}
    fields, reasons = MisconceptionItemGenerator(chat, None).verify("verb_choice", verb)
    assert fields is not None and "VERB:마시>먹" in fields[4], (fields, reasons)
    print("all smoke tests passed")
