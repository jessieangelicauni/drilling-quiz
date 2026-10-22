"""
Item foundry: Korean grammar drill items, one correct answer and three
wrong options, each with a short reasoning string explaining why it's
right or wrong.

Code VERIFIES what a model writes rather than trusting it:

  1. Item: for every skill but `subj_topic`, EXAONE writes only ONE natural
     correct sentence; the three wrong options are built DETERMINISTICALLY
     from that sentence's own morphology or a curated lexical pair
     (distractor_validator.deterministic_distractors_for_sentence) --
     validated at 77.3% item acceptance vs. 28.8% when a model wrote every
     distractor freely (it reproduces the exact broken pattern it is shown
     as "do not do this", verbatim). `subj_topic` has no such rule (a
     pedagogical convention, not a hard grammar rule) and keeps the older
     free-generation path: EXAONE writes one correct sentence and three
     wrong copies of it, each with a different error, and the blank is
     found from where they differ (_blank).
  2. Verification, none of it by asking a model to judge (Qwen3-4B could
     not judge acceptability reliably in any of three framings, PRD
     §9.16/§9.31): the free-generation path requires each wrong sentence to
     be a minimal pair of the key (exactly one edit, in the skill's KAGAS
     category, kagas_annotator.annotate) with the three edits distinct;
     every path requires each wrong form to be one coherent Korean word
     (distractor_validator.morphologically_broken) and, except for
     `subj_topic`, a strictly lower log-probability than the key under a
     different model family from the generator (BLiMP-style minimal pairs).
  3. Reasoning: once an item is assembled, one more model call explains
     each option in one sentence, quoting its own Korean form -- verified
     to actually mention that text, with a stock fallback if it doesn't, so
     authoring never blocks on this field.
  4. Critic (free-generation path only): a failed item is sent back to
     EXAONE with the exact reasons, for up to MAX_REVISIONS revisions.

Models are injected as plain callables (chat, sentence_logprob), so the
pipeline logic is tested on CPU below with stand-ins; ExaoneChat and
solver_check.SolverCheckProvider are the real ones.
"""
from __future__ import annotations

import itertools
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from distractor_validator import (
    DETERMINISTIC_SKILLS,
    deterministic_distractors_for_sentence,
    morphologically_broken,
    sentence_constraint,
)
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


@dataclass(frozen=True)
class Curriculum:
    """Admin-supplied scope material (e.g. "smart factory" course content):
    used as a reference so generated sentences stay inside the domain a
    course is actually teaching, rather than generic unrelated vocabulary.
    One curriculum applies across every skill -- set/replace it on a
    generator at any time (`gen.curriculum = ...`) when the admin uploads
    new material; `None` (the default) means no scope material at all,
    and generation is unaffected."""
    context: str
    modules: dict[str, list[str]]  # module name -> topic/description strings

    @staticmethod
    def from_json(path: str | Path) -> "Curriculum":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return Curriculum(context=data["context"], modules=data["modules"])

    def random_setting(self, rng: random.Random) -> str:
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
    # One short sentence per option label explaining why it's right or
    # wrong, written by the model after the item is assembled.
    reasoning: dict[str, str] = field(default_factory=dict)


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

_ITEM_PROMPT = """Grammar point: {description}.
{setting}
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

# The deterministic path's own, smaller prompt: EXAONE writes only the
# correct sentence (distractor_validator builds the wrong forms from it),
# so there is no wrong_sentences format to teach or a critic loop to run.
_SENTENCE_ONLY_PROMPT = """You write Korean sentences for Indonesian learners at {level} level.

Write ONE natural, everyday Korean sentence for the given grammar point.

Rules:
1. Hangul only: no romanisation, no translation, no brackets.
2. Output one JSON object and nothing else: no explanation, no comments, no markdown.

Output format:
{{"sentence": "..."}}"""

# The reasoning follow-up, run once per assembled item regardless of which
# path built it. Starts with "Exercise:" so a scripted chat stand-in in the
# smoke test below can tell it apart from every other kind of call.
_REASONING_PROMPT = """Exercise: {stem}
Options:
{options_list}
Correct answer: {answer}) {correct_text}

For EACH option, write ONE short sentence in Indonesian explaining why it is correct, or what
mistake it represents if it is wrong. Quote the option's own Korean form in your sentence.
Output one JSON object mapping each option's letter to its sentence, nothing else.

Output format:
{{"A": "...", "B": "...", ...}}"""


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

    def __init__(self, chat: Chat, sentence_logprob: Scorer | None, seed: int | None = None,
                 curriculum: Curriculum | None = None):
        self.chat = chat
        self.sentence_logprob = sentence_logprob
        self.rng = random.Random(seed)
        self.curriculum = curriculum
        self.items: dict[str, GeneratedItem] = {}
        self.stats: dict[str, FoundryStats] = {}
        self._recent: dict[str, list[str]] = {}
        self._example: dict[str, str] = {}
        self._counter = 0

    # -- verification (free-generation path: subj_topic only) --------------
    def verify(self, skill_id: str, reply: dict):
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
        if len(codes) < N_DISTRACTORS:
            return None, reasons
        wrongs = wrongs[:N_DISTRACTORS]  # more valid wrong sentences than needed: keep the first ones
        span = _blank(sentence, wrongs)
        if span is None:
            return None, ["target: the mistakes must all be in the same word of the sentence"]
        before, target, after = sentence[:span[0]], sentence[span[0]:span[1]], sentence[span[1]:]
        texts = [w[len(before):len(w) - len(after)] for w in wrongs]
        reasons = []

        # Found empirically (2026-10-22, tested with a real EXAONE-3.5-7.8B model, not the
        # scripted_chat smoke test): single_edit()'s "exactly one edit, right KAGAS category"
        # check does not catch a malformed span like two particles glued together (을에서,
        # 이에서) or a duplicated ending (습니다요, 으요) -- both were accepted by this function
        # before this check existed. morphologically_broken() is the same kiwipiepy check
        # benchmark_v2.py validated for exactly this failure class.
        #
        # Checked on the WHOLE WORD the blank sits in (adjacent word from `before`/`after` +
        # the candidate text), not the candidate text alone: _blank() can find a blank that is
        # only a trailing fragment of a word (e.g. "어요" after an already-complete "읽어요"),
        # and kiwipiepy cannot tell a bare fragment like "어어요" is broken in isolation -- it
        # only shows up as an EF-not-last violation once reattached to the real stem it follows
        # (found empirically: "어어요"/"요요" alone tokenize as harmless IC/JX; "읽어요어어요"/
        # "읽어요요요" correctly tokenize as broken).
        word_before = before.rsplit(" ", 1)[-1] if before else ""
        word_after = after.split(" ", 1)[0] if after else ""
        broken = [text for text in texts if morphologically_broken(word_before + text + word_after)]
        if broken:
            reasons.extend(f"malformed: '{word_before}{t}{word_after}' is not one coherent Korean word "
                           f"(kiwipiepy tokenization)" for t in broken)
            return None, reasons
        if morphologically_broken(word_before + target + word_after):
            reasons.append(f"malformed: the correct form '{word_before}{target}{word_after}' is not one "
                           f"coherent Korean word (kiwipiepy tokenization)")
            return None, reasons

        if skill.logprob_gate and self.sentence_logprob is not None:
            key_score = self.sentence_logprob(sentence)
            for wrong, text in zip(wrongs, texts):
                if self.sentence_logprob(wrong) >= key_score:
                    reasons.append(f"ambiguous: '{wrong}' reads at least as naturally as the correct "
                                   f"sentence; it may also be correct")
            if reasons:
                return None, reasons
        return (before, target, after, texts), []

    # -- generation: free path (subj_topic only) ----------------------------
    def _prompt(self, skill_id: str) -> str:
        skill = SKILLS[skill_id]
        recent = self._recent.get(skill_id, [])[-8:]
        avoid = f" Use a different sentence from these: {'; '.join(recent)}" if recent else ""
        example = self._example.get(skill_id)
        note = ("A good exercise for this grammar point:" if example else
                "Example of the output shape only (it tests a different grammar point):")
        setting = self.curriculum.random_setting(self.rng) if self.curriculum else ""
        return _ITEM_PROMPT.format(description=skill.description, setting=setting,
                                   avoid=avoid, example_note=note, example=example or _FORMAT_EXAMPLE)

    def _system(self, skill_id: str) -> str:
        return _SYSTEM_PROMPT.format(level=SKILLS[skill_id].level)

    def _generate_free(self, skill_id: str) -> GeneratedItem | None:
        stats = self.stats.setdefault(skill_id, FoundryStats())
        stats.attempts += 1
        prompt = [{"role": "system", "content": self._system(skill_id)},
                  {"role": "user", "content": self._prompt(skill_id)}]
        messages = prompt
        for revision in range(MAX_REVISIONS + 1):
            reply_text = self.chat(messages)
            fields, reasons = self.verify(skill_id, extract_json(reply_text))
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

    # -- generation: deterministic path (every skill but subj_topic) -------
    def _deterministic_item(self, skill_id: str):
        """(item, "") on success, else (None, reason) -- same shape as
        verify(), so FoundryStats stays as diagnosable as the free-
        generation path."""
        skill = SKILLS[skill_id]
        recent = self._recent.get(skill_id, [])[-8:]
        avoid = f" Use a different sentence from these: {'; '.join(recent)}" if recent else ""
        description = sentence_constraint(skill_id, skill.description, self.rng)
        setting = f"\n{self.curriculum.random_setting(self.rng)}" if self.curriculum else ""
        messages = [{"role": "system", "content": _SENTENCE_ONLY_PROMPT.format(level=skill.level)},
                    {"role": "user", "content": f"Grammar point: {description}{setting}\n"
                                                f"Write one sentence.{avoid}"}]
        reply = extract_json(self.chat(messages))
        if not isinstance(reply, dict) or not isinstance(reply.get("sentence"), str):
            return None, 'format: the reply was not a JSON object with a "sentence" string'
        sentence = korean_only(re.sub(r"[\[\]*]", "", reply["sentence"]).strip())
        if not sentence:
            return None, "format: the sentence must be Korean only, without romanisation or translation"
        located = deterministic_distractors_for_sentence(skill_id, sentence, self.rng)
        if located is None:
            return None, f"target: no word of '{sentence}' is a usable answer for this skill"
        start, end, wrong_texts, _mistakes = located
        before, target, after = sentence[:start], sentence[start:end], sentence[end:]
        word_before = before.rsplit(" ", 1)[-1] if before else ""
        word_after = after.split(" ", 1)[0] if after else ""
        if morphologically_broken(word_before + target + word_after):
            return None, (f"malformed: the correct form '{word_before}{target}{word_after}' is not one "
                          f"coherent Korean word (kiwipiepy tokenization)")
        broken = [t for t in wrong_texts if morphologically_broken(word_before + t + word_after)]
        if broken:
            return None, (f"malformed: '{word_before}{broken[0]}{word_after}' is not one coherent Korean "
                          f"word (kiwipiepy tokenization)")
        if skill.logprob_gate and self.sentence_logprob is not None:
            key_score = self.sentence_logprob(sentence)
            for t in wrong_texts:
                if self.sentence_logprob(before + t + after) >= key_score:
                    return None, (f"ambiguous: '{before}{t}{after}' reads at least as naturally as the "
                                  f"correct sentence; it may also be correct")
        return self._assemble(skill_id, before, target, after, wrong_texts), ""

    def _generate_deterministic(self, skill_id: str) -> GeneratedItem | None:
        stats = self.stats.setdefault(skill_id, FoundryStats())
        stats.attempts += 1
        for revision in range(MAX_REVISIONS + 1):
            item, reason = self._deterministic_item(skill_id)
            if item is not None:
                stats.accepted += 1
                stats.revisions_used += revision
                return item
            stats.reject(reason)
            if len(stats.rejected) < 10:
                stats.rejected.append({"reply": "", "reasons": [reason]})
        return None

    def generate(self, skill_id: str) -> GeneratedItem | None:
        if skill_id not in SKILLS:
            return None
        if skill_id in DETERMINISTIC_SKILLS:
            return self._generate_deterministic(skill_id)
        return self._generate_free(skill_id)

    # -- reasoning (every path) ---------------------------------------------
    def _reasoning(self, stem: str, options: dict[str, str], answer: str) -> dict[str, str]:
        options_list = "\n".join(f"{label}) {text}" for label, text in options.items())
        messages = [{"role": "user", "content": _REASONING_PROMPT.format(
            stem=stem, options_list=options_list, answer=answer, correct_text=options[answer])}]
        for _ in range(MAX_REVISIONS + 1):
            reply = extract_json(self.chat(messages))
            if isinstance(reply, dict) and self._reasoning_plausible(options, reply):
                return {label: str(reply[label]) for label in options}
        # Never block item authoring on this field: fall back to a minimal
        # stock explanation if the model keeps failing to produce one that
        # actually mentions its own option's text.
        return {label: f"'{text}' is {'correct' if label == answer else 'incorrect'} here."
                for label, text in options.items()}

    @staticmethod
    def _reasoning_plausible(options: dict[str, str], reply: dict) -> bool:
        return set(reply) == set(options) and all(
            isinstance(reply[label], str) and options[label] in reply[label] for label in options)

    def _assemble(self, skill_id, before, target, after, texts) -> GeneratedItem:
        labels = ["A", "B", "C", "D"]
        self.rng.shuffle(labels)
        options = {labels[0]: target, **dict(zip(labels[1:], texts))}
        options = dict(sorted(options.items()))
        answer = labels[0]
        stem = before + BLANK + after
        self._recent.setdefault(skill_id, []).append(before + target + after)
        self._example[skill_id] = json.dumps(
            {"sentence": before + target + after, "wrong_sentences": [before + t + after for t in texts]},
            ensure_ascii=False)
        reasoning = self._reasoning(stem, options, answer)
        return GeneratedItem(skill_id, stem, BLANK, options, answer, reasoning)

    def next_new_grammar_item(self, skill_id: str, learner_urn=None,
                              exclude: set[str] = frozenset()) -> str | None:
        item = self.generate(skill_id)
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


def _echo_reasoning(messages: list[dict]) -> str:
    """A chat stand-in for the reasoning follow-up call only: echoes each
    option's own text back as its 'reason', parsed straight out of the
    prompt -- so a test doesn't need to know the random label shuffle in
    advance to supply a reply that will pass _reasoning_plausible."""
    content = messages[-1]["content"]
    labels_texts = re.findall(r"^([A-D])\) (.+)$", content, re.MULTILINE)
    return json.dumps({label: f"'{text}' is the point being tested here." for label, text in labels_texts},
                      ensure_ascii=False)


def _make_chat(*replies):
    """A chat stand-in: answers the reasoning follow-up generically (see
    _echo_reasoning above), and returns each of `replies` (JSON-encoded) in
    turn for every other call, repeating the last one once exhausted (a
    retry loop may call this more times than a test cares to script).
    `chat.seen` collects the content of each non-reasoning call, so a test
    can inspect what the model was told."""
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

    # -- deterministic path: particle_loc writes only the sentence, the
    # three wrong options are built by distractor_validator, not the model.
    chat = _make_chat({"sentence": "동생이 공원에서 축구를 해요."})
    gen = MisconceptionItemGenerator(chat, lambda s: 0.0 if "공원에서" in s else -5.0, seed=0)
    item_id = gen.next_new_grammar_item("particle_loc")
    assert item_id is not None, gen.stats
    item = gen.items[item_id]
    # The deterministic path blanks the WHOLE word (noun+particle), same as benchmark_v2.py's
    # validated approach -- not just the bare particle the old free-generation path isolated.
    assert item.stem == "동생이 ___ 축구를 해요." and item.options[item.answer] == "공원에서", item
    assert set(item.reasoning) == set(item.options), item
    assert all(item.options[label] in item.reasoning[label] for label in item.options), item.reasoning
    assert gen.stats["particle_loc"].accepted == 1 and gen.stats["particle_loc"].attempts == 1

    # Curriculum (admin-supplied scope material): the chosen setting line must actually
    # reach the model's prompt, for both the deterministic and the free-generation path.
    curriculum = Curriculum(context="a smart factory course", modules={
        "Module 1": ["predictive maintenance using sensor data"]})
    cur_chat = _make_chat({"sentence": "동생이 공원에서 축구를 해요."})
    cur_gen = MisconceptionItemGenerator(cur_chat, lambda s: 0.0 if "공원에서" in s else -5.0,
                                         seed=0, curriculum=curriculum)
    assert cur_gen.next_new_grammar_item("particle_loc") is not None, cur_gen.stats
    assert any("a smart factory course" in p and "predictive maintenance" in p for p in cur_chat.seen), (
        "the curriculum's setting line must reach the deterministic path's sentence prompt")
    subj_cur_chat = _make_chat({"sentence": "동생이 공원에서 축구를 해요.",
                                "wrong_sentences": ["동생이 공원에 축구를 해요.", "동생이 공원을 축구를 해요.",
                                                     "동생이 공원이 축구를 해요."]})
    subj_cur_gen = MisconceptionItemGenerator(subj_cur_chat, None, seed=0, curriculum=curriculum)
    subj_cur_gen.generate("subj_topic")
    assert any("a smart factory course" in p for p in subj_cur_chat.seen), (
        "the curriculum's setting line must reach the free-generation path's prompt too")
    no_curriculum_prompt = MisconceptionItemGenerator(lambda m: "", None)._prompt("subj_topic")
    assert "Setting:" not in no_curriculum_prompt, "no curriculum set must mean no setting line at all"

    # Ambiguity is caught by scoring: every wrong form reads as naturally as the key under
    # this scorer, so every retry fails and the attempt is given up on, not shipped anyway.
    flat = MisconceptionItemGenerator(_make_chat({"sentence": "동생이 공원에서 축구를 해요."}), lambda s: 0.0, seed=0)
    assert flat.generate("particle_loc") is None
    assert flat.stats["particle_loc"].reasons.get("ambiguous", 0) > 0

    # A sentence with no word this skill's builder can use is retried, then given up on.
    no_particle = MisconceptionItemGenerator(_make_chat({"sentence": "안녕하세요."}), None, seed=0)
    assert no_particle.generate("particle_loc") is None
    assert no_particle.stats["particle_loc"].reasons.get("target", 0) > 0

    # -- deterministic path: a lexical (vocabulary) skill, same mechanism.
    verb_chat = _make_chat({"sentence": "저는 데이터를 분석해요."})
    verb_gen = MisconceptionItemGenerator(verb_chat, None, seed=0)
    verb_item_id = verb_gen.next_new_grammar_item("verb_choice")
    assert verb_item_id is not None, verb_gen.stats
    verb_item = verb_gen.items[verb_item_id]
    assert verb_item.options[verb_item.answer] == "분석해요", verb_item
    assert len(set(verb_item.options.values())) == 4, "the 3 wrong verbs and the key must all differ"

    # -- free-generation path: subj_topic only, no deterministic rule exists for it.
    W = lambda *ps: ["동생이 공원" + p + " 축구를 해요." for p in ps]
    good = {"sentence": "동생이 공원에서 축구를 해요.", "wrong_sentences": W("에", "을", "이")}
    repeated = {**good, "wrong_sentences": W("에", "에", "이")}
    subj_chat = _make_chat(repeated, good)
    subj_gen = MisconceptionItemGenerator(subj_chat, None, seed=0)  # subj_topic: logprob_gate=False
    item_id = subj_gen.next_new_grammar_item("subj_topic")
    assert item_id is not None, subj_gen.stats
    item = subj_gen.items[item_id]
    # The blank is found from where the wrong sentences differ, not marked by the generator.
    assert item.stem == "동생이 공원___ 축구를 해요." and item.options[item.answer] == "에서", item
    assert set(item.reasoning) == set(item.options), item
    assert "duplicate:" in subj_chat.seen[1], "the critic must be told exactly why the first reply failed"
    assert subj_gen.stats["subj_topic"].rejected and subj_gen.stats["subj_topic"].rejected[0]["reasons"]
    assert subj_gen.stats["subj_topic"].revisions_used == 1

    plain = MisconceptionItemGenerator(_make_chat(good), None)
    # All three start with 에 (에, 에게, 에다): the blank still covers the whole particle.
    fields, reasons = plain.verify("subj_topic", {**good, "wrong_sentences": W("에", "에게", "을")})
    assert fields is not None and fields[1] == "에서" and fields[3] == ["에", "에게", "을"], (fields, reasons)
    # A wrong sentence that only leaves something out blanks the whole word (no empty option).
    past = {"sentence": "어제 영화를 봤어요.", "wrong_sentences": ["어제 영화를 봐요.", "어제 영화를 보았어요요.",
                                                                "어제 영화를 볼 거예요."]}
    fields, reasons = plain.verify("tense_past", past)
    assert fields is None or all(fields[3]), (fields, reasons)
    # Mistakes in different words cannot share one blank.
    fields, reasons = plain.verify("subj_topic", {"sentence": "동생이 공원에서 축구를 해요.", "wrong_sentences": [
        "동생이 공원에 축구를 해요.", "동생이 공원을 축구를 해요.", "동생이 공원에서 축구가 해요."]})
    assert fields is None, fields
    # Brackets or markdown the generator adds anyway are ignored.
    fields, _ = plain.verify("subj_topic", {**good, "sentence": "동생이 공원[에서] 축구를 해요."})
    assert fields is not None and fields[1] == "에서", fields
    # A fourth valid wrong sentence is tolerated; only N_DISTRACTORS are kept.
    fields, reasons = plain.verify("subj_topic", {**good, "wrong_sentences": W("에", "을", "도", "이")})
    assert fields is not None and len(fields[3]) == N_DISTRACTORS, (fields, reasons)
    # A wrong sentence that changes something else as well is rejected.
    _, reasons = plain.verify("subj_topic", {**good, "wrong_sentences": W("에", "을") + ["형이 공원에서 야구를 해요."]})
    assert any(r.startswith("not-minimal") for r in reasons), reasons

    # Two wrong sentences that make the same mistake (allomorphs 을/를) are rejected as
    # non-distinct, so every pick a learner could make stays informative.
    _, reasons = plain.verify("subj_topic", {**good, "wrong_sentences": W("을", "를", "이")})
    assert any(r.startswith("distinct") for r in reasons), reasons

    # A wrong sentence that edits something other than the tested category is rejected.
    _, reasons = plain.verify("subj_topic", {**good, "wrong_sentences": W("에", "을") + ["동생이 공원에서 축구를 해."]})
    assert any(r.startswith("wrong-category") for r in reasons), reasons

    print("all smoke tests passed")
