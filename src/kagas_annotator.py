"""
KAGAS-style edit annotator (Yoon et al., 2023, "Towards standardizing
Korean grammatical error correction"): given a correct sentence and a
wrong one, find WHERE they differ and classify each difference by the
grammatical category it touches -- the Korean counterpart of ERRANT.

Used by item_foundry.py to verify, without asking another LLM, that a
generated distractor is a genuine minimal pair of the key (exactly one
edit), that the edit sits in the grammatical category the skill tests
(a particle item must differ in a particle, PART), and, for functional
morphemes, WHAT the edit is (에서 -> 에): the edit's code (PART:에서>에) is
the distractor's misconception label.

How it works:
  1. Analyse both sentences with kiwipiepy and align their morpheme
     sequences, after merging phonologically conditioned allomorphs.
  2. Classify each changed run of morphemes by its Sejong tag (below).
  3. Sentences whose morpheme analysis is identical but whose surface
     differs are conjugation / spelling errors, found word by word:
     kiwipiepy deliberately re-analyses a malformed form like 부르어요
     exactly as 불러요, so this case is detected from the surface, not
     from the tags.
  4. The opposite case: a malformed word the tagger cannot analyse
     at all (먹았어요 comes out as one unknown noun 먹았 + 어요), which
     step 2 can only call MULTI. When that is the only edit and it lies
     inside one word, the edit is read from the surface of that word
     instead: what is left after the common prefix and suffix. In a
     predicate (먹[었>았]어요) it is an ending error, END:었>았, or a
     conjugation error (CONJ) when the correct stem is irregular
     (아름다워요, tagged VA-I). After a host noun (학교[에>으로], which the
     tagger splits into 학교 + 으 + 로) it takes the category of the
     correct morphemes it replaces, here PART:에>으로.

KAGAS's own implementation uses a different tagger; the category set and
the align-then-classify procedure follow the paper, the tagger is
kiwipiepy, which the rest of this project already uses.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass

from kiwipiepy import Kiwi

_KIWI = Kiwi()

# Sejong tag prefix -> KAGAS category.
_CATEGORY = (
    ("J", "PART"),       # particles (JKS, JKO, JKB, JX, ...)
    ("EP", "END"), ("EF", "END"), ("EC", "END"), ("ETM", "END"), ("ETN", "END"),
    ("NNB", "NOUN"), ("NNG", "NOUN"), ("NNP", "NOUN"), ("NP", "NOUN"), ("NR", "NOUN"),
    ("VV", "VERB"), ("VX", "VERB"), ("XSV", "VERB"),
    ("VA", "ADJ"), ("XSA", "ADJ"),
    ("VCP", "PART"), ("VCN", "ADJ"),
    ("MM", "MOD"),
    ("MAG", "ADV"), ("MAJ", "CONJ"),
    ("XSN", "NOUN"), ("XPN", "NOUN"),
    ("S", "PUNCT"),
)
# Categories whose forms are a closed class, so the edit itself (에서 -> 에)
# names the misconception and generalises across items. Open-class edits
# (one verb for another) are labelled by their lexical pair (VERB:마시>먹),
# which recurs less often across items.
FUNCTIONAL = frozenset({"PART", "END"})
FUNCTIONAL_TAGS = frozenset({"NNB", "MM"})

# Phonologically conditioned allomorphs are one morpheme: picking 를 where
# 을 belongs is an allomorph error, not a different particle.
_CONTENT = frozenset({"NOUN", "VERB", "ADJ", "ADV", "MOD"})
_PREDICATE_TAGS = ("VV", "VA", "VX", "XSV", "XSA", "VCP")

_ALLOMORPH = {"를": "을", "가": "이", "는": "은", "로": "으로", "와": "과", "랑": "이랑",
              "야": "이야", "나": "이나", "며": "이며"}


def category(tag: str) -> str:
    base = tag.split("-")[0]
    for prefix, cat in _CATEGORY:
        if base.startswith(prefix):
            return cat
    return "OTHER"


@dataclass(frozen=True)
class Edit:
    category: str          # KAGAS category (PART, END, NOUN, VERB, CONJ, SPELL, ...)
    op: str                # "REP", "INS" (only in wrong), "DEL" (only in correct)
    correct: str           # changed morphemes on the correct side, "+"-joined ("" for INS)
    wrong: str             # changed morphemes on the wrong side ("" for DEL)
    functional: bool       # closed-class edit whose forms name the misconception

    @property
    def code(self) -> str:
        """Canonical form of a functional edit, e.g. "PART:에서>에"."""
        return f"{self.category}:{self.correct}>{self.wrong}"


def _morphs(text: str) -> list[tuple[str, str]]:
    return [(t.form, t.tag) for t in _KIWI.tokenize(text)]


def _norm(form: str, tag: str) -> str:
    return _ALLOMORPH.get(form, form) if tag.startswith("J") else form


def _classify(c_side, w_side, c_form, w_form, kind) -> Edit:
    tags = [t for _, t in c_side + w_side]
    cats = {category(t) for t in tags}
    cat = cats.pop() if len(cats) == 1 else "MULTI"
    functional = cat in FUNCTIONAL or (bool(tags) and all(t.split("-")[0] in FUNCTIONAL_TAGS for t in tags))
    return Edit(cat, kind, c_form, w_form, functional)


def _common_affixes(a: str, b: str) -> tuple[int, int]:
    p = 0
    while p < min(len(a), len(b)) and a[p] == b[p]:
        p += 1
    q = 0
    while q < min(len(a), len(b)) - p and a[-1 - q] == b[-1 - q]:
        q += 1
    return p, q


def _surface_word_edit(correct: str, wrong: str) -> Edit | None:
    """Step 4 of the module docstring: the edit read from the surface of
    the one word that differs, or None if the sentences differ in more than
    one word or the word's start (its stem or host noun) differs."""
    cw, ww = correct.split(), wrong.split()
    ops = [o for o in difflib.SequenceMatcher(a=cw, b=ww, autojunk=False).get_opcodes() if o[0] != "equal"]
    if len(ops) != 1 or ops[0][0] != "replace" or ops[0][2] - ops[0][1] != 1 or ops[0][4] - ops[0][3] != 1:
        return None
    c_word, w_word = cw[ops[0][1]], ww[ops[0][3]]
    tokens = _KIWI.tokenize(c_word)
    tags = [t.tag for t in tokens]
    p, q = _common_affixes(c_word, w_word)
    if p == 0:  # the stem itself differs: a different word, not a form of it
        return None
    c_diff, w_diff = c_word[p:len(c_word) - q], w_word[p:len(w_word) - q]
    # A whole word pasted into another (늦[게] -> 늦일찍) is not a slip in
    # the word's form: the inserted piece is a content word of its own.
    inserted = [t for t in _KIWI.tokenize(w_word)
                if t.start >= p and t.start + t.len <= len(w_word) - q and t.len >= 2]
    if any(category(t.tag) in _CONTENT for t in inserted):
        return None
    op = "REP" if c_diff and w_diff else ("DEL" if c_diff else "INS")
    if any(t.split("-")[0] in _PREDICATE_TAGS for t in tags):
        if any(t.endswith("-I") for t in tags):
            return Edit("CONJ", "REP", c_word, w_word, False)
        return Edit("END", op, c_diff, w_diff, True)
    # Host noun + particle (학교에 written 학교으로 / 학교이, which the
    # tagger splits into nonsense): the category of the changed morphemes.
    changed = [t.tag for t in tokens if t.start < len(c_word) - q and t.start + t.len > p]
    cats = {category(t) for t in changed}
    if len(cats) != 1 or not c_diff:
        return None
    cat = cats.pop()
    return Edit(cat, op, _ALLOMORPH.get(c_diff, c_diff) if cat == "PART" else c_diff,
                _ALLOMORPH.get(w_diff, w_diff) if cat == "PART" else w_diff, cat in FUNCTIONAL)


def annotate(correct: str, wrong: str) -> list[Edit]:
    """All edits turning `correct` into `wrong`, in sentence order. Both
    sentences are analysed whole, so every morpheme is tagged in context
    (개 in 세 개를 is a counter; analysed alone it would be "dog")."""
    cm, wm = _morphs(correct), _morphs(wrong)
    if cm == wm:
        # Same analysis, different surface: a conjugation slip on a
        # predicate (불러요 vs 부르어요), a spelling slip otherwise.
        edits = []
        cw, ww = correct.split(), wrong.split()
        matcher = difflib.SequenceMatcher(a=cw, b=ww, autojunk=False)
        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op == "equal":
                continue
            c_text, w_text = " ".join(cw[i1:i2]), " ".join(ww[j1:j2])
            cat = "CONJ" if any(category(t) in ("VERB", "ADJ") for _, t in _morphs(c_text)) else "SPELL"
            edits.append(Edit(cat, "REP", c_text, w_text, False))
        return edits
    cn = [(_norm(f, t), t) for f, t in cm]
    wn = [(_norm(f, t), t) for f, t in wm]
    if cn == wn:
        # Differs only in allomorph choice (라면를 for 라면을).
        return [_classify([a], [b], a[0], b[0], "REP") for a, b in zip(cm, wm) if a != b]
    # Aligned on allomorph-normalised morphemes, so an allomorph that only
    # changed because its host changed (개를 / 명을) is not a second edit.
    edits = []
    matcher = difflib.SequenceMatcher(a=cn, b=wn, autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            continue
        c_side, w_side = cn[i1:i2], wn[j1:j2]
        kind = "REP" if c_side and w_side else ("DEL" if c_side else "INS")
        edits.append(_classify(c_side, w_side, "+".join(f for f, _ in c_side),
                               "+".join(f for f, _ in w_side), kind))
    if len(edits) == 1 and edits[0].category == "MULTI":
        return [_surface_word_edit(correct, wrong) or edits[0]]
    return edits


if __name__ == "__main__":
    def one(c, w):
        e = annotate(c, w)
        assert len(e) == 1, (c, w, e)
        return e[0]

    e = one("저는 도서관에서 공부해요.", "저는 도서관에 공부해요.")
    assert (e.category, e.op, e.code, e.functional) == ("PART", "REP", "PART:에서>에", True), e

    # Allomorphs normalise, so the same misconception gets one code whatever the noun.
    assert one("책을 읽어요.", "책에 읽어요.").code == one("커피를 마셔요.", "커피에 마셔요.").code == "PART:을>에"
    e = one("라면을 먹어요.", "라면를 먹어요.")
    assert (e.category, e.correct, e.wrong) == ("PART", "을", "를"), e

    e = one("어제 영화를 봤어요.", "어제 영화를 봐요.")
    assert (e.category, e.op, e.code) == ("END", "DEL", "END:었>"), e

    e = one("선생님께서 가르치세요.", "선생님께서 가르쳐요.")
    assert e.category == "END" and e.functional, e

    e = one("노래를 불러요.", "노래를 부르어요.")
    assert e.category == "CONJ" and not e.functional, e

    e = one("사과 세 개를 샀어요.", "사과 세 명을 샀어요.")
    assert e.functional and e.correct == "개" and e.wrong == "명", e

    e = one("매운 음식을 좋아해요.", "맵는 음식을 좋아해요.")
    assert e.code == "END:은>는", e

    e = one("밥을 먹어요.", "밥을 마셔요.")
    assert e.category == "VERB" and not e.functional, e

    # Malformed predicates the tagger cannot analyse (step 4).
    assert one("저는 저녁을 먹었어요.", "저는 저녁을 먹았어요.").code == "END:었>았"
    assert one("정말 좋았어요.", "정말 좋았아요.").code == "END:어>아"
    assert one("조용한 친구예요.", "조용 친구예요.").code == "END:한>"
    # A particle the tagger misreads after its host (학교으로, 학교이).
    assert one("학생이 학교에 가요.", "학생이 학교으로 가요.").code == "PART:에>으로"
    assert one("학생이 학교에 가요.", "학생이 학교이 가요.").code == "PART:에>이"
    # ...but a content word pasted into another word is not read as one edit.
    assert annotate("친구가 늦게 왔어요.", "친구가 늦일찍 왔어요.")[0].category == "MULTI"
    e = one("꽃이 아름다워요.", "꽃이 아름답어요.")
    assert e.category == "CONJ" and not e.functional, e

    # Not a minimal pair: two separate edits.
    assert len(annotate("저는 도서관에서 공부해요.", "저가 도서관에 공부해요.")) == 2
    assert annotate("저는 학생이에요.", "저는 학생이에요.") == []
    print("all smoke tests passed")
