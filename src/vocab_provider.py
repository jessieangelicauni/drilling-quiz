"""
Vocab ContentProvider (PRD §2.3): vocab items come from an admin-maintained
curriculum ("Maintain curriculum per subject" / "upload curriculum"), not
generation -- there is no "new word" to invent, only a closed word list an
LMS admin owns.

Distractors for a vocab MC question are other curriculum meanings, not
LLM-generated -- picking plausible wrong meanings from the same bank is a
closed, deterministic operation: vocab distractors carry no misconception,
unlike grammar items (item_foundry.py).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class VocabEntry:
    word: str
    meaning: str
    topic: str = "general"


class VocabCurriculum:
    """Admin-maintained word bank."""

    def __init__(self):
        self._entries: dict[str, VocabEntry] = {}

    def add(self, item_id: str, word: str, meaning: str, topic: str = "general") -> None:
        self._entries[item_id] = VocabEntry(word, meaning, topic)

    def get(self, item_id: str) -> VocabEntry:
        return self._entries[item_id]

    def ids(self) -> list[str]:
        return list(self._entries.keys())

    def __len__(self) -> int:
        return len(self._entries)


@dataclass
class VocabMCQuestion:
    word_id: str
    stem: str
    options: dict[str, str]
    answer: str


class VocabContentProvider:
    """Implements the vocab half of daily_drill.ContentProvider.

    Returns the word_id itself as the item id (not a separate question id):
    daily_drill.py keys vocab_cards by whatever next_new_vocab_item returns,
    so the FSRS card created for this word once it's answered must use the
    same id to be found as "due" again later. The MC question (stem +
    randomized options) is re-rolled each time a word is presented and kept
    only as display-time metadata, not identity.
    """

    def __init__(self, curriculum: VocabCurriculum, n_options: int = 4, seed: int | None = None):
        if n_options < 2:
            raise ValueError("n_options must be at least 2")
        self.curriculum = curriculum
        self.n_options = n_options
        self.rng = random.Random(seed)
        self.questions: dict[str, VocabMCQuestion] = {}

    def next_new_vocab_item(self, user_id: str, exclude: set[str]) -> str | None:
        candidates = [i for i in self.curriculum.ids() if i not in exclude]
        if not candidates:
            return None
        word_id = self.rng.choice(candidates)
        entry = self.curriculum.get(word_id)

        pool = [i for i in self.curriculum.ids()
                if i != word_id and self.curriculum.get(i).topic == entry.topic]
        if len(pool) < self.n_options - 1:
            pool = [i for i in self.curriculum.ids() if i != word_id]
        n_distractors = min(self.n_options - 1, len(pool))
        distractor_meanings = [self.curriculum.get(i).meaning
                                for i in self.rng.sample(pool, n_distractors)]

        option_texts = [entry.meaning] + distractor_meanings
        labels = ["A", "B", "C", "D"][:len(option_texts)]
        self.rng.shuffle(labels)
        answer_label = labels[0]
        options = dict(zip(labels, option_texts))

        self.questions[word_id] = VocabMCQuestion(
            word_id=word_id, stem=f'"{entry.word}"의 뜻은 무엇입니까?',
            options=options, answer=answer_label)
        return word_id

    def next_new_grammar_item(self, skill_id: str) -> str | None:
        raise NotImplementedError("grammar items come from ExaoneGrammarProvider, not here")


if __name__ == "__main__":
    curriculum = VocabCurriculum()
    curriculum.add("w1", "도서관", "library", topic="places")
    curriculum.add("w2", "병원", "hospital", topic="places")
    curriculum.add("w3", "공원", "park", topic="places")
    curriculum.add("w4", "사과", "apple", topic="food")

    provider = VocabContentProvider(curriculum, n_options=3, seed=0)
    word_id = provider.next_new_vocab_item("u1", exclude=set())
    assert word_id is not None
    q = provider.questions[word_id]
    assert q.word_id == word_id
    assert len(q.options) == 3
    assert q.options[q.answer] == curriculum.get(word_id).meaning

    seen: set[str] = set()
    for _ in range(4):
        wid = provider.next_new_vocab_item("u1", exclude=seen)
        if wid is None:
            break
        seen.add(wid)
    assert len(seen) == 4, "excluding seen words must eventually exhaust the 4-word curriculum"
    assert provider.next_new_vocab_item("u1", exclude=seen) is None, "no words left -> None"

    print("all smoke tests passed")
