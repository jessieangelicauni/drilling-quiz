from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

_TARGET_SUCCESS = 0.85
_LOGIT_OFFSET = math.log(_TARGET_SUCCESS / (1.0 - _TARGET_SUCCESS))

import numpy as np

from .answer_processing import AnswerResult, submit_grammar_answer
from .daily_drill import UserDrillState
from ..urnings.difficulty_heuristic import estimate_difficulty

from ..generation.item_foundry import Curriculum, CurriculumItem, GeneratedItem, MisconceptionItemGenerator
from ..urnings.urnings_tracker import UrnState, seed_item_from_encoder, selection_probs


class ReviewStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class BankEntry:
    item: GeneratedItem
    status: ReviewStatus = ReviewStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    item_urn: UrnState | None = None

    def __post_init__(self):
        if self.item_urn is None:
            self.item_urn = seed_item_from_encoder(*estimate_difficulty(self.item.stem))


class QuestionBank:
    def __init__(self):
        self.entries: dict[str, BankEntry] = {}

    def batch_generate(self, provider: MisconceptionItemGenerator, skill_id: str | None,
                        n_target: int, max_attempts: int | None = None) -> int:
        max_attempts = max_attempts or n_target * 10
        added, attempts = 0, 0
        while added < n_target and attempts < max_attempts:
            item_id = provider.next_new_grammar_item(skill_id)
            attempts += 1
            if item_id is None:
                continue
            self.entries[item_id] = BankEntry(item=provider.items[item_id])
            added += 1
        return added

    def approve(self, item_id: str) -> None:
        self.entries[item_id].status = ReviewStatus.APPROVED

    def reject(self, item_id: str) -> None:
        self.entries[item_id].status = ReviewStatus.REJECTED

    def edit(self, item_id: str, item: GeneratedItem) -> None:
        self.entries[item_id].item = item
        self.entries[item_id].item_urn = seed_item_from_encoder(*estimate_difficulty(item.stem))

    def submit_answer(self, item_id: str, state: UserDrillState, chosen_option: str,
                      rng: np.random.Generator, now: datetime | None = None) -> AnswerResult:
        entry = self.entries[item_id]
        skill_id = entry.item.skill_id
        candidates = self.candidates(skill_id)

        def selection_prob(learner: UrnState, item: UrnState) -> float:
            return float(self.selection_probs(candidates, learner, (item_id, item))[candidates.index(item_id)])

        result = submit_grammar_answer(state, entry.item, chosen_option, rng,
                                       now=now, item_urn=entry.item_urn,
                                       selection_prob=selection_prob if item_id in candidates else None)
        entry.item_urn = result.updated_item_urn
        return result

    def candidates(self, skill_id: str, exclude: set[str] = frozenset()) -> list[str]:
        return [iid for iid in self.approved_ids(skill_id) if iid not in exclude]

    def selection_probs(self, candidates: list[str], learner_urn: UrnState,
                        override: tuple[str, UrnState] | None = None) -> np.ndarray:
        urns = [self.entries[iid].item_urn for iid in candidates]
        if override is not None:
            urns[candidates.index(override[0])] = override[1]
        return selection_probs(learner_urn.logit_estimate(), [u.logit_estimate() for u in urns],
                               target=_TARGET_SUCCESS)

    def approved_ids(self, skill_id: str) -> list[str]:
        return [iid for iid, e in self.entries.items()
                if e.item.skill_id == skill_id and e.status == ReviewStatus.APPROVED]

    def pending_ids(self, skill_id: str | None = None) -> list[str]:
        return [iid for iid, e in self.entries.items()
                if e.status == ReviewStatus.PENDING and (skill_id is None or e.item.skill_id == skill_id)]


class BankContentProvider:
    def __init__(self, bank: QuestionBank, seed: int | None = None):
        self.bank = bank
        self.rng = random.Random(seed)

    def next_new_grammar_item(self, skill_id: str, learner_urn: UrnState | None = None,
                               exclude: set[str] = frozenset()) -> str | None:
        candidates = self.bank.candidates(skill_id, exclude)
        if not candidates:
            return None
        if learner_urn is None:
            return self.rng.choice(candidates)
        probs = self.bank.selection_probs(candidates, learner_urn)
        return candidates[int(np.searchsorted(np.cumsum(probs), self.rng.random() * probs.sum()))]

    def next_new_vocab_item(self, user_id: str, exclude: set[str]) -> str | None:
        raise NotImplementedError("vocab items come from the admin-maintained curriculum, not this bank")


if __name__ == "__main__":
    import json

    places = ["공원", "도서관", "회사", "카페", "식당"]

    def scripted_chat(messages, _n=iter(range(10 ** 6))):
        content = messages[-1]["content"]
        if content.startswith("Exercise:"):
            return json.dumps({l: "reason" for l in ("A", "B", "C", "D")}, ensure_ascii=False)
        if content.startswith("Task: write three"):
            return json.dumps({"wrong": ["에", "을", "이"]}, ensure_ascii=False)
        place = places[next(_n) % len(places)]
        return json.dumps({"sentence": f"친구가 {place}에서 축구를 해요.", "target": "에서"}, ensure_ascii=False)

    test_curriculum = Curriculum(items={"particle_loc": CurriculumItem(
        "location particles: 에서 marks where an action takes place, 에 marks a destination or "
        "where something is or exists")})

    bank = QuestionBank()
    provider = MisconceptionItemGenerator(scripted_chat, test_curriculum, seed=0)

    added = bank.batch_generate(provider, "particle_loc", n_target=5)
    assert added == 5, f"every scripted item passes verification, got {added}/5"
    assert len(bank.pending_ids("particle_loc")) == added

    serving = BankContentProvider(bank, seed=0)
    assert serving.next_new_grammar_item("particle_loc") is None, (
        "nothing has been approved yet -- serving must not leak pending items")

    for item_id in bank.pending_ids("particle_loc"):
        bank.approve(item_id)
    assert len(bank.approved_ids("particle_loc")) == added
    assert len(bank.pending_ids("particle_loc")) == 0

    served_id = serving.next_new_grammar_item("particle_loc")
    assert served_id is not None
    assert served_id in bank.approved_ids("particle_loc")

    some_id = bank.approved_ids("particle_loc")[0]
    bank.reject(some_id)
    assert some_id not in bank.approved_ids("particle_loc"), "a rejected item must stop being served"

    edited_item = GeneratedItem(
        skill_id="particle_loc", stem="저는 도서관___ 공부해요.",
        blank_marker="___", options={"A": "에서", "B": "에"}, answer="A",
    )
    another_id = bank.approved_ids("particle_loc")[0]
    bank.edit(another_id, edited_item)
    assert bank.entries[another_id].item.stem == "저는 도서관___ 공부해요."

    assert serving.next_new_grammar_item("honorific") is None, "an unfilled skill must not serve anything"

    fresh_id = bank.approved_ids("particle_loc")[0]
    fresh_entry = bank.entries[fresh_id]
    assert fresh_entry.item_urn is not None
    assert 0 <= fresh_entry.item_urn.k <= fresh_entry.item_urn.n
    urn_before_edit = fresh_entry.item_urn
    bank.edit(fresh_id, GeneratedItem(
        skill_id="particle_loc",
        stem="이 문제가 학생들에 의해 풀릴 수 있었던 것으로 보인다 선생님께서 말씀하셨습니다 ___.",
        blank_marker="___", options={"A": "에서", "B": "에"}, answer="A",
    ))
    urn_after_edit = bank.entries[fresh_id].item_urn
    assert urn_after_edit is not urn_before_edit, "edit() must re-seed, not keep the old item's urn object"
    assert urn_after_edit.k > urn_before_edit.k, (
        "a much longer/more complex edited stem must seed a higher (harder) item urn, "
        f"got {urn_after_edit} vs original {urn_before_edit}"
    )

    adapt_bank = QuestionBank()
    adapt_provider = BankContentProvider(adapt_bank, seed=0)
    difficulty_spread = [
        ("easy-1", UrnState(k=4, n=32)),
        ("easy-2", UrnState(k=8, n=32)),
        ("medium", UrnState(k=16, n=32)),
        ("hard-1", UrnState(k=24, n=32)),
        ("hard-2", UrnState(k=28, n=32)),
    ]
    for label, item_urn in difficulty_spread:
        item = GeneratedItem(skill_id="adapt_test", stem=f"stem-{label}", blank_marker="___",
                              options={"A": "x", "B": "y"}, answer="A")
        adapt_bank.entries[label] = BankEntry(item=item, status=ReviewStatus.APPROVED, item_urn=item_urn)

    weak_learner, strong_learner = UrnState(k=4, n=32), UrnState(k=28, n=32)
    from collections import Counter
    weak_picks = Counter(adapt_provider.next_new_grammar_item("adapt_test", learner_urn=weak_learner)
                         for _ in range(400))
    strong_picks = Counter(adapt_provider.next_new_grammar_item("adapt_test", learner_urn=strong_learner)
                           for _ in range(400))
    assert weak_picks.most_common(1)[0][0] == "easy-1", weak_picks
    assert strong_picks.most_common(1)[0][0] == "medium", strong_picks
    mean_k = lambda picks: sum(adapt_bank.entries[i].item_urn.k * c for i, c in picks.items()) / sum(picks.values())
    assert mean_k(weak_picks) < mean_k(strong_picks), "weak learners must get easier items on average"

    cands = adapt_bank.candidates("adapt_test")
    p0 = adapt_bank.selection_probs(cands, strong_learner)
    p1 = adapt_bank.selection_probs(cands, strong_learner, ("medium", UrnState(k=30, n=32)))
    assert abs(p0.sum() - 1) < 1e-9 and p1[cands.index("medium")] < p0[cands.index("medium")]

    seen = set()
    for _ in range(5):
        iid = adapt_provider.next_new_grammar_item("adapt_test", learner_urn=weak_learner, exclude=seen)
        assert iid is not None and iid not in seen
        seen.add(iid)
    assert adapt_provider.next_new_grammar_item("adapt_test", learner_urn=weak_learner, exclude=seen) is None, (
        "once every candidate is excluded, must return None, not repeat")

    assert adapt_provider.next_new_grammar_item("adapt_test") in dict(difficulty_spread), (
        "no learner_urn given must still fall back to a valid (uniform) choice")

    writeback_id = bank.approved_ids("particle_loc")[0]
    writeback_entry = bank.entries[writeback_id]
    urn_before = writeback_entry.item_urn
    learner_state = UserDrillState(user_id="writeback-test", skill_ids=["particle_loc"])
    answer_rng = np.random.default_rng(0)
    for _ in range(30):
        bank.submit_answer(writeback_id, learner_state, writeback_entry.item.answer, rng=answer_rng)
    assert writeback_entry.item_urn is not urn_before, "the updated urn must be written back to the bank entry"
    assert writeback_entry.item_urn.n == urn_before.n
    assert writeback_entry.item_urn.k <= urn_before.k, (
        f"correct answers must not raise the item urn's k: {urn_before} -> {writeback_entry.item_urn}")

    print(f"batch_generate added {added}/5 for particle_loc")
    print("all smoke tests passed")
