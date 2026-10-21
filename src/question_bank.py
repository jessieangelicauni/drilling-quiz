"""
Question bank (PRD §9.3's "Review topic & Question Bank" / "Set batch
publish time" admin features, application-logic side): decouples
GENERATING items (needs the LLM, slow, can fail validation) from SERVING
items (must be fast and reliable every time daily_drill.py assembles a
quiz -- it should never depend on a live LLM call succeeding at request
time, and it should never start "from zero" per request).

batch_generate() is meant to run ahead of time (a scheduled job, or an
admin clicking "generate more"), filling the bank with PENDING items;
approve()/reject()/edit() are the admin review workflow; serving (via
BankContentProvider) only ever reads APPROVED items.

Storage here is a plain in-memory dict with dataclasses, same pattern as
urnings_tracker.py/streak.py -- swappable for a real database later
(explicitly deferred, PRD §9.3) without changing this module's interface.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Section IV-C Eq. 4's actual target (85% success, not maximum-information
# 50%) -- the logit offset such that sigma(theta - (theta - _LOGIT_OFFSET))
# = 0.85. Shared constant so the offset used here can never silently drift
# from the one validate_item_discrimination.py/validate_novelty2.py assume
# when they report this policy's measured cost.
_TARGET_SUCCESS = 0.85
_LOGIT_OFFSET = math.log(_TARGET_SUCCESS / (1.0 - _TARGET_SUCCESS))

import numpy as np

from answer_processing import AnswerResult, submit_grammar_answer
from daily_drill import UserDrillState
from difficulty_heuristic import estimate_difficulty
from typing import TYPE_CHECKING

from item_foundry import GeneratedItem, MisconceptionItemGenerator
from urnings_tracker import UrnState, seed_item_from_encoder, selection_probs


if TYPE_CHECKING:  # loads a GPU model; only the type is needed here
    from qwen_feedback import QwenFeedbackProvider


class ReviewStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class BankEntry:
    """item_urn is Novelty 1's item-level Urnings calibration (PRD §3.1,
    wired in for real at §9.12): unlike generated items used
    directly (one-shot, never reused), a bank entry genuinely IS served to
    many different learners once approved -- the reasoning in §9.2 that
    item urns don't need persisting predates this bank existing, and
    doesn't hold for entries that live here. Seeded once from
    estimate_difficulty() at creation (or re-seeded by edit(), since the
    estimate is of the TEXT, which just changed); accumulates real
    evidence afterwards via answer_processing.submit_grammar_answer's
    returned AnswerResult.updated_item_urn, which the caller writes back
    here directly (this class stays a plain mutable state object, same
    convention as the rest of this codebase -- no persistence layer of
    its own).
    """
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

    def batch_generate(self, provider: MisconceptionItemGenerator, skill_id: str,
                        n_target: int, max_attempts: int | None = None) -> int:
        """Add up to n_target NEW validated items for skill_id as PENDING
        (awaiting admin review). Returns how many were actually added --
        can be less than n_target if the generator keeps failing
        verification (item_foundry.MisconceptionItemGenerator.verify).
        """
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
        """Admin edits an item's content directly ("edit" in PRD's approve/edit/drop).

        Re-seeds item_urn from scratch rather than keeping whatever evidence
        had accumulated: the old urn's accumulated evidence was calibrating
        the OLD text, which no longer exists -- carrying it forward would
        silently misattribute responses to the wrong item content.
        """
        self.entries[item_id].item = item
        self.entries[item_id].item_urn = seed_item_from_encoder(*estimate_difficulty(item.stem))

    def submit_answer(self, item_id: str, state: UserDrillState, chosen_option: str,
                      feedback_provider: QwenFeedbackProvider, rng: np.random.Generator,
                      now: datetime | None = None) -> AnswerResult:
        """submit_grammar_answer for an item served from this bank, with the
        item's own urn passed in AND the updated urn written back. Before
        this existed nothing wrote AnswerResult.updated_item_urn back (review
        2026-10-21), so every approved item kept its estimate_difficulty()
        seed forever and BankContentProvider's adaptive selection never saw
        real response evidence.
        """
        entry = self.entries[item_id]
        skill_id = entry.item.skill_id
        # Served adaptively by BankContentProvider, so the Urnings update gets
        # the selection correction. Candidates are recomputed here with no
        # within-drill exclusions, which the serving call may have had; the
        # target misconception is the one the drill served it for.
        candidates = self.candidates(skill_id, target_misconception=state.misconceptions.dominant(skill_id))
        if item_id not in candidates:
            candidates = self.candidates(skill_id)

        def selection_prob(learner: UrnState, item: UrnState) -> float:
            return float(self.selection_probs(candidates, learner, (item_id, item))[candidates.index(item_id)])

        result = submit_grammar_answer(state, entry.item, chosen_option, feedback_provider, rng,
                                       now=now, item_urn=entry.item_urn,
                                       selection_prob=selection_prob if item_id in candidates else None)
        entry.item_urn = result.updated_item_urn
        return result

    def candidates(self, skill_id: str, exclude: set[str] = frozenset(),
                   target_misconception: str | None = None) -> list[str]:
        """Approved items for the skill; when a target misconception is given
        (the learner's current dominant one, misconception_tracker), only the
        items with a distractor realising it, if there are any."""
        ids = [iid for iid in self.approved_ids(skill_id) if iid not in exclude]
        if target_misconception is not None:
            targeted = [iid for iid in ids
                        if target_misconception in (self.entries[iid].item.option_codes or {}).values()]
            ids = targeted or ids
        return ids

    def selection_probs(self, candidates: list[str], learner_urn: UrnState,
                        override: tuple[str, UrnState] | None = None) -> np.ndarray:
        """BankContentProvider's probability of serving each candidate, with
        one item's urn optionally replaced (the proposed state in the
        correction)."""
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
    """Implements daily_drill.ContentProvider's grammar half by serving
    from already-APPROVED entries -- no live LLM call, no GPU, at request
    time. This is the fix for "starts from zero every time": the bank is
    filled by batch_generate() ahead of time, separately from serving.

    next_new_grammar_item is also where Novelty 1's "Select" step actually
    happens now (PRD §9.15): §9.12 wired item_urn seeding/persistence in,
    but nothing used it to pick WHICH item to serve until this -- before,
    an item's calibration existed but had zero effect on the learner's
    actual experience. Found directly from the user's own description of
    the behaviour they expected (an easy-for-you item should adapt harder
    next, a hard-for-your-friend item should adapt easier), which is
    standard adaptive-item-selection and was simply never connected.
    """
    def __init__(self, bank: QuestionBank, seed: int | None = None):
        self.bank = bank
        self.rng = random.Random(seed)

    def next_new_grammar_item(self, skill_id: str, learner_urn: UrnState | None = None,
                               exclude: set[str] = frozenset(),
                               target_misconception: str | None = None) -> str | None:
        candidates = self.bank.candidates(skill_id, exclude, target_misconception)
        if not candidates:
            return None  # bank empty (or exhausted by exclude) for this skill
        if learner_urn is None:
            return self.rng.choice(candidates)  # no ability info -- fall back to uniform
        # Adaptive selection targeting 85% success (Section IV-C Eq. 4),
        # sampled from the selection kernel rather than "one of the closest
        # three", so that the probability of serving an item is known and
        # submit_answer() can correct the Urnings update for it (PRD §9.68).
        probs = self.bank.selection_probs(candidates, learner_urn)
        return candidates[int(np.searchsorted(np.cumsum(probs), self.rng.random() * probs.sum()))]

    def next_new_vocab_item(self, user_id: str, exclude: set[str]) -> str | None:
        raise NotImplementedError("vocab items come from the admin-maintained curriculum, not this bank")


if __name__ == "__main__":
    import json

    # CPU stand-ins for EXAONE (scripted replies) and Qwen3-4B (a scorer that
    # prefers the key); item_foundry.py's own smoke test covers verification.
    places = ["공원", "도서관", "회사", "카페", "식당"]

    def scripted_chat(messages, _n=iter(range(10 ** 6))):
        place = places[next(_n) % len(places)]
        return json.dumps({"sentence": f"친구가 {place}에서 축구를 해요.",
                           "wrong_sentences": [f"친구가 {place}{p} 축구를 해요." for p in ("에", "을", "이")]},
                          ensure_ascii=False)

    bank = QuestionBank()
    provider = MisconceptionItemGenerator(scripted_chat, lambda s: 0.0 if "에서" in s else -5.0,
                                          seed=0)

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

    # Novelty 1 (§9.12): every entry must get a real encoder-seeded item_urn,
    # not a flat/missing one, and edit() must re-seed it for the new text.
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
    # estimate_difficulty's sigma_hat2 is fixed (difficulty_heuristic.py), so
    # seed_item_from_encoder's n (driven only by sigma_hat2) stays constant;
    # a much harder stem shows up as a HIGHER k (the item urn is a difficulty).
    assert urn_after_edit.k > urn_before_edit.k, (
        "a much longer/more complex edited stem must seed a higher (harder) item urn, "
        f"got {urn_after_edit} vs original {urn_before_edit}"
    )

    # Misconception targeting: only items offering the target misconception
    # are candidates, when the bank has any.
    targeted = BankContentProvider(bank, seed=0).next_new_grammar_item(
        "particle_loc", target_misconception="PART:에서>이")
    assert "PART:에서>이" in bank.entries[targeted].item.option_codes.values()

    # Novelty 1's "Select" step (§9.15): adaptive item selection must
    # actually match item difficulty to the learner's ability, not serve
    # uniformly at random once a learner_urn is given. Hand-constructed
    # entries (bypassing the LLM) for precise control over item_urn.
    adapt_bank = QuestionBank()
    adapt_provider = BankContentProvider(adapt_bank, seed=0)
    difficulty_spread = [
        ("easy-1", UrnState(k=4, n=32)),    # very easy item (low k/n: item urn = difficulty)
        ("easy-2", UrnState(k=8, n=32)),
        ("medium", UrnState(k=16, n=32)),
        ("hard-1", UrnState(k=24, n=32)),
        ("hard-2", UrnState(k=28, n=32)),   # very hard item (high k/n)
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
    # Selection is sampled from the 85%-success kernel (PRD §9.32, §9.68),
    # so check where the mass goes, not a fixed set. At 85% the target is an
    # item a bit EASIER than the learner: logit(k/n) - logit(0.85). For the
    # weak learner that is easier than every item, so the easiest one
    # dominates; for the strong learner it is the medium item.
    assert weak_picks.most_common(1)[0][0] == "easy-1", weak_picks
    assert strong_picks.most_common(1)[0][0] == "medium", strong_picks
    mean_k = lambda picks: sum(adapt_bank.entries[i].item_urn.k * c for i, c in picks.items()) / sum(picks.values())
    assert mean_k(weak_picks) < mean_k(strong_picks), "weak learners must get easier items on average"

    # The selection correction (urnings_tracker.update) needs selection_probs
    # to be a proper distribution that responds to the proposed urn state.
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

    # submit_answer must write the item's updated urn back to the bank, not
    # just the learner's. In urnings_tracker.update() a correct response
    # moves a ball OUT of the item urn (ki - x + y), so 30 correct answers
    # can only lower the item's k (make it look easier), never raise it.
    writeback_id = bank.approved_ids("particle_loc")[0]
    writeback_entry = bank.entries[writeback_id]
    urn_before = writeback_entry.item_urn
    learner_state = UserDrillState(user_id="writeback-test", skill_ids=["particle_loc"])
    answer_rng = np.random.default_rng(0)
    for _ in range(30):
        bank.submit_answer(writeback_id, learner_state, writeback_entry.item.answer,
                           feedback_provider=None, rng=answer_rng)  # correct -> no feedback call
    assert writeback_entry.item_urn is not urn_before, "the updated urn must be written back to the bank entry"
    assert writeback_entry.item_urn.n == urn_before.n
    assert writeback_entry.item_urn.k <= urn_before.k, (
        f"correct answers must not raise the item urn's k: {urn_before} -> {writeback_entry.item_urn}")

    print(f"batch_generate added {added}/5 for particle_loc")
    print("all smoke tests passed")
