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

import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from difficulty_heuristic import estimate_difficulty
from exaone_provider import ExaoneGrammarProvider, GeneratedItem
from urnings_tracker import UrnState, seed_item_from_encoder


class ReviewStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class BankEntry:
    """item_urn is Novelty 1's item-level Urnings calibration (PRD §3.1,
    wired in for real at §9.12): unlike exaone_provider's items used
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

    def batch_generate(self, provider: ExaoneGrammarProvider, skill_id: str,
                        n_target: int, max_attempts: int | None = None) -> int:
        """Add up to n_target NEW validated items for skill_id as PENDING
        (awaiting admin review). Returns how many were actually added --
        can be less than n_target if the generator keeps failing
        validation (e.g. all curated slots already exhausted).
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
    """
    def __init__(self, bank: QuestionBank, seed: int | None = None):
        self.bank = bank
        self.rng = random.Random(seed)

    def next_new_grammar_item(self, skill_id: str) -> str | None:
        candidates = self.bank.approved_ids(skill_id)
        if not candidates:
            return None  # bank empty for this skill -- batch_generate() needs to run
        return self.rng.choice(candidates)

    def next_new_vocab_item(self, user_id: str, exclude: set[str]) -> str | None:
        raise NotImplementedError("vocab items come from the admin-maintained curriculum, not this bank")


if __name__ == "__main__":
    bank = QuestionBank()
    provider = ExaoneGrammarProvider(seed=0)

    added = bank.batch_generate(provider, "particle_loc", n_target=5)
    assert added >= 4, f"expected the validated pipeline to fill most of the batch, got {added}/5"
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
    # a much harder stem shows up as a LOWER k (lower implied pass-probability).
    assert urn_after_edit.k < urn_before_edit.k, (
        "a much longer/more complex edited stem must seed a lower (harder) item urn, "
        f"got {urn_after_edit} vs original {urn_before_edit}"
    )

    print(f"batch_generate added {added}/5 for particle_loc")
    print("all smoke tests passed")
