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

# Section IV-D Eq. 2's actual target (85% success, not maximum-information
# 50%) -- the logit offset such that sigma(theta - (theta - _LOGIT_OFFSET))
# = 0.85. Shared constant so the offset used here can never silently drift
# from the one validate_item_discrimination.py/validate_novelty2.py assume
# when they report this policy's measured cost.
_TARGET_SUCCESS = 0.85
_LOGIT_OFFSET = math.log(_TARGET_SUCCESS / (1.0 - _TARGET_SUCCESS))

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
                               exclude: set[str] = frozenset()) -> str | None:
        candidates = [iid for iid in self.bank.approved_ids(skill_id) if iid not in exclude]
        if not candidates:
            return None  # bank empty (or exhausted by exclude) for this skill
        if learner_urn is None:
            return self.rng.choice(candidates)  # no ability info -- fall back to uniform

        # Adaptive selection targeting 85% success (Section IV-D Eq. 2),
        # NOT the 50%/maximum-information target simulate_trackers.pick_adaptive
        # uses -- found and fixed as a real bug (PRD §9.32): this used to
        # sort by raw |beta_hat - theta| with no offset at all, i.e. silently
        # implemented the 50% target the paper explicitly argues against,
        # while validate_item_discrimination.py/validate_novelty2.py (and
        # Eq. 2 itself) all assumed 0.85. item_urn's logit, negated, is the
        # item's implied difficulty (same sign convention as
        # simulate_trackers.run_urnings's beta_hat = -_logit_urn(...)).
        theta = learner_urn.logit_estimate()
        want = theta - _LOGIT_OFFSET
        candidates.sort(key=lambda iid: abs(-self.bank.entries[iid].item_urn.logit_estimate() - want))
        # Pick randomly among the closest few, not always the single best
        # match -- a single best-match item_urn estimate is noisy (small
        # urn, possibly just seeded), and always returning the literal
        # closest would also make repeated calls for the same learner+
        # skill in one drill degenerate without relying on `exclude` alone.
        pool = candidates[:min(3, len(candidates))]
        return self.rng.choice(pool)

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

    # Novelty 1's "Select" step (§9.15): adaptive item selection must
    # actually match item difficulty to the learner's ability, not serve
    # uniformly at random once a learner_urn is given. Hand-constructed
    # entries (bypassing the LLM) for precise control over item_urn.
    adapt_bank = QuestionBank()
    adapt_provider = BankContentProvider(adapt_bank, seed=0)
    difficulty_spread = [
        ("easy-1", UrnState(k=28, n=32)),   # very easy item (high k/n)
        ("easy-2", UrnState(k=24, n=32)),
        ("medium", UrnState(k=16, n=32)),
        ("hard-1", UrnState(k=8, n=32)),
        ("hard-2", UrnState(k=4, n=32)),    # very hard item (low k/n)
    ]
    for label, item_urn in difficulty_spread:
        item = GeneratedItem(skill_id="adapt_test", stem=f"stem-{label}", blank_marker="___",
                              options={"A": "x", "B": "y"}, answer="A")
        adapt_bank.entries[label] = BankEntry(item=item, status=ReviewStatus.APPROVED, item_urn=item_urn)

    weak_learner, strong_learner = UrnState(k=4, n=32), UrnState(k=28, n=32)
    weak_picks = {adapt_provider.next_new_grammar_item("adapt_test", learner_urn=weak_learner)
                  for _ in range(20)}
    strong_picks = {adapt_provider.next_new_grammar_item("adapt_test", learner_urn=strong_learner)
                    for _ in range(20)}
    # Expected sets recomputed for the 85%-target offset (PRD §9.32), not
    # guessed: at 85%, a learner's closest-3 by |beta_hat - (theta - logit_offset)|
    # skews toward items a bit EASIER than the learner, not items matched to
    # their raw ability -- e.g. a strong learner's own ability-matched item
    # ("hard-2") only gives them ~50% success, which Eq. 2 deliberately
    # avoids; "easy-2" at 85% is closer to their actual 85%-success target
    # than "hard-2" is, so it replaces "hard-2" in the closest-3 set.
    assert weak_picks <= {"easy-1", "easy-2", "medium"}, f"a weak learner must be served easier items, got {weak_picks}"
    assert strong_picks <= {"medium", "hard-1", "easy-2"}, f"a strong learner must be served items near their 85%-success point, got {strong_picks}"
    assert weak_picks != strong_picks, "weak and strong learners must not get the same item distribution"

    seen = set()
    for _ in range(5):
        iid = adapt_provider.next_new_grammar_item("adapt_test", learner_urn=weak_learner, exclude=seen)
        assert iid is not None and iid not in seen
        seen.add(iid)
    assert adapt_provider.next_new_grammar_item("adapt_test", learner_urn=weak_learner, exclude=seen) is None, (
        "once every candidate is excluded, must return None, not repeat")

    assert adapt_provider.next_new_grammar_item("adapt_test") in dict(difficulty_spread), (
        "no learner_urn given must still fall back to a valid (uniform) choice")

    print(f"batch_generate added {added}/5 for particle_loc")
    print("all smoke tests passed")
