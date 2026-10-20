"""
Serves curriculum-generated grammar items (curriculum_item_generator.py)
through the same ContentProvider protocol daily_drill.py already defines
for exaone_provider.ExaoneGrammarProvider (PRD §9.49).

skill_id is f"curriculum_{number}" -- one skill per grammar point (user's
explicit choice over a single shared bucket, PRD §9.49), so each of the
524 curriculum points gets its own independent Urnings mastery track
through the existing per-skill scheduling machinery, same as the 11
hand-coded skills. next_new_grammar_item looks up that single
CurriculumEntry by number and tries to generate+validate one MCQ item
for it on demand -- same lazy, on-demand generation pattern as
ExaoneGrammarProvider, just with a much lower per-call success rate.

Production rate is NOT ~100% like the hand-coded skills (PRD §9.46/9.48:
~34% of entries produce a usable item even after all 6 checks in
curriculum_item_generator.py) -- a caller must expect
next_new_grammar_item to return None often for a GIVEN skill_id and move
on, which assemble_daily_drill's existing `attempts` retry loop already
does (that loop was written for the 11 skills' occasional miss, not a
~66% miss rate -- worth revisiting the retry budget if curriculum skills
end up dominating state.skill_ids, not addressed here).
"""
from __future__ import annotations

from curriculum_item_generator import CurriculumItem, CurriculumItemGenerator
from curriculum_provider import CurriculumEntry, parse_curriculum_docx

_SKILL_PREFIX = "curriculum_"


class CurriculumGrammarProvider:
    def __init__(self, docx_path: str = "Korean_Grammar_Curriculum.docx", seed: int | None = None):
        entries = parse_curriculum_docx(docx_path)
        self.entries_by_number: dict[str, CurriculumEntry] = {e.number: e for e in entries}
        self.generator = CurriculumItemGenerator(seed=seed)
        self.items: dict[str, CurriculumItem] = {}
        self._counter = 0

    def next_new_grammar_item(self, skill_id: str, learner_urn=None,
                               exclude: set[str] = frozenset()) -> str | None:
        if not skill_id.startswith(_SKILL_PREFIX):
            return None
        number = skill_id[len(_SKILL_PREFIX):]
        entry = self.entries_by_number.get(number)
        if entry is None:
            return None
        item = self.generator.generate(entry)
        if item is None:
            return None
        item_id = f"{skill_id}-{self._counter}"
        self._counter += 1
        self.items[item_id] = item
        return item_id


if __name__ == "__main__":
    entries = parse_curriculum_docx("Korean_Grammar_Curriculum.docx")
    # Entry "40" (-았/었/했-) is one of the reliably-produced points from
    # the §9.48 sweep -- used here so this smoke test doesn't depend on
    # GPU luck on an untested entry.
    provider = CurriculumGrammarProvider(seed=0)

    item_id = provider.next_new_grammar_item("curriculum_40")
    assert item_id is not None, "entry 40 is a known-reliable point from the §9.48 sweep"
    item = provider.items[item_id]
    assert item.skill_id == "curriculum_40"
    assert len(item.options) == 4
    assert item.answer in item.options
    print(f"curriculum_40 -> {item_id}: stem={item.stem!r} options={item.options!r} answer={item.answer!r}")

    # A skill_id this provider doesn't own must return None, not raise or
    # silently hand back someone else's item -- the dispatcher in
    # content_provider.py depends on this to route correctly.
    assert provider.next_new_grammar_item("particle_loc") is None
    assert provider.next_new_grammar_item("curriculum_99999") is None, "a nonexistent entry number must not crash"

    print("all smoke tests passed")
