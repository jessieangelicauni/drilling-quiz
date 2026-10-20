"""
Admin-uploaded grammar curriculum (user request, 2026-10-20): an admin
provides a .docx like Korean_Grammar_Curriculum.docx (524 grammar points,
TOPIK levels 1-6, one table per unit) and items should be generated from
it, rather than only from the 11 hand-coded skills in exaone_provider.py.

This module does ONLY the parsing -- turning the uploaded document into
structured CurriculumEntry rows -- which every downstream piece (kind
classification, item generation, domain-vocabulary substitution) depends
on, so it is built and verified first rather than guessed at alongside
everything else.

Document structure (verified directly against the real file, not
assumed): the body is a flat sequence of paragraphs and tables. A
"Heading 1" paragraph names the TOPIK level, a "Heading 2" paragraph
names the unit within it, and the table immediately following has the
fixed header row ["No.", "Grammar", "Meaning / Usage", "Example",
"Translation"]. The very first table (before any heading) is the
overview/summary table (Level/Focus/Units/Grammar counts) and is
skipped -- it has a different header and no per-point data.
"""
from __future__ import annotations

from dataclasses import dataclass

import docx

_EXPECTED_HEADER = ["No.", "Grammar", "Meaning / Usage", "Example", "Translation"]


@dataclass
class CurriculumEntry:
    level: str       # e.g. "Level 1 — Beginner (TOPIK I, Level 1)"
    unit: str        # e.g. "Unit 1. Sentence Basics & Speech Levels"
    number: str      # the document's own "No." column, kept as-is (not re-indexed)
    grammar: str     # e.g. "N이/가", "-았/었/했-"
    meaning: str     # e.g. "Subject particle", "Past tense"
    example: str     # the curriculum author's own Korean sentence
    translation: str


def parse_curriculum_docx(path: str) -> list[CurriculumEntry]:
    """Walks the document body in order (not docx.Document.tables, which
    loses the surrounding heading context) so every entry can be tagged
    with which level/unit it came from.
    """
    document = docx.Document(path)
    entries: list[CurriculumEntry] = []
    current_level = current_unit = ""

    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            paragraph = docx.text.paragraph.Paragraph(child, document)
            style = paragraph.style.name if paragraph.style else ""
            text = paragraph.text.strip()
            if not text:
                continue
            if style == "Heading 1":
                current_level = text
            elif style == "Heading 2":
                current_unit = text
        elif tag == "tbl":
            table = docx.table.Table(child, document)
            rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
            if not rows or rows[0] != _EXPECTED_HEADER:
                continue  # the overview table, or any other non-grammar table
            for row in rows[1:]:
                if len(row) != 5 or not row[0]:
                    continue  # defensive: a malformed or blank row, skip rather than crash
                number, grammar, meaning, example, translation = row
                entries.append(CurriculumEntry(current_level, current_unit, number,
                                                grammar, meaning, example, translation))
    return entries


if __name__ == "__main__":
    entries = parse_curriculum_docx("Korean_Grammar_Curriculum.docx")
    print(f"parsed {len(entries)} entries")
    assert len(entries) > 500, f"expected ~524 entries per the document's own overview table, got {len(entries)}"

    levels = sorted(set(e.level for e in entries))
    assert len(levels) == 6, f"expected 6 TOPIK levels, got {len(levels)}: {levels}"

    units = sorted(set(e.unit for e in entries))
    assert len(units) == 36, f"expected 36 units per the overview table's total, got {len(units)}"

    # Spot-check real rows seen directly in the document, not invented.
    first = entries[0]
    assert first.level.startswith("Level 1"), first.level
    assert first.grammar == "N입니다 / N입니까?", first.grammar
    assert first.example == "저는 학생입니다.", first.example

    by_number = {e.number: e for e in entries}
    assert by_number["17"].grammar == "N이/가", by_number["17"]
    assert by_number["40"].grammar == "-았/었/했-", by_number["40"]

    # Every number must be unique -- a parsing bug could duplicate rows
    # across tables without this check ever being noticed.
    numbers = [e.number for e in entries]
    assert len(numbers) == len(set(numbers)), "duplicate grammar-point numbers found -- parsing bug"

    print("all smoke tests passed")
    print(f"\nlevels: {levels}")
    print(f"first entry: {first}")
