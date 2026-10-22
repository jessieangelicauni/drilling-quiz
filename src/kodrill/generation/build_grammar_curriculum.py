import json
import re

import docx
from docx.table import Table
from docx.text.paragraph import Paragraph

SRC = "data/Korean_Grammar_Curriculum.docx"
OUT = "data/korean_grammar_curriculum.json"


def iter_block_items(doc):
    body = doc.element.body
    for child in body.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, doc)
        elif child.tag.endswith("}tbl"):
            yield Table(child, doc)


def level_label(heading: str) -> str:
    m = re.search(r"—\s*([\w-]+)\s*\((.+)\)", heading)
    return f"{m.group(1).lower()} ({m.group(2)})" if m else heading


def build(src: str = SRC) -> dict:
    doc = docx.Document(src)
    items = {}
    current_level, table_index = "", -1
    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            if block.style.name == "Heading 1":
                current_level = level_label(block.text.strip())
            continue
        table_index += 1
        if table_index == 0:
            continue
        header = [c.text.strip() for c in block.rows[0].cells]
        if header[:2] != ["No.", "Grammar"]:
            continue
        for row in block.rows[1:]:
            no, grammar, meaning, example, translation = (c.text.strip() for c in row.cells)
            item_id = f"gp_{int(no):03d}"
            description = f"{grammar} -- {meaning}. Example: {example} ({translation})"
            items[item_id] = {"description": description, "level": current_level}
    return items


if __name__ == "__main__":
    items = build()
    assert len(items) == 524, f"expected 524 grammar points, got {len(items)}"
    assert all(v["description"].strip() and v["level"].strip() for v in items.values())
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"items": items}, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(items)} items to {OUT}")
