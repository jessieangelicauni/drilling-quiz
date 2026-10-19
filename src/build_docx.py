# -*- coding: utf-8 -*-
"""Build the IEEE-format DOCX from content.py."""
import re, sys, os
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING, WD_TAB_ALIGNMENT
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import content as C

SERIF = "Times New Roman"
BODY_PT, SMALL_PT, REF_PT = 10, 9, 8


# ----------------------------------------------------------------- helpers
def set_cols(section, n, space_tw=360):
    cols = section._sectPr.xpath("./w:cols")[0]
    cols.set(qn("w:num"), str(n))
    cols.set(qn("w:space"), str(space_tw))
    cols.set(qn("w:equalWidth"), "1")


def east_asian(run, font=SERIF):
    """Ensure Hangul picks a font that actually has the glyphs."""
    rPr = run._element.get_or_add_rPr()
    rf = rPr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts"); rPr.append(rf)
    rf.set(qn("w:eastAsia"), "Malgun Gothic")


def style_run(r, size=BODY_PT, bold=False, italic=False, caps=False):
    r.font.name = SERIF
    r.font.size = Pt(size)
    r.bold = bold
    r.italic = italic
    r.font.color.rgb = RGBColor(0, 0, 0)
    if caps:
        r.font.small_caps = True
    east_asian(r)
    return r


def para(doc, space_after=0, space_before=0, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
         first_indent=None, line=1.0):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(space_before)
    pf.alignment = align
    pf.line_spacing = line
    pf.widow_control = True
    if first_indent is not None:
        pf.first_line_indent = Inches(first_indent)
    return p


_TAG = re.compile(r"(<i>.*?</i>)", re.S)
_MATH = re.compile(r"(_\{[^}]*\}|\^\{[^}]*\})")


def _emit(p, text, size, bold, italic):
    """Emit text, turning _{x} / ^{x} into real sub/superscript runs."""
    for piece in _MATH.split(text):
        if not piece:
            continue
        if piece.startswith("_{") or piece.startswith("^{"):
            r = style_run(p.add_run(piece[2:-1]), size, bold, italic)
            if piece[0] == "_":
                r.font.subscript = True
            else:
                r.font.superscript = True
        else:
            style_run(p.add_run(piece.replace("&amp;", "&")), size, bold, italic)


def rich(p, text, size=BODY_PT, bold=False, italic=False):
    """Write text honouring <i>...</i> spans, &amp;, and _{}/^{} math markup."""
    for chunk in _TAG.split(text):
        if not chunk:
            continue
        if chunk.startswith("<i>"):
            _emit(p, chunk[3:-4], size, bold, True)
        else:
            _emit(p, chunk, size, bold, italic)
    return p


def add_equation(doc, text):
    """Centred formula with a right-aligned equation number after a tab."""
    p = para(doc, space_before=5, space_after=5, align=WD_ALIGN_PARAGRAPH.LEFT)
    pf = p.paragraph_format
    pf.left_indent = Inches(0.10)
    tabs = pf.tab_stops
    tabs.add_tab_stop(Inches(3.37), WD_TAB_ALIGNMENT.RIGHT)
    rich(p, text, BODY_PT, italic=True)
    return p


# ----------------------------------------------------------------- document
doc = Document()

st = doc.styles["Normal"]
st.font.name = SERIF
st.font.size = Pt(BODY_PT)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Malgun Gothic")

s0 = doc.sections[0]
s0.page_width, s0.page_height = Inches(8.5), Inches(11)
s0.top_margin = Inches(0.75)
s0.bottom_margin = Inches(1.0)
s0.left_margin = s0.right_margin = Inches(0.625)
set_cols(s0, 1)

# ---- title block (full width)
p = para(doc, space_after=6, align=WD_ALIGN_PARAGRAPH.CENTER)
style_run(p.add_run(C.TITLE), 20)

p = para(doc, space_after=2, align=WD_ALIGN_PARAGRAPH.CENTER)
style_run(p.add_run(C.AUTHORS), 11)

for line in C.AFFIL.split("\n"):
    p = para(doc, space_after=0, align=WD_ALIGN_PARAGRAPH.CENTER)
    style_run(p.add_run(line), 9, italic=True)

para(doc, space_after=4)

# ---- switch to two columns
s1 = doc.add_section(WD_SECTION.CONTINUOUS)
s1.top_margin, s1.bottom_margin = Inches(0.75), Inches(1.0)
s1.left_margin = s1.right_margin = Inches(0.625)
set_cols(s1, 2)

# ---- abstract + index terms
p = para(doc, space_after=6)
style_run(p.add_run("Abstract—"), SMALL_PT, bold=True, italic=True)
style_run(p.add_run(C.ABSTRACT), SMALL_PT, bold=True)

p = para(doc, space_after=10)
style_run(p.add_run("Index Terms—"), SMALL_PT, bold=True, italic=True)
style_run(p.add_run(C.INDEX_TERMS), SMALL_PT, bold=True)


def add_fullwidth_figure(path, width_in, caption):
    """Figure spanning both columns, via continuous 1-column section."""
    a = doc.add_section(WD_SECTION.CONTINUOUS)
    a.top_margin, a.bottom_margin = Inches(0.75), Inches(1.0)
    a.left_margin = a.right_margin = Inches(0.625)
    set_cols(a, 1)

    p = para(doc, space_before=6, space_after=3, align=WD_ALIGN_PARAGRAPH.CENTER)
    p.add_run().add_picture(path, width=Inches(width_in))

    p = para(doc, space_after=8, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    rich(p, caption, SMALL_PT - 1)

    b = doc.add_section(WD_SECTION.CONTINUOUS)
    b.top_margin, b.bottom_margin = Inches(0.75), Inches(1.0)
    b.left_margin = b.right_margin = Inches(0.625)
    set_cols(b, 2)


ANCHORS = [
    ("Fig. 1 shows the architecture.", 0),
    ("This inverts the usual relationship", 1),
    ("That is, when the system is genuinely uncertain", 2),
    ("Two remedies are available.", 3),
    ("Participants complete a TOPIK-aligned pre-test", 4),
]
anchor_map = {}
for prefix, fidx in ANCHORS:
    anchor_map.setdefault(prefix, []).append(fidx)

# ---- body
first_para_of_section = True
for kind, text in C.BODY:
    if kind == "H1":
        p = para(doc, space_before=10, space_after=4, align=WD_ALIGN_PARAGRAPH.CENTER)
        style_run(p.add_run(text.upper()), BODY_PT, caps=False)
        first_para_of_section = True

    elif kind == "H2":
        p = para(doc, space_before=7, space_after=3, align=WD_ALIGN_PARAGRAPH.LEFT)
        style_run(p.add_run(text), BODY_PT, italic=True)
        first_para_of_section = True

    elif kind == "EQ":
        add_equation(doc, text)
        first_para_of_section = False

    elif kind == "L":
        p = para(doc, space_after=3, first_indent=None)
        p.paragraph_format.left_indent = Inches(0.17)
        p.paragraph_format.first_line_indent = Inches(-0.17)
        style_run(p.add_run("•  "), BODY_PT)
        rich(p, text)
        first_para_of_section = False

    else:  # "P"
        p = para(doc, space_after=3,
                 first_indent=None if first_para_of_section else 0.17)
        rich(p, text)
        first_para_of_section = False

        for prefix, idxs in anchor_map.items():
            if text.startswith(prefix):
                for fi in idxs:
                    path, w, cap = C.FIGURES[fi]
                    add_fullwidth_figure(path, w, cap)
                first_para_of_section = True

# ---- references
p = para(doc, space_before=10, space_after=4, align=WD_ALIGN_PARAGRAPH.CENTER)
style_run(p.add_run("REFERENCES"), BODY_PT)

for i, ref in enumerate(C.REFERENCES, 1):
    p = para(doc, space_after=2)
    p.paragraph_format.left_indent = Inches(0.22)
    p.paragraph_format.first_line_indent = Inches(-0.22)
    style_run(p.add_run(f"[{i}]\t"), REF_PT)
    rich(p, ref, REF_PT)

out = "paper/KO-DRILL_IEEE.docx"
doc.save(out)
print("saved:", out)
