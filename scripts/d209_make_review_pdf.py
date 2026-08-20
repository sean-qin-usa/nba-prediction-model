#!/usr/bin/env python3
"""D209 — render docs/REVIEW.md to nba_model_and_strategy_review.pdf.

Replaces the owner's original PDF, whose figures predate D186 (coverage frame),
D199 (bet-time leak), D207 (trading frame) and D208 (window selection).

Deliberately a RENDERER, not a second source of truth: it reads the published
markdown, so the PDF can never drift from the repo. Regenerate after any edit to
REVIEW.md.

PATH CONTRACT — DO NOT RENAME `OUT`. That filename is a PUBLISHED URL:

    https://github.com/sean-qin-usa/nba-prediction-model/blob/main/nba_model_and_strategy_review.pdf

It is the repo's main summary document and is linked from the top of README.md.
Renaming the file, moving it into a subdirectory, or deleting it in favour of a
generated artefact breaks that link for anyone holding it. Re-rendering in place
is always correct and always preserves the URL; replacing the path is never a
cosmetic change.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent / "nba-prediction-model"
SRC = REPO / "docs" / "REVIEW.md"
OUT = REPO / "nba_model_and_strategy_review.pdf"

from reportlab import rl_config                                   # noqa: E402
from reportlab.lib import colors                                  # noqa: E402
from reportlab.pdfbase import pdfmetrics                          # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont                      # noqa: E402

# FONT EMBEDDING. ReportLab's default base-14 Helvetica/Courier are NOT embedded
# (pdffonts reports emb=no); the viewer substitutes, and on many systems that
# renders as merged or wildly spaced glyphs. Register DejaVu as a real TrueType
# so the file carries its own outlines. Verified after every build with
# `pdffonts`, which must report emb=yes for every face.
_FD = Path("/usr/share/fonts/truetype/dejavu")
_FACES = [("DejaVu", "DejaVuSans.ttf"), ("DejaVu-Bold", "DejaVuSans-Bold.ttf"),
          ("DejaVu-Oblique", "DejaVuSans-Oblique.ttf"),
          ("DejaVu-BoldOblique", "DejaVuSans-BoldOblique.ttf"),
          ("DejaVuMono", "DejaVuSansMono.ttf"),
          ("DejaVuMono-Bold", "DejaVuSansMono-Bold.ttf")]
for _n, _f in _FACES:
    pdfmetrics.registerFont(TTFont(_n, str(_FD / _f)))
from reportlab.pdfbase.pdfmetrics import registerFontFamily        # noqa: E402
registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold",
                   italic="DejaVu-Oblique", boldItalic="DejaVu-BoldOblique")
# ReportLab registers its base-14 Helvetica as the canvas base font BEFORE any
# document exists, so setFont() on the page callback is too late — the
# unembedded resource is already in the file. This is the switch that removes it.
rl_config.canvas_basefontname = "DejaVu"
from reportlab.lib.enums import TA_LEFT                           # noqa: E402
from reportlab.lib.pagesizes import LETTER                        # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import inch                              # noqa: E402
from reportlab.platypus import (BaseDocTemplate, Frame, HRFlowable,  # noqa: E402
                                Image, PageBreak, PageTemplate, Paragraph,
                                Spacer,
                                Table, TableStyle)

NAVY = colors.HexColor("#1f3864")
INK = colors.HexColor("#111111")
INK2 = colors.HexColor("#444444")
RULE = colors.HexColor("#c9d3e3")
BAND = colors.HexColor("#eef2f8")

ss = getSampleStyleSheet()
S = {
    "h1": ParagraphStyle("h1", parent=ss["Title"], fontName="DejaVu-Bold",
                         fontSize=14, leading=17, textColor=NAVY,
                         alignment=TA_LEFT, spaceAfter=2),
    "h2": ParagraphStyle("h2", fontName="DejaVu-Bold", fontSize=10.5,
                         leading=12.5, textColor=NAVY, spaceBefore=7,
                         spaceAfter=1.5, keepWithNext=1, borderWidth=0, borderPadding=0,
                         underlineWidth=0),
    "h3": ParagraphStyle("h3", fontName="DejaVu-Bold", fontSize=8.8,
                         leading=11, textColor=INK, spaceBefore=5,
                         spaceAfter=2, keepWithNext=1),
    "p": ParagraphStyle("p", fontName="DejaVu", fontSize=7.5, leading=9.9,
                        textColor=INK, spaceAfter=3.4),
    "li": ParagraphStyle("li", fontName="DejaVu", fontSize=7.5, leading=9.9,
                         textColor=INK, leftIndent=10, bulletIndent=3,
                         spaceAfter=1.4, bulletFontName="DejaVu",
                         bulletFontSize=7.5),
    "code": ParagraphStyle("code", fontName="DejaVuMono", fontSize=7.2, leading=9.4,
                           textColor=INK, backColor=BAND, leftIndent=6,
                           spaceBefore=3, spaceAfter=5, borderPadding=4),
    "sub": ParagraphStyle("sub", fontName="DejaVu", fontSize=7.8,
                          leading=10, textColor=INK2, spaceAfter=8),
    "quote": ParagraphStyle("quote", fontName="DejaVu-Oblique", fontSize=7.0,
                            leading=9.2, textColor=INK2, leftIndent=8,
                            spaceAfter=5),
}


def inline(s: str) -> str:
    """markdown inline -> reportlab mini-HTML."""
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)
    # markdown link -> a real PDF link annotation
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
               lambda m: (f'<link href="{m.group(2)}" color="#1f3864">'
                          f'{m.group(1)}</link>')
               if m.group(2).startswith(("http", "mailto")) else m.group(1), s)
    # bare github.com/... -> clickable, without duplicating an existing <link>
    if "<link" not in s:
        s = re.sub(r"(?<![\w/])((?:https?://)?github\.com/[\w./-]+)",
                   lambda m: (f'<link href="https://'
                              f'{m.group(1).replace("https://", "")}" '
                              f'color="#1f3864">{m.group(1)}</link>'), s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", s)
    s = re.sub(r"`([^`]+)`", r'<font face="DejaVuMono">\1</font>', s)
    return s


def make_table(rows):
    hdr, body = rows[0], rows[1:]
    data = [[Paragraph(f"<b>{inline(c)}</b>",
                       ParagraphStyle("th", parent=S["p"], fontSize=6.8,
                                      leading=8.6, textColor=colors.white))
             for c in hdr]]
    for r in body:
        data.append([Paragraph(inline(c),
                               ParagraphStyle("td", parent=S["p"], fontSize=6.8,
                                              leading=8.6, spaceAfter=0))
                     for c in r])
    t = Table(data, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        # without an explicit FONTNAME the table declares an unembedded
        # Helvetica resource even though every cell is a DejaVu Paragraph
        ("FONTNAME", (0, 0), (-1, -1), "DejaVu"),
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BAND]),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 1.8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8),
    ]))
    return t


def build():
    md = SRC.read_text().split("\n")
    flow, i, n_tbl = [], 0, 0
    while i < len(md):
        ln = md[i]
        if ln.startswith("|") and i + 1 < len(md) and set(md[i + 1].replace("|", "").strip()) <= set("-: "):
            rows = []
            while i < len(md) and md[i].startswith("|"):
                cells = [c.strip() for c in md[i].strip().strip("|").split("|")]
                if not set("".join(cells)) <= set("-: "):
                    rows.append(cells)
                i += 1
            flow.append(make_table(rows))
            flow.append(Spacer(1, 3))
            n_tbl += 1
            continue
        if ln.startswith("```"):
            i += 1
            buf = []
            while i < len(md) and not md[i].startswith("```"):
                buf.append(md[i]); i += 1
            i += 1
            flow.append(Paragraph("<br/>".join(
                x.replace(" ", "&nbsp;") for x in buf), S["code"]))
            continue
        m = re.match(r"^(#{1,3})\s+(.*)", ln)
        if m:
            lvl = len(m.group(1))
            flow.append(Paragraph(inline(m.group(2)),
                                  S["h1" if lvl == 1 else "h2" if lvl == 2 else "h3"]))
            if lvl == 1:
                # the owner's original carries a rule directly under the title
                flow.append(HRFlowable(width="100%", thickness=1.1, color=NAVY,
                                       spaceBefore=2, spaceAfter=4))
            if lvl == 2:
                hr = HRFlowable(width="100%", thickness=0.7, color=RULE,
                                spaceBefore=1, spaceAfter=3.5)
                hr.keepWithNext = 1
                flow.append(hr)
            i += 1
            continue
        if ln.startswith("!["):
            mm = re.match(r"!\[[^\]]*\]\(([^)]+)\)", ln.strip())
            if mm:
                p = (SRC.parent / mm.group(1)).resolve()
                if p.exists():
                    from reportlab.lib.utils import ImageReader
                    iw, ih = ImageReader(str(p)).getSize()
                    w = 6.55 * inch
                    flow.append(Image(str(p), width=w, height=w * ih / iw))
                    flow.append(Spacer(1, 4))
            i += 1
            continue
        if ln.startswith(">"):
            buf = []
            while i < len(md) and md[i].startswith(">"):
                buf.append(md[i].lstrip("> ").rstrip()); i += 1
            flow.append(Paragraph(inline(" ".join(buf)), S["quote"]))
            continue
        if re.match(r"^\s*[-*]\s+", ln):
            # A markdown bullet may wrap over several lines, its continuations
            # INDENTED and not themselves bullets. Without folding them in they
            # render as separate unindented paragraphs, which is what the first
            # PDF build did to every multi-line bullet on page 3.
            while i < len(md) and re.match(r"^\s*[-*]\s+", md[i]):
                parts = [re.sub(r"^\s*[-*]\s+", "", md[i])]
                i += 1
                while (i < len(md) and md[i].strip()
                       and md[i].startswith((" ", "\t"))
                       and not re.match(r"^\s*[-*]\s+", md[i])):
                    parts.append(md[i].strip())
                    i += 1
                flow.append(Paragraph(inline(" ".join(parts)), S["li"],
                                      bulletText="\u2022"))
            flow.append(Spacer(1, 2))
            continue
        if ln.strip() == "<!--PAGEBREAK-->":
            flow.append(PageBreak())
            i += 1
            continue
        if ln.strip() in ("", "---"):
            i += 1
            continue
        buf = []
        while i < len(md) and md[i].strip() and not md[i].startswith(("|", "#", ">", "!", "```")) \
                and not re.match(r"^\s*[-*]\s+", md[i]):
            buf.append(md[i].strip()); i += 1
        flow.append(Paragraph(inline(" ".join(buf)), S["p"]))

    def deco(canv, doc):
        # ReportLab opens each page with its default Helvetica in the resource
        # dict. Setting the font here means the only face the file declares is
        # an embedded one; pdffonts must come back with zero emb=no rows.
        canv.setFont("DejaVu", 7.0)
        canv.saveState()
        # The owner's original carries NO running header — the github line is a
        # subtitle under the title on page 1. Page number only.
        canv.setFont("DejaVu", 7.0); canv.setFillColor(INK2)
        canv.drawCentredString(LETTER[0] / 2, 0.4 * inch, f"{doc.page}")
        canv.restoreState()

    doc = BaseDocTemplate(str(OUT), pagesize=LETTER,
                          leftMargin=0.58 * inch, rightMargin=0.58 * inch,
                          topMargin=0.48 * inch, bottomMargin=0.45 * inch,
                          title="NBA opening-spread relative value — review",
                          author="sean-qin-usa")
    fr = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    doc.addPageTemplates([PageTemplate(id="main", frames=[fr], onPage=deco)])
    doc.build(flow)
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB, {n_tbl} tables)")


if __name__ == "__main__":
    build()
