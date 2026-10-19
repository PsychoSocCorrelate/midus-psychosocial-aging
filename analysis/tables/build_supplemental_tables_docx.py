#!/usr/bin/env python3
"""Render the Supplementary Tables workbook as a Word document for the SI Appendix.

Reads `manuscript/supplement/supplementary_tables_combined.xlsx` (step `supplement_tables`) and
writes `manuscript/supplement/supplementary_tables.docx`: one table per sheet, captioned the way
the SI captions its figures ("SI Table S1. Title."), in the SI's Times New Roman, with the
sheet's column spanners, stub sections, panels, bold cells and notes. Every cell reads as the
workbook displays it. The changes are typographic only: exponents and note markers become
superscript runs, the W and B of the within- and between-family betas and the u and e of the
variance components become subscripts, a negative number takes a true minus sign, and an
interval or a power of ten does not break across lines.

Each table opens a page, in portrait when its columns fit there unwrapped and in landscape
otherwise. Header rows repeat on every page and no row splits across pages. Table S17's six
outcome-by-layer column groups do not fit a landscape page at a readable size, so it is set as
two panels, cross-sectional and 9-year change, each with the three outcomes; every cell is kept.

The document carries direct formatting only and no styles, so its body can be spliced into
the SI document unchanged. Column
widths are measured with Liberation Serif, which is metric-compatible with Times New Roman.
"""

from __future__ import annotations

import argparse
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from lxml import etree
from openpyxl import load_workbook
from openpyxl.cell.rich_text import CellRichText, TextBlock
from PIL import ImageFont

ROOT = Path(__file__).resolve().parents[2]
WORKBOOK = ROOT / "manuscript" / "supplement" / "supplementary_tables_combined.xlsx"
OUT = ROOT / "manuscript" / "supplement" / "supplementary_tables.docx"

FONT = "Times New Roman"
CAPTION_PT = 10
SIZES_PT = (9, 8.5, 8)          # the table's font, tried in order until its columns fit
TWIP = 20                       # twips per point
PAGE_SHORT, PAGE_LONG = 12240, 15840   # US Letter, twips
MARGIN = 1440                   # 1 inch, as the SI
CELL_PAD = 58                   # left and right cell margin, twips (0.04 in)
CELL_VPAD = 16                  # top and bottom cell margin, twips
RULE_OUTER, RULE_INNER = 6, 4   # border widths, eighths of a point


# ── Workbook model ──────────────────────────────────────────────────────────
@dataclass
class Cell:
    text: str
    span: int = 1
    bold: bool = False
    left: bool = False


@dataclass
class Section:
    """A stub heading spanning the table, in italic."""
    text: str


@dataclass
class Block:
    """One header and body. `head` rows are spanners then column heads; a row holds one slot
    per column, None where a span covers it."""
    ncols: int
    head: list[list[Cell | None]]
    body: list[list[Cell | None] | Section]
    panel: str | None = None


@dataclass
class Table:
    key: str
    title: str
    blocks: list[Block]
    notes: list[list[tuple[str, bool]]]      # paragraphs of (text, italic) runs


_SECTION = re.compile(r"^(?P<lit>[-+]?)(?P<int>[#0,]*)(?:\.(?P<dec>0+))?$")


def excel_display(value: float | int, fmt: str) -> str:
    """The text Excel shows for a number under the number formats the builder writes."""
    sections = fmt.split(";")
    if value < 0 and len(sections) > 1:
        sec, sign = sections[1], ""
    elif value == 0 and len(sections) > 2:
        sec, sign = sections[2], ""
    else:
        sec, sign = sections[0], "-" if value < 0 else ""
    m = _SECTION.match(sec)
    if not m:
        raise ValueError(f"unsupported number format {fmt!r}")
    d = len(m["dec"] or "")
    text = f"{abs(value):,.{d}f}" if "," in m["int"] else f"{abs(value):.{d}f}"
    if "0" not in m["int"] and text.startswith("0."):
        text = text[1:]
    return m["lit"] + sign + text


def display(cell) -> str:
    v = cell.value
    if v is None:
        return ""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return excel_display(v, cell.number_format)
    return str(v)


def _thin(side) -> bool:
    return side is not None and side.style is not None


def parse_sheet(ws) -> Table:
    if ws.cell(1, 1).value != f"Table {ws.title}":
        raise ValueError(f"sheet {ws.title}: A1 is {ws.cell(1, 1).value!r}")
    title = str(ws.cell(2, 1).value)
    ncols, nrows = ws.max_column, ws.max_row
    spans: dict[tuple[int, int], int] = {}
    for rng in ws.merged_cells.ranges:
        if rng.min_row != rng.max_row:
            raise ValueError(f"sheet {ws.title}: vertical merge {rng}")
        spans[(rng.min_row, rng.min_col)] = rng.max_col - rng.min_col + 1
    covered = {(r, c0 + k) for (r, c0), n in spans.items() for k in range(1, n)}

    def ruled(r: int, side: str) -> bool:
        return all(_thin(getattr(ws.cell(r, c).border, side)) for c in range(1, ncols + 1))

    def empty(r: int) -> bool:
        return all(ws.cell(r, c).value is None for c in range(1, ncols + 1))

    def full_width(r: int) -> bool:
        return spans.get((r, 1)) == ncols

    def slots(r: int, width: int) -> list[Cell | None]:
        row: list[Cell | None] = []
        for c in range(1, width + 1):
            if (r, c) in covered:
                row.append(None)
                continue
            cell = ws.cell(r, c)
            row.append(Cell(display(cell), span=min(spans.get((r, c), 1), width - c + 1),
                            bold=bool(cell.font.b),
                            left=(cell.alignment.horizontal == "left")))
        for c in range(width + 1, ncols + 1):
            if (r, c) not in covered and ws.cell(r, c).value is not None:
                raise ValueError(f"sheet {ws.title}: {ws.cell(r, c).coordinate} outside the block")
        return row

    blocks: list[Block] = []
    notes: list[list[tuple[str, bool]]] = []
    panel = None
    r = 3
    while r <= nrows:
        if empty(r):
            r += 1
            continue
        if _thin(ws.cell(r, 1).border.top):
            start = r
            while not ruled(r, "bottom"):
                r += 1
            head_rows = list(range(start, r + 1))
            last_head = head_rows[-1]
            width = max(c for c in range(1, ncols + 1)
                        if ws.cell(last_head, c).value is not None)
            r += 1
            body_start = r
            while not ruled(r, "bottom"):
                r += 1
            body: list[list[Cell | None] | Section] = []
            for br in range(body_start, r + 1):
                if full_width(br) and ws.cell(br, 1).font.i:
                    body.append(Section(display(ws.cell(br, 1))))
                else:
                    body.append(slots(br, width))
            blocks.append(Block(width, [slots(hr, width) for hr in head_rows], body, panel))
            panel = None
            r += 1
            continue
        value = ws.cell(r, 1).value
        if full_width(r) and ws.cell(r, 1).font.b and not isinstance(value, CellRichText):
            panel = str(value)
        elif blocks:
            if isinstance(value, CellRichText):
                notes.append([(str(b.text) if isinstance(b, TextBlock) else str(b),
                               bool(isinstance(b, TextBlock) and b.font.i)) for b in value])
            else:
                notes.append([(str(value), False)])
        else:
            raise ValueError(f"sheet {ws.title}: unplaced row {r}")
        r += 1
    if not blocks or panel is not None:
        raise ValueError(f"sheet {ws.title}: no table body, or a panel heading without one")
    return Table(ws.title, title, blocks, notes)


def split_s17_by_layer(t: Table) -> Table:
    """Set S17's outcome-by-layer column groups as two panels, one per layer.

    The sheet has a stub, six groups of three columns (outcomes over layers), and a last column
    counting top-20 lists over all six. Each panel keeps the stub and one layer's three groups
    under their outcome; the list count goes with the second. Rows with nothing in a panel's
    columns (the cross-layer rows, in the cross-sectional panel) are left out of that panel.
    """
    (b,) = t.blocks
    outcomes, layers, heads = b.head
    if b.ncols != 20 or len(b.head) != 3:
        raise ValueError("S17 no longer has the layout this split expects")
    names = [c.text for c in outcomes if c is not None and c.text]
    layer_names = [c.text for c in layers if c is not None and c.text][:2]
    if len(names) != 3 or len(layer_names) != 2:
        raise ValueError("S17 spanners changed")
    blocks = []
    for li, layer in enumerate(layer_names):
        cols = [0] + [1 + (o * 2 + li) * 3 + k for o in range(3) for k in range(3)]
        if li == 1:
            cols.append(19)

        def take(row: list[Cell | None]) -> list[Cell | None]:
            out: list[Cell | None] = []
            for c in cols:
                cell = row[c]
                if cell is not None and cell.span > 1 and not all(
                        c + k in cols for k in range(cell.span)):
                    raise ValueError("S17 span crosses a panel")
                out.append(cell)
            return out

        spanner: list[Cell | None] = [Cell("")]
        for name in names:
            spanner += [Cell(name, span=3), None, None]
        if li == 1:
            spanner.append(Cell(""))
        body: list[list[Cell | None] | Section] = []
        for row in b.body:
            if isinstance(row, Section):
                body.append(row)
                continue
            picked = take(row)
            if any(c is not None and c.text for c in picked[1:]):
                body.append(picked)
        blocks.append(Block(len(cols), [spanner, take(heads)], body,
                            panel=f"Panel {'AB'[li]}. {layer}"))
    return Table(t.key, t.title, blocks, t.notes)


def read_tables(path: Path) -> list[Table]:
    wb = load_workbook(path, rich_text=True)
    tables = [parse_sheet(ws) for ws in wb.worksheets if ws.title != "Contents"]
    return [split_s17_by_layer(t) if t.key == "S17" else t for t in tables]


# ── Typography ──────────────────────────────────────────────────────────────
SUPER = dict(zip("⁰¹²³⁴⁵⁶⁷⁸⁹⁻ᵃᵇᶜᵈ", "0123456789−abcd"))
_SUPER_RUN = re.compile(f"[{''.join(SUPER)}]+")
_MINUS = re.compile(r"(?<![\w.\])])-(?=\.?\d)")
_INTERVAL = re.compile(r"\[([^\[\],]+), ([^\[\],]+)\]")
_ABS = re.compile(r"\|[^|\s]+ [^|\s]+\|")            # |βW std| stays on one line
_RELATION = re.compile(r" ([=<>≤≥]) (?=[.\d])")       # q ≤ .05, n = 836
_SUB = re.compile(r"(?<=β)([WB])(?![a-z])|(?<=σ²)([ue])(?![a-z])")
NBSP = "\u00a0"


def typeset(text: str) -> str:
    text = _MINUS.sub("−", text)
    text = _INTERVAL.sub(lambda m: f"[{m[1]},{NBSP}{m[2]}]", text)
    text = _ABS.sub(lambda m: m[0].replace(" ", NBSP), text)
    return _RELATION.sub(lambda m: f"{NBSP}{m[1]}{NBSP}", text.replace(" × 10", f"{NBSP}×{NBSP}10"))


def runs(text: str) -> list[tuple[str, str | None]]:
    """Split typeset text into (text, vertAlign) runs."""
    text = typeset(text)
    marks: list[tuple[int, int, str, str]] = []
    for m in _SUPER_RUN.finditer(text):
        if set(m[0]) - set("²³"):          # ² and ³ are in the font; leave them as glyphs
            marks.append((m.start(), m.end(), "".join(SUPER[ch] for ch in m[0]), "superscript"))
    for m in _SUB.finditer(text):
        marks.append((m.start(), m.end(), m[0], "subscript"))
    out: list[tuple[str, str | None]] = []
    pos = 0
    for s, e, rep, vert in sorted(marks):
        if s > pos:
            out.append((text[pos:s], None))
        out.append((rep, vert))
        pos = e
    if pos < len(text):
        out.append((text[pos:], None))
    return out


# ── Measurement ─────────────────────────────────────────────────────────────
_FONT_DIRS = [Path("/usr/share/fonts/truetype/liberation"), Path("/usr/share/fonts/liberation"),
              Path("/Library/Fonts"), Path("C:/Windows/Fonts")]
_FONT_FILES = {False: ("LiberationSerif-Regular.ttf", "times.ttf", "Times New Roman.ttf"),
               True: ("LiberationSerif-Bold.ttf", "timesbd.ttf", "Times New Roman Bold.ttf")}
_fonts: dict[bool, ImageFont.FreeTypeFont | None] = {}


def _font(bold: bool):
    if bold not in _fonts:
        _fonts[bold] = None
        for d in _FONT_DIRS:
            for name in _FONT_FILES[bold]:
                if (d / name).exists():
                    _fonts[bold] = ImageFont.truetype(str(d / name), 100)
                    break
            if _fonts[bold]:
                break
    return _fonts[bold]


def text_width(text: str, pt: float, bold: bool = False) -> float:
    """Width in twips of text set at pt points, sub- and superscripts counted at full size."""
    font = _font(bold)
    plain = text.replace(NBSP, " ")
    em = font.getlength(plain) / 100 if font else 0.5 * len(plain)
    return em * pt * TWIP


def tokens(text: str) -> list[str]:
    return [tok for line in text.split("\n") for tok in line.split(" ") if tok]


_NUMERIC = re.compile(r"^[\s\d.,+\-−×⁰¹²³⁴⁵⁶⁷⁸⁹⁻ᵃ\[\]()%/]+$")


def column_widths(b: Block, pt: float, avail: int) -> tuple[list[float], float, float]:
    """Widths for a block's columns.

    Each column has a minimum (its longest unbreakable token) and a natural width (its longest
    line). Numeric columns do not wrap while the table still fits. When the natural widths do
    not fit, the space is water-filled: every column is held to one common width or its natural
    width, whichever is smaller, never below its minimum, so short columns stay on one line and
    only the long text columns wrap. Returns the widths and the total minimum and natural widths.
    """
    pad = 2 * CELL_PAD + 2 * TWIP
    lo = [0.0] * b.ncols
    hi = [0.0] * b.ncols
    body_hi = [0.0] * b.ncols          # natural width of the body alone; a header may wrap
    numeric_hits = [[0, 0] for _ in range(b.ncols)]
    rows = [r for r in b.body if not isinstance(r, Section)]
    for row in rows + [b.head[-1]]:
        for j, cell in enumerate(row):
            if cell is None or cell.span != 1 or not cell.text:
                continue
            txt = typeset(cell.text)
            lo[j] = max(lo[j], max(text_width(t, pt, cell.bold) for t in tokens(txt)) + pad)
            hi[j] = max(hi[j], text_width(txt, pt, cell.bold) + pad)
            if row is not b.head[-1]:
                body_hi[j] = max(body_hi[j], text_width(txt, pt, cell.bold) + pad)
                numeric_hits[j][1] += 1
                numeric_hits[j][0] += bool(_NUMERIC.match(cell.text))
    numeric = [bool(n) and h / n >= 0.9 for h, n in numeric_hits]
    floor = [max(body_hi[j] if numeric[j] else lo[j], lo[j]) for j in range(b.ncols)]
    if sum(floor) > avail:
        floor = lo
    total_lo, total_hi = sum(floor), sum(hi)
    if total_hi <= avail:
        target = avail if total_hi >= 0.55 * avail else min(avail, total_hi * 1.25)
        widths = [h + (target - total_hi) * h / total_hi for h in hi]
    elif total_lo <= avail:
        widths = shortest_table(b, rows, pt, floor, hi, avail, pad)
    else:
        return floor, total_lo, total_hi
    # A spanning cell must hold its longest token.
    for row in b.head + rows:
        for j, cell in enumerate(row):
            if cell is None or cell.span == 1 or not cell.text:
                continue
            need = max(text_width(t, pt, cell.bold) for t in tokens(typeset(cell.text))) + pad
            have = sum(widths[j:j + cell.span])
            if have < need:
                for k in range(j, j + cell.span):
                    widths[k] += (need - have) / cell.span
    return widths, total_lo, total_hi


def shortest_table(b: Block, rows, pt: float, floor: list[float], hi: list[float],
                   avail: float, pad: float) -> list[float]:
    """Hand out the width above the minimums in small steps, each to the column that most
    shortens the table: the sum over rows of the row's tallest cell, in estimated lines. Lines
    are counted continuously and the tallest cell is a soft maximum (a 6-norm), so a step that
    only brings a cell closer to losing a line still counts and no column starves."""
    text = np.zeros((len(rows) + 1, b.ncols))
    for i, row in enumerate(rows + [b.head[-1]]):
        for j, cell in enumerate(row):
            if cell is not None and cell.span == 1 and cell.text:
                text[i, j] = text_width(typeset(cell.text), pt, cell.bold) * 1.08

    def height(ws: np.ndarray) -> float:
        lines = np.maximum(text / np.maximum(ws - pad, 1.0), 1.0)
        return ((lines ** 6).sum(axis=1) ** (1 / 6)).sum()

    widths = np.array(floor, dtype=float)
    cap = np.array(hi, dtype=float)
    step = (avail - widths.sum()) / 400
    while avail - widths.sum() > 1e-6:
        step = min(step, avail - widths.sum())
        base = height(widths)
        best, gain = None, -1.0
        for j in range(b.ncols):
            if widths[j] + step > cap[j] + 1e-6:
                continue
            trial = widths.copy()
            trial[j] += step
            g = base - height(trial)
            if g > gain:
                best, gain = j, g
        if best is None:           # every column at its natural width
            break
        widths[best] += step
    if avail - widths.sum() > 1:   # leftover goes where the text is
        widths += (avail - widths.sum()) * cap / cap.sum()
    return list(widths)


@dataclass
class Layout:
    landscape: bool
    pt: float
    widths: list[list[int]] = field(default_factory=list)


def layout(t: Table) -> Layout:
    portrait = PAGE_SHORT - 2 * MARGIN
    landscape = PAGE_LONG - 2 * MARGIN
    first = [column_widths(b, SIZES_PT[0], portrait) for b in t.blocks]
    if all(hi <= portrait for _, _, hi in first):
        return Layout(False, SIZES_PT[0], [[round(w) for w in ws] for ws, _, _ in first])
    for pt in SIZES_PT:
        fits = [column_widths(b, pt, landscape) for b in t.blocks]
        if all(lo <= landscape for _, lo, _ in fits):
            return Layout(True, pt, [[round(w) for w in ws] for ws, _, _ in fits])
    raise ValueError(f"Table {t.key} does not fit a landscape page at {SIZES_PT[-1]} pt")


# ── WordprocessingML ────────────────────────────────────────────────────────
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NSMAP = {"w": W_NS, "r": R_NS}


def w(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


def el(tag: str, attrs: dict | None = None, *children) -> etree._Element:
    e = etree.Element(w(tag), nsmap=NSMAP)
    for k, v in (attrs or {}).items():
        e.set(f"{{{R_NS}}}{k[2:]}" if k.startswith("r:") else w(k), str(v))
    for c in children:
        if c is not None:
            e.append(c)
    return e


def run(text: str, pt: float, bold=False, italic=False, vert: str | None = None):
    rpr = el("rPr", None,
             el("rFonts", {"ascii": FONT, "hAnsi": FONT, "cs": FONT, "eastAsia": FONT}),
             el("b") if bold else None, el("bCs") if bold else None,
             el("i") if italic else None, el("iCs") if italic else None,
             el("sz", {"val": round(pt * 2)}), el("szCs", {"val": round(pt * 2)}),
             el("vertAlign", {"val": vert}) if vert else None)
    t = el("t", None)
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return el("r", None, rpr, t)


def paragraph(content: list[tuple[str, bool, bool]], pt: float, *, align="left", before=0.0,
              after=0.0, keep_next=False, keep_lines=False, page_break=False, rule_below=False,
              sect=None):
    """content: (text, bold, italic) pieces, each typeset into runs."""
    ppr = el("pPr", None,
             el("keepNext") if keep_next else None,
             el("keepLines") if keep_lines else None,
             el("pageBreakBefore") if page_break else None,
             el("pBdr", None, el("bottom", {"val": "single", "sz": RULE_INNER, "space": 1,
                                            "color": "000000"})) if rule_below else None,
             el("spacing", {"before": round(before * TWIP), "after": round(after * TWIP),
                            "line": 240, "lineRule": "auto"}),
             el("ind", {"left": 0, "right": 0, "firstLine": 0}),
             el("jc", {"val": align}),
             # The paragraph mark's size sets an empty cell's height, so it must not fall back
             # to the host document's default size once spliced into the SI.
             el("rPr", None,
                el("rFonts", {"ascii": FONT, "hAnsi": FONT, "cs": FONT, "eastAsia": FONT}),
                el("sz", {"val": round(pt * 2)}), el("szCs", {"val": round(pt * 2)})),
             sect)
    p = el("p", None, ppr)
    for text, bold, italic in content:
        for piece, vert in runs(text):
            p.append(run(piece, pt, bold, italic, vert))
    return p


def section_properties(landscape: bool) -> etree._Element:
    wd, ht = (PAGE_LONG, PAGE_SHORT) if landscape else (PAGE_SHORT, PAGE_LONG)
    size = {"w": wd, "h": ht}
    if landscape:
        size["orient"] = "landscape"
    return el("sectPr", None,
              el("footerReference", {"type": "default", "r:id": "rIdFooter1"}),
              el("pgSz", size),
              el("pgMar", {"top": MARGIN, "right": MARGIN, "bottom": MARGIN, "left": MARGIN,
                           "header": 720, "footer": 720, "gutter": 0}),
              el("cols", {"space": 720}))


def cell_xml(cell: Cell, width: int, pt: float, *, top=0, bottom=0, valign="top",
             align=None, italic=False, rule_below=False, keep_next=False, before=0.0):
    borders = el("tcBorders", None,
                 el("top", {"val": "single", "sz": top, "space": 0, "color": "000000"})
                 if top else None,
                 el("bottom", {"val": "single", "sz": bottom, "space": 0, "color": "000000"})
                 if bottom else None)
    tcpr = el("tcPr", None, el("tcW", {"w": width, "type": "dxa"}),
              el("gridSpan", {"val": cell.span}) if cell.span > 1 else None,
              borders if len(borders) else None,
              el("vAlign", {"val": valign}))
    align = align or ("left" if cell.left else "center")
    p = paragraph([(cell.text, cell.bold, italic)], pt, align=align, before=before,
                  keep_next=keep_next, rule_below=rule_below and bool(cell.text))
    return el("tc", None, tcpr, p)


def table_xml(b: Block, widths: list[int], pt: float) -> etree._Element:
    tbl = el("tbl", None,
             el("tblPr", None,
                el("tblW", {"w": sum(widths), "type": "dxa"}),
                el("jc", {"val": "left"}),
                el("tblInd", {"w": 0, "type": "dxa"}),
                el("tblBorders", None, *[el(s, {"val": "nil"}) for s in
                                         ("top", "left", "bottom", "right", "insideH", "insideV")]),
                el("tblLayout", {"type": "fixed"}),
                el("tblCellMar", None, el("top", {"w": CELL_VPAD, "type": "dxa"}),
                   el("left", {"w": CELL_PAD, "type": "dxa"}),
                   el("bottom", {"w": CELL_VPAD, "type": "dxa"}),
                   el("right", {"w": CELL_PAD, "type": "dxa"})),
                el("tblLook", {"val": "0000", "firstRow": 0, "lastRow": 0, "firstColumn": 0,
                               "lastColumn": 0, "noHBand": 1, "noVBand": 1})),
             el("tblGrid", None, *[el("gridCol", {"w": x}) for x in widths]))
    nhead = len(b.head)
    for i, row in enumerate(b.head):
        tr = el("tr", None, el("trPr", None, el("cantSplit"), el("tblHeader")))
        col_heads = i == nhead - 1
        for j, cell in enumerate(row):
            if cell is None:
                continue
            tr.append(cell_xml(cell, sum(widths[j:j + cell.span]), pt,
                               top=RULE_OUTER if i == 0 else 0,
                               bottom=RULE_INNER if col_heads else 0, valign="bottom",
                               align=None if col_heads else "center",
                               rule_below=not col_heads, keep_next=True))
        tbl.append(tr)
    last = len(b.body) - 1
    for i, row in enumerate(b.body):
        tr = el("tr", None, el("trPr", None, el("cantSplit")))
        bottom = RULE_OUTER if i == last else 0
        # The first rows stay with the header; the last stays with the note.
        keep = i < 2 or i >= last - 1
        if isinstance(row, Section):
            tr.append(cell_xml(Cell(row.text, span=b.ncols, left=True), sum(widths), pt,
                               bottom=bottom, italic=True, keep_next=True,
                               before=2 if i else 0))
        else:
            for j, cell in enumerate(row):
                if cell is not None:
                    tr.append(cell_xml(cell, sum(widths[j:j + cell.span]), pt, bottom=bottom,
                                       keep_next=keep))
        tbl.append(tr)
    return tbl


def table_body(t: Table, lay: Layout, page_break: bool) -> list[etree._Element]:
    out = [paragraph([(f"SI Table {t.key}. {t.title}.", True, False)], CAPTION_PT,
                     after=6, keep_next=True, page_break=page_break)]
    for bi, (b, widths) in enumerate(zip(t.blocks, lay.widths)):
        if b.panel:
            out.append(paragraph([(b.panel, True, False)], lay.pt, before=9 if bi else 0,
                                 after=3, keep_next=True))
        elif bi:
            out.append(paragraph([("", False, False)], lay.pt, after=6, keep_next=True))
        out.append(table_xml(b, widths, lay.pt))
    for k, note in enumerate(t.notes):     # notes a point below the body, as table notes are set
        out.append(paragraph([(text, False, italic) for text, italic in note], lay.pt - 1,
                             align="both", before=4 if k == 0 else 1,
                             keep_next=k < len(t.notes) - 1, keep_lines=True))
    return out


# ── Package ─────────────────────────────────────────────────────────────────
CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
<Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>
<Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>
</Types>"""

PACKAGE_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

DOCUMENT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
<Relationship Id="rIdSettings" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>
<Relationship Id="rIdFooter1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/>
</Relationships>"""

STYLES = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{W_NS}">
<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="{FONT}" w:hAnsi="{FONT}" w:cs="{FONT}" w:eastAsia="{FONT}"/><w:sz w:val="20"/><w:szCs w:val="20"/><w:lang w:val="en-US"/></w:rPr></w:rPrDefault>
<w:pPrDefault><w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
<w:style w:type="table" w:default="1" w:styleId="TableNormal"><w:name w:val="Normal Table"/><w:tblPr><w:tblInd w:w="0" w:type="dxa"/><w:tblCellMar><w:top w:w="0" w:type="dxa"/><w:left w:w="108" w:type="dxa"/><w:bottom w:w="0" w:type="dxa"/><w:right w:w="108" w:type="dxa"/></w:tblCellMar></w:tblPr></w:style>
</w:styles>"""

SETTINGS = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:settings xmlns:w="{W_NS}"><w:defaultTabStop w:val="720"/><w:characterSpacingControl w:val="doNotCompress"/><w:compat><w:compatSetting w:name="compatibilityMode" w:uri="http://schemas.microsoft.com/office/word" w:val="15"/></w:compat></w:settings>"""

FOOTER = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:ftr xmlns:w="{W_NS}"><w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:rPr><w:rFonts w:ascii="{FONT}" w:hAnsi="{FONT}"/><w:sz w:val="20"/></w:rPr><w:fldChar w:fldCharType="begin"/></w:r><w:r><w:rPr><w:rFonts w:ascii="{FONT}" w:hAnsi="{FONT}"/><w:sz w:val="20"/></w:rPr><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r><w:r><w:rPr><w:rFonts w:ascii="{FONT}" w:hAnsi="{FONT}"/><w:sz w:val="20"/></w:rPr><w:fldChar w:fldCharType="separate"/></w:r><w:r><w:rPr><w:rFonts w:ascii="{FONT}" w:hAnsi="{FONT}"/><w:sz w:val="20"/></w:rPr><w:t>1</w:t></w:r><w:r><w:rPr><w:rFonts w:ascii="{FONT}" w:hAnsi="{FONT}"/><w:sz w:val="20"/></w:rPr><w:fldChar w:fldCharType="end"/></w:r></w:p></w:ftr>"""


def build_document(tables: list[Table]) -> tuple[bytes, list[tuple[str, Layout]]]:
    body = el("body")
    layouts = [(t, layout(t)) for t in tables]
    prev_last = None
    for i, (t, lay) in enumerate(layouts):
        turn = i > 0 and lay.landscape != layouts[i - 1][1].landscape
        if turn:   # the previous table's last paragraph closes its section
            prev_last.find(w("pPr")).append(section_properties(layouts[i - 1][1].landscape))
        parts = table_body(t, lay, page_break=i > 0 and not turn)
        body.extend(parts)
        prev_last = parts[-1]
    body.append(section_properties(layouts[-1][1].landscape))
    doc = el("document", None, body)
    etree.cleanup_namespaces(doc)
    xml = etree.tostring(doc, xml_declaration=True, encoding="UTF-8", standalone=True)
    return xml, [(t.key, lay) for t, lay in layouts]


def write_docx(path: Path, document_xml: bytes) -> None:
    """A deterministic package: fixed member order and timestamps."""
    parts = [("[Content_Types].xml", CONTENT_TYPES.encode()), ("_rels/.rels", PACKAGE_RELS.encode()),
             ("word/document.xml", document_xml),
             ("word/_rels/document.xml.rels", DOCUMENT_RELS.encode()),
             ("word/styles.xml", STYLES.encode()), ("word/settings.xml", SETTINGS.encode()),
             ("word/footer1.xml", FOOTER.encode())]
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workbook", type=Path, default=WORKBOOK)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    if _font(False) is None:
        print("warning: Liberation Serif or Times New Roman not found; widths are estimated")
    xml, layouts = build_document(read_tables(args.workbook))
    write_docx(args.out, xml)
    for key, lay in layouts:
        print(f"Table {key}: {'landscape' if lay.landscape else 'portrait'}, {lay.pt:g} pt, "
              f"{len(lay.widths)} block(s)")
    print(f"wrote {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out}")


if __name__ == "__main__":
    main()
