"""The SI tables document carries every title, cell and note of the workbook, unchanged.

Reads the sheets directly, not through the renderer's parser, and compares them row by row with
the rendered document. Text must match exactly once typographic forms are folded back (true
minus to hyphen-minus, no-break space to space, superscript and subscript runs to the characters
they replace). A number must equal the stored value at exactly the decimals its Excel format
shows, with the format's leading-zero and thousands rules.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest
from lxml import etree
from openpyxl import load_workbook

from analysis.tables.build_supplemental_tables_docx import WORKBOOK, build_document, read_tables

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
TO_SUPER = str.maketrans("0123456789−-abcd", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁻ᵃᵇᶜᵈ")
STRAY = re.compile("[⁰¹⁴⁵⁶⁷⁸⁹⁻ᵃᵇᶜᵈ]")      # characters that must have become superscript runs
# S17 is set as two panels; its sheet columns by panel (stub first).
S17_PANELS = ("ABCDHIJNOP", "AEFGKLMQRST")

# The workbook is built by step supplement_tables from the MIDUS data, so a clone without
# the data has nothing to compare.
pytestmark = pytest.mark.skipif(
    not WORKBOOK.exists(), reason="build step supplement_tables first (needs the MIDUS data)")


def fold(text: str) -> str:
    return text.replace("−", "-").replace(" ", " ")


def run_text(r) -> tuple[str, bool]:
    text = "".join(t.text or "" for t in r.iter(f"{W}t"))
    vert = r.find(f"{W}rPr/{W}vertAlign")
    if vert is not None and vert.get(f"{W}val") == "superscript":
        text = text.translate(TO_SUPER)
    return text, r.find(f"{W}rPr/{W}b") is not None


def para_text(p) -> str:
    return "".join(run_text(r)[0] for r in p.iter(f"{W}r"))


@pytest.fixture(scope="module")
def rendered():
    xml, _ = build_document(read_tables(WORKBOOK))
    body = etree.fromstring(xml).find(f"{W}body")
    tables: dict[str, dict] = {}
    current = None
    for child in body:
        if child.tag == f"{W}p":
            text = para_text(child)
            m = re.match(r"SI Table (S\d+)\. ", text)
            if m:
                current = tables.setdefault(m[1], {"caption": text, "rows": [], "head": [],
                                                   "bold": []})
            elif text and current is not None:
                current["rows"].append([text])
        elif child.tag == f"{W}tbl":
            for tr in child.iter(f"{W}tr"):
                cells = [para_text(tc) for tc in tr.iter(f"{W}tc")]
                header = tr.find(f"{W}trPr/{W}tblHeader") is not None
                (current["head"] if header else current["rows"]).append([c for c in cells if c])
                if not header:
                    current["bold"] += [para_text(tc) for tc in tr.iter(f"{W}tc")
                                        if tc.find(f".//{W}b") is not None]
    return tables


def number_matches(text: str, value: float, fmt: str) -> bool:
    sections = fmt.split(";")
    sec = sections[1] if value < 0 and len(sections) > 1 else (
        sections[2] if value == 0 and len(sections) > 2 else sections[0])
    decimals = len(sec.split(".")[1]) if "." in sec else 0
    t = fold(text)
    if not re.fullmatch(r"-?(\d{1,3}(,\d{3})+|\d*)(\.\d+)?", t):
        return False
    shown = len(t.split(".")[1]) if "." in t else 0
    if shown != decimals or abs(float(t.replace(",", "")) - value) > 0.5 * 10 ** -decimals + 1e-12:
        return False
    int_part = sec.lstrip("+-").split(".")[0]
    if "0" not in int_part and abs(value) < 1 and t.lstrip("-").startswith("0"):
        return False
    return not ("," in int_part and abs(value) >= 1000 and "," not in t)


def cell_matches(text: str, cell) -> bool:
    v = cell.value
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return number_matches(text, v, cell.number_format)
    return fold(text) == fold(str(v))


def sheet_rows(ws, first: int, cols: str | None = None) -> list[list]:
    letters = cols or "".join(chr(65 + i) for i in range(ws.max_column))
    rows = []
    for r in range(first, ws.max_row + 1):
        cells = [ws[f"{c}{r}"] for c in letters]
        cells = [c for c in cells if c.value is not None]
        if cells:
            rows.append(cells)
    return rows


def assert_rows(actual: list[list[str]], expected: list[list], where: str) -> None:
    assert len(actual) == len(expected), f"{where}: {len(actual)} rows, sheet has {len(expected)}"
    for k, (a, e) in enumerate(zip(actual, expected)):
        assert len(a) == len(e), f"{where} row {k}: {a} against {[c.value for c in e]}"
        for text, cell in zip(a, e):
            assert cell_matches(text, cell), f"{where} {cell.coordinate}: {text!r} is not {cell.value!r}"


@pytest.fixture(scope="module")
def workbook():
    return load_workbook(WORKBOOK, rich_text=True)


def test_every_table_is_captioned_in_order(rendered, workbook):
    sheets = [ws for ws in workbook.worksheets if ws.title != "Contents"]
    assert list(rendered) == [ws.title for ws in sheets]
    for ws in sheets:
        assert rendered[ws.title]["caption"] == f"SI Table {ws.title}. {ws['A2'].value}."


def test_every_row_cell_and_note_matches_the_sheet(rendered, workbook):
    for ws in workbook.worksheets:
        if ws.title in ("Contents", "S17"):
            continue
        got = rendered[ws.title]
        # Header rows come first in each block, so interleave them back in sheet order by
        # comparing the whole sequence with headers removed, then the headers alone.
        expected = sheet_rows(ws, 4)
        head_texts = Counter(fold(t) for row in got["head"] for t in row)
        body = [row for row in expected
                if not all(fold(str(c.value)) in head_texts for c in row)
                or any(isinstance(c.value, (int, float)) for c in row)]
        heads = [row for row in expected if row not in body]
        assert_rows(got["rows"], body, ws.title)
        assert Counter(fold(str(c.value)) for row in heads for c in row) == head_texts, ws.title


def test_s17_panels_hold_every_cell(rendered, workbook):
    ws = workbook["S17"]
    got = rendered["S17"]
    rows = []
    for k, cols in enumerate(S17_PANELS):
        layer = ws[f"{cols[1]}5"].value
        rows.append([f"Panel {'AB'[k]}. {layer}"])
        for r in range(7, ws.max_row):
            cells = [ws[f"{c}{r}"] for c in cols if ws[f"{c}{r}"].value is not None]
            section = len(cells) == 1 and ws[f"A{r}"].font.i
            if section or len(cells) > 1:
                rows.append(cells)
    actual = got["rows"]
    expected = rows + sheet_rows(ws, ws.max_row)
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        if isinstance(e[0], str):
            assert a == e
        else:
            assert len(a) == len(e) and all(cell_matches(t, c) for t, c in zip(a, e)), a
    head = Counter(fold(t) for row in got["head"] for t in row)
    for r in (4, 6):
        for c in sheet_rows(ws, r)[0]:
            assert fold(str(c.value)) in head, c.coordinate


def test_bold_cells_stay_bold(rendered, workbook):
    for ws in workbook.worksheets:
        if ws.title == "Contents":
            continue
        expected = Counter()
        for row in ws.iter_rows(min_row=4):
            for c in row:
                full_width_panel = c.column == 1 and str(c.value).startswith("Panel ")
                if c.value is not None and c.font.b and not full_width_panel:
                    expected[fold(str(c.value))] += 1
        assert Counter(fold(t) for t in rendered[ws.title]["bold"]) == expected, ws.title


def test_no_unicode_superscript_left_in_plain_runs():
    xml, _ = build_document(read_tables(WORKBOOK))
    for r in etree.fromstring(xml).iter(f"{W}r"):
        if r.find(f"{W}rPr/{W}vertAlign") is None:
            text = "".join(t.text or "" for t in r.iter(f"{W}t"))
            assert not STRAY.search(text), text
