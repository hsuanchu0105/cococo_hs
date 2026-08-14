"""
Extract the schedule-length benchmark table out of the analysis workbook.

    python extract_schedule_lengths.py [--xlsx ../analysis/data_analysis.xlsx]

The `fgsa(move_upd)` sheet holds two `Len(schedule)` blocks -- one per `j` --
each laid out as four side-by-side layout groups:

    Len(schedule)        j=8   max_ov = 8
    Reduction
    single cs fg cs+sa fg+sa    pair cs fg cs+sa fg+sa    triple ...   hex ...
         1 248 226   217   217     1 267 244   233   217        ...       ...

which is melted into a tidy long CSV next to the workbook:

    layout,j,max_ov,seed,method,length
    single,8,8,1,cs,248

Blocks are located by scanning column A for the `Len(schedule)` marker rather
than by hardcoded row numbers, so the sheet can grow without breaking this.

openpyxl is not installed and is not worth a dependency for one sheet, so the
xlsx is unzipped directly: an xlsx is a zip of XML, cell values live in
`xl/worksheets/sheetN.xml`, and strings are indirected through
`xl/sharedStrings.xml` (a cell carrying `t="s"` holds an index into it).
"""

import argparse
import csv
import re
import sys
import zipfile
from pathlib import Path

LAYOUTS = ("single", "pair", "triple", "hex")
METHODS = ("cs", "fg", "cs+sa", "fg+sa")
N_SEEDS = 10

project_root = Path(__file__).resolve().parent.parent
DEFAULT_XLSX = project_root.parent / "analysis" / "data_analysis.xlsx"
DEFAULT_OUT = project_root.parent / "analysis" / "schedule_lengths.csv"


# --------------------------------------------------------------------------- #
# minimal xlsx reader
# --------------------------------------------------------------------------- #

_CELL_RE = re.compile(r'<c r="([A-Z]+)(\d+)"([^>]*)>(.*?)</c>', re.S)
_VALUE_RE = re.compile(r"<v>(.*?)</v>", re.S)
_SI_RE = re.compile(r"<si>(.*?)</si>", re.S)
_TEXT_RE = re.compile(r"<t[^>]*>(.*?)</t>", re.S)
_SHEET_RE = re.compile(r'<sheet name="([^"]+)"[^>]*r:id="([^"]+)"')
_REL_RE = re.compile(r'Id="([^"]+)"[^>]*Target="([^"]+)"')


def col_index(letters):
    """'A' -> 0, 'Z' -> 25, 'AA' -> 26."""
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def read_sheet(xlsx_path, sheet_name):
    """Return {row_number: {col_index: value}} for one sheet, values as str."""
    with zipfile.ZipFile(xlsx_path) as z:
        shared = [
            "".join(_TEXT_RE.findall(si))
            for si in _SI_RE.findall(z.read("xl/sharedStrings.xml").decode("utf8"))
        ]
        # The sheetN.xml numbering does not follow sheet order -- go through the
        # relationship ids, which is the only mapping the format guarantees.
        rels = dict(_REL_RE.findall(z.read("xl/_rels/workbook.xml.rels").decode("utf8")))
        sheets = dict(_SHEET_RE.findall(z.read("xl/workbook.xml").decode("utf8")))
        if sheet_name not in sheets:
            raise SystemExit(
                f"sheet {sheet_name!r} not in {xlsx_path.name}; have: {', '.join(sheets)}"
            )
        target = rels[sheets[sheet_name]].lstrip("/")
        xml = z.read("xl/" + target).decode("utf8")

    grid = {}
    for col, row, attrs, body in _CELL_RE.findall(xml):
        value = _VALUE_RE.search(body)
        if value is None:
            continue
        text = value.group(1)
        if 't="s"' in attrs:
            text = shared[int(text)]
        grid.setdefault(int(row), {})[col_index(col)] = text.strip()
    return grid


# --------------------------------------------------------------------------- #
# block parsing
# --------------------------------------------------------------------------- #


def as_int(text, where):
    try:
        number = float(text)
    except (TypeError, ValueError):
        raise SystemExit(f"{where}: expected a number, got {text!r}")
    if number <= 0 or number != int(number):
        raise SystemExit(f"{where}: expected a positive integer, got {text!r}")
    return int(number)


def parse_block(grid, marker_row):
    """Melt one `Len(schedule)` block into records."""
    row_text = " ".join(grid[marker_row].values())
    j_match = re.search(r"j\s*=\s*(\d+)", row_text)
    ov_match = re.search(r"max_ov\s*=\s*(\d+)", row_text)
    if not (j_match and ov_match):
        raise SystemExit(f"row {marker_row}: cannot read j / max_ov from {row_text!r}")
    j, max_ov = int(j_match.group(1)), int(ov_match.group(1))

    # The header is the next row that starts with a layout name; between the
    # marker and it sits a stray "Reduction" label.
    header_row = next(
        (r for r in range(marker_row + 1, marker_row + 6) if grid.get(r, {}).get(0) in LAYOUTS),
        None,
    )
    if header_row is None:
        raise SystemExit(f"row {marker_row}: no layout header row beneath the marker")

    records = []
    for col, label in sorted(grid[header_row].items()):
        if label not in LAYOUTS:
            continue
        found = tuple(grid[header_row].get(col + k) for k in range(1, 5))
        if found != METHODS:
            raise SystemExit(
                f"row {header_row}, {label}: expected method columns {METHODS}, found {found}"
            )
        for offset in range(1, N_SEEDS + 1):
            row = header_row + offset
            where = f"row {row}, layout {label} (j={j})"
            seed = as_int(grid.get(row, {}).get(col), f"{where}: seed")
            if seed != offset:
                raise SystemExit(f"{where}: expected seed {offset}, found {seed}")
            for k, method in enumerate(METHODS, start=1):
                length = as_int(grid[row].get(col + k), f"{where}: {method}")
                records.append(
                    {
                        "layout": label,
                        "j": j,
                        "max_ov": max_ov,
                        "seed": seed,
                        "method": method,
                        "length": length,
                    }
                )
    return records


def extract(xlsx_path, sheet_name):
    grid = read_sheet(xlsx_path, sheet_name)
    markers = [r for r, cells in sorted(grid.items()) if cells.get(0) == "Len(schedule)"]
    if not markers:
        raise SystemExit(f"no 'Len(schedule)' marker in column A of sheet {sheet_name!r}")

    records = []
    for marker_row in markers:
        records.extend(parse_block(grid, marker_row))

    expected = len(markers) * len(LAYOUTS) * N_SEEDS * len(METHODS)
    if len(records) != expected:
        raise SystemExit(f"expected {expected} records, built {len(records)}")
    keys = {(r["layout"], r["j"], r["seed"], r["method"]) for r in records}
    if len(keys) != len(records):
        raise SystemExit("duplicate (layout, j, seed, method) rows -- blocks overlap?")
    return records, markers


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--sheet", default="fgsa(move_upd)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.xlsx.exists():
        raise SystemExit(f"workbook not found: {args.xlsx}")

    records, markers = extract(args.xlsx, args.sheet)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["layout", "j", "max_ov", "seed", "method", "length"])
        writer.writeheader()
        writer.writerows(records)

    js = sorted({r["j"] for r in records})
    print(f"{args.xlsx.name} [{args.sheet}]: {len(markers)} blocks at rows {markers}")
    print(f"{len(records)} records  j={js}  layouts={list(LAYOUTS)}  methods={list(METHODS)}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
