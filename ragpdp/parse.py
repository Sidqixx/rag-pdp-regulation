"""Step 2: raw text -> one Chunk per Pasal (+ Penjelasan + references).

Input : data/text/<stem>.txt (output of `python -m ragpdp.extract`)
Output: data/chunks.jsonl, one Chunk per line (JSON)
Run   : python -m ragpdp.parse
"""
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# file stem -> (document code for id, label for citations)
SOURCES = {
    "uu-27-2022-pdp": ("uu27", "UU 27/2022"),
    "pp-33-2026-pdp": ("pp33", "PP 33/2026"),
}


@dataclass
class Chunk:
    id: str  # "<code>:<pasal>", e.g. "uu27:4"
    source: str  # "UU 27/2022"
    pasal: str  # "4"
    bab: str  # "BAB III JENIS DATA PRIBADI"
    text: str  # body of the pasal, stripped of page numbers & signature blocks
    penjelasan: str  # this pasal's PENJELASAN; "" if it only says "Cukup jelas."
    refs: list[str] = field(default_factory=list)  # other pasal numbers in the same document that this one references


# Page numbers: UU "3 / 38", PP "- 5 -".
HALAMAN = re.compile(r"\d+ / \d+|- \d+ -")
PASAL = re.compile(r"Pasal (\d+)")
BAB = re.compile(r"BAB [IVXLC]+")
# Headings under a BAB. The title is sometimes on the next line ("Bagian Kesatu" / "Umum"), sometimes inline ("Paragraf 1 Umum").
SUB = re.compile(r"(Bagian|Paragraf) \S+")
# End of body / penjelasan: signature block and Tambahan Lembaran Negara number.
AKHIR = re.compile(r"(Disahkan|Ditetapkan) [Dd]i|TAMBAHAN LEMBARAN NEGARA")
# Item markers at line start: ayat "(1)", huruf "a.", number "1.", and penjelasan sub-headings.
BUTIR = re.compile(r"\(\d+\)|[a-z]\.|\d+\.|Ayat \(\d+\)|Huruf [a-z]|Angka \d+")


def _blocks(lines: list[str]) -> dict[str, tuple[str, list[str]]]:
    """Group lines by 'Pasal N' heading -> {number: (bab, body lines)}."""
    blocks: dict[str, tuple[str, list[str]]] = {}
    cur, bab, i = None, "", 0
    while i < len(lines):
        s = lines[i]
        if BAB.fullmatch(s):
            bab = f"{s} {lines[i + 1]}"
            i += 2
        elif SUB.fullmatch(s):
            i += 2
        elif SUB.match(s):
            i += 1
        elif AKHIR.match(s):
            cur, i = None, i + 1
        elif m := PASAL.fullmatch(s):
            cur, i = m[1], i + 1
            blocks[cur] = (bab, [])
        else:
            if cur:
                blocks[cur][1].append(s)
            i += 1
    return blocks


def _join(lines: list[str]) -> str:
    """Join lines broken by the layout; each ayat/huruf starts on a new line."""
    out: list[str] = []
    for s in lines:
        prev = out[-1] if out else ""
        if BUTIR.fullmatch(prev) and not prev[0].isupper():  # lone "(1)" / "a.", its content is on the next line
            out[-1] += " " + s
        elif not out or BUTIR.fullmatch(prev) or BUTIR.fullmatch(s) or BUTIR.fullmatch(s.split(" ", 1)[0]):
            out.append(s)
        else:
            out[-1] += " " + s
    return "\n".join(out)


def parse(raw: str, doc: str, source: str) -> list[Chunk]:
    """Split one document's text into a list of Chunks, ordered from Pasal 1."""
    raw = raw.replace("(I)", "(1)")  # OCR error in the PP PDF
    lines = [s for line in raw.splitlines() if (s := line.strip()) and not HALAMAN.fullmatch(s)]
    split = lines.index("PENJELASAN")
    body, penjelasan = _blocks(lines[:split]), _blocks(lines[split:])
    chunks = []
    for no, (bab, isi) in body.items():
        text = _join(isi)
        pj = _join(penjelasan.get(no, ("", []))[1])
        refs = list(dict.fromkeys(r for r in PASAL.findall(text) if r != no))
        chunks.append(Chunk(f"{doc}:{no}", source, no, bab, text, "" if pj == "Cukup jelas." else pj, refs))
    return chunks


def main() -> None:
    chunks: list[Chunk] = []
    for stem, (doc, source) in SOURCES.items():
        raw = (ROOT / "data/text" / f"{stem}.txt").read_text(encoding="utf-8")
        chunks += parse(raw, doc, source)
    out = ROOT / "data/chunks.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    print(f"{len(chunks)} chunks -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
