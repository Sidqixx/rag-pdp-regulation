"""Step 2: teks mentah -> satu Chunk per Pasal (+ Penjelasan + rujukan).

Input : data/text/<stem>.txt (hasil `python -m ragpdp.extract`)
Output: data/chunks.jsonl, satu Chunk per baris (JSON)
Jalan : python -m ragpdp.parse
"""
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# stem file -> (kode dokumen untuk id, label untuk sitasi)
SOURCES = {
    "uu-27-2022-pdp": ("uu27", "UU 27/2022"),
    "pp-33-2026-pdp": ("pp33", "PP 33/2026"),
}


@dataclass
class Chunk:
    id: str  # "<kode>:<pasal>", mis. "uu27:4"
    source: str  # "UU 27/2022"
    pasal: str  # "4"
    bab: str  # "BAB III JENIS DATA PRIBADI"
    text: str  # isi batang tubuh pasal, bersih dari nomor halaman & blok tanda tangan
    penjelasan: str  # isi PENJELASAN pasal ini; "" kalau isinya cuma "Cukup jelas."
    refs: list[str] = field(default_factory=list)  # nomor pasal lain di dokumen yang sama yang dirujuk


# Nomor halaman: UU "3 / 38", PP "- 5 -".
HALAMAN = re.compile(r"\d+ / \d+|- \d+ -")
PASAL = re.compile(r"Pasal (\d+)")
BAB = re.compile(r"BAB [IVXLC]+")
# Heading di bawah BAB. Judulnya kadang di baris berikut ("Bagian Kesatu" / "Umum"), kadang sebaris ("Paragraf 1 Umum").
SUB = re.compile(r"(Bagian|Paragraf) \S+")
# Akhir batang tubuh / penjelasan: blok tanda tangan dan nomor Tambahan Lembaran Negara.
AKHIR = re.compile(r"(Disahkan|Ditetapkan) [Dd]i|TAMBAHAN LEMBARAN NEGARA")
# Penanda butir di awal baris: ayat "(1)", huruf "a.", angka "1.", dan sub-heading penjelasan.
BUTIR = re.compile(r"\(\d+\)|[a-z]\.|\d+\.|Ayat \(\d+\)|Huruf [a-z]|Angka \d+")


def _blocks(lines: list[str]) -> dict[str, tuple[str, list[str]]]:
    """Kelompokkan baris per heading 'Pasal N' -> {nomor: (bab, baris isi)}."""
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
    """Gabung baris yang terpotong layout; tiap ayat/huruf mulai di baris baru."""
    out: list[str] = []
    for s in lines:
        prev = out[-1] if out else ""
        if BUTIR.fullmatch(prev) and not prev[0].isupper():  # "(1)" / "a." sendirian, isinya di baris berikut
            out[-1] += " " + s
        elif not out or BUTIR.fullmatch(prev) or BUTIR.fullmatch(s) or BUTIR.fullmatch(s.split(" ", 1)[0]):
            out.append(s)
        else:
            out[-1] += " " + s
    return "\n".join(out)


def parse(raw: str, doc: str, source: str) -> list[Chunk]:
    """Pecah teks satu dokumen jadi list Chunk, urut dari Pasal 1."""
    raw = raw.replace("(I)", "(1)")  # salah OCR di PDF PP
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
