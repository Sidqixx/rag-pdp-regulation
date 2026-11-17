"""Step 1: PDF -> teks mentah. Satu file .txt per PDF di data/text/.

Jalankan: python -m ragpdp.extract
"""
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]


def extract(pdf: Path) -> str:
    with pymupdf.open(pdf) as doc:
        return "\n".join(page.get_text() for page in doc)


def main() -> None:
    out = ROOT / "data/text"
    out.mkdir(parents=True, exist_ok=True)
    for pdf in sorted((ROOT / "data/raw").glob("*.pdf")):
        txt = out / f"{pdf.stem}.txt"
        txt.write_text(extract(pdf), encoding="utf-8")
        print(f"{pdf.name} -> {txt.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
