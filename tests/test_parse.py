"""Definisi "parser selesai". Semua hijau = Step 2 beres."""
import re

import pytest

from ragpdp.parse import ROOT, SOURCES, parse


def load(stem: str) -> dict:
    doc, source = SOURCES[stem]
    raw = (ROOT / "data/text" / f"{stem}.txt").read_text(encoding="utf-8")
    return {c.pasal: c for c in parse(raw, doc, source)}


@pytest.fixture(scope="module")
def uu():
    return load("uu-27-2022-pdp")


@pytest.fixture(scope="module")
def pp():
    return load("pp-33-2026-pdp")


def test_semua_pasal_ada_dan_urut(uu, pp):
    assert list(uu) == [str(i) for i in range(1, 77)]
    assert list(pp) == [str(i) for i in range(1, 226)]


def test_isi_dan_metadata(uu):
    c = uu["4"]
    assert (c.id, c.source) == ("uu27:4", "UU 27/2022")
    assert c.bab == "BAB III JENIS DATA PRIBADI"
    assert "data biometrik" in c.text
    assert "BAB IV" not in c.text  # tidak bocor ke heading berikutnya


def test_penjelasan_dipasangkan_ke_pasal_yang_benar(uu):
    assert "daktiloskopi" in uu["4"].penjelasan
    assert "daktiloskopi" not in uu["4"].text  # penjelasan tidak tercampur batang tubuh
    assert uu["1"].penjelasan == ""  # "Cukup jelas." -> kosong


def test_bersih_dari_noise(uu, pp):
    for c in [*uu.values(), *pp.values()]:
        assert not re.search(r"\d+ / 38", c.text), c.id  # nomor halaman UU
        assert "JOKO WIDODO" not in c.text and "PRABOWO" not in c.text, c.id  # blok tanda tangan
    assert pp["6"].text.startswith("(1)")  # PDF PP salah OCR: "(I)" harusnya "(1)"


def test_refs_pasal_yang_dirujuk(uu):
    assert {"20", "21", "56"} <= set(uu["57"].refs)
    assert "57" not in uu["57"].refs
