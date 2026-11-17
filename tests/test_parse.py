"""Definition of "parser done". All green = Step 2 finished."""
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


def test_all_pasal_present_in_order(uu, pp):
    assert list(uu) == [str(i) for i in range(1, 77)]
    assert list(pp) == [str(i) for i in range(1, 226)]


def test_body_and_metadata(uu):
    c = uu["4"]
    assert (c.id, c.source) == ("uu27:4", "UU 27/2022")
    assert c.bab == "BAB III JENIS DATA PRIBADI"
    assert "data biometrik" in c.text
    assert "BAB IV" not in c.text  # does not leak into the next heading


def test_penjelasan_paired_with_correct_pasal(uu):
    assert "daktiloskopi" in uu["4"].penjelasan
    assert "daktiloskopi" not in uu["4"].text  # penjelasan not mixed into the body
    assert uu["1"].penjelasan == ""  # "Cukup jelas." -> empty


def test_free_of_noise(uu, pp):
    for c in [*uu.values(), *pp.values()]:
        assert not re.search(r"\d+ / 38", c.text), c.id  # UU page numbers
        assert "JOKO WIDODO" not in c.text and "PRABOWO" not in c.text, c.id  # signature block
    assert pp["6"].text.startswith("(1)")  # OCR error in PP PDF: "(I)" should be "(1)"


def test_refs_to_referenced_pasal(uu):
    assert {"20", "21", "56"} <= set(uu["57"].refs)
    assert "57" not in uu["57"].refs
