from datetime import date

from ragpdp.answer import invalid_citations, status_pp

HITS = [{"source": "UU 27/2022", "pasal": "4"}, {"source": "PP 33/2026", "pasal": "6"}]


def test_citation_outside_sources_flagged():
    text = "Data biometrik termasuk spesifik [UU 27/2022 Pasal 4] [PP 33/2026 Pasal 6], sanksinya [UU 27/2022 Pasal 57]."
    assert invalid_citations(text, HITS) == ["UU 27/2022 Pasal 57"]


def test_same_pasal_different_document_not_confused():
    assert invalid_citations("[PP 33/2026 Pasal 4]", HITS) == ["PP 33/2026 Pasal 4"]


def test_citation_with_ayat_still_checked():
    text = "[UU 27/2022 Pasal 4 Ayat (2)] [PP 33/2026 Pasal 9 ayat (1) huruf a] dan **UU 27/2022 Pasal 40**"
    assert invalid_citations(text, HITS) == ["PP 33/2026 Pasal 9", "UU 27/2022 Pasal 40"]


def test_reversed_citation_order_still_checked():
    text = "diatur Pasal 120 ayat (1) UU No. 27 Tahun 2022 dan Pasal 6 PP 33/2026"
    assert invalid_citations(text, HITS) == ["UU 27/2022 Pasal 120"]


def test_grouped_citations_each_checked():
    text = "[UU 27/2022 Pasal 4, Pasal 57] dan [PP 33/2026 Pasal 6 ayat (1) dan ayat (2)]"
    assert invalid_citations(text, HITS) == ["UU 27/2022 Pasal 57"]


def test_status_pp_changes_when_in_force():
    assert "BELUM berlaku" in status_pp(date(2027, 1, 15))
    assert "BELUM" not in status_pp(date(2027, 1, 16))
