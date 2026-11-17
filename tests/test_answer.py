from datetime import date

from ragpdp.answer import invalid_citations, status_pp

HITS = [{"source": "UU 27/2022", "pasal": "4"}, {"source": "PP 33/2026", "pasal": "6"}]


def test_sitasi_di_luar_sumber_ditandai():
    text = "Data biometrik termasuk spesifik [UU 27/2022 Pasal 4] [PP 33/2026 Pasal 6], sanksinya [UU 27/2022 Pasal 57]."
    assert invalid_citations(text, HITS) == ["UU 27/2022 Pasal 57"]


def test_pasal_sama_beda_dokumen_tidak_tertukar():
    assert invalid_citations("[PP 33/2026 Pasal 4]", HITS) == ["PP 33/2026 Pasal 4"]


def test_sitasi_dengan_ayat_tetap_dicek():
    text = "[UU 27/2022 Pasal 4 Ayat (2)] [PP 33/2026 Pasal 9 ayat (1) huruf a] dan **UU 27/2022 Pasal 40**"
    assert invalid_citations(text, HITS) == ["PP 33/2026 Pasal 9", "UU 27/2022 Pasal 40"]


def test_sitasi_urutan_terbalik_tetap_dicek():
    text = "diatur Pasal 120 ayat (1) UU No. 27 Tahun 2022 dan Pasal 6 PP 33/2026"
    assert invalid_citations(text, HITS) == ["UU 27/2022 Pasal 120"]


def test_status_pp_berubah_saat_berlaku():
    assert "BELUM berlaku" in status_pp(date(2027, 1, 15))
    assert "BELUM" not in status_pp(date(2027, 1, 16))
