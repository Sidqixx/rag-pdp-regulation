from datetime import date
from types import SimpleNamespace as NS

from ragpdp.answer import CONTINUE, CUTOFF_NOTICE, invalid_citations, resume_if_cut_off, status_pp

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


def test_multi_ayat_citation_parsed_as_one():
    assert invalid_citations("[PP 33/2026 Pasal 6 ayat (1) dan (2)] [UU 27/2022 Pasal 4 ayat (1), (2)]", HITS) == []


def test_status_pp_changes_when_in_force():
    assert "BELUM berlaku" in status_pp(date(2027, 1, 15))
    assert "BELUM" not in status_pp(date(2027, 1, 16))


def _chunks(*parts, finish):
    return [NS(choices=[NS(delta=NS(content=p), finish_reason=None)]) for p in parts] + [
        NS(choices=[NS(delta=NS(content=None), finish_reason=finish)])
    ]


def _resume(*streams):
    """Run resume_if_cut_off where the first stream is given and later ones come from fake API calls."""
    calls = []
    later = iter(streams[1:])
    create = lambda msgs: calls.append(msgs) or next(later)
    text = "".join(resume_if_cut_off(streams[0], create, [{"role": "user", "content": "q"}]))
    return text, calls


def test_complete_stream_is_passed_through():
    assert _resume(_chunks("Halo", " dunia.", finish="stop")) == ("Halo dunia.", [])


def test_cut_off_mid_word_is_continued_seamlessly():
    text, calls = _resume(_chunks("karena pem", finish=None), _chunks("pemrosesan", " data.", finish="stop"))
    assert text == "karena pemrosesan data."
    assert calls[0][-2:] == [{"role": "assistant", "content": "karena pem"}, {"role": "user", "content": CONTINUE}]


def test_cut_off_between_words_keeps_the_space():
    text, _ = _resume(_chunks("Berikut", finish=None), _chunks("Berikut ketentuannya.", finish="stop"))
    assert text == "Berikut ketentuannya."


def test_continuation_without_repeated_word_gets_a_space():
    text, _ = _resume(_chunks("Berikut", finish=None), _chunks("ketentuannya.", finish="stop"))
    assert text == "Berikut ketentuannya."


def test_repeated_phrase_completes_a_cut_word():
    text, _ = _resume(_chunks("harus memenuhi persyar", finish=None), _chunks("memenuhi persyaratan ketat.", finish="stop"))
    assert text == "harus memenuhi persyaratan ketat."


def test_flagged_when_continuation_also_cut_off():
    text, _ = _resume(_chunks("karena", finish=None), _chunks("karena pem", finish=None))
    assert text == "karena pem" + CUTOFF_NOTICE
