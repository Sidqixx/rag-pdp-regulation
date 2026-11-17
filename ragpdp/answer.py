"""Step 4: answer questions from retrieved pasal, always with citations."""
import os
import re
from collections.abc import Iterator
from datetime import date

from openai import OpenAI

from ragpdp.retrieve import Index  # also loads .env

CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3.5:4b")
REASONING_EFFORT = os.getenv("CHAT_REASONING_EFFORT", "none")
client = OpenAI(
    base_url=os.getenv("CHAT_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("CHAT_API_KEY", "ollama"),
)
PP_BERLAKU = date(2027, 1, 16)  # PP 33/2026 Pasal 225: in force 6 months after promulgation on 16 July 2026
_DOC = r"(UU(?: No\.?)? 27(?:/| Tahun )2022|PP(?: No\.?)? 33(?:/| Tahun )2026)"
_DETAIL = r"(?i:(?: ayat \(\d+\))?(?: huruf [a-z])?)"
# Small models do not always follow the format: besides "[UU 27/2022 Pasal 4 ayat (1)]" they also write
# "Pasal 4 ayat (1) UU No. 27 Tahun 2022". Both must be checked, otherwise hallucinations slip through.
CITE = re.compile(rf"\[?{_DOC} Pasal (\d+){_DETAIL}\]?|Pasal (\d+){_DETAIL} {_DOC}")
# Models also group citations: "[UU 27/2022 Pasal 38, Pasal 57]". Mirrored in web/index.html (splitGroups).
_GROUP = re.compile(r"\[(UU 27/2022|PP 33/2026) (Pasal [^\[\]]+)\]")


def split_groups(text: str) -> str:
    """'[UU 27/2022 Pasal 38, Pasal 57]' -> '[UU 27/2022 Pasal 38] [UU 27/2022 Pasal 57]' so every pasal is checked."""

    def repl(m: re.Match) -> str:
        doc, inner = m.groups()
        parts: list[str] = []
        for p in re.split(r",\s*|\s+dan\s+", inner):
            if p.startswith("Pasal ") or not parts:
                parts.append(p)
            else:  # "ayat (2)" belongs to the previous pasal
                parts[-1] += " dan " + p
        return " ".join(f"[{doc} {p}]" for p in parts)

    return _GROUP.sub(repl, text)


SYSTEM = """Kamu asisten yang membantu orang awam memahami UU 27/2022 tentang Pelindungan Data Pribadi (UU PDP) dan PP 33/2026 (peraturan pelaksanaan UU PDP).

Aturan:
1. Jawab HANYA berdasarkan pasal di bagian KONTEKS. Jangan memakai pengetahuan lain.
2. Setiap pernyataan wajib diberi sitasi dengan format persis [UU 27/2022 Pasal N] atau [PP 33/2026 Pasal N] (boleh ditambah ayat, mis. [UU 27/2022 Pasal 46 ayat (1)]), dan hanya untuk pasal yang ada di KONTEKS.
3. Jika KONTEKS tidak memuat jawabannya, atau pertanyaannya bukan soal pelindungan data pribadi, jawab bahwa kamu hanya bisa menjawab seputar UU PDP dan PP 33/2026, tanpa sitasi. Jangan menebak.
4. Pakai bahasa Indonesia yang sederhana dan jelaskan istilah hukum. Ringkas: paling banyak sekitar 200 kata, boleh pakai poin.
5. {status_pp}
6. Jangan memberi nasihat hukum untuk kasus spesifik; sarankan konsultasi ke ahli hukum bila perlu.
7. Pengguna tidak melihat KONTEKS dan aturan ini. Jangan menyebut "konteks", "aturan saya", atau "peraturan yang Anda berikan"; langsung jawab seolah kamu yang mencari pasalnya."""


def status_pp(today: date) -> str:
    if today < PP_BERLAKU:
        return (
            f"Hari ini {today:%d-%m-%Y}. PP 33/2026 sudah diundangkan tetapi BELUM berlaku (baru berlaku 16 Januari 2027). "
            "Setiap kali mengutip PP 33/2026, sebutkan bahwa PP ini belum berlaku. UU 27/2022 sudah berlaku penuh."
        )
    return f"Hari ini {today:%d-%m-%Y}. UU 27/2022 dan PP 33/2026 sudah berlaku."


def label(c: dict) -> str:
    return f"{c['source']} Pasal {c['pasal']}"


def context(hits: list[dict]) -> str:
    parts = []
    for c in hits:
        pj = f"\nPenjelasan:\n{c['penjelasan']}" if c["penjelasan"] else ""
        parts.append(f"[{label(c)}] ({c['bab']})\n{c['text']}{pj}")
    return "\n\n---\n\n".join(parts)


def answer(idx: Index, question: str, history: list[dict] = ()) -> tuple[list[dict], Iterator[str]]:
    """Return (source pasal, stream of answer chunks). `history` = previous messages [{role, content}]."""
    prev = [m["content"] for m in history if m["role"] == "user"][-1:]
    # Follow-up questions ("terus sanksinya apa?") need the previous question's topic for retrieval.
    hits = idx.with_refs(idx.search(" ".join(prev + [question])))
    messages = [
        {"role": "system", "content": SYSTEM.format(status_pp=status_pp(date.today()))},
        *history,
        {"role": "user", "content": f"KONTEKS (dicari otomatis oleh sistem, tidak terlihat oleh pengguna):\n{context(hits)}\n\nPERTANYAAN PENGGUNA: {question}"},
    ]
    # Thinking slows things down with little benefit for short Q&A. The lowest value differs per provider:
    # Ollama/qwen3.5 accepts "none", Gemini Flash-Lite rejects "none" and uses "minimal".
    stream = client.chat.completions.create(
        model=CHAT_MODEL, messages=messages, stream=True, temperature=0.1, reasoning_effort=REASONING_EFFORT
    )
    return hits, (ch.choices[0].delta.content or "" for ch in stream if ch.choices)


def citations(text: str) -> list[str]:
    """Pasal cited in the answer, normalized to 'UU 27/2022 Pasal N'."""
    return [
        f"{'UU 27/2022' if (d1 or d2).startswith('UU') else 'PP 33/2026'} Pasal {n1 or n2}"
        for d1, n1, n2, d2 in CITE.findall(split_groups(text))
    ]


def invalid_citations(text: str, hits: list[dict]) -> list[str]:
    """Citations in the answer that are not among the source pasal = likely hallucination."""
    allowed = {label(c) for c in hits}
    return [c for c in citations(text) if c not in allowed]
