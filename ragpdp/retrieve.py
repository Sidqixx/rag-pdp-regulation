"""Step 3: cari pasal yang relevan.

Default: embedding (makna). BM25 (kata persis) dan hybrid (RRF) tetap ada untuk perbandingan di evaluate.py.
"""
import hashlib
import json
import os
import re
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from openai import OpenAI, RateLimitError
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]

# .env (tidak di-commit) berisi API key & pilihan provider. Env yang sudah di-set di terminal tetap menang.
if (ROOT / ".env").exists():
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

# Ollama, Gemini, Groq, dll. sama-sama menyediakan API format OpenAI: ganti provider cukup lewat env.
# Embedding dan chat dikonfigurasi terpisah (lihat answer.py), jadi bisa beda provider.
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
embed_client = OpenAI(
    base_url=os.getenv("EMBED_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("EMBED_API_KEY", "ollama"),
)

STOPWORDS = set(
    "yang dan di ke dari ini itu untuk dengan atau pada dalam adalah sebagaimana dimaksud ayat huruf "
    "tersebut oleh akan apa apakah bagaimana gimana kalau kalo jika gua gue aku saya kita kami mereka "
    "bisa boleh harus ada tidak gak nggak ga enggak sih ya dong kah lah nya juga sudah udah mau "
    "berapa kapan siapa mana".split()
)
RRF_K = 60  # konstanta standar dari paper RRF
PASAL_DISEBUT = re.compile(r"\bpasal\s+(\d+)", re.I)


def tokenize(s: str) -> list[str]:
    return [w for w in re.findall(r"\w+", s.lower()) if w not in STOPWORDS]


def doc_text(c: dict) -> str:
    """Teks yang di-index: label + BAB + isi + penjelasan."""
    return f"{c['source']} Pasal {c['pasal']} - {c['bab']}\n{c['text']}\nPenjelasan: {c['penjelasan']}"


def embed(texts: list[str], wait_on_limit: bool = False) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), 32):
        for attempt in range(6):
            try:
                res = embed_client.embeddings.create(model=EMBED_MODEL, input=texts[i : i + 32])
                break
            except RateLimitError:  # free tier Gemini: 100 teks/menit
                if not wait_on_limit or attempt == 5:
                    raise
                time.sleep(20)
        vecs += [d.embedding for d in res.data]
    v = np.array(vecs, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)  # normalisasi -> dot product = cosine similarity


@lru_cache(maxsize=1024)
def embed_query(query: str) -> np.ndarray:
    """Pertanyaan yang sama tidak di-embed ulang (hemat kuota API, mis. di evaluate.py)."""
    return embed([query])[0]


class Index:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.by_id = {c["id"]: c for c in chunks}
        docs = [doc_text(c) for c in chunks]
        self.bm25 = BM25Okapi([tokenize(d) for d in docs])
        # Cache embedding: nama file memuat hash isi, jadi otomatis dihitung ulang kalau chunk/model berubah.
        h = hashlib.sha1((EMBED_MODEL + "".join(docs)).encode()).hexdigest()[:10]
        cache = ROOT / "data" / f"emb-{re.sub(r'\W', '_', EMBED_MODEL)}-{h}.npy"
        if cache.exists():
            self.emb = np.load(cache)
        else:
            self.emb = embed(docs, wait_on_limit=True)  # 301 pasal sekaligus: bisa kena limit per menit
            np.save(cache, self.emb)

    @classmethod
    def load(cls) -> "Index":
        with open(ROOT / "data/chunks.jsonl", encoding="utf-8") as f:
            return cls([json.loads(line) for line in f])

    def rank(self, query: str, method: str = "embed") -> list[int]:
        """Urutan indeks chunk dari paling relevan. Default embedding: BM25 & hybrid kalah di eval (lihat README)."""
        if method == "bm25":
            return list(np.argsort(-self.bm25.get_scores(tokenize(query))))
        if method == "embed":
            return list(np.argsort(-(self.emb @ embed_query(query))))
        score = np.zeros(len(self.chunks))
        for m in ("bm25", "embed"):
            for r, i in enumerate(self.rank(query, m)):
                score[i] += 1 / (RRF_K + r + 1)
        return list(np.argsort(-score))

    def search(self, query: str, k: int = 8, method: str = "embed") -> list[dict]:
        # Pasal yang disebut eksplisit ("isi Pasal 20 PP?") langsung diambil; embedding lemah untuk nomor.
        docs = ["pp33"] if re.search(r"\bPP\b", query, re.I) else ["uu27"] if re.search(r"\bUU\b", query, re.I) else ["uu27", "pp33"]
        named = [self.by_id[f"{d}:{n}"] for n in PASAL_DISEBUT.findall(query) for d in docs if f"{d}:{n}" in self.by_id]
        ranked = [self.chunks[i] for i in self.rank(query, method)[:k]]
        return named + [c for c in ranked if c not in named][: max(0, k - len(named))]

    def with_refs(self, hits: list[dict], max_extra: int = 3) -> list[dict]:
        """Tambah pasal yang dirujuk hasil pencarian, supaya LLM tahu isi 'sebagaimana dimaksud dalam Pasal X'.
        Pasal yang merujuk >3 pasal (mis. daftar sanksi) dilewati: terlalu banyak untuk konteks."""
        seen = {c["id"] for c in hits}
        extra = []
        for c in hits:
            if len(c["refs"]) > 3:
                continue
            for r in c["refs"]:
                rid = f"{c['id'].split(':')[0]}:{r}"
                if rid not in seen and len(extra) < max_extra:
                    seen.add(rid)
                    extra.append(self.by_id[rid])
        return hits + extra
