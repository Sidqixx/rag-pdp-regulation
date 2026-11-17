# Tanya UU PDP

Chatbot RAG yang membantu orang awam memahami **UU 27/2022 tentang Pelindungan Data Pribadi** dan **PP 33/2026** (peraturan pelaksanaannya). Setiap jawaban menyebut pasal sumbernya, dan sitasi yang tidak ada di sumber otomatis ditandai.

> Bukan nasihat hukum. Percakapan tidak disimpan. Versi online memproses pertanyaan lewat API LLM pihak ketiga; untuk penggunaan yang sepenuhnya offline, jalankan versi lokal (Ollama).

## Cara kerja

1. **Ingestion:** PDF resmi → teks (`pymupdf`) → 1 chunk per Pasal, lengkap dengan Penjelasan dan daftar pasal yang dirujuk.
2. **Retrieval:** embedding + cosine similarity (menang telak atas BM25 dan hybrid di eval, lihat tabel), pasal yang disebut eksplisit diambil langsung, lalu pasal yang dirujuk ikut ditarik. Embedding: `gemini-embedding-2` (online) atau `bge-m3` lokal (offline).
3. **Generation:** LLM menjawab hanya dari pasal hasil retrieval, dengan format sitasi `[UU 27/2022 Pasal N]` yang dicek ulang di kode. Dua mode: **Gemini Flash-Lite** (API free tier, cepat) atau **qwen3.5:4b lokal via Ollama** (100% offline).
4. **Evaluasi:** 50 pertanyaan berbahasa awam beserta pasal yang benar.

## Hasil evaluasi

50 pertanyaan berbahasa awam (44 di lingkup UU/PP, 6 di luar lingkup), `eval/questions.jsonl`.

**Retrieval** (`python -m ragpdp.evaluate`):

| Metode | recall@5 | recall@8 | MRR |
|---|---|---|---|
| BM25 | 0.500 | 0.530 | 0.374 |
| Hybrid (BM25 + bge-m3, RRF) | 0.686 | 0.754 | 0.599 |
| Embedding `bge-m3` (lokal) | 0.833 | 0.924 | 0.766 |
| Hybrid (BM25 + gemini-embedding-2, RRF) | 0.712 | 0.777 | 0.635 |
| **Embedding `gemini-embedding-2` (online)** | **0.951** | **0.981** | **0.860** |

Hybrid kalah dari embedding saja dengan kedua model: pertanyaan orang awam jarang memakai kata yang sama dengan teks pasal, jadi BM25 malah menambah noise.

**Jawaban** (`python -m ragpdp.evaluate --answers`, semuanya memakai retrieval `bge-m3`):

| Metrik | **Gemini 3.5 Flash-Lite (API)** | qwen3.5:4b (lokal) | qwen3.5:2b (lokal) |
|---|---|---|---|
| Jawaban dengan sitasi | **43/44** | 38/44 | 37/44 |
| Menyitasi pasal yang benar | **41/44** | 35/44 | 27/44 |
| Ada sitasi halusinasi (ditandai merah di UI) | **0/44** | 2/44 | 6/44 |
| Menolak pertanyaan di luar lingkup | **6/6** | 6/6 | 2/6 |
| Waktu jawaban lengkap | **±15 detik** | ±45–55 detik | ±20–25 detik |

Model lokal diukur di GTX 1650 4GB. Angka 2b berasal dari prompt versi sebelumnya (dihitung ulang dengan pengecek sitasi yang sama). Hasil LLM bervariasi antar-run: run 4b sebelumnya mendapat 38/44 pasal benar tetapi 4/6 penolakan, jadi selisih ±3 pertanyaan masih dalam batas noise.

## Menjalankan

Butuh Python 3.12+ dan [Ollama](https://ollama.com). Set env `OLLAMA_CONTEXT_LENGTH=8192` sebelum Ollama dinyalakan; default 4096 terlalu kecil dan prompt akan terpotong diam-diam.

```powershell
ollama pull bge-m3
ollama pull qwen3.5:4b

python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt

.venv\Scripts\python.exe -m uvicorn ragpdp.app:app --port 8000   # buka http://localhost:8000
```

Membangun ulang data dan mengukur (pakai `.venv\Scripts\python.exe`):

```powershell
python -m ragpdp.extract              # PDF -> data/text/*.txt
python -m ragpdp.parse                # teks -> data/chunks.jsonl
python -m pytest
python -m ragpdp.evaluate             # retrieval
python -m ragpdp.evaluate --answers   # jawaban LLM (lama: ±50 jawaban)
```

Ganti provider (mis. Gemini free tier untuk deploy): salin `.env.example` jadi `.env` lalu isi API key. Chat (`CHAT_BASE_URL`, `CHAT_API_KEY`, `CHAT_MODEL`) dan embedding (`EMBED_BASE_URL`, `EMBED_API_KEY`, `EMBED_MODEL`) dikonfigurasi terpisah. `.env` tidak ikut di-commit.

## Struktur

```
data/raw/          PDF resmi
data/text/         hasil extract
data/chunks.jsonl  1 baris = 1 pasal (+ penjelasan, rujukan)
ragpdp/            extract, parse, retrieve, answer, evaluate, app
web/index.html     UI chat
eval/              pertanyaan uji + jawaban model
docs/LEARNING.md   peta belajar, keputusan desain, log
```
