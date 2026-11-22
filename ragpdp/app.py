"""Step 5: web API + UI.

Run: uvicorn ragpdp.app:app --port 8000
"""
import json
import time
from collections import defaultdict, deque
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from openai import RateLimitError
from pydantic import BaseModel, Field

from ragpdp.answer import answer, label
from ragpdp.retrieve import ROOT, Index

app = FastAPI()
idx = Index.load()

# Per-visitor limit so one person cannot burn the shared free Gemini quota for everyone.
# shortcut: in-memory and per server instance (Vercel may run several, and a cold start resets it);
# move to a shared store such as Upstash Redis if real abuse shows up.
LIMITS = ((60, 6), (24 * 3600, 40))  # (window in seconds, max questions per IP)
_asked: dict[str, deque] = defaultdict(deque)


def client_ip(request: Request) -> str:
    # Vercel sets these headers itself, so clients cannot spoof them there.
    forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    return request.headers.get("x-real-ip") or forwarded or (request.client.host if request.client else "unknown")


def over_limit(ip: str) -> bool:
    now, times = time.time(), _asked[ip]
    while times and now - times[0] > LIMITS[-1][0]:
        times.popleft()
    if any(sum(now - t < window for t in times) >= limit for window, limit in LIMITS):
        return True
    times.append(now)
    return False


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class Ask(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    history: list[Message] = Field(default=[], max_length=20)


@app.post("/api/ask")
def ask(req: Ask, request: Request) -> StreamingResponse:
    if over_limit(client_ip(request)):
        raise HTTPException(429, "Kamu sudah sampai batas: 6 pertanyaan per menit dan 40 per hari untuk tiap pengunjung. Tunggu semenit lalu coba lagi, ya. Kalau jatah hari ini sudah habis, besok kamu bisa bertanya lagi 🙏")
    history = [m.model_dump() for m in req.history[-4:]]  # last 2 turns are enough for follow-up questions
    try:
        hits, stream = answer(idx, req.question, history)
    except RateLimitError:  # free-tier per-minute or per-day limit
        raise HTTPException(429, "Kuota gratis Google Gemini untuk situs ini sedang habis, karena semua pengunjung memakai kuota yang sama. Coba lagi semenit lagi. Kalau masih gagal, jatah hari ini sudah habis dan terisi lagi besok siang 🙏")
    sources = [{"label": label(c), "bab": c["bab"], "text": c["text"], "penjelasan": c["penjelasan"]} for c in hits]

    def body():
        # First line: JSON of source pasal. The rest: streamed answer text.
        yield json.dumps({"sources": sources}, ensure_ascii=False) + "\n"
        yield from stream

    return StreamingResponse(body(), media_type="text/plain; charset=utf-8")


@app.get("/")
def home() -> FileResponse:
    # no-cache = revalidate every time, so visitors get the new UI right after a deploy.
    return FileResponse(ROOT / "web/index.html", headers={"Cache-Control": "no-cache"})
