"""Step 5: web API + UI.

Run: uvicorn ragpdp.app:app --port 8000
"""
import json
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from openai import RateLimitError
from pydantic import BaseModel, Field

from ragpdp.answer import answer, label
from ragpdp.retrieve import ROOT, Index

app = FastAPI()
idx = Index.load()


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class Ask(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    history: list[Message] = Field(default=[], max_length=20)


@app.post("/api/ask")
def ask(req: Ask) -> StreamingResponse:
    history = [m.model_dump() for m in req.history[-4:]]  # last 2 turns are enough for follow-up questions
    try:
        hits, stream = answer(idx, req.question, history)
    except RateLimitError:  # free-tier per-minute or per-day limit
        raise HTTPException(429, "Maaf, lagi banyak yang bertanya nih, jadi kuota AI gratisnya sedang habis. Coba lagi sekitar semenit lagi, ya. Kalau masih belum bisa, berarti jatah hari ini sudah habis dan besok siang bisa dipakai lagi 🙏")
    sources = [{"label": label(c), "bab": c["bab"], "text": c["text"], "penjelasan": c["penjelasan"]} for c in hits]

    def body():
        # First line: JSON of source pasal. The rest: streamed answer text.
        yield json.dumps({"sources": sources}, ensure_ascii=False) + "\n"
        yield from stream

    return StreamingResponse(body(), media_type="text/plain; charset=utf-8")


@app.get("/")
def home() -> FileResponse:
    return FileResponse(ROOT / "web/index.html")
