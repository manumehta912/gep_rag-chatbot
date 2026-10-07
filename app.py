from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from rag import RAG

app = FastAPI(title="World Bank GEP June 2026 - Advanced RAG")
rag = RAG()


class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []
    chapter: Optional[str] = None
    rerank: bool = False


@app.post("/api/chat")
def chat(req: ChatRequest):
    try:
        return rag.answer(req.question, req.history, req.chapter, req.rerank)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:300])


@app.get("/api/meta")
def meta():
    return {"chapters": rag.chapters, "chunks": len(rag.chunks)}


@app.get("/")
def home():
    return FileResponse(config.BASE_DIR / "static" / "index.html")


app.mount("/static", StaticFiles(directory=config.BASE_DIR / "static"), name="static")
