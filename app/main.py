# -*- coding: utf-8 -*-
from fastapi import FastAPI
from pydantic import BaseModel

# 检索器：优先向量检索
try:
    from src.rag_embed import RAG
    USE_EMBED = True
except Exception:
    from src.rag_simple import RAG
    USE_EMBED = False

# 生成器：优先小模型
try:
    from src.generator import Generator
    GEN = Generator()
    USE_GEN = True
except Exception:
    GEN = None
    USE_GEN = False

# 护栏
from src.guard import apply_guard

app = FastAPI(title="RAG Demo (Embed + Small LLM + Guard)")

rag = RAG()
try:
    rag.load()
except Exception:
    rag.build("data/docs")

class AskReq(BaseModel):
    question: str
    topk: int = 3

@app.get("/")
def root():
    return {
        "status": "ok",
        "retriever": "embed" if USE_EMBED else "tf-idf",
        "generator": "flan-t5-small" if USE_GEN else "template",
        "guard": "on"
    }

@app.post("/ask")
def ask(req: AskReq):
    res = rag.answer(req.question, topk=req.topk)
    hits = res.get("hits", [])
    # 护栏判定
    blocked, msg = apply_guard(req.question, hits)
    if blocked:
        return {"answer": msg, "hits": hits}

    # 生成或模板
    context = res.get("context") or "\n".join([f"- {h['chunk']}（来源：{h['source']}）" for h in hits])
    if USE_GEN:
        answer = GEN.generate(req.question, context, max_new_tokens=192)
    else:
        answer = f"基于资料：\n{context}\n（模板回答）"

    return {"answer": answer, "hits": hits}
