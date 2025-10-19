# -*- coding: utf-8 -*-
"""
升级版 RAG：Sentence-Transformers 向量检索 + 余弦近邻
"""
from __future__ import annotations
import os, re, json, glob, argparse, pickle
from typing import List, Tuple
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data/docs")
INDEX_DIR = Path("artifacts")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

def read_docs(folder: Path) -> List[Tuple[str, str]]:
    paths = sorted(glob.glob(str(folder / "*.txt")))
    docs = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append((os.path.basename(p), text))
    return docs

def chunk_text(text: str, max_chars=400) -> List[str]:
    import re
    sents = re.split(r"(?<=[。！？；.!?])", text)
    buf, out = "", []
    for s in sents:
        if len(buf) + len(s) <= max_chars:
            buf += s
        else:
            if buf: out.append(buf.strip())
            buf = s
    if buf: out.append(buf.strip())
    return [c for c in out if c.strip()]

class RAG:
    def __init__(self, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embed_model_name = embed_model
        self.embedder: SentenceTransformer | None = None
        self.nn = None
        self.chunks: List[str] = []
        self.meta: List[str] = []
        self.embeddings: np.ndarray | None = None

    def _ensure_embedder(self):
        if self.embedder is None:
            # device='cpu' 保证所有机器都能跑
            self.embedder = SentenceTransformer(self.embed_model_name, device="cpu")

    def build(self, folder=DATA_DIR):
        docs = read_docs(Path(folder))
        chunks, meta = [], []
        for fname, text in docs:
            for ch in chunk_text(text):
                chunks.append(ch)
                meta.append(fname)
        self._ensure_embedder()
        # 归一化嵌入以便用余弦相似度
        emb = self.embedder.encode(
            chunks, batch_size=32, show_progress_bar=False, normalize_embeddings=True
        )
        emb = np.asarray(emb, dtype=np.float32)
        self.nn = NearestNeighbors(n_neighbors=5, metric="cosine").fit(emb)
        self.chunks, self.meta, self.embeddings = chunks, meta, emb

        # 持久化
        with open(INDEX_DIR / "chunks.json", "w", encoding="utf-8") as f:
            json.dump({"chunks": self.chunks, "meta": self.meta}, f, ensure_ascii=False)
        np.save(INDEX_DIR / "embeddings.npy", emb)
        with open(INDEX_DIR / "embed_model.txt", "w", encoding="utf-8") as f:
            f.write(self.embed_model_name)

    def load(self):
        # 读元数据与嵌入
        with open(INDEX_DIR / "chunks.json", "r", encoding="utf-8") as f:
            obj = json.load(f)
        self.chunks, self.meta = obj["chunks"], obj["meta"]
        self.embeddings = np.load(INDEX_DIR / "embeddings.npy")
        # 恢复近邻
        self.nn = NearestNeighbors(n_neighbors=5, metric="cosine").fit(self.embeddings)
        # 恢复同款嵌入模型（若不存在就用默认）
        try:
            model_name = Path(INDEX_DIR / "embed_model.txt").read_text(encoding="utf-8").strip()
        except Exception:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.embed_model_name = model_name
        self._ensure_embedder()

    def retrieve(self, question: str, topk=3):
        self._ensure_embedder()
        q = self.embedder.encode([question], normalize_embeddings=True)
        distances, indices = self.nn.kneighbors(np.asarray(q), n_neighbors=topk)
        hits = []
        for dist, idx in zip(distances[0], indices[0]):
            hits.append({
                "chunk": self.chunks[idx],
                "source": self.meta[idx],
                "score": 1 - float(dist)
            })
        return hits

    def answer(self, question: str, topk=3) -> dict:
        # 这里只负责检索与拼接，真正“生成”交给 generator.py
        hits = self.retrieve(question, topk=topk)
        context = "\n".join([f"- {h['chunk']}（来源：{h['source']}）" for h in hits])
        return {"context": context, "hits": hits}

def cli():
    import argparse, pprint
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-index", type=str, default="")
    parser.add_argument("--ask", type=str, default="")
    args = parser.parse_args()
    rag = RAG()
    if args.build_index:
        rag.build(args.build_index)
        print("✅ 向量索引已构建到 artifacts/")
    elif args.ask:
        rag.load()
        out = rag.answer(args.ask)
        pprint.pp(out)
    else:
        print("用法：\n  python -m src.rag_embed --build-index data/docs\n  python -m src.rag_embed --ask \"什么是 Git？\"")

if __name__ == "__main__":
    cli()
