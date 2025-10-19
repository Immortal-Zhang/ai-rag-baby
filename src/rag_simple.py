# -*- coding: utf-8 -*-
"""
一个最小可用的 RAG（不连大模型）：用 TF-IDF 做向量化，最近邻做检索，
把检索到的片段用模板拼接成“基于资料的回答 + 引用”。
"""
from __future__ import annotations
import os, re, json, argparse, glob, pickle
from typing import List, Tuple
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

DATA_DIR = Path("data/docs")
INDEX_DIR = Path("artifacts")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

def normalize_text(s: str) -> str:
    """极简清洗：小写 + 压空白"""
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def read_docs(folder: Path) -> List[Tuple[str, str]]:
    """读取 docs 文件夹下的 .txt 文档，返回 [(文件名, 内容), ...]"""
    paths = sorted(glob.glob(str(folder / "*.txt")))
    docs = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append((os.path.basename(p), text))
    return docs

def chunk_text(text: str, max_chars=400) -> List[str]:
    """把长文本切成不超过 max_chars 的小段，尽量按句号等断开"""
    sents = re.split(r"(?<=[。！？；.!?])", text)
    buf, out = "", []
    for s in sents:
        if len(buf) + len(s) <= max_chars:
            buf += s
        else:
            if buf:
                out.append(buf.strip())
            buf = s
    if buf:
        out.append(buf.strip())
    return [c for c in out if c.strip()]

class RAG:
    def __init__(self):
        self.vectorizer = None
        self.nn = None
        self.chunks: List[str] = []
        self.meta: List[str] = []  # 与 chunks 对应的来源文件名

    def build(self, folder=DATA_DIR):
        """从原始文档构建索引：切块 → TF-IDF → 最近邻"""
        docs = read_docs(Path(folder))
        chunks, meta = [], []
        for fname, text in docs:
            for ch in chunk_text(text):
                chunks.append(ch)
                meta.append(fname)

        # 用 1-2 元语法，效果比纯 1 元更稳一点
        self.vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
        X = self.vectorizer.fit_transform(chunks)

        self.nn = NearestNeighbors(n_neighbors=5, metric="cosine").fit(X)
        self.chunks = chunks
        self.meta = meta

        # 持久化到 artifacts/，下次直接加载
        with open(INDEX_DIR / "tfidf.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(INDEX_DIR / "nn.pkl", "wb") as f:
            pickle.dump(self.nn, f)
        with open(INDEX_DIR / "chunks.json", "w", encoding="utf-8") as f:
            json.dump({"chunks": self.chunks, "meta": self.meta}, f, ensure_ascii=False)

    def load(self):
        """加载已保存的索引"""
        with open(INDEX_DIR / "tfidf.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(INDEX_DIR / "nn.pkl", "rb") as f:
            self.nn = pickle.load(f)
        with open(INDEX_DIR / "chunks.json", "r", encoding="utf-8") as f:
            obj = json.load(f)
        self.chunks = obj["chunks"]
        self.meta = obj["meta"]

    def retrieve(self, question: str, topk=3):
        """用余弦相似度检索 topk 个最相关文本块"""
        Xq = self.vectorizer.transform([question])
        distances, indices = self.nn.kneighbors(Xq, n_neighbors=topk)
        hits = []
        for dist, idx in zip(distances[0], indices[0]):
            hits.append({
                "chunk": self.chunks[idx],
                "source": self.meta[idx],
                "score": 1 - float(dist)  # 余弦相似度 = 1 - 距离
            })
        return hits

    def answer(self, question: str, topk=3) -> dict:
        """模板回答：把检索到的片段按“引用”方式拼出来"""
        hits = self.retrieve(question, topk=topk)
        context = "\n".join([f"- {h['chunk']}（来源：{h['source']}）" for h in hits])
        answer = (
            f"基于资料，关于“{question}”我找到了这些要点：\n"
            f"{context}\n\n"
            f"请根据以上资料综合作答（这是一份演示，未调用大模型）。"
        )
        return {"answer": answer, "hits": hits}

def cli():
    """命令行模式：--build-index 构建索引；--ask 提问"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-index", type=str, default="")
    parser.add_argument("--ask", type=str, default="")
    args = parser.parse_args()

    rag = RAG()
    if args.build_index:
        rag.build(args.build_index)
        print("✅ 索引已构建到 artifacts/")
    elif args.ask:
        rag.load()
        out = rag.answer(args.ask)
        print(out["answer"])
        print("\n引用：")
        for h in out["hits"]:
            print(f"{h['source']}  score={h['score']:.3f}")
    else:
        print("用法示例：\n"
              "  python -m src.rag_simple --build-index data/docs\n"
              "  python -m src.rag_simple --ask \"什么是 Git？\"")

if __name__ == "__main__":
    cli()
