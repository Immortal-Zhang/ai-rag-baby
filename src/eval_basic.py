# -*- coding: utf-8 -*-
"""
最小离线评测：
- 检索命中率（Top-k 内是否包含预期文件名）
- 答案关键词覆盖率（命中关键词个数 / 关键词总数）
"""
from __future__ import annotations
import json, os
from pathlib import Path

# 优先用“向量检索 + 小模型”，不可用则自动回退
try:
    from src.rag_embed import RAG
    USE_EMBED = True
except Exception:
    from src.rag_simple import RAG
    USE_EMBED = False

try:
    from src.generator import Generator
    GEN = Generator()
    USE_GEN = True
except Exception:
    GEN = None
    USE_GEN = False

ROOT = Path(__file__).resolve().parents[1]
EVAL_SET = ROOT / "eval" / "eval_set.jsonl"
OUT_DIR = ROOT / "eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def build_context(hits):
    return "\n".join([f"- {h['chunk']}（来源：{h['source']}）" for h in hits])

def ask_once(rag: RAG, question: str, topk: int = 3) -> tuple[str, list]:
    res = rag.answer(question, topk=topk)
    hits = res.get("hits", [])
    context = res.get("context") or build_context(hits)
    if USE_GEN:
        from src.generator import Generator  # 避免导入顺序问题
        gen = GEN or Generator()
        answer = gen.generate(question, context, max_new_tokens=192)
    else:
        answer = f"基于资料：\n{context}\n（模板回答）"
    return answer, hits

def evaluate():
    rag = RAG()
    # 若没有索引就先构建
    try:
        rag.load()
    except Exception:
        rag.build(ROOT / "data" / "docs")

    n = 0
    hit_ok = 0
    kw_covered_sum = 0.0

    rows = []
    with open(EVAL_SET, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            q = obj["question"]
            exp_src = obj["expected_source"]
            exp_kws = obj["expected_keywords"]

            answer, hits = ask_once(rag, q, topk=3)
            sources = [h["source"] for h in hits]
            # 检索命中：Top-3 任意命中目标文件即记 1
            hit = 1 if exp_src in sources else 0
            # 关键词覆盖率：答案中出现了多少预期关键词
            covered = sum(1 for kw in exp_kws if kw in answer)
            cover_rate = covered / max(len(exp_kws), 1)

            n += 1
            hit_ok += hit
            kw_covered_sum += cover_rate

            rows.append({
                "question": q,
                "expected_source": exp_src,
                "top3_sources": "|".join(sources),
                "retrieval_hit": hit,
                "keyword_cover_rate": round(cover_rate, 3)
            })

    # 汇总
    retrieval_hit_rate = hit_ok / n if n else 0.0
    avg_kw_cover = kw_covered_sum / n if n else 0.0

    # 输出到控制台
    print("\n===== Eval Summary =====")
    print(f"Retriever: {'embed' if USE_EMBED else 'tf-idf'}; Generator: {'flan-t5-small' if USE_GEN else 'template'}")
    print(f"Samples: {n}")
    print(f"Top-3 Retrieval Hit Rate: {retrieval_hit_rate:.3f}")
    print(f"Answer Keyword Coverage (avg): {avg_kw_cover:.3f}")

    # 存 CSV
    csv_path = OUT_DIR / "results.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("question,expected_source,top3_sources,retrieval_hit,keyword_cover_rate\n")
        for r in rows:
            f.write(f"{r['question']},{r['expected_source']},{r['top3_sources']},{r['retrieval_hit']},{r['keyword_cover_rate']}\n")
    # 存 summary
    with open(OUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write(
            f"Retriever: {'embed' if USE_EMBED else 'tf-idf'}; "
            f"Generator: {'flan-t5-small' if USE_GEN else 'template'}\n"
            f"Samples: {n}\n"
            f"Top-3 Retrieval Hit Rate: {retrieval_hit_rate:.3f}\n"
            f"Answer Keyword Coverage (avg): {avg_kw_cover:.3f}\n"
        )
    print(f"\n明细已保存：{csv_path}\n汇总：{OUT_DIR / 'summary.txt'}")

if __name__ == "__main__":
    evaluate()
