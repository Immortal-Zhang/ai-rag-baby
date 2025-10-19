# -*- coding: utf-8 -*-
"""
离线评测（改进版）：
- Top-3 检索命中率（按来源文件名）
- 关键词覆盖率：在 “答案 + 上下文” 中找关键词；支持简单同义词
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any

# —— 检索器：优先向量版 —— #
try:
    from src.rag_embed import RAG
    USE_EMBED = True
except Exception:
    from src.rag_simple import RAG
    USE_EMBED = False

# —— 生成器：优先小模型 —— #
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

def build_context(hits: List[Dict[str, Any]]) -> str:
    return "\n".join([f"- {h['chunk']}（来源：{h['source']}）" for h in hits])

# —— 同义词表（可按需扩展） —— #
SYNONYMS = {
    "回滚": ["回退", "撤销"],
    "持续集成": ["CI", "持续集成(CI)"],
    "镜像": ["image", "镜像文件"],
    "容器": ["container", "容器实例"],
    "自动化测试": ["测试自动化"],
    "分支": ["branch"],
    "合并": ["merge"],
    "断言": ["assert"],
    "Dockerfile": ["dockerfile"],
    "记录修改": ["记录变更", "变更记录"],
    "版本控制": ["版本管理"],
}

def keyword_hit(text: str, kw: str) -> bool:
    if kw in text:
        return True
    for alt in SYNONYMS.get(kw, []):
        if alt in text:
            return True
    return False

def ask_once(rag: RAG, question: str, topk: int = 3) -> tuple[str, List[Dict[str, Any]], str]:
    res = rag.answer(question, topk=topk)
    hits = res.get("hits", [])
    context = res.get("context") or build_context(hits)
    if USE_GEN:
        gen = GEN or Generator()
        answer = gen.generate(question, context, max_new_tokens=192)
    else:
        answer = f"基于资料：\n{context}\n（模板回答）"
    return answer, hits, context

def evaluate():
    rag = RAG()
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

            answer, hits, context = ask_once(rag, q, topk=3)
            sources = [h["source"] for h in hits]
            hit = 1 if exp_src in sources else 0

            # 关键：在 “答案 + 上下文” 中查找关键词/同义词
            combined = (answer + "\n" + context)
            covered = sum(1 for kw in exp_kws if keyword_hit(combined, kw))
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

    retrieval_hit_rate = hit_ok / n if n else 0.0
    avg_kw_cover = kw_covered_sum / n if n else 0.0

    print("\n===== Eval Summary =====")
    print(f"Retriever: {'embed' if USE_EMBED else 'tf-idf'}; Generator: {'mt5-small' if USE_GEN else 'template'}")
    print(f"Samples: {n}")
    print(f"Top-3 Retrieval Hit Rate: {retrieval_hit_rate:.3f}")
    print(f"Answer Keyword Coverage (avg): {avg_kw_cover:.3f}")

    # 保存结果
    csv_path = OUT_DIR / "results.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("question,expected_source,top3_sources,retrieval_hit,keyword_cover_rate\n")
        for r in rows:
            f.write(f"{r['question']},{r['expected_source']},{r['top3_sources']},{r['retrieval_hit']},{r['keyword_cover_rate']}\n")
    with open(OUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write(
            f"Retriever: {'embed' if USE_EMBED else 'tf-idf'}; "
            f"Generator: {'mt5-small' if USE_GEN else 'template'}\n"
            f"Samples: {n}\n"
            f"Top-3 Retrieval Hit Rate: {retrieval_hit_rate:.3f}\n"
            f"Answer Keyword Coverage (avg): {avg_kw_cover:.3f}\n"
        )
    print(f"\n明细已保存：{csv_path}\n汇总：{OUT_DIR / 'summary.txt'}")

if __name__ == "__main__":
    evaluate()
