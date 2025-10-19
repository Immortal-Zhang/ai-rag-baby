# -*- coding: utf-8 -*-
from src.rag_simple import RAG

def test_build_and_query(tmp_path):
    # 在临时目录里写两个小文档
    d = tmp_path / "docs"; d.mkdir()
    (d / "a.txt").write_text("Git 可以管理代码的历史与分支。", encoding="utf-8")
    (d / "b.txt").write_text("Docker 能把应用和依赖打包为镜像。", encoding="utf-8")

    r = RAG(); r.build(d)
    out = r.answer("如何管理代码历史？", topk=2)

    assert "Git" in out["answer"]
    assert len(out["hits"]) == 2
