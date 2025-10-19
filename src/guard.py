# -*- coding: utf-8 -*-
from __future__ import annotations
import re
from typing import List, Tuple, Dict, Any

# 命中这些注入/高风险指令就拒答（示例，实际可扩展）
INJECTION_PATTERNS = [
    r"忽略(以上|之前).*指令",
    r"越狱", r"脱离(限制|规则)",
    r"系统提示", r"提示注入",
    r"DROP\s+TABLE", r"DELETE\s+FROM", r"rm\s+-rf\s+/",
]

def apply_guard(question: str, hits: List[Dict[str, Any]], min_score: float = 0.35) -> Tuple[bool, str]:
    """
    返回：(blocked, message)
    - 若最高相关度 < min_score，则认为资料支撑不足 → 拒答
    - 若命中注入/危险操作关键词 → 拒答
    """
    # 低相关拒答
    top_score = max([h.get("score", 0.0) for h in hits], default=0.0)
    if top_score < min_score:
        return True, "无法在资料中找到可靠答案，请补充资料或调整问题。"

    # 注入/危险操作拒答
    for pat in INJECTION_PATTERNS:
        if re.search(pat, question, flags=re.IGNORECASE):
            return True, "问题包含越界或危险操作指令，已拒绝处理。"

    return False, ""
