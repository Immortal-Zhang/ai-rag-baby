# -*- coding: utf-8 -*-
"""
最小可用生成器：用 flan-t5-small 把“问题 + 检索上下文”生成自然语言答案。
CPU 可跑，首次会自动下载模型到本地缓存。
"""
from __future__ import annotations
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Generator:
    def __init__(self, model_name: str = "google/flan-t5-small"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(self, question: str, context: str, max_new_tokens: int = 256) -> str:
        prompt = (
            "你是一个严谨的中文助手。请严格基于给定资料回答问题；"
            "若资料中没有答案，请明确说“无法在资料中找到答案”。\n\n"
            f"【资料】\n{context}\n\n"
            f"【问题】{question}\n\n"
            "【回答】"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        outputs = self.model.generate(
            **inputs,
            do_sample=False,            # 先走确定性输出，便于复现
            max_new_tokens=max_new_tokens
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
