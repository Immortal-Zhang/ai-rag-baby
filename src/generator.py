# -*- coding: utf-8 -*-
from __future__ import annotations
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Generator:
    def __init__(self, model_name: str = "google/mt5-small"):
        # mT5 更适合多语言（含中文）
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(self, question: str, context: str, max_new_tokens: int = 192) -> str:
        # 明确要求复用术语、条目式输出
        prompt = (
            "你是一个严谨的中文技术写手，请严格遵守：\n"
            "1) 仅根据【资料】回答，不得编造资料外内容；\n"
            "2) 保留资料中的关键术语（如：版本控制、分支、合并、回滚、断言、持续集成、镜像、容器、Dockerfile 等）；\n"
            "3) 用编号要点输出，每条尽量含术语；最后给一行“结论：…”。\n"
            "4) 若资料不足，请回答：无法在资料中找到答案。\n\n"
            f"【资料】\n{context}\n\n"
            f"【问题】{question}\n\n"
            "【按上述要求用中文作答】"
        )
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        out = self.model.generate(
            **enc,
            do_sample=False,      # 先走确定性，便于复现
            num_beams=4,          # 提升覆盖
            length_penalty=0.0,
            max_new_tokens=max_new_tokens,
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
