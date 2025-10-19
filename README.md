# AI RAG Baby（向量检索 + 小模型生成）

最小可复现的 RAG Demo：Sentence-Transformers 向量检索（all-MiniLM-L6-v2） + `flan-t5-small` 生成，含**离线评测**与**基本安全护栏**。

## 快速开始
```bash
# 进入并激活虚拟环境
cd ai-rag-baby
source .venv/bin/activate

# 依赖
pip install -r requirements.txt
pip install transformers==4.45.2 accelerate==0.34.2 sentence-transformers==2.7.0 torch==2.4.1

# 构建向量索引
python -m src.rag_embed --build-index data/docs

# 启动服务
uvicorn app.main:app --reload
# 浏览器 http://127.0.0.1:8000/docs → POST /ask
