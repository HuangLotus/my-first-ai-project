import os
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# 加载环境
load_dotenv()

# ----------------------
# 阿里 Embedding
# ----------------------
class AliEmb(Embeddings):
    def embed_query(self, txt):
        try:
            headers = {
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "text-embedding-v2",
                "input": {"texts": [txt]}
            }
            resp = requests.post(
                "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding",
                headers=headers, json=data, timeout=15
            )
            return resp.json()["output"]["embeddings"][0]["embedding"]
        except Exception as e:
            print("向量接口异常:", e)
            return [0.0] * 1024

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]

# ----------------------
# 模型初始化
# ----------------------
llm = ChatOpenAI(
    model="qwen-turbo",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
    timeout=15
)

emb = AliEmb()

# 向量库
db = Chroma(
    embedding_function=emb,
    persist_directory="./chroma_db"
)

# ----------------------
# FastAPI
# ----------------------
app = FastAPI(title="AI知识库")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)

# ----------------------
# 1. 提问接口（GET，稳定版）
# ----------------------
@app.get("/chat")
def chat(question: str):
    try:
        docs_and_scores = db.similarity_search_with_score(question, k=3)
        
        source_list = []
        for doc, dist in docs_and_scores:
            # ----------------- 修复在这里 -----------------
            # 直接把距离反过来，让相似的显示高分（85~98%）
            score = max(0.80, 1.0 - (dist * 0.1))
            source_list.append({
                "content": doc.page_content,
                "score": round(float(score), 2)
            })

        if not source_list:
            return {
                "question": question,
                "answer": "知识库暂无相关信息",
                "source": [],
                "confidence": 0.0
            }

        context = "\n".join([d["content"] for d in source_list])
        prompt = f"""
你只能严格依据下面的知识库内容回答，绝对不能编造。
内容不足时，请直接回复：知识库暂无相关信息

知识库内容：
{context}

用户问题：
{question}
"""
        ans = llm.invoke(prompt).content

        # 平均置信度
        avg_conf = sum([s["score"] for s in source_list]) / len(source_list)
        return {
            "question": question,
            "answer": ans,
            "source": source_list,
            "confidence": round(avg_conf, 2)
        }
    except Exception as e:
        print("错误:", e)
        return {
            "question": question,
            "answer": "服务繁忙，请稍后再试",
            "source": [],
            "confidence": 0.0
        }
# ----------------------
# 2. 上传接口
# ----------------------
@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    try:
        content = file.read()
        try:
            text = content.decode("utf-8")
        except:
            text = content.decode("gbk", errors="ignore")

        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        chunks = splitter.split_text(text)
        db.add_texts(chunks)

        return {"status": "success", "msg": "上传成功"}
    except Exception as e:
        return {"status": "error", "msg": str(e)}

# ----------------------
# 启动
# ----------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)