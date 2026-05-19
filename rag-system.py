import os
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings.base import Embeddings

load_dotenv()

# 阿里云 Embedding
class AliEmb(Embeddings):
    def embed_query(self, txt):
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
            headers=headers, json=data
        )
        return resp.json()["output"]["embeddings"][0]["embedding"]

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]

# 大模型
llm = ChatOpenAI(
    model="qwen-turbo",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

# 向量模型
emb = AliEmb()

# 读取知识库
with open("test.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 切分文本
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
texts = splitter.split_text(text)

# ===================== Chroma 向量库（持久化存储） =====================
db = Chroma.from_texts(
    texts=texts,
    embedding=emb,
    persist_directory="./chroma_db"
)

# 问答链
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 2})
)

# 开始对话
if __name__ == "__main__":
    print("✅ 项目启动成功！Chroma 持久化知识库已就绪")
    while True:
        q = input("\n请输入问题：")
        if q in ["exit", "quit"]: break
        print("AI 回答：", qa.run(q))