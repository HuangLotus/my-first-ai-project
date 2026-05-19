# AI 智能知识库问答系统

基于 **RAG（检索增强生成）** 的本地知识库问答 Demo：上传文档 → 向量检索 → 大模型按知识库内容回答。

## 功能

- 上传文本文档，自动分块并写入向量库
- 自然语言提问，检索最相关的片段作为上下文
- 展示回答、置信度及参考片段来源
- 向量数据持久化到 `chroma_db/`，重启后仍可使用

## 技术栈

| 组件 | 说明 |
|------|------|
| 后端 | [FastAPI](https://fastapi.tiangolo.com/) + [LangChain](https://python.langchain.com/) |
| 大模型 | 通义千问 `qwen-turbo`（OpenAI 兼容接口） |
| 向量模型 | 阿里云 DashScope `text-embedding-v2` |
| 向量库 | [ChromaDB](https://www.trychroma.com/) |
| 前端 | 纯 HTML（`index.html`） |

## 项目结构

```
.
├── rag-system.py      # FastAPI 后端（检索 + 问答 + 上传）
├── index.html         # 前端聊天界面
├── chroma_db/         # Chroma 向量库（自动生成，勿提交敏感数据）
├── uploads/           # 上传文件存放目录
├── docs/              # 示例学习资料
├── .env               # API 配置（需自行创建，勿提交）
└── README.md
```

## 环境要求

- Python 3.10+
- 阿里云 DashScope API Key（用于 Embedding 与通义千问对话）

## 快速开始

### 1. 安装依赖

```bash
pip install fastapi uvicorn python-dotenv requests \
  langchain langchain-openai langchain-chroma chromadb python-multipart
```

### 2. 配置环境变量

在项目根目录创建 `.env` 文件：

```env
OPENAI_API_KEY=你的_DashScope_API_Key
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
```

> API Key 可在 [阿里云百炼 / DashScope 控制台](https://dashscope.console.aliyun.com/) 获取。

### 3. 启动后端

```bash
python rag-system.py
```

服务默认运行在 `http://127.0.0.1:8000`。

### 4. 打开前端

用浏览器直接打开 `index.html`，或通过本地静态服务访问：

```bash
# 可选：避免部分浏览器的 file:// 跨域限制
python -m http.server 5500
# 然后访问 http://127.0.0.1:5500/index.html
```

## API 说明

### `GET /chat`

根据问题检索知识库并生成回答。

| 参数 | 说明 |
|------|------|
| `question` | 用户问题（Query String） |

**响应示例：**

```json
{
  "question": "什么是 RAG？",
  "answer": "...",
  "confidence": 0.92,
  "source": [
    { "content": "检索到的片段...", "score": 0.95 }
  ]
}
```

### `POST /upload`

上传文档并写入向量库。

- Content-Type: `multipart/form-data`
- 字段名: `file`

**响应示例：**

```json
{ "status": "success", "msg": "上传成功" }
```

## 使用说明

1. 在页面上传 `.txt` 等**纯文本**文件（当前实现按 UTF-8 / GBK 解码，**暂不支持 PDF**）。
2. 在输入框提问，系统会检索 Top 3 相关片段并调用大模型回答。
3. 若知识库中无相关内容，模型会回复「知识库暂无相关信息」。

## 注意事项

- `.env` 含 API Key，请勿提交到公开仓库；`chroma_db/` 已在 `.ignore` 中忽略。
- 仓库内 `docs/`、`uploads/` 下的 PDF 需先转为文本才能通过当前上传接口入库。
- 后端需先启动，前端才能正常调用 `http://127.0.0.1:8000` 接口。

## 后续可改进

- [ ] 支持 PDF / Word 等格式解析
- [ ] 添加 `requirements.txt` 锁定依赖版本
- [ ] 流式输出（SSE）提升对话体验
- [ ] Docker 一键部署

## License

个人学习项目，按需使用。
