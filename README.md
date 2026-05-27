<p align="center">
  <h1 align="center">DocuMind</h1>
  <p align="center">
    <strong>Multi-Agent RAG Document Q&A System</strong><br>
    基于多 Agent 协作的智能文档问答系统
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.9-blue" alt="Python">
    <img src="https://img.shields.io/badge/LangChain-0.2-green" alt="LangChain">
    <img src="https://img.shields.io/badge/Gradio-4.29-orange" alt="Gradio">
    <img src="https://img.shields.io/badge/ChromaDB-0.5-purple" alt="ChromaDB">
    <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
  </p>
</p>

---

## 项目简介

DocuMind 是一个基于**多 Agent 协作**的 RAG（Retrieval-Augmented Generation）文档问答系统。系统通过三个专职 Agent 协同工作，实现从文档上传、智能解析、向量索引到语义检索、生成回答的全自动化流水线。

### 核心特性

- **Multi-Agent 架构** — Parser / Indexer / QA 三 Agent 各司其职，通过编排器协调完成端到端文档处理
- **本地 Embedding** — 使用 BGE-Small-ZH 模型本地生成向量，零 API 成本完成文档索引
- **云端 LLM** — 接入 DeepSeek / OpenAI API 生成高质量回答，按需付费
- **智能分块** — 多级分块策略（段落优先 → 句子分割 → 重叠保留），保持语义完整性
- **溯源引用** — 回答自动标注来源文档、页码、相似度分数，可追溯可验证
- **多格式支持** — PDF、Word、TXT、Markdown 一键处理
- **多轮对话** — 保留最近 5 轮上下文，支持连续追问
- **Web 界面** — 基于 Gradio 的交互式界面，开箱即用

## 系统架构

```
用户操作
   │
   ▼
┌──────────────────────────────────────────────────────────────┐
│                    RAGWorkflow (编排器)                       │
│                                                              │
│  文档上传流水线:                                              │
│    ParserAgent.parse() ──► IndexerAgent.index()              │
│                                                              │
│  问答流水线:                                                  │
│    IndexerAgent.search() ──► QAAgent.answer()                │
└──────────────────────────────────────────────────────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
  │ ParserAgent │    │IndexerAgent │    │   QAAgent   │
  │  文档解析    │    │  向量索引    │    │  问答生成    │
  ├─────────────┤    ├─────────────┤    ├─────────────┤
  │ PDF 解析    │    │ BGE Embed   │    │ RAG 检索    │
  │ Word 解析   │    │ Chroma DB   │    │ LLM 生成    │
  │ 智能分块    │    │ 语义检索    │    │ 溯源引用    │
  │ 去重索引    │    │ MD5 去重    │    │ 多轮对话    │
  └─────────────┘    └─────────────┘    └─────────────┘
```

## 技术栈

| 层次 | 技术 | 说明 |
|------|------|------|
| Agent 框架 | LangChain 0.2 | 多 Agent 编排与 LLM 调用 |
| Embedding | BGE-Small-ZH (BAAI) | 本地中文向量模型，通过 sentence-transformers 加载 |
| 向量数据库 | ChromaDB | 轻量级本地持久化，HNSW 索引 |
| LLM | DeepSeek / OpenAI API | 通过 langchain-openai 的 ChatOpenAI 调用 |
| Web UI | Gradio 4.29 | 交互式 Web 界面 |
| 文档解析 | pypdf / python-docx | PDF 和 Word 格式支持 |

## 快速开始

### 环境要求

- Python 3.9+
- CUDA（可选，用于 GPU 加速 Embedding）

### 安装

```bash
# 克隆项目
git clone https://github.com/your-username/documind.git
cd documind

# 创建并激活虚拟环境
conda create -n documind python=3.9 -y
conda activate documind

# 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 配置 API Key

复制环境变量模板并填写：

```bash
cp .env.example .env
```

编辑 `.env`：

```env
# DeepSeek API（推荐，国内可用）
DEEPSEEK_API_KEY=sk-your-key-here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

# 或使用 OpenAI API
# OPENAI_API_KEY=sk-your-key-here
# OPENAI_BASE_URL=https://api.openai.com/v1
```

### 启动

```bash
# 设置 HuggingFace 镜像（国内用户）
export HF_ENDPOINT=https://hf-mirror.com   # Linux/macOS
# $env:HF_ENDPOINT="https://hf-mirror.com"  # Windows PowerShell

# 启动服务
python main.py
```

浏览器访问 `http://127.0.0.1:7860` 即可使用。

> **首次启动**会自动下载 BGE-Small-ZH 模型（约 100MB），后续启动使用本地缓存。

## 使用指南

### Web 界面

1. **上传文档** — 左侧拖拽或点击上传 PDF / Word / TXT 文件
2. **自动处理** — 系统自动解析文档、智能分块、构建向量索引
3. **开始提问** — 右侧输入框输入问题，点击发送
4. **查看来源** — 回答底部自动标注参考文档、页码和相关度

### 代码调用

```python
from orchestrator.workflow import RAGWorkflow

# 初始化
workflow = RAGWorkflow(
    api_key="your-api-key",
    base_url="https://api.deepseek.com/v1",
    llm_model="deepseek-chat"
)

# 上传文档
result = workflow.upload_document("report.pdf")
print(result.message)

# 提问
answer = workflow.ask("这篇文档的核心结论是什么？")
print(answer.content)

# 查看溯源
for src in answer.sources:
    print(f"[{src['source']}] 第{src['page_num']}页 (相关度: {src['similarity_score']:.2f})")
```

## 项目结构

```
documind/
├── main.py                     # 入口，Gradio 兼容性补丁 + 启动 Web UI
├── requirements.txt            # Python 依赖
├── .env.example                # 环境变量模板
│
├── agents/                     # Agent 模块
│   ├── parser_agent/
│   │   └── document_parser.py  # ParserAgent — 多格式解析 + 智能分块
│   ├── indexer_agent/
│   │   └── embedding_indexer.py # IndexerAgent — BGE Embedding + Chroma
│   └── qa_agent/
│       └── rag_qa.py           # QAAgent — RAG 检索 + LLM 生成
│
├── orchestrator/
│   └── workflow.py             # RAGWorkflow — Agent 编排调度器
│
├── web_ui/
│   └── app.py                  # Gradio Web 界面
│
└── data/
    ├── documents/              # 上传文档存储
    └── vector_db/              # Chroma 向量数据库持久化
```

## 核心设计

### 智能分块策略

ParserAgent 采用多级分块：

1. **段落优先** — 按 `\n` 分割，保持段落语义完整
2. **句子兜底** — 超长段落按 `。！？.!?` 分割，避免截断
3. **重叠保留** — 块间保留最后 2 个段落/句子，确保上下文连贯

### 混合部署

| 组件 | 部署方式 | 原因 |
|------|---------|------|
| Embedding 模型 | 本地 (CPU/GPU) | 文档索引频繁调用，本地部署零成本 |
| 向量数据库 | 本地 (ChromaDB) | 数据隐私，无需外部服务 |
| LLM | 云端 API | 生成质量高，按需付费 |

### 溯源机制

全链路保留来源信息：解析时记录文件路径和页码 → 索引时存储元数据 → 检索时返回相似度 → 生成时标注引用来源。

## 配置选项

```python
workflow = RAGWorkflow(
    chunk_size=500,          # 分块大小（字符数）
    chunk_overlap=50,        # 分块重叠（字符数）
    embedding_model="BAAI/bge-small-zh-v1.5",  # Embedding 模型
    llm_model="deepseek-chat",                  # LLM 模型
    db_path="./data/vector_db",                 # 向量库路径
    documents_path="./data/documents",          # 文档存储路径
)
```


## License

[MIT](LICENSE)

## 致谢

- [LangChain](https://github.com/langchain-ai/langchain) — Agent 框架
- [BAAI/bge](https://github.com/FlagOpen/FlagEmbedding) — Embedding 模型
- [Chroma](https://github.com/chroma-core/chroma) — 向量数据库
- [Gradio](https://github.com/gradio-app/gradio) — Web UI 框架
