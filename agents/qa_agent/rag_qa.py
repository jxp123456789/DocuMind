"""
QA Agent - 问答 Agent
负责基于检索结果生成回答，支持溯源引用
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))


@dataclass
class Answer:
    """回答数据结构"""
    content: str
    sources: List[Dict[str, Any]]
    confidence: float
    query: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class QAAgent:
    """
    问答 Agent
    
    功能：
    1. 基于检索结果生成回答
    2. 支持溯源引用（显示信息来源）
    3. 支持多轮对话上下文
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.3
    ):
        # 从环境变量读取配置
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or "sk-0ec59622a0684268a2fa57d95bff444f"
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1"
        
        if not self.api_key:
            raise ValueError("请设置 API Key（OPENAI_API_KEY 或 DEEPSEEK_API_KEY）")
        
        # 自动检测模型提供商
        if "deepseek" in (self.base_url or "").lower():
            self.provider = "deepseek"
            self.model = model if model != "gpt-3.5-turbo" else "deepseek-chat"
        else:
            self.provider = "openai"
            self.model = model
        
        # 初始化 LLM
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            temperature=temperature,
            max_tokens=2000
        )
        
        # 对话历史
        self.chat_history = []
        self.max_history = 5  # 保留最近5轮对话
    
    def answer(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        use_history: bool = True
    ) -> Answer:
        """
        基于检索结果生成回答
        
        Args:
            query: 用户问题
            retrieved_docs: 检索到的文档片段
            use_history: 是否使用对话历史
            
        Returns:
            Answer: 包含回答内容和溯源信息
        """
        if not retrieved_docs:
            return Answer(
                content="抱歉，未能在文档中找到相关信息。请尝试用不同的方式提问，或上传包含相关内容的文档。",
                sources=[],
                confidence=0.0,
                query=query
            )
        
        # 构建上下文
        context = self._build_context(retrieved_docs)
        
        # 构建提示词
        messages = self._build_messages(query, context, use_history)
        
        # 调用 LLM
        try:
            response = self.llm.invoke(messages)
            answer_content = response.content
            
            # 更新对话历史
            if use_history:
                self._update_history(query, answer_content)
            
            # 计算置信度（基于检索分数）
            confidence = self._calculate_confidence(retrieved_docs)
            
            # 格式化溯源信息
            sources = self._format_sources(retrieved_docs)
            
            return Answer(
                content=answer_content,
                sources=sources,
                confidence=confidence,
                query=query,
                metadata={
                    "model": self.model,
                    "provider": self.provider,
                    "retrieved_count": len(retrieved_docs)
                }
            )
            
        except Exception as e:
            return Answer(
                content=f"生成回答时出错: {str(e)}",
                sources=[],
                confidence=0.0,
                query=query,
                metadata={"error": str(e)}
            )
    
    def _build_context(self, docs: List[Dict[str, Any]]) -> str:
        """构建检索上下文"""
        context_parts = []
        
        for i, doc in enumerate(docs, 1):
            content = doc.get("content", "")
            source = doc.get("source", "未知来源")
            page = doc.get("page_num", -1)
            score = doc.get("similarity_score", 0)
            
            # 格式化来源信息
            source_info = f"[来源: {Path(source).name}"
            if page and page > 0:
                source_info += f" 第{page}页"
            source_info += f" | 相关度: {score:.2f}]"
            
            context_parts.append(f"【片段{i}】{source_info}\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _build_messages(
        self,
        query: str,
        context: str,
        use_history: bool
    ) -> List:
        """构建消息列表"""
        
        # 系统提示词
        system_prompt = """你是一个专业的文档问答助手。你的任务是基于提供的文档内容回答用户问题。

要求：
1. 只基于提供的文档内容回答，不要添加外部知识
2. 如果文档中没有相关信息，明确告知用户
3. 回答要准确、简洁、有条理
4. 适当引用文档中的关键信息
5. 如果涉及多个方面，请分点说明

文档内容如下：
{context}
"""
        
        messages = [
            SystemMessage(content=system_prompt.format(context=context))
        ]
        
        # 添加历史对话
        if use_history and self.chat_history:
            for human_msg, ai_msg in self.chat_history:
                messages.append(HumanMessage(content=human_msg))
                messages.append(ai_msg)
        
        # 添加当前问题
        messages.append(HumanMessage(content=query))
        
        return messages
    
    def _update_history(self, query: str, answer: str):
        """更新对话历史"""
        from langchain_core.messages import AIMessage
        
        self.chat_history.append((query, AIMessage(content=answer)))
        
        # 保持历史长度限制
        if len(self.chat_history) > self.max_history:
            self.chat_history = self.chat_history[-self.max_history:]
    
    def _calculate_confidence(self, docs: List[Dict[str, Any]]) -> float:
        """计算回答置信度"""
        if not docs:
            return 0.0
        
        # 基于检索分数计算（使用余弦相似度，范围通常是0-1）
        scores = [doc.get("similarity_score", 0) for doc in docs]
        avg_score = sum(scores) / len(scores)
        
        # 归一化到 0-1 范围（余弦相似度通常已经在这个范围）
        confidence = min(max(avg_score, 0), 1)
        
        return round(confidence, 2)
    
    def _format_sources(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """格式化溯源信息"""
        sources = []
        
        for doc in docs:
            source_info = {
                "content": doc.get("content", "")[:200] + "...",  # 截断显示
                "source": doc.get("source", "unknown"),
                "page_num": doc.get("page_num", -1),
                "similarity_score": doc.get("similarity_score", 0)
            }
            sources.append(source_info)
        
        return sources
    
    def clear_history(self):
        """清空对话历史"""
        self.chat_history.clear()
        print(f"[{self.__class__.__name__}] 对话历史已清空")
    
    def get_history(self) -> List[Dict[str, str]]:
        """获取对话历史"""
        return [
            {"role": "user", "content": h[0]}
            for h in self.chat_history
        ]


# 测试代码
if __name__ == '__main__':
    # 需要设置 API Key 才能运行测试
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("DEEPSEEK_API_KEY"):
        print("请设置 OPENAI_API_KEY 或 DEEPSEEK_API_KEY 环境变量后运行测试")
        exit(1)
    
    # 模拟检索结果
    test_docs = [
        {
            "content": "人工智能（AI）是计算机科学的一个分支，致力于创造能够模拟人类智能的系统。AI 可以执行通常需要人类智能的任务，如视觉识别、语言理解、决策等。",
            "source": "AI_Introduction.pdf",
            "page_num": 1,
            "similarity_score": 0.92
        },
        {
            "content": "机器学习是人工智能的核心技术之一。它通过数据训练模型，使计算机能够从经验中学习并改进性能，而无需明确编程。",
            "source": "AI_Introduction.pdf",
            "page_num": 2,
            "similarity_score": 0.85
        }
    ]
    
    # 初始化 Agent
    agent = QAAgent()
    
    # 测试问答
    print("\n--- 测试问答 ---")
    query = "什么是人工智能？"
    answer = agent.answer(query, test_docs)
    
    print(f"问题: {answer.query}")
    print(f"\n回答:\n{answer.content}")
    print(f"\n置信度: {answer.confidence}")
    print(f"\n溯源信息:")
    for i, src in enumerate(answer.sources, 1):
        print(f"  [{i}] {src['source']} 第{src['page_num']}页 (相关度: {src['similarity_score']:.2f})")
