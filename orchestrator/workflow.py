"""
Orchestrator - Agent 协作调度器
协调 ParserAgent、IndexerAgent、QAAgent 完成文档问答流水线
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

from agents.parser_agent.document_parser import ParserAgent
from agents.indexer_agent.embedding_indexer import IndexerAgent
from agents.qa_agent.rag_qa import QAAgent, Answer


@dataclass
class PipelineResult:
    """流水线执行结果"""
    success: bool
    stage: str
    message: str
    data: Any = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class RAGWorkflow:
    """
    RAG 工作流调度器
    
    协调三个 Agent 完成：
    文档上传 → 解析 → 索引 → 问答检索
    """
    
    def __init__(
        self,
        db_path: str = "./data/vector_db",
        documents_path: str = "./data/documents",
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        llm_model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        初始化 RAG 工作流
        
        Args:
            db_path: 向量数据库路径
            documents_path: 文档存储路径
            embedding_model: Embedding 模型名称
            llm_model: LLM 模型名称
            api_key: API Key（可选，默认从环境变量读取）
            base_url: API 基础 URL（可选）
            chunk_size: 文档分块大小
            chunk_overlap: 分块重叠大小
        """
        self.documents_path = Path(documents_path)
        self.documents_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化三个 Agent
        print("[RAGWorkflow] 正在初始化 Agent...")
        
        self.parser = ParserAgent(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        self.indexer = IndexerAgent(
            db_path=db_path,
            model_name=embedding_model
        )
        
        self.qa = QAAgent(
            api_key=api_key,
            base_url=base_url,
            model=llm_model
        )
        
        print("[RAGWorkflow] 所有 Agent 初始化完成")
        
        # 回调函数（用于进度通知）
        self.progress_callback: Optional[Callable[[str, str], None]] = None
    
    def set_progress_callback(self, callback: Callable[[str, str], None]):
        """设置进度回调函数"""
        self.progress_callback = callback
    
    def _notify(self, stage: str, message: str):
        """通知进度"""
        print(f"[{stage}] {message}")
        if self.progress_callback:
            self.progress_callback(stage, message)
    
    def upload_document(self, file_path: str, move_to_storage: bool = True) -> PipelineResult:
        """
        上传并处理文档
        
        Args:
            file_path: 文档路径
            move_to_storage: 是否移动到文档存储目录
            
        Returns:
            PipelineResult: 处理结果
        """
        try:
            source_path = Path(file_path)
            
            if not source_path.exists():
                return PipelineResult(
                    success=False,
                    stage="upload",
                    message=f"文件不存在: {file_path}"
                )
            
            # 移动到存储目录
            if move_to_storage:
                dest_path = self.documents_path / source_path.name
                # 如果已存在，添加数字后缀
                counter = 1
                while dest_path.exists():
                    stem = source_path.stem
                    suffix = source_path.suffix
                    dest_path = self.documents_path / f"{stem}_{counter}{suffix}"
                    counter += 1
                
                import shutil
                shutil.copy2(file_path, dest_path)
                file_path = str(dest_path)
                self._notify("upload", f"文档已保存到: {dest_path.name}")
            
            # Step 1: 解析文档
            self._notify("parse", "开始解析文档...")
            chunks = self.parser.parse(file_path)
            self._notify("parse", f"文档解析完成，共 {len(chunks)} 个片段")
            
            # Step 2: 索引文档
            self._notify("index", "开始构建向量索引...")
            index_result = self.indexer.index(chunks)
            self._notify("index", f"索引完成: {index_result['message']}")
            
            return PipelineResult(
                success=True,
                stage="upload",
                message=f"文档处理成功: {Path(file_path).name}",
                data={
                    "filename": Path(file_path).name,
                    "chunks": len(chunks),
                    "indexed": index_result["indexed"],
                    "path": file_path
                }
            )
            
        except Exception as e:
            return PipelineResult(
                success=False,
                stage="upload",
                message=f"文档处理失败: {str(e)}"
            )
    
    def ask(self, question: str, top_k: int = 5) -> Answer:
        """
        问答入口
        
        Args:
            question: 用户问题
            top_k: 检索结果数量
            
        Returns:
            Answer: 回答对象
        """
        self._notify("qa", f"收到问题: {question}")
        
        # Step 1: 检索相关文档
        self._notify("retrieve", "正在检索相关文档...")
        retrieved_docs = self.indexer.search(question, top_k=top_k)
        
        if not retrieved_docs:
            self._notify("retrieve", "未找到相关文档")
            return Answer(
                content="抱歉，知识库中没有找到相关信息。请先上传包含相关内容的文档。",
                sources=[],
                confidence=0.0,
                query=question
            )
        
        self._notify("retrieve", f"找到 {len(retrieved_docs)} 个相关片段")
        
        # Step 2: 生成回答
        self._notify("generate", "正在生成回答...")
        answer = self.qa.answer(question, retrieved_docs)
        self._notify("generate", "回答生成完成")
        
        return answer
    
    def batch_upload(self, file_paths: List[str]) -> List[PipelineResult]:
        """
        批量上传文档
        
        Args:
            file_paths: 文档路径列表
            
        Returns:
            List[PipelineResult]: 每个文档的处理结果
        """
        results = []
        for i, path in enumerate(file_paths, 1):
            self._notify("batch", f"处理第 {i}/{len(file_paths)} 个文档...")
            result = self.upload_document(path)
            results.append(result)
        
        success_count = sum(1 for r in results if r.success)
        self._notify("batch", f"批量处理完成: {success_count}/{len(file_paths)} 成功")
        
        return results
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """列出已上传的文档"""
        docs = []
        for file_path in self.documents_path.iterdir():
            if file_path.is_file():
                stat = file_path.stat()
                docs.append({
                    "name": file_path.name,
                    "path": str(file_path),
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        return sorted(docs, key=lambda x: x["modified"], reverse=True)
    
    def delete_document(self, filename: str) -> PipelineResult:
        """
        删除文档及其索引
        
        Args:
            filename: 文档文件名
            
        Returns:
            PipelineResult: 删除结果
        """
        try:
            file_path = self.documents_path / filename
            
            if not file_path.exists():
                return PipelineResult(
                    success=False,
                    stage="delete",
                    message=f"文档不存在: {filename}"
                )
            
            # 删除文件
            file_path.unlink()
            
            # 删除索引
            full_path = str(file_path.absolute())
            self.indexer.delete_by_source(full_path)
            
            return PipelineResult(
                success=True,
                stage="delete",
                message=f"已删除文档: {filename}"
            )
            
        except Exception as e:
            return PipelineResult(
                success=False,
                stage="delete",
                message=f"删除失败: {str(e)}"
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        index_stats = self.indexer.get_stats()
        documents = self.list_documents()
        
        return {
            "index": index_stats,
            "documents": {
                "count": len(documents),
                "list": [d["name"] for d in documents]
            },
            "config": {
                "documents_path": str(self.documents_path),
                "embedding_model": self.indexer.model_name,
                "llm_model": self.qa.model
            }
        }
    
    def clear_all(self):
        """清空所有数据和索引（谨慎使用）"""
        self._notify("clear", "正在清空所有数据...")
        
        # 清空索引
        self.indexer.clear_all()
        
        # 清空文档
        for file_path in self.documents_path.iterdir():
            if file_path.is_file():
                file_path.unlink()
        
        # 清空对话历史
        self.qa.clear_history()
        
        self._notify("clear", "所有数据已清空")


# 测试代码
if __name__ == '__main__':
    # 创建工作流实例
    workflow = RAGWorkflow()
    
    # 设置进度回调
    def on_progress(stage, message):
        print(f"  >> {stage}: {message}")
    
    workflow.set_progress_callback(on_progress)
    
    # 创建测试文档
    test_doc = "./test_upload.txt"
    with open(test_doc, 'w', encoding='utf-8') as f:
        f.write("""人工智能是计算机科学的一个分支。

机器学习是人工智能的核心技术。

深度学习是机器学习的子领域。""")
    
    print("\n=== 测试文档上传 ===")
    result = workflow.upload_document(test_doc, move_to_storage=False)
    print(f"结果: {result.message}")
    
    if result.success:
        print("\n=== 测试问答 ===")
        answer = workflow.ask("什么是机器学习？")
        print(f"\n回答: {answer.content}")
        print(f"置信度: {answer.confidence}")
        
        print("\n=== 系统统计 ===")
        stats = workflow.get_stats()
        print(f"索引文档数: {stats['index']['total_documents']}")
        print(f"存储文档数: {stats['documents']['count']}")
    
    # 清理
    if os.path.exists(test_doc):
        os.remove(test_doc)
    workflow.clear_all()
