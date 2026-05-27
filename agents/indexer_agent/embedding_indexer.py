"""
Indexer Agent - 向量索引 Agent
负责生成 Embedding 并构建向量数据库索引
"""

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from agents.parser_agent.document_parser import DocumentChunk


class IndexerAgent:
    """
    向量索引 Agent
    
    功能：
    1. 使用本地 Embedding 模型生成向量（默认 bge-small-zh-v1.5）
    2. 使用 Chroma 存储向量索引
    3. 支持增量更新和重复检测
    """
    
    def __init__(
        self,
        db_path: str = "./data/vector_db",
        model_name: str = "BAAI/bge-small-zh-v1.5",
        device: str = "auto"
    ):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name
        self.device = self._get_device(device)
        
        # 仅从本地缓存加载，不联网下载（模型需已缓存到 ~/.cache/huggingface/hub）
        local_only = os.getenv("HF_HUB_OFFLINE", "1") == "1"
        print(f"[{self.__class__.__name__}] 正在加载本地 Embedding 模型: {model_name} (offline={local_only})")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={
                "device": self.device,
                "local_files_only": local_only,
            },
            encode_kwargs={"normalize_embeddings": True},
        )
        print(f"[{self.__class__.__name__}] Embedding 模型加载完成")
        
        # 初始化或加载向量数据库
        self.vector_store = self._init_vector_store()
        
        # 用于追踪已索引的文档
        self.indexed_docs = set()
    
    def _get_device(self, device: str) -> str:
        """自动选择设备"""
        if device != "auto":
            return device
        
        try:
            import torch
            if torch.cuda.is_available():
                print(f"[{self.__class__.__name__}] 检测到 CUDA，使用 GPU 加速")
                return "cuda"
            elif torch.backends.mps.is_available():
                print(f"[{self.__class__.__name__}] 检测到 MPS，使用 Apple Silicon 加速")
                return "mps"
        except ImportError:
            pass
        
        print(f"[{self.__class__.__name__}] 使用 CPU 运行")
        return "cpu"
    
    def _init_vector_store(self) -> Chroma:
        """初始化 Chroma 向量数据库"""
        return Chroma(
            persist_directory=str(self.db_path),
            embedding_function=self.embeddings
        )
    
    def _compute_content_hash(self, content: str) -> str:
        """计算内容哈希，用于去重"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def index(self, chunks: List[DocumentChunk], collection_name: str = "default") -> Dict[str, Any]:
        """
        将文档片段索引到向量数据库
        
        Args:
            chunks: 文档片段列表
            collection_name: 集合名称（用于区分不同文档集）
            
        Returns:
            索引统计信息
        """
        if not chunks:
            return {"indexed": 0, "skipped": 0, "message": "没有可索引的内容"}
        
        # 过滤已索引的内容
        new_chunks = []
        skipped = 0
        
        for chunk in chunks:
            content_hash = self._compute_content_hash(chunk.content)
            doc_id = f"{chunk.source}_{content_hash}"
            
            if doc_id not in self.indexed_docs:
                new_chunks.append(chunk)
                self.indexed_docs.add(doc_id)
            else:
                skipped += 1
        
        if not new_chunks:
            return {"indexed": 0, "skipped": skipped, "message": "所有内容已存在"}
        
        # 准备数据
        texts = [chunk.content for chunk in new_chunks]
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(new_chunks):
            metadata = {
                "source": chunk.source,
                "page_num": chunk.page_num if chunk.page_num else -1,
                "chunk_index": chunk.chunk_index,
                "content_hash": self._compute_content_hash(chunk.content)
            }
            metadata.update(chunk.metadata or {})
            metadatas.append(metadata)
            
            # 生成唯一 ID
            doc_id = f"{Path(chunk.source).stem}_{chunk.chunk_index}_{metadata['content_hash'][:8]}"
            ids.append(doc_id)
        
        # 添加到向量数据库
        self.vector_store.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        # 持久化
        self.vector_store.persist()
        
        return {
            "indexed": len(new_chunks),
            "skipped": skipped,
            "total_in_db": len(self.indexed_docs),
            "message": f"成功索引 {len(new_chunks)} 个片段"
        }
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        语义检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            filter_dict: 过滤条件
            
        Returns:
            检索结果列表，包含内容和相似度分数
        """
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=top_k,
            filter=filter_dict
        )
        
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": float(score),
                "source": doc.metadata.get("source", "unknown"),
                "page_num": doc.metadata.get("page_num", -1)
            })
        
        return formatted_results
    
    def delete_by_source(self, source: str) -> int:
        """
        删除指定来源的所有文档
        
        Args:
            source: 文档路径
            
        Returns:
            删除的文档数量
        """
        # Chroma 的删除功能
        try:
            self.vector_store._collection.delete(
                where={"source": source}
            )
            self.vector_store.persist()
            
            # 更新索引记录
            self.indexed_docs = {
                doc_id for doc_id in self.indexed_docs 
                if not doc_id.startswith(source)
            }
            
            return 1
        except Exception as e:
            print(f"删除失败: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        try:
            count = self.vector_store._collection.count()
            return {
                "total_documents": count,
                "indexed_sources": len(set(
                    doc_id.split('_')[0] 
                    for doc_id in self.indexed_docs
                )),
                "db_path": str(self.db_path),
                "embedding_model": self.model_name
            }
        except Exception as e:
            return {
                "error": str(e),
                "db_path": str(self.db_path)
            }
    
    def clear_all(self):
        """清空所有索引（谨慎使用）"""
        self.vector_store.delete_collection()
        self.vector_store = self._init_vector_store()
        self.indexed_docs.clear()
        print(f"[{self.__class__.__name__}] 已清空所有索引")


# 测试代码
if __name__ == '__main__':
    # 创建测试数据
    test_chunks = [
        DocumentChunk(
            content="人工智能是计算机科学的一个分支，致力于创造能够模拟人类智能的系统。",
            source="test_doc.txt",
            page_num=1,
            chunk_index=0
        ),
        DocumentChunk(
            content="机器学习是人工智能的核心技术之一，通过数据训练模型来实现预测和决策。",
            source="test_doc.txt",
            page_num=1,
            chunk_index=1
        ),
        DocumentChunk(
            content="深度学习是机器学习的一个子领域，使用多层神经网络来处理复杂的数据模式。",
            source="test_doc.txt",
            page_num=2,
            chunk_index=2
        ),
    ]
    
    # 初始化 Agent
    agent = IndexerAgent(db_path="./test_vector_db")
    
    # 测试索引
    print("\n--- 测试索引 ---")
    result = agent.index(test_chunks)
    print(f"索引结果: {result}")
    
    # 测试检索
    print("\n--- 测试检索 ---")
    query = "什么是机器学习？"
    results = agent.search(query, top_k=2)
    
    print(f"查询: {query}")
    for i, r in enumerate(results, 1):
        print(f"\n结果 {i}:")
        print(f"  内容: {r['content'][:50]}...")
        print(f"  相似度: {r['similarity_score']:.4f}")
        print(f"  来源: {r['source']} 第{r['page_num']}页")
    
    # 查看统计
    print("\n--- 索引统计 ---")
    print(agent.get_stats())
    
    # 清理测试数据
    import shutil
    if os.path.exists("./test_vector_db"):
        shutil.rmtree("./test_vector_db")
        print("\n已清理测试数据")
