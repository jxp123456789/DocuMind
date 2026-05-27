"""
Parser Agent - 文档解析 Agent
负责解析 PDF、Word 等格式文档，提取结构化文本
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class DocumentChunk:
    """文档片段数据结构"""
    content: str
    source: str
    page_num: Optional[int] = None
    chunk_index: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ParserAgent:
    """
    文档解析 Agent
    
    功能：
    1. 支持 PDF、Word (.docx)、TXT 格式
    2. 提取文本内容并保留页码/段落信息
    3. 智能分块，保持语义完整性
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.supported_formats = {'.pdf', '.docx', '.txt', '.md'}
    
    def parse(self, file_path: str) -> List[DocumentChunk]:
        """
        解析文档入口方法
        
        Args:
            file_path: 文档路径
            
        Returns:
            List[DocumentChunk]: 文档片段列表
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        if path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"不支持的文件格式: {path.suffix}")
        
        # 根据格式选择解析器
        if path.suffix.lower() == '.pdf':
            raw_text = self._parse_pdf(file_path)
        elif path.suffix.lower() == '.docx':
            raw_text = self._parse_docx(file_path)
        else:
            raw_text = self._parse_text(file_path)
        
        # 分块处理
        chunks = self._chunk_text(raw_text, file_path)
        
        return chunks
    
    def _parse_pdf(self, file_path: str) -> List[tuple]:
        """解析 PDF 文件，返回 [(页码, 文本), ...]"""
        try:
            from pypdf import PdfReader
        except ImportError:
            from PyPDF2 import PdfReader
            
        reader = PdfReader(file_path)
        pages = []
        
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text.strip():
                pages.append((i, text))
        
        return pages
    
    def _parse_docx(self, file_path: str) -> List[tuple]:
        """解析 Word 文件"""
        from docx import Document
        
        doc = Document(file_path)
        full_text = []
        
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text)
        
        # Word 没有页码概念，用 None 表示
        return [(None, '\n'.join(full_text))]
    
    def _parse_text(self, file_path: str) -> List[tuple]:
        """解析纯文本文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return [(None, content)]
    
    def _chunk_text(self, pages: List[tuple], source: str) -> List[DocumentChunk]:
        """
        将文本智能分块
        
        策略：
        1. 优先按段落分割
        2. 段落过长时按句子分割
        3. 保持上下文重叠
        """
        chunks = []
        chunk_index = 0
        
        for page_num, text in pages:
            # 先按段落分割
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
            
            current_chunk = []
            current_length = 0
            
            for para in paragraphs:
                para_length = len(para)
                
                # 如果单个段落就超过 chunk_size，需要进一步分割
                if para_length > self.chunk_size:
                    # 先把当前积累的段落保存
                    if current_chunk:
                        chunk_text = '\n'.join(current_chunk)
                        chunks.append(DocumentChunk(
                            content=chunk_text,
                            source=source,
                            page_num=page_num,
                            chunk_index=chunk_index
                        ))
                        chunk_index += 1
                        current_chunk = []
                        current_length = 0
                    
                    # 按句子分割长段落
                    sentences = re.split(r'([。！？.!?])', para)
                    sentences = [''.join(i) for i in zip(sentences[0::2], sentences[1::2] + [''])]
                    
                    temp_chunk = []
                    temp_length = 0
                    
                    for sent in sentences:
                        if temp_length + len(sent) > self.chunk_size and temp_chunk:
                            chunks.append(DocumentChunk(
                                content=''.join(temp_chunk),
                                source=source,
                                page_num=page_num,
                                chunk_index=chunk_index
                            ))
                            chunk_index += 1
                            # 保留重叠部分
                            temp_chunk = temp_chunk[-2:] if len(temp_chunk) > 2 else temp_chunk
                            temp_length = sum(len(s) for s in temp_chunk)
                        
                        temp_chunk.append(sent)
                        temp_length += len(sent)
                    
                    if temp_chunk:
                        current_chunk = temp_chunk
                        current_length = temp_length
                
                # 正常段落处理
                elif current_length + para_length > self.chunk_size:
                    # 保存当前 chunk
                    chunk_text = '\n'.join(current_chunk)
                    chunks.append(DocumentChunk(
                        content=chunk_text,
                        source=source,
                        page_num=page_num,
                        chunk_index=chunk_index
                    ))
                    chunk_index += 1
                    
                    # 保留重叠段落
                    overlap_count = min(len(current_chunk), 2)
                    current_chunk = current_chunk[-overlap_count:] + [para]
                    current_length = sum(len(p) for p in current_chunk)
                else:
                    current_chunk.append(para)
                    current_length += para_length
            
            # 保存最后一个 chunk
            if current_chunk:
                chunk_text = '\n'.join(current_chunk)
                chunks.append(DocumentChunk(
                    content=chunk_text,
                    source=source,
                    page_num=page_num,
                    chunk_index=chunk_index
                ))
                chunk_index += 1
        
        return chunks
    
    def get_document_info(self, file_path: str) -> Dict[str, Any]:
        """获取文档基本信息"""
        path = Path(file_path)
        
        info = {
            'filename': path.name,
            'format': path.suffix.lower(),
            'size_bytes': path.stat().st_size,
        }
        
        try:
            chunks = self.parse(file_path)
            info.update({
                'total_chunks': len(chunks),
                'total_chars': sum(len(c.content) for c in chunks),
                'parsed': True
            })
        except Exception as e:
            info.update({
                'parsed': False,
                'error': str(e)
            })
        
        return info


# 测试代码
if __name__ == '__main__':
    agent = ParserAgent(chunk_size=300, chunk_overlap=30)
    
    # 创建一个测试文本文件
    test_content = """这是第一段内容。用于测试文档解析功能。

这是第二段内容。包含更多的文字，用来测试分块逻辑是否能够正确处理段落边界。

这是第三段内容。智能文档处理流水线需要能够处理各种格式的文档，包括PDF、Word和纯文本。"""
    
    test_file = 'test_document.txt'
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    # 测试解析
    chunks = agent.parse(test_file)
    
    print(f"解析完成，共生成 {len(chunks)} 个片段：")
    for i, chunk in enumerate(chunks):
        print(f"\n--- 片段 {i+1} ---")
        print(f"内容: {chunk.content[:100]}...")
        print(f"长度: {len(chunk.content)} 字符")
    
    # 清理测试文件
    os.remove(test_file)
