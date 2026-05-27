"""
智能文档问答系统 - Web 界面
基于 Gradio 构建
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import gradio as gr
from dotenv import load_dotenv

from orchestrator.workflow import RAGWorkflow

# 加载环境变量
load_dotenv(project_root / ".env")


class RAGWebApp:
    """RAG Web 应用"""
    
    def __init__(self):
        self.workflow = None
        self.init_message = "⏳ 系统正在初始化，请稍候..."
    
    def _ensure_workflow(self):
        """懒加载工作流，避免阻塞 Web 服务启动"""
        if self.workflow is not None:
            return self.init_message
        
        try:
            print("[RAGWebApp] 正在初始化工作流...")
            api_key = (
                os.getenv("DEEPSEEK_API_KEY")
                or os.getenv("OPENAI_API_KEY")
            )
            base_url = (
                os.getenv("DEEPSEEK_BASE_URL")
                or os.getenv("OPENAI_BASE_URL")
                or "https://api.deepseek.com/v1"
            )
            llm_model = os.getenv("LLM_MODEL", "deepseek-chat")
            
            if not api_key:
                self.init_message = "❌ 未配置 API Key，请在 .env 中设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY"
                return self.init_message
            
            print(f"[RAGWebApp] API Key: 已配置")
            print(f"[RAGWebApp] Base URL: {base_url}")
            
            self.workflow = RAGWorkflow(
                api_key=api_key,
                base_url=base_url,
                llm_model=llm_model
            )
            print("[RAGWebApp] 工作流初始化成功")
            self.init_message = "✅ 系统初始化成功，请上传文档开始使用"
        except Exception as e:
            print(f"[RAGWebApp] 初始化失败: {str(e)}")
            import traceback
            traceback.print_exc()
            self.init_message = f"❌ 初始化失败: {str(e)}"
        
        return self.init_message
    
    def upload_file(self, file):
        """上传文件"""
        if file is None:
            return "请先选择文件", self.update_doc_list()
        
        status = self._ensure_workflow()
        if self.workflow is None:
            return status, self.update_doc_list()
        
        try:
            result = self.workflow.upload_document(file.name, move_to_storage=True)
            if result.success:
                return f"✅ {result.message}\n\n📊 处理详情:\n- 生成 {result.data['chunks']} 个文本片段\n- 成功索引 {result.data['indexed']} 个片段", self.update_doc_list()
            else:
                return f"❌ {result.message}", self.update_doc_list()
        except Exception as e:
            return f"❌ 上传失败: {str(e)}", self.update_doc_list()
    
    def ask_question(self, question, history):
        """回答问题"""
        if not question.strip():
            return history, ""
        
        status = self._ensure_workflow()
        if self.workflow is None:
            history.append([question, status])
            return history, ""
        
        try:
            answer = self.workflow.ask(question)
            
            # 构建回答文本
            response = answer.content
            
            # 添加溯源信息
            if answer.sources:
                response += "\n\n📚 **参考来源**:\n"
                for i, src in enumerate(answer.sources[:3], 1):
                    filename = Path(src['source']).name
                    page = f"第{src['page_num']}页" if src['page_num'] > 0 else ""
                    score = src['similarity_score']
                    response += f"{i}. {filename} {page} (相关度: {score:.2f})\n"
            
            # 更新历史记录
            history.append([question, response])
            return history, ""
            
        except Exception as e:
            error_msg = f"抱歉，处理问题时出错: {str(e)}"
            history.append([question, error_msg])
            return history, ""
    
    def update_doc_list(self):
        """更新文档列表"""
        if self.workflow is None:
            return self.init_message
        
        try:
            docs = self.workflow.list_documents()
            if not docs:
                return "暂无文档"
            
            doc_info = []
            for doc in docs:
                size_kb = doc['size'] / 1024
                doc_info.append(f"📄 {doc['name']} ({size_kb:.1f} KB)")
            
            return "\n".join(doc_info)
        except Exception as e:
            return f"获取文档列表失败: {str(e)}"
    
    def get_stats(self):
        """获取系统统计"""
        if self.workflow is None:
            return self.init_message
        
        try:
            stats = self.workflow.get_stats()
            
            info = f"""📊 **系统状态**

🗄️ **向量数据库**:
- 索引文档数: {stats['index'].get('total_documents', 0)}
- Embedding模型: {stats['config']['embedding_model']}

📁 **文档存储**:
- 存储文档数: {stats['documents']['count']}
- 存储路径: {stats['config']['documents_path']}

🤖 **语言模型**:
- 当前模型: {stats['config']['llm_model']}
"""
            return info
        except Exception as e:
            return f"获取统计信息失败: {str(e)}"
    
    def clear_all(self):
        """清空所有数据"""
        status = self._ensure_workflow()
        if self.workflow is None:
            return status, self.update_doc_list()
        
        try:
            self.workflow.clear_all()
            return "✅ 所有数据已清空", self.update_doc_list()
        except Exception as e:
            return f"❌ 清空失败: {str(e)}", self.update_doc_list()
    
    def create_ui(self):
        """创建 Gradio 界面"""
        
        with gr.Blocks(title="智能文档问答系统", theme=gr.themes.Soft()) as app:
            gr.Markdown("""
            # 🤖 智能文档问答系统
            
            基于多 Agent 协作的 RAG 系统，支持 PDF、Word、TXT 格式文档的智能问答。
            
            **工作流程**: 文档上传 → 智能解析 → 向量索引 → 语义检索 → 生成回答
            """)
            
            with gr.Row():
                # 左侧：文档管理
                with gr.Column(scale=1):
                    gr.Markdown("## 📁 文档管理")
                    
                    file_input = gr.File(
                        label="上传文档",
                        file_types=[".pdf", ".docx", ".txt", ".md"]
                    )
                    upload_btn = gr.Button("📤 上传并处理", variant="primary")
                    upload_output = gr.Textbox(label="上传状态", lines=3)
                    
                    gr.Markdown("### 已上传文档")
                    doc_list = gr.Textbox(
                        label="",
                        value=self.update_doc_list(),
                        lines=8,
                        interactive=False
                    )
                    
                    refresh_btn = gr.Button("🔄 刷新列表")
                    
                    gr.Markdown("### 系统状态")
                    stats_output = gr.Textbox(
                        label="",
                        value=self.get_stats(),
                        lines=10,
                        interactive=False
                    )
                    
                    refresh_stats_btn = gr.Button("🔄 刷新状态")
                    
                    clear_btn = gr.Button("🗑️ 清空所有数据", variant="stop")
                    clear_output = gr.Textbox(label="清空状态")
                
                # 右侧：问答界面
                with gr.Column(scale=2):
                    gr.Markdown("## 💬 智能问答")
                    
                    chatbot = gr.Chatbot(
                        label="对话历史",
                        height=500,
                    )
                    
                    with gr.Row():
                        msg_input = gr.Textbox(
                            label="输入问题",
                            placeholder="请输入您的问题...",
                            scale=8
                        )
                        send_btn = gr.Button("发送", variant="primary", scale=1)
                    
                    gr.Examples(
                        examples=[
                            "这篇文档的主要内容是什么？",
                            "请总结一下关键要点",
                            "文档中提到了哪些重要数据？",
                        ],
                        inputs=msg_input,
                        label="示例问题"
                    )
            
            # 事件绑定
            upload_btn.click(
                fn=self.upload_file,
                inputs=file_input,
                outputs=[upload_output, doc_list]
            )
            
            refresh_btn.click(
                fn=self.update_doc_list,
                outputs=doc_list
            )
            
            refresh_stats_btn.click(
                fn=self.get_stats,
                outputs=stats_output
            )
            
            clear_btn.click(
                fn=self.clear_all,
                outputs=[clear_output, doc_list]
            )
            
            send_btn.click(
                fn=self.ask_question,
                inputs=[msg_input, chatbot],
                outputs=[chatbot, msg_input]
            )
            
            msg_input.submit(
                fn=self.ask_question,
                inputs=[msg_input, chatbot],
                outputs=[chatbot, msg_input]
            )
            
            gr.Markdown("""
            ---
            💡 **使用提示**:
            1. 上传 PDF、Word 或文本文档
            2. 系统会自动解析并建立索引
            3. 在右侧输入问题，获取基于文档内容的回答
            4. 回答会标注参考来源，方便溯源
            """)
        
        return app


def main():
    """主函数"""
    app = RAGWebApp()
    ui = app.create_ui()
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    ui.launch(share=False, server_name="127.0.0.1", server_port=port)


if __name__ == "__main__":
    main()
