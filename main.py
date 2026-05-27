"""
智能文档问答系统 - 主入口
"""

import os
import sys
from pathlib import Path

# 默认离线加载 Embedding，避免启动时访问 HuggingFace
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

# 修复 Gradio 4.29 + gradio_client 0.16 的 schema 解析 bug（如 Chatbot 的 bool 参数）
# 未修复时 /info 返回 500，前端会显示 "No API found"
try:
    import gradio_client.utils as _gradio_utils

    _original_get_type = _gradio_utils.get_type

    def _patched_get_type(schema):
        if isinstance(schema, bool):
            return "boolean"
        if not isinstance(schema, dict):
            return "Any"
        return _original_get_type(schema)

    _original_json_schema_to_python_type = _gradio_utils._json_schema_to_python_type

    def _patched_json_schema_to_python_type(schema, defs):
        if isinstance(schema, bool):
            return "bool"
        if not isinstance(schema, dict):
            return "Any"
        return _original_json_schema_to_python_type(schema, defs)

    _gradio_utils.get_type = _patched_get_type
    _gradio_utils._json_schema_to_python_type = _patched_json_schema_to_python_type
    print("[Patch] gradio_client schema 解析已修复")
except Exception:
    pass

from web_ui.app import main

if __name__ == "__main__":
    main()
