"""检查当前 Python 环境的 langchain 状态"""
import sys
print("=" * 60)
print("当前 Python 环境")
print("=" * 60)
print(f"可执行文件: {sys.executable}")
print(f"版本: {sys.version}")
print()

print("=" * 60)
print("检查 langchain_core")
print("=" * 60)
try:
    import langchain_core
    print(f"✓ langchain_core 已安装")
    print(f"  位置: {langchain_core.__file__}")
    print(f"  版本: {langchain_core.__version__}")
except Exception as e:
    print(f"✗ 导入失败: {e}")

print()
print("=" * 60)
print("检查 LanguageModelInput")
print("=" * 60)
try:
    from langchain_core.language_models import LanguageModelInput
    print(f"✓ LanguageModelInput 导入成功")
except ImportError as e:
    print(f"✗ LanguageModelInput 导入失败: {e}")
    print()
    print("检查 base.py 文件...")
    import os
    base_path = os.path.join(os.path.dirname(langchain_core.__file__), "language_models", "base.py")
    if os.path.exists(base_path):
        print(f"  base.py 存在: {base_path}")
        with open(base_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'LanguageModelInput' in content:
                print("  ✓ base.py 中包含 LanguageModelInput")
            else:
                print("  ✗ base.py 中不包含 LanguageModelInput")
    else:
        print(f"  ✗ base.py 不存在")

print()
print("=" * 60)
print("检查 langchain_openai")
print("=" * 60)
try:
    from langchain_openai import ChatOpenAI
    print(f"✓ langchain_openai 导入成功")
except Exception as e:
    print(f"✗ 导入失败: {e}")

print()
print("=" * 60)
print("诊断完成")
print("=" * 60)
