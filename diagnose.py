"""诊断脚本 - 检查 langchain 环境"""
import sys
print("=" * 60)
print("Python 环境诊断")
print("=" * 60)
print(f"Python 可执行文件: {sys.executable}")
print(f"Python 版本: {sys.version}")
print(f"Python 路径: {sys.path}")
print()

print("=" * 60)
print("检查 langchain_core")
print("=" * 60)
try:
    import langchain_core
    print(f"langchain_core 位置: {langchain_core.__file__}")
    print(f"langchain_core 版本: {langchain_core.__version__}")
except Exception as e:
    print(f"导入 langchain_core 失败: {e}")

print()
print("=" * 60)
print("检查 LanguageModelInput")
print("=" * 60)
try:
    from langchain_core.language_models import LanguageModelInput
    print(f"✓ LanguageModelInput 导入成功: {LanguageModelInput}")
except ImportError as e:
    print(f"✗ LanguageModelInput 导入失败: {e}")
    print("\n尝试直接检查 base.py...")
    try:
        from langchain_core.language_models.base import LanguageModelInput
        print(f"✓ 从 base.py 导入成功: {LanguageModelInput}")
    except ImportError as e2:
        print(f"✗ 从 base.py 也失败: {e2}")

print()
print("=" * 60)
print("检查 langchain_openai")
print("=" * 60)
try:
    from langchain_openai import ChatOpenAI
    print(f"✓ ChatOpenAI 导入成功")
except Exception as e:
    print(f"✗ ChatOpenAI 导入失败: {e}")

print()
print("=" * 60)
print("检查 site-packages 中的 langchain_core")
print("=" * 60)
import os
import glob

# 查找所有 langchain_core 安装
paths = glob.glob(os.path.join(sys.prefix, "lib", "site-packages", "langchain_core*"))
if paths:
    print(f"找到 {len(paths)} 个 langchain_core 安装:")
    for p in paths:
        print(f"  - {p}")
else:
    print("在 site-packages 中未找到 langchain_core")

print()
print("=" * 60)
print("诊断完成")
print("=" * 60)
