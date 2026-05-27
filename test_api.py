"""
测试 DeepSeek API 是否可用
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# API 配置
API_KEY = "sk-0ec59622a0684268a2fa57d95bff444f"
BASE_URL = "https://api.deepseek.com/v1"

print("=" * 50)
print("测试 DeepSeek API")
print("=" * 50)
print(f"API Key: {API_KEY[:10]}...{API_KEY[-4:]}")
print(f"Base URL: {BASE_URL}")
print()

try:
    # 初始化 LLM
    print("[1] 正在初始化 ChatOpenAI...")
    llm = ChatOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        model="deepseek-chat",
        temperature=0.3,
        max_tokens=500
    )
    print("[✓] ChatOpenAI 初始化成功")
    print()
    
    # 发送测试消息
    print("[2] 正在发送测试请求...")
    messages = [HumanMessage(content="你好，请用一句话介绍自己")]
    response = llm.invoke(messages)
    print("[✓] API 请求成功")
    print()
    
    # 显示结果
    print("=" * 50)
    print("API 响应:")
    print("=" * 50)
    print(response.content)
    print()
    print("[✓] API 测试通过！")
    
except Exception as e:
    print("[✗] 错误:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    print()
    print("[✗] API 测试失败")

print("=" * 50)
