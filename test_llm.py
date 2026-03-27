import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

def test_llm():
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL")
    
    print(f"Testing LLM Connection...")
    print(f"Model: {model}")
    print(f"Base URL: {base_url}")
    print(f"API Key configured: {'Yes' if api_key and api_key.startswith('sk-') else 'No'}")
    
    try:
        llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=0.1
        )
        response = llm.invoke("你好，请做一个简短的自我介绍。")
        print("\n=== 测试成功 ===")
        print("LLM 回复:", response.content)
    except Exception as e:
        print("\n=== 测试失败 ===")
        print(f"错误信息: {e}")

if __name__ == "__main__":
    test_llm()
