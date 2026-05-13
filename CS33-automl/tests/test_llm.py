import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_llm():
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL")

    print("Testing LLM Connection...")
    print(f"Model: {model}")
    print(f"Base URL: {base_url}")
    print(f"API Key configured: {'Yes' if api_key and api_key.startswith('sk-') else 'No'}")

    try:
        llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=0.1,
        )
        response = llm.invoke("Hello, please give a brief self-introduction.")
        print("\n=== Test succeeded ===")
        print("LLM reply:", response.content)
    except Exception as e:
        print("\n=== Test failed ===")
        print(f"Error: {e}")


if __name__ == "__main__":
    test_llm()
