from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.service import retrieve_knowledge, upload_knowledge_base


def main() -> None:

    print("\nRetrieving knowledge...")
    retrieve_result = retrieve_knowledge(
        query="What do Doraemon's height, weight, and horsepower have in common?",
        top_k=3,
    )
    pprint(retrieve_result)


if __name__ == "__main__":
    main()
