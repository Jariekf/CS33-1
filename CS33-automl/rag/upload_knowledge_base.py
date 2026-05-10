from __future__ import annotations

import logging
import sys
from pathlib import Path
from pprint import pprint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.service import upload_knowledge_base

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

# Manually edit this list when you want to upload different knowledge files.
KNOWLEDGE_FILES = "tests/doramon.txt"

# Set to True when you want to rebuild the collection from scratch.
DROP_OLD_COLLECTION = False


def main() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    logger = logging.getLogger(__name__)

    logger.info("Uploading knowledge files: %s", KNOWLEDGE_FILES)
    logger.info("Drop old collection: %s", DROP_OLD_COLLECTION)

    result = upload_knowledge_base(
        file_paths=KNOWLEDGE_FILES,
        drop_old=DROP_OLD_COLLECTION,
    )

    print("\n[Knowledge Base Upload Result]")
    pprint(result)


if __name__ == "__main__":
    main()
