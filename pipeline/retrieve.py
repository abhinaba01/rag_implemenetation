import sys

import config
import data_utils
import retrieval

DEFAULT_QUESTION = "How do I validate a Pydantic model ?"


def main(question=DEFAULT_QUESTION, k=config.TOP_K):
    chunk_ids = retrieval.dense_search(question, k=k)

    print(f"Q: {question}\n")
    for chunk in data_utils.get_chunks_by_ids(chunk_ids):
        print(f"{chunk['chunk_id']} | {chunk['heading_path']}")
        print(chunk['text'][:200])
        print('---')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION)
