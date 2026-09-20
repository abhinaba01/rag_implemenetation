import sys

import config
import retrieval

DEFAULT_QUESTION = (
    "There's a newer way to chain parsing, transformation and checking steps "
    "directly inside a type annotation. What is it called, what kinds of steps "
    "can it hold, and is it stable?"
)


def main(question=DEFAULT_QUESTION, k=config.TOP_K, candidate_k=config.CANDIDATE_K):
    candidate_ids = retrieval.dense_search(question, k=candidate_k)

    print(f"Q: {question}\n")
    for chunk, score in retrieval.rerank_with_scores(question, candidate_ids, k=k):
        print(f"{score:.2f}  {chunk['chunk_id']}  |  {chunk['heading_path']}")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION)
