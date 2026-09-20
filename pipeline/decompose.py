import sys

import config
import retrieval
import llm

DECOMPOSE_PROMPT = """Does this question ask about more than one distinct thing that would need
different information to answer? If yes, split it into separate, self-contained sub-questions —
each one should make sense on its own without the others. If the question is already a single,
atomic ask, return it unchanged as the only item.

Output ONLY a JSON array of strings, no markdown fences, no other text.

QUESTION: {question}"""


def decompose_question(question):
    return llm.chat_json(DECOMPOSE_PROMPT.format(question=question))


def retrieve_decomposed(question, k=config.TOP_K, search_k=config.SUB_QUESTION_K,
                        verbose=False):
    sub_questions = decompose_question(question)

    per_sub_results = []
    for sub_q in sub_questions:
        sub_ids = retrieval.dense_search(sub_q, k=search_k)
        if verbose:
            print(f"'{sub_q}'")
            print(f"  -> {sub_ids}")
        per_sub_results.append(sub_ids)

    interleaved = []
    seen = set()
    for round_idx in range(search_k):
        for sub_ids in per_sub_results:
            if round_idx < len(sub_ids):
                cid = sub_ids[round_idx]
                if cid not in seen:
                    seen.add(cid)
                    interleaved.append(cid)

    return interleaved[:k]


def main(question):
    ids = retrieve_decomposed(question, verbose=True)
    print(f"\nfinal: {ids}")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else
         "What is the difference between model_validate and model_validate_json, "
         "and how do I make a field immutable?")
