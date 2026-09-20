import sys

import config
import data_utils
import retrieval
import llm

PROMPT_TEMPLATE = """Answer the question using ONLY the context below. If the context does not contain enough information to answer, say "I don't have enough information to answer that" rather than guessing.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


def generate_answer(question, retrieved_chunks, model=config.CHAT_MODEL):
    prompt = PROMPT_TEMPLATE.format(
        context=llm.build_context(retrieved_chunks),
        question=question,
    )
    return llm.chat(prompt, model=model)


def answer(question, k=config.TOP_K):
    """Retrieve with the default strategy, then generate."""
    chunk_ids = retrieval.dense_search(question, k=k)
    return generate_answer(question, data_utils.get_chunks_by_ids(chunk_ids))


def main(question):
    print(answer(question))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else "How does the sun create energy ?")
