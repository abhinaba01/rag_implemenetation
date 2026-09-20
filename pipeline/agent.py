from typing import List, TypedDict

from langgraph.graph import END, StateGraph

import config
import data_utils
import retrieval
import llm
from generate import generate_answer


class CRAGState(TypedDict):
    question: str
    retrieved_chunks: List[dict]
    answer: str
    grade: str
    retry_count: int


GRADE_PROMPT = """Given this question and these retrieved passages, judge whether the passages
contain enough information to answer the question well.

Respond with ONLY one word: "correct" (fully sufficient), "ambiguous" (partially relevant but
incomplete), or "incorrect" (not relevant / not sufficient at all).

QUESTION: {question}

PASSAGES:
{passages}
"""

REWRITE_PROMPT = """This search query didn't retrieve fully relevant results. Rewrite it to be
clearer and more specific, using precise technical terminology where possible, so that
retrieval is more likely to find the right documentation. Output ONLY the rewritten question,
nothing else.

ORIGINAL QUESTION: {question}
"""

REFUSAL = "I don't have enough information in the documentation to answer that question."

PASSAGE_PREVIEW_CHARS = 500



def retrieve_node(state: CRAGState) -> dict:
    chunk_ids = retrieval.dense_search(state['question'], k=config.TOP_K)
    return {'retrieved_chunks': data_utils.get_chunks_by_ids(chunk_ids)}


def grade_retrieval_node(state: CRAGState) -> dict:
    passages = '\n\n---\n\n'.join(
        c['text'][:PASSAGE_PREVIEW_CHARS] for c in state['retrieved_chunks']
    )
    grade = llm.chat(GRADE_PROMPT.format(question=state['question'], passages=passages))
    return {'grade': grade.lower()}


def generate_node(state: CRAGState) -> dict:
    return {'answer': generate_answer(state['question'], state['retrieved_chunks'])}


def refuse_node(state: CRAGState) -> dict:
    return {'answer': REFUSAL}


def rewrite_query_node(state: CRAGState) -> dict:
    rewritten = llm.chat(REWRITE_PROMPT.format(question=state['question']))
    return {'question': rewritten, 'retry_count': state.get('retry_count', 0) + 1}



def _route(grade, on_ambiguous):
    if grade == 'correct':
        return 'generate'
    if grade == 'incorrect':
        return 'refuse'
    return on_ambiguous


def route_after_first_grading(state: CRAGState) -> str:
    return _route(state['grade'], on_ambiguous='rewrite')


def route_after_second_grading(state: CRAGState) -> str:
    return _route(state['grade'], on_ambiguous='generate')


def build_graph():
    graph = StateGraph(CRAGState)

    graph.add_node('retrieve', retrieve_node)
    graph.add_node('grade1', grade_retrieval_node)
    graph.add_node('rewrite', rewrite_query_node)
    graph.add_node('retrieve2', retrieve_node)
    graph.add_node('grade2', grade_retrieval_node)
    graph.add_node('generate', generate_node)
    graph.add_node('refuse', refuse_node)

    graph.set_entry_point('retrieve')
    graph.add_edge('retrieve', 'grade1')
    graph.add_conditional_edges(
        'grade1',
        route_after_first_grading,
        {'generate': 'generate', 'refuse': 'refuse', 'rewrite': 'rewrite'},
    )
    graph.add_edge('rewrite', 'retrieve2')
    graph.add_edge('retrieve2', 'grade2')
    graph.add_conditional_edges(
        'grade2',
        route_after_second_grading,
        {'generate': 'generate', 'refuse': 'refuse'},
    )
    graph.add_edge('generate', END)
    graph.add_edge('refuse', END)

    return graph.compile()


def run(question, app=None):
    app = app or build_graph()
    return app.invoke({
        'question': question,
        'retrieved_chunks': [],
        'grade': '',
        'answer': '',
        'retry_count': 0,
    })


def evaluate_crag(q, app):
    result = run(q['question'], app=app)
    return {
        'question': q['question'],
        'category': q['category'],
        'expected_chunk_ids': q['expected_chunk_ids'],
        'grade': result['grade'],
        'answer': result['answer'],
        'retry_count': result.get('retry_count', 0),
    }


def report(results):
    unanswerable = [r for r in results if not r['expected_chunk_ids']]
    answerable = [r for r in results if r['expected_chunk_ids']]

    correctly_refused = sum(1 for r in unanswerable if r['grade'] == 'incorrect')
    print(f"Correctly refused unanswerable: {correctly_refused}/{len(unanswerable)}")

    wrongly_refused = sum(1 for r in answerable if r['grade'] == 'incorrect')
    print(f"Wrongly refused answerable: {wrongly_refused}/{len(answerable)}")

    print("\n=== Unanswerable that were NOT refused ===")
    for r in unanswerable:
        if r['grade'] != 'incorrect':
            print(r['question'])
            print('  grade:', r['grade'])
            print()

    print("=== Answerable that WERE refused ===")
    for r in answerable:
        if r['grade'] == 'incorrect':
            print(r['question'])
            print('  expected:', r['expected_chunk_ids'])
            print()


def main():
    app = build_graph()
    gold_questions = data_utils.load_gold_questions()
    results = [evaluate_crag(q, app) for q in gold_questions]
    report(results)


if __name__ == '__main__':
    main()
