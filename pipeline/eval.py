import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import config
import data_utils
import retrieval
from decompose import retrieve_decomposed



def recall_at_k(expected_ids, retrieved_ids):
    return 1 if any(eid in retrieved_ids for eid in expected_ids) else 0


def reciprocal_rank(expected_ids, retrieved_ids):
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in expected_ids:
            return 1 / rank
    return 0



STRATEGIES = {
    'Baseline (dense only)': lambda q: retrieval.dense_search(q, k=config.TOP_K),
    'Reranked':              lambda q: retrieval.dense_reranked(q),
    'Hybrid (dense + BM25)': lambda q: retrieval.hybrid_search(q),
    'Hybrid + Reranked':     lambda q: retrieval.hybrid_reranked(q),
    'Decomposed':            lambda q: retrieve_decomposed(q),
}


def evaluate_question(q, retrieve_fn):
    started = time.perf_counter()
    retrieved_ids = retrieve_fn(q['question'])
    latency_s = time.perf_counter() - started

    return {
        'question': q['question'],
        'category': q['category'],
        'expected_chunk_ids': q['expected_chunk_ids'],
        'retrieved_chunk_ids': retrieved_ids,
        'recall_at_k': recall_at_k(q['expected_chunk_ids'], retrieved_ids),
        'reciprocal_rank': reciprocal_rank(q['expected_chunk_ids'], retrieved_ids),
        'latency_s': round(latency_s, 4),
    }


def run_strategy(gold_questions, retrieve_fn):
    return [evaluate_question(q, retrieve_fn) for q in gold_questions]


def summarize(results, label):
    """Print the aggregate metrics for one strategy and return them."""
    answerable = [r for r in results if r['expected_chunk_ids']]
    unanswerable = [r for r in results if not r['expected_chunk_ids']]

    avg_recall = sum(r['recall_at_k'] for r in answerable) / len(answerable)
    avg_mrr = sum(r['reciprocal_rank'] for r in answerable) / len(answerable)
    latencies = sorted(r['latency_s'] for r in results)

    by_category = defaultdict(list)
    for r in answerable:
        by_category[r['category']].append(r['recall_at_k'])
    per_category = {
        cat: {'recall_at_k': sum(s) / len(s), 'n': len(s)}
        for cat, s in sorted(by_category.items())
    }

    print(f"\n=== {label} ===")
    print(f"Recall@{config.TOP_K}: {avg_recall:.3f}")
    print(f"MRR: {avg_mrr:.3f}")
    for cat, m in per_category.items():
        print(f"  {cat}: {m['recall_at_k']:.3f} (n={m['n']})")

    return {
        'label': label,
        'n_questions': len(results),
        'n_answerable': len(answerable),
        'n_unanswerable': len(unanswerable),
        'recall_at_k': avg_recall,
        'mrr': avg_mrr,
        'per_category': per_category,
        'latency_s': {
            'total': round(sum(latencies), 3),
            'mean': round(sum(latencies) / len(latencies), 4),
            'median': round(latencies[len(latencies) // 2], 4),
            'max': round(latencies[-1], 4),
        },
    }



def slugify(label):
    """'Hybrid (dense + BM25)' -> 'hybrid_dense_bm25'"""
    return re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_')


def run_metadata(gold_questions, started):
    """Everything needed to tell later what produced these numbers."""
    return {
        'run_id': started.strftime('%Y%m%d_%H%M%S'),
        'timestamp': started.isoformat(timespec='seconds'),
        'embed_model': config.EMBED_MODEL,
        'rerank_model': config.RERANK_MODEL,
        'chat_model': config.CHAT_MODEL,
        'top_k': config.TOP_K,
        'candidate_k': config.CANDIDATE_K,
        'sub_question_k': config.SUB_QUESTION_K,
        'n_chunks': len(data_utils.get_chunk_ids()),
        'n_gold_questions': len(gold_questions),
        'chunks_path': config.CHUNKS_PATH,
        'gold_path': config.GOLD_PATH,
    }


def save_run(label, results, summary, metadata, root=config.RESULTS_DIR):
    """Write one strategy's traces + summary into its own run folder."""
    folder = Path(root) / f"{slugify(label)}_{metadata['run_id']}"
    folder.mkdir(parents=True, exist_ok=True)

    with open(folder / 'traces.jsonl', 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    with open(folder / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump({'run': metadata, 'metrics': summary}, f, indent=2, ensure_ascii=False)

    return folder


def load_run(folder):
    """Read back a saved run as (metadata, metrics, traces)."""
    folder = Path(folder)
    with open(folder / 'summary.json', encoding='utf-8') as f:
        saved = json.load(f)
    with open(folder / 'traces.jsonl', encoding='utf-8') as f:
        traces = [json.loads(line) for line in f if line.strip()]
    return saved['run'], saved['metrics'], traces



def report_regressions(before, after, label_before, label_after, category='multi-hop'):
    """Questions in `category` that `before` retrieved but `after` lost."""
    before_by_q = {r['question']: r for r in before}
    after_by_q = {r['question']: r for r in after}

    print(f"\n=== {category} regressions ({label_before} hit, {label_after} missed) ===")
    for question, b in before_by_q.items():
        a = after_by_q.get(question)
        if a is None or b['category'] != category:
            continue
        if b['recall_at_k'] == 1 and a['recall_at_k'] == 0:
            print(question)
            print('  expected:', b['expected_chunk_ids'])
            print(f'  {label_before} got:', b['retrieved_chunk_ids'])
            print(f'  {label_after} got:', a['retrieved_chunk_ids'])
            print()


def select_strategies(names):
    """Match CLI arguments against strategy names by case-insensitive prefix."""
    if not names:
        return dict(STRATEGIES)
    chosen = {
        label: fn
        for label, fn in STRATEGIES.items()
        if any(label.lower().startswith(n.lower()) for n in names)
    }
    if not chosen:
        raise SystemExit(f"No strategy matched {names}. Available: {list(STRATEGIES)}")
    return chosen


def warmup():
    """Load the index and both models up front.

    Without this the first question measured is also paying for the lazy model
    load, which skews its latency by seconds.
    """
    ids = retrieval.dense_search('warmup', k=1)
    retrieval.rerank('warmup', ids, k=1)
    retrieval.bm25_search('warmup', k=1)


def main(names=(), save=True):
    started = datetime.now().astimezone()
    gold_questions = data_utils.load_gold_questions()
    strategies = select_strategies(names)
    metadata = run_metadata(gold_questions, started)
    warmup()

    print(f"run_id {metadata['run_id']} | {metadata['n_gold_questions']} questions "
          f"| {metadata['n_chunks']} chunks | strategies: {list(strategies)}")

    results = {}
    for label, retrieve_fn in strategies.items():
        results[label] = run_strategy(gold_questions, retrieve_fn)
        summary = summarize(results[label], label)
        if save:
            print(f"  saved -> {save_run(label, results[label], summary, metadata)}")

    baseline_label, reranked_label = 'Baseline (dense only)', 'Reranked'
    if baseline_label in results and reranked_label in results:
        report_regressions(results[baseline_label], results[reranked_label],
                           baseline_label, reranked_label)

    return results


if __name__ == '__main__':
    args = sys.argv[1:]
    main([a for a in args if not a.startswith('--')], save='--no-save' not in args)
