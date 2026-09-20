from functools import lru_cache

from sentence_transformers import CrossEncoder, SentenceTransformer

import config
import data_utils
from bm25_utils import reciprocal_rank_fusion, tokenize



@lru_cache(maxsize=1)
def get_embed_model():
    return SentenceTransformer(config.EMBED_MODEL)


@lru_cache(maxsize=1)
def get_reranker():
    return CrossEncoder(config.RERANK_MODEL)


def embed(texts):
    """Encode a list of texts into normalized vectors (cosine == inner product)."""
    return get_embed_model().encode(list(texts), normalize_embeddings=True)



def dense_search(question, k=config.TOP_K):
    chunk_ids = data_utils.get_chunk_ids()
    _, positions = data_utils.get_index().search(embed([question]), k=k)
    return [chunk_ids[p] for p in positions[0]]


def bm25_search(question, k=config.TOP_K):
    bm25, bm25_chunk_ids = data_utils.get_bm25()
    scores = bm25.get_scores(tokenize(question))
    ranked = sorted(zip(bm25_chunk_ids, scores), key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in ranked[:k]]


def rerank_with_scores(question, candidate_ids, k=config.TOP_K):
    """Re-score a shortlist with the cross-encoder. Returns (chunk, score) pairs."""
    candidates = data_utils.get_chunks_by_ids(candidate_ids)
    scores = get_reranker().predict([(question, c['text']) for c in candidates])
    ordered = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return ordered[:k]


def rerank(question, candidate_ids, k=config.TOP_K):
    """As `rerank_with_scores`, but returns just the chunk_ids."""
    return [c['chunk_id'] for c, _ in rerank_with_scores(question, candidate_ids, k=k)]


def dense_reranked(question, k=config.TOP_K, candidate_k=config.CANDIDATE_K):
    return rerank(question, dense_search(question, k=candidate_k), k=k)


def hybrid_search(question, k=config.TOP_K, candidate_k=config.CANDIDATE_K):
    """Dense and BM25 shortlists merged with reciprocal rank fusion."""
    fused = reciprocal_rank_fusion(
        dense_search(question, k=candidate_k),
        bm25_search(question, k=candidate_k),
    )
    return [cid for cid, _ in fused[:k]]


def hybrid_reranked(question, k=config.TOP_K, candidate_k=config.CANDIDATE_K):
    fused_ids = hybrid_search(question, k=candidate_k, candidate_k=candidate_k)
    return rerank(question, fused_ids, k=k)
