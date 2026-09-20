import json
import pickle
from functools import lru_cache

import faiss

import config



def load_chunks(path=config.CHUNKS_PATH):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def load_chunk_ids(path=config.CHUNK_IDS_PATH):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_faiss_index(path=config.FAISS_PATH):
    return faiss.read_index(path)


def load_bm25(path=config.BM25_PATH):
    """Returns (bm25_model, chunk_ids) as written by bm25.py."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['bm25'], data['chunk_ids']


def load_gold_questions(path=config.GOLD_PATH):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]



@lru_cache(maxsize=1)
def get_chunks():
    return load_chunks()


@lru_cache(maxsize=1)
def get_chunk_ids():
    return load_chunk_ids()


@lru_cache(maxsize=1)
def get_index():
    return load_faiss_index()


@lru_cache(maxsize=1)
def get_bm25():
    return load_bm25()


@lru_cache(maxsize=1)
def _chunks_by_id():
    """chunk_id -> chunk. Built once; lookups are O(1) instead of a linear scan."""
    return {c['chunk_id']: c for c in get_chunks()}


def get_chunk_by_id(chunk_id):
    return _chunks_by_id()[chunk_id]


def get_chunks_by_ids(chunk_ids):
    """Chunks for these ids, in the order given (i.e. rank order)."""
    by_id = _chunks_by_id()
    return [by_id[cid] for cid in chunk_ids]
