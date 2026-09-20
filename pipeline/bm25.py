import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

import config
import data_utils
from bm25_utils import tokenize


def main():
    chunks = data_utils.load_chunks()
    chunk_ids = [c['chunk_id'] for c in chunks]

    bm25 = BM25Okapi([tokenize(c['text']) for c in chunks])

    Path(config.BM25_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(config.BM25_PATH, 'wb') as f:
        pickle.dump({'bm25': bm25, 'chunk_ids': chunk_ids}, f)

    print(f"Indexed {len(chunks)} chunks")


if __name__ == '__main__':
    main()
