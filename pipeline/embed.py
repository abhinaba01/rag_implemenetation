import json
from pathlib import Path

import faiss

import config
import data_utils
import retrieval


def main():
    chunks = data_utils.load_chunks()
    chunk_ids = [c['chunk_id'] for c in chunks]

    vectors = retrieval.embed([c['text'] for c in chunks])

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    assert len(chunk_ids) == index.ntotal, f"{len(chunk_ids)} ids vs {index.ntotal} vectors"

    Path(config.FAISS_PATH).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, config.FAISS_PATH)

    with open(config.CHUNK_IDS_PATH, 'w', encoding='utf-8') as f:
        json.dump(chunk_ids, f)

    print(f"Indexed {index.ntotal} chunks as {vectors.shape[1]}-dim vectors")


if __name__ == '__main__':
    main()
