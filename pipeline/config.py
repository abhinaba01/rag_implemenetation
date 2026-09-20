RAW_DOCS_ROOT = 'data/raw_docs'
RAW_DOCS_GLOB = 'data/raw_docs/**/*.md'

CHUNKS_PATH = 'index/chunks.jsonl'
CHUNK_IDS_PATH = 'index/chunk_ids.json'
FAISS_PATH = 'index/faiss.index'
BM25_PATH = 'index/bm25.pkl'

GOLD_PATH = 'data/gold_questions.jsonl'
RESULTS_DIR = 'results'

EMBED_MODEL = 'all-MiniLM-L6-v2'
RERANK_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
CHAT_MODEL = 'gpt-4o-mini'

MAX_CHARS = 2500

TOP_K = 5
CANDIDATE_K = 20
SUB_QUESTION_K = 10
