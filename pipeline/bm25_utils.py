import re
from nltk.corpus import stopwords

try:
    ENGLISH_STOPWORDS = set(stopwords.words('english'))
except LookupError:
    import nltk
    nltk.download('stopwords')
    ENGLISH_STOPWORDS = set(stopwords.words('english'))

DOMAIN_STOPWORDS = {'pydantic', 'python'}
CODE_STOPWORDS = {'import', 'from', 'class', 'print', 'str', 'int', 'def'}

ALL_STOPWORDS = ENGLISH_STOPWORDS | DOMAIN_STOPWORDS | CODE_STOPWORDS


def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if t not in ALL_STOPWORDS]


def reciprocal_rank_fusion(dense_ids, bm25_ids, k=60):
    """Merge two ranked ID lists into one, using RRF. Returns a list of
    (chunk_id, fused_score) sorted best-first."""
    scores = {}

    for rank, cid in enumerate(dense_ids, start=1):
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank)

    for rank, cid in enumerate(bm25_ids, start=1):
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)