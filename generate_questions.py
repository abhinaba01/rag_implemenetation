import json
import os
import random
import time
from collections import defaultdict

from dotenv import load_dotenv
from google import genai

load_dotenv()
API_KEY = os.environ["GOOGLE_API_KEY"]
MODEL = "gemini-3.6-flash"

N_MULTIHOP_PAIRS = 12
N_UNANSWERABLE = 12
N_AMBIGUOUS_SOURCES = 8

client = genai.Client(api_key=API_KEY)


def load_chunks(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def call_gemini(prompt):
    try:
        response = client.models.generate_content(model=MODEL, contents=prompt)
        raw = response.text.strip()
        if raw.startswith('```'):
            raw = raw.strip('`')
            raw = raw.replace('json', '', 1).strip()
        return json.loads(raw)
    except Exception as e:
        if "404" in str(e) or "NOT_FOUND" in str(e):
            raise SystemExit(f"Model '{MODEL}' unavailable — check ai.google.dev. {e}")
        print(f"  FAILED: {e}")
        return None



MULTIHOP_PROMPT = """You will be shown two chunks of Pydantic documentation, A and B.

Only generate a question if answering it PROPERLY REQUIRES INFORMATION FROM BOTH chunks —
not a question answerable from either one alone with the other just providing flavor.
Good multi-hop questions usually combine a concept from A with a concept from B, e.g.
"how do I use concept X (from A) together with feature Y (from B)" or a genuine
comparison/tradeoff between them.

If no genuine two-chunk question exists between these two, return exactly: null

Otherwise, output ONLY a JSON object, no markdown fences:
{{"question": "...", "expected_answer": "one sentence stating what a correct answer must cover from BOTH chunks"}}

CHUNK A — heading: {heading_a}
{text_a}

CHUNK B — heading: {heading_b}
{text_b}
"""


def generate_multihop(all_chunks, n_pairs):
    by_folder = defaultdict(list)
    for c in all_chunks:
        if len(c['text']) < 200:
            continue
        folder = c['source_file'].replace('\\', '/').split('/')[-2] if '/' in c['source_file'].replace('\\', '/') else 'root'
        by_folder[folder].append(c)

    results = []
    attempts = 0
    while len(results) < n_pairs and attempts < n_pairs * 3:
        attempts += 1
        folder = random.choice([f for f, cs in by_folder.items() if len(cs) >= 2])
        a, b = random.sample(by_folder[folder], 2)

        print(f"[multi-hop {attempts}] {a['chunk_id']} + {b['chunk_id']}")

        prompt = MULTIHOP_PROMPT.format(
            heading_a=a['heading_path'], text_a=a['text'],
            heading_b=b['heading_path'], text_b=b['text']
        )
        result = call_gemini(prompt)

        if result is None or result == "null" or result is None:
            continue
        if isinstance(result, dict) and 'question' in result:
            results.append({
                "question": result["question"],
                "expected_chunk_ids": [a['chunk_id'], b['chunk_id']],
                "category": "multi-hop",
                "expected_answer": result["expected_answer"]
            })
        time.sleep(1)

    return results



UNANSWERABLE_PROMPT = """Here is a list of real section headings from the Pydantic documentation
(so you know what topics ARE covered):

{heading_list}

Write {n} questions that SOUND like real Pydantic questions but are NOT answered anywhere
in Pydantic's documentation. Use these strategies, mixed:
- A real Pydantic V1 feature that was removed in V2 (asked as if it still exists)
- A feature that actually belongs to an adjacent library (SQLAlchemy, FastAPI, dataclasses)
  described as if it's a Pydantic feature
- A plausible but fictional method/function name in Pydantic's naming style
  (e.g. model_validate_strict() — sounds real, does not exist)
- A real Pydantic concept but asking for an undocumented implementation detail
  (exact internal limits, unpublished performance numbers)

Do NOT reuse a real documented feature by accident — cross-check against the heading list above.

Output ONLY a JSON array, no markdown fences:
[{{"question": "...", "expected_answer": "note explaining why this is NOT answerable from the docs"}}]
"""


def generate_unanswerable(all_chunks, n):
    headings = list({c['heading_path'] for c in all_chunks})
    sample_headings = random.sample(headings, min(150, len(headings)))
    heading_list = '\n'.join(f"- {h}" for h in sample_headings)

    prompt = UNANSWERABLE_PROMPT.format(heading_list=heading_list, n=n)
    result = call_gemini(prompt)

    if not result:
        return []

    return [{
        "question": q["question"],
        "expected_chunk_ids": [],
        "category": "unanswerable",
        "expected_answer": q["expected_answer"]
    } for q in result]



AMBIGUOUS_PROMPT = """Here is one chunk of Pydantic documentation.

Write ONE question that this chunk answers, but phrase it VAGUELY and CASUALLY —
the way someone half-remembering a term or unsure of exact vocabulary might type it.
Avoid naming the exact method/class/parameter the chunk uses. Use looser, more general
language a beginner might use instead.

If this chunk's content doesn't lend itself to a vague version (e.g. it's already very
general), return exactly: null

Otherwise output ONLY a JSON object, no markdown fences:
{{"question": "...", "expected_answer": "one sentence stating what a correct answer must contain"}}

HEADING: {heading_path}
CHUNK TEXT:
{text}
"""


def generate_ambiguous(all_chunks, n_sources):
    candidates = [c for c in all_chunks if len(c['text']) >= 200]
    sample = random.sample(candidates, min(n_sources, len(candidates)))

    results = []
    for c in sample:
        print(f"[ambiguous] {c['chunk_id']}")
        prompt = AMBIGUOUS_PROMPT.format(heading_path=c['heading_path'], text=c['text'])
        result = call_gemini(prompt)
        if result and isinstance(result, dict) and 'question' in result:
            results.append({
                "question": result["question"],
                "expected_chunk_ids": [c['chunk_id']],
                "category": "ambiguous",
                "expected_answer": result["expected_answer"]
            })
        time.sleep(1)

    return results


def main():
    all_chunks = load_chunks('index/chunks.jsonl')

    print("=== Generating multi-hop ===")
    multihop = generate_multihop(all_chunks, N_MULTIHOP_PAIRS)

    print("\n=== Generating unanswerable ===")
    unanswerable = generate_unanswerable(all_chunks, N_UNANSWERABLE)

    print("\n=== Generating ambiguous ===")
    ambiguous = generate_ambiguous(all_chunks, N_AMBIGUOUS_SOURCES)

    all_new = multihop + unanswerable + ambiguous

    with open('data/gold_questions_other_types.jsonl', 'w', encoding='utf-8') as f:
        for q in all_new:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')

    print(f"\nMulti-hop: {len(multihop)}")
    print(f"Unanswerable: {len(unanswerable)}")
    print(f"Ambiguous: {len(ambiguous)}")
    print(f"Total written to data/gold_questions_other_types.jsonl: {len(all_new)}")


if __name__ == '__main__':
    main()