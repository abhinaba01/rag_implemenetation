import json

with open('index/chunks.jsonl', 'r', encoding='utf-8') as f:
    valid_ids = {json.loads(line)['chunk_id'] for line in f}

with open('data/gold_questions.jsonl', 'r', encoding='utf-8') as f:
    gold = [json.loads(line) for line in f]

missing = 0
for q in gold:
    for cid in q['expected_chunk_ids']:
        if cid not in valid_ids:
            print('MISSING:', cid, '|', q['question'][:60])
            missing += 1

print(f"\n{missing} missing out of {len(gold)} questions")