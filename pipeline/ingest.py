import glob
import json
from pathlib import Path

import config


def _group_into_blocks(lines):
    """Group lines into indivisible blocks: one paragraph, or one whole code fence."""
    blocks = []
    current = []
    in_code_block = False

    for line in lines:
        if line.strip().startswith("```"):
            if not in_code_block:

                if current:
                    blocks.append(current)
                    current = []
                in_code_block = True
                current.append(line)
            else:

                current.append(line)
                blocks.append(current)
                current = []
                in_code_block = False
            continue

        if in_code_block:
            current.append(line)
            continue

        if line.strip() == "":
            if current:
                blocks.append(current)
                current = []
            continue

        current.append(line)

    if current:
        blocks.append(current)

    return blocks


def split_large_chunk(chunk, max_chars=config.MAX_CHARS):

    lines = chunk["text"]

    if len("\n".join(lines)) <= max_chars:
        return [chunk]

    heading_line = lines[0] if lines and lines[0].lstrip().startswith("#") else None

    blocks = _group_into_blocks(lines)

    parts = []
    current = []
    current_size = 0

    for block in blocks:
        block_size = len("\n".join(block)) + 1

        if current and current_size + block_size > max_chars:
            parts.append(current)
            current = []
            current_size = 0

        current.extend(block)
        current.append("")
        current_size += block_size + 1

    if current:
        parts.append(current)

    out = []
    for i, part in enumerate(parts):
        text = part if i == 0 or heading_line is None else [heading_line, ""] + part
        out.append(
            {
                **chunk,
                "text": text,
                "chunk_id": f"{chunk['chunk_id']}_{chr(ord('a') + i)}",
            }
        )

    return out


def doc_slug(file_path):
    """Folder-qualified stem, so api/config.md and concepts/config.md differ."""
    rel = Path(file_path).relative_to(config.RAW_DOCS_ROOT).with_suffix('')
    return '_'.join(rel.parts)


def chunk_document(file_path):
    """Split one markdown file into chunks."""
    with open(file_path, "r", encoding="utf-8") as file:
        doc = file.read()

    lines = doc.split("\n")
    slug = doc_slug(file_path)

    chunks = []
    chunk_number = 1

    chunk = {
        'text': [],
        'source_file': file_path,
        'heading_path': '(preamble)',
        'chunk_id': f"{slug}_{chunk_number:03d}",
    }
    chunk_number += 1

    in_code_block = False

    for line in lines:

        if line.strip().startswith("```"):
            in_code_block = not in_code_block

        if in_code_block:
            chunk["text"].append(line)
            continue

        if line.startswith("#"):

            if chunk.get("text"):
                chunks.extend(split_large_chunk(chunk))

            chunk = {
                "text": [line],
                "source_file": file_path,
                "heading_path": line.strip(" #"),
                "chunk_id": f"{slug}_{chunk_number:03d}",
            }
            chunk_number += 1
            continue

        chunk["text"].append(line)

    if chunk.get("text"):
        chunks.extend(split_large_chunk(chunk))

    return chunks


def build_chunks(pattern=config.RAW_DOCS_GLOB):
    chunks = []
    for file_path in glob.glob(pattern, recursive=True):
        chunks.extend(chunk_document(file_path))
    return chunks


def write_chunks(chunks, path=config.CHUNKS_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            record = {**chunk, 'text': '\n'.join(chunk['text'])}
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def main():
    chunks = build_chunks()
    write_chunks(chunks)

    ids = [c['chunk_id'] for c in chunks]
    duplicates = len(ids) - len(set(ids))
    files = len({c['source_file'] for c in chunks})
    print(f"Wrote {len(chunks)} chunks from {files} files to {config.CHUNKS_PATH}")
    print(f"Duplicate chunk_ids: {duplicates}")


if __name__ == '__main__':
    main()
