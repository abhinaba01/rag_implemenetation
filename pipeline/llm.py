import json
import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

import config


@lru_cache(maxsize=1)
def get_client():
    load_dotenv()
    return OpenAI(api_key=os.environ['OPENAI_API_KEY'])


def chat(prompt, model=config.CHAT_MODEL):
    """Single-turn completion. Returns the stripped message text."""
    response = get_client().chat.completions.create(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
    )
    return response.choices[0].message.content.strip()


def strip_code_fence(raw):
    """Models often wrap JSON in ```json ... ``` despite being told not to."""
    if raw.startswith('```'):
        raw = raw.strip('`').replace('json', '', 1).strip()
    return raw


def chat_json(prompt, model=config.CHAT_MODEL):
    """Completion whose output is parsed as JSON, tolerating markdown fences."""
    return json.loads(strip_code_fence(chat(prompt, model=model)))


def build_context(chunks):
    """Format retrieved chunks as the CONTEXT block of a prompt."""
    return '\n\n---\n\n'.join(
        f"[Source: {c['heading_path']}]\n{c['text']}" for c in chunks
    )
