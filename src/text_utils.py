"""Shared text processing utilities."""

import re


def split_into_chunks(text: str, max_chars: int) -> list:
    """Split text into chunks of at most max_chars, breaking at sentence boundaries."""
    if len(text) <= max_chars:
        return [text]

    # Split into sentences (period/question/exclamation followed by space or newline)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current = ""
    for sentence in sentences:
        # If a single sentence exceeds max_chars, split it by newlines as fallback
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            parts = sentence.split("\n")
            for part in parts:
                if not part.strip():
                    continue
                if current and len(current) + len(part) + 1 > max_chars:
                    chunks.append(current)
                    current = part
                else:
                    current = current + "\n" + part if current else part
            continue

        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = current + " " + sentence if current else sentence

    if current:
        chunks.append(current)

    return chunks
