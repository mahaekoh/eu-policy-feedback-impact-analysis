"""Shared text processing utilities."""

import logging
import re

logger = logging.getLogger(__name__)

MIN_PRINTABLE_RATIO = 0.5


def should_skip_text(text: str, label: str = "") -> bool:
    """Return True if text should not be summarized."""
    tag = f" [{label}]" if label else ""
    if not text or not text.strip():
        logger.warning("Empty or whitespace-only text, skipping%s", tag)
        return True
    if text.startswith("%PDF-"):
        logger.warning("Raw PDF binary detected, skipping%s", tag)
        return True
    # Skip text that is mostly non-printable (failed PDF extraction)
    printable = sum(1 for c in text if c.isprintable() or c in '\n\r\t')
    if printable / len(text) < MIN_PRINTABLE_RATIO:
        logger.warning("Text is only %.0f%% printable (%d/%d chars), skipping%s",
                       printable / len(text) * 100, printable, len(text), tag)
        return True
    return False


def group_by_char_budget(texts: list, max_chars: int) -> list:
    """Group texts greedily so each group's combined length fits within max_chars."""
    groups = []
    current_group = []
    current_len = 0
    separator_len = 2  # "\n\n" between items

    for text in texts:
        added_len = len(text) + (separator_len if current_group else 0)
        if current_group and current_len + added_len > max_chars:
            groups.append(current_group)
            current_group = [text]
            current_len = len(text)
        else:
            current_group.append(text)
            current_len += added_len

    if current_group:
        groups.append(current_group)

    return groups


def split_into_chunks(text: str, max_chars: int, label: str = "") -> list:
    """Split text into chunks of at most max_chars, breaking at sentence boundaries.

    Args:
        text: The text to split.
        max_chars: Maximum characters per chunk.
        label: Optional identifier for log messages (e.g. "init=12096 fb=503089").
    """
    if len(text) <= max_chars:
        return [text]

    tag = f" [{label}]" if label else ""

    # Split into sentences (period/question/exclamation followed by space or newline)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current = ""
    for sentence in sentences:
        # If a single sentence exceeds max_chars, split it by newlines as fallback
        if len(sentence) > max_chars:
            logger.warning("Sentence exceeds max_chars (%d > %d), falling back to newline split%s",
                           len(sentence), max_chars, tag)
            if current:
                chunks.append(current)
                current = ""
            parts = sentence.split("\n")
            for part in parts:
                if not part.strip():
                    continue
                # If a single line still exceeds max_chars, split at word
                # boundaries first, then hard-split as last resort
                if len(part) > max_chars:
                    logger.warning("Line exceeds max_chars (%d > %d), falling back to word boundary split%s",
                                   len(part), max_chars, tag)
                    if current:
                        chunks.append(current)
                        current = ""
                    words = re.split(r'\s+', part)
                    for word in words:
                        if len(word) > max_chars:
                            logger.warning("Word exceeds max_chars (%d > %d), hard-splitting%s",
                                           len(word), max_chars, tag)
                            if current:
                                chunks.append(current)
                                current = ""
                            for i in range(0, len(word), max_chars):
                                chunks.append(word[i:i + max_chars])
                        elif current and len(current) + len(word) + 1 > max_chars:
                            chunks.append(current)
                            current = word
                        else:
                            current = current + " " + word if current else word
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
