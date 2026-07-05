"""Small, dependency-free text cleanup, equivalent in spirit to Natively's
TranscriptNormalizer: strip filler words and squash the repetition loops
Whisper occasionally produces on noisy/silent audio."""

import re

FILLER_RE = re.compile(r"\b(uh|um|erm?|hmm|you know|i mean)\b[,.]?\s*", re.IGNORECASE)
REPEAT_RE = re.compile(r"\b(\w+)(\s+\1\b){2,}", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = REPEAT_RE.sub(r"\1", text)
    text = FILLER_RE.sub("", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def format_line(speaker: str, start_ts: float, text: str) -> str:
    return f"[{start_ts:6.1f}s] {speaker}: {text}"
