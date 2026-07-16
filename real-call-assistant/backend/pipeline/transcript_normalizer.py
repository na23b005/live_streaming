"""Small, dependency-free text cleanup, equivalent in spirit to Natively's
TranscriptNormalizer: strip filler words and squash the repetition loops
Whisper occasionally produces on noisy/silent audio."""

import re

FILLER_RE = re.compile(r"\b(uh|um|erm?|hmm|you know|i mean)\b[,.]?\s*", re.IGNORECASE)
REPEAT_RE = re.compile(r"\b(\w+)(\s+\1\b){2,}", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")


def strip_trailing_hallucinations(text: str) -> str:
    """Strips common trailing Whisper hallucinations (like 'thank you') from the end
    of a transcription segment.
    """
    cleaned = text.strip()
    if not cleaned:
        return cleaned

    # Normalization helper to compare suffixes
    def get_norm_words(s: str) -> list[str]:
        return s.lower().translate(str.maketrans("", "", ".,?!")).split()

    # Trailing hallucination phrases as word lists
    hallucination_suffixes = [
        ["thank", "you", "everybody"],
        ["thank", "you", "everyone"],
        ["thank", "you", "very", "much"],
        ["thank", "you", "so", "much"],
        ["thank", "you"],
        ["thanks", "for", "watching"],
        ["thank", "you", "for", "watching"],
        ["please", "subscribe"],
        ["subscribe"],
        ["thank", "you", "bye"],
        ["bye", "bye"],
    ]

    modified = True
    while modified:
        modified = False
        norm_words = get_norm_words(cleaned)
        
        for suffix in hallucination_suffixes:
            s_len = len(suffix)
            if len(norm_words) >= s_len and norm_words[-s_len:] == suffix:
                words_in_orig = cleaned.split()
                if len(words_in_orig) >= s_len:
                    cleaned = " ".join(words_in_orig[:-s_len]).strip()
                    cleaned = cleaned.rstrip(".,?! ")
                    modified = True
                    break
                    
    return cleaned


def clean_text(text: str) -> str:
    text = REPEAT_RE.sub(r"\1", text)
    text = FILLER_RE.sub("", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    text = strip_trailing_hallucinations(text)
    return text


def is_whisper_hallucination(text: str, rms: float) -> bool:
    """
    Detects if the transcribed text is a common Whisper hallucination
    on silent or noisy audio segments, based on RMS energy of the audio segment.
    """
    # Normalize text to lower-case with no punctuation
    cleaned = text.lower().strip().replace("’", "'").translate(str.maketrans("", "", ".,?!"))
    
    # Absolute hallucination patterns (rarely said as a single segment in a call, and/or standard Whisper failures)
    # If the segment contains these phrases or their variants, we filter them.
    absolute_hallucinations = {
        "thank you for watching",
        "thanks for watching",
        "please subscribe",
        "subscribe",
        "subtitles by",
        "subtitles by opensubtitles",
        "subtitles by opensubtitles.org",
        "downloaded from",
        "translated by",
        "sh",
        "hmmm",
        "use code",
        "click the link",
        "click on this video",
        "click on your favorite button",
        "click on your favorite",
        "favorite button",
        "leave them in the comments below",
        "leave them in the comments",
        "in the comments below",
        "comments below",
    }
    
    for phrase in absolute_hallucinations:
        if len(phrase) <= 4:
            # Short phrases (e.g. "sh", "hmmm") must be exact matches or separate words
            if cleaned == phrase or f" {phrase} " in f" {cleaned} ":
                return True
        else:
            # Long phrases can be substring matched safely
            if phrase in cleaned:
                return True

    # Punctuation-only or dots-only check (e.g. ". . . . . . . . .")
    if not any(c.isalnum() for c in cleaned):
        return True
            
    # Conditional hallucination patterns (words/phrases that are valid, but frequently hallucinated on quiet audio chunks like breaths, clicks, keyboard noise)
    conditional_hallucinations = {
        "thank you",
        "thank you so much",
        "thank you very much",
        "i don't know",
        "you",
        "yeah",
        "yes",
        "oh",
        "bye",
        "hello",
    }
    
    if cleaned in conditional_hallucinations:
        # If the energy is low, we treat it as a suspected Whisper hallucination
        if rms < 0.015:
            return True
            
    return False


def longest_common_prefix(s1: str, s2: str) -> str:
    """Finds the longest common prefix of two strings."""
    match_len = 0
    min_len = min(len(s1), len(s2))
    for i in range(min_len):
        if s1[i].lower() == s2[i].lower():
            match_len += 1
        else:
            break
    return s1[:match_len]


def remove_boundary_overlap(prev_text: str, next_text: str, max_overlap_words: int = 4) -> str:
    """Compares the suffix of the previous text with the prefix of the new text,
    and removes any matching overlap of up to `max_overlap_words`.
    """
    words1 = prev_text.strip().split()
    words2 = next_text.strip().split()

    if not words1 or not words2:
        return next_text

    # Search for the longest matching overlap (e.g., from 4 words down to 1)
    for n in range(min(len(words1), len(words2), max_overlap_words), 0, -1):
        suffix1 = [w.lower().strip(".,?!:;\"'") for w in words1[-n:]]
        prefix2 = [w.lower().strip(".,?!:;\"'") for w in words2[:n]]

        if suffix1 == prefix2:
            return " ".join(words2[n:])

    return next_text


def format_line(speaker: str, start_ts: float, text: str) -> str:
    return f"[{start_ts:6.1f}s] {speaker}: {text}"

