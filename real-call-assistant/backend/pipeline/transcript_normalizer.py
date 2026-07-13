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


def is_whisper_hallucination(text: str, rms: float) -> bool:
    """
    Detects if the transcribed text is a common Whisper hallucination
    on silent or noisy audio segments, based on RMS energy of the audio segment.
    """
    # Normalize text to lower-case with no punctuation
    cleaned = text.lower().strip().replace("’", "'").translate(str.maketrans("", "", ".,?!"))
    
    # Absolute hallucination patterns (rarely said as a single segment in a call, and/or standard Whisper failures)
    # If the segment contains *only* these phrases or their variants, we filter them.
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
    }
    
    if cleaned in absolute_hallucinations:
        return True
        
    for phrase in absolute_hallucinations:
        if phrase in cleaned and len(cleaned) < len(phrase) + 10:
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
        if rms < 0.035:
            return True
            
    return False


def format_line(speaker: str, start_ts: float, text: str) -> str:
    return f"[{start_ts:6.1f}s] {speaker}: {text}"
