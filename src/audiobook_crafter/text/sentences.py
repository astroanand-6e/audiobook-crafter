"""Sentence splitting and TTS-sized chunking.

TTS models are most reliable on a sentence or two at a time, so long passages are cut at sentence
ends first, then at clause boundaries (; : , —), and only as a last resort at a space.
"""

from __future__ import annotations

import re

_ABBREVIATIONS = {
    "mr", "mrs", "ms", "mx", "dr", "st", "prof", "sr", "jr", "rev", "hon", "capt", "col", "gen", "lt", "sgt",
    "cpl", "maj", "adm", "gov", "pres", "messrs", "mme", "mlle", "esq", "vs", "etc", "e.g", "i.e", "viz",
    "cf", "approx", "dept", "est", "fig", "no", "nos", "vol", "vols", "ch", "pp", "inc", "ltd", "co", "corp",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "mt", "ft", "ave",
    "blvd", "rd", "a.m", "p.m", "u.s", "u.k", "ph.d", "b.a", "m.a",
}  # fmt: skip
_BOUNDARY = re.compile(r"([.!?…]+|[。！？।॥]+)([\"”’»)\]]*)(\s+|$)")
_CJK_ENDERS = set("。！？।॥")
_CLAUSE = re.compile(r"(?<=[;:,—–])\s*|(?<=\))\s+")


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    sentences: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        end = match.end(2)
        punct = match.group(1)
        rest = text[match.end() :]
        if not rest:
            break
        if not match.group(3) and punct[-1] not in _CJK_ENDERS:
            continue
        if punct.endswith(".") and len(punct) == 1:
            word = re.findall(r"([\w.]+)\.$", text[start : match.start() + 1])
            token = word[0].lower() if word else ""
            if token in _ABBREVIATIONS or re.fullmatch(r"[A-Za-z]", token):  # "Mr." / initials "J."
                continue
        first = rest.lstrip("\"“‘'«(—–- ")[:1]
        if first and first.islower() and punct[-1] not in _CJK_ENDERS:
            continue  # "… he said. and then" / "e.g. this" — not a real boundary
        sentences.append(text[start:end].strip())
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _split_long(sentence: str, max_chars: int) -> list[str]:
    if len(sentence) <= max_chars:
        return [sentence]
    middle = len(sentence) / 2
    cuts = [
        m.start() for m in _CLAUSE.finditer(sentence) if 0.2 * len(sentence) < m.start() < 0.8 * len(sentence)
    ]
    if not cuts:
        cuts = [
            m.start()
            for m in re.finditer(r"\s", sentence)
            if 0.3 * len(sentence) < m.start() < 0.7 * len(sentence)
        ]
    if not cuts:
        return [sentence]
    cut = min(cuts, key=lambda c: abs(c - middle))
    left, right = sentence[:cut].strip(), sentence[cut:].strip()
    return _split_long(left, max_chars) + _split_long(right, max_chars)


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Greedily pack whole sentences into chunks of at most `max_chars` characters."""
    pieces: list[str] = []
    for sentence in split_sentences(text):
        pieces.extend(_split_long(sentence, max_chars))
    chunks: list[str] = []
    for piece in pieces:
        if chunks and len(chunks[-1]) + 1 + len(piece) <= max_chars:
            chunks[-1] = f"{chunks[-1]} {piece}"
        else:
            chunks.append(piece)
    return chunks
