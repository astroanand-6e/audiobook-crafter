"""Split paragraphs into narration and quoted dialogue.

Handles “curly”, "straight", ‘single’ (British), «guillemets», »reversed«, „German“, 「CJK」 quotes
and em-dash dialogue (—Hola —dijo Juan—.). The splitter is deterministic so the LLM never has to
copy book text back, only label numbered spans.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Literal

QuoteStyle = Literal["double", "single", "guillemets", "reversed_guillemets", "german", "cjk", "dash"]

_PAIRS: dict[str, dict[str, str]] = {
    "double": {"“": '”"', '"': '"”'},
    "single": {"‘": "’"},
    "guillemets": {"«": "»"},
    "reversed_guillemets": {"»": "«"},
    "german": {"„": "“”"},
    "cjk": {"「": "」", "『": "』"},
}
_DASHES = "—–―"
_OPENING_CONTEXT = set(" \t\n([{—–-\"“'«„")
_CLOSING_CONTEXT = set(' \t\n,.;:!?)]}—–-"”»…')


@dataclass
class Span:
    kind: Literal["narration", "dialogue"]
    text: str
    open_end: bool = False  # dialogue still open at the end of the paragraph (continues in the next)


def detect_quote_style(paragraphs: list[str]) -> QuoteStyle:
    counts: Counter[str] = Counter()
    for text in paragraphs[:400]:
        stripped = text.lstrip()
        if stripped[:1] in _DASHES:
            counts["dash"] += 1
        counts["double"] += text.count("“") + text.count('"') // 2
        counts["single"] += len(re.findall(r"(?:^|[\s(—-])‘\w", text))
        counts["german"] += text.count("„")
        counts["cjk"] += text.count("「") + text.count("『")
        guillemets = text.count("«")
        if guillemets:
            first_open, first_close = text.find("«"), text.find("»")
            reversed_style = first_close != -1 and first_close < first_open
            counts["reversed_guillemets" if reversed_style else "guillemets"] += guillemets
    if not counts or max(counts.values()) == 0:
        return "double"
    return counts.most_common(1)[0][0]  # type: ignore[return-value]


def _can_open(text: str, i: int, style: QuoteStyle) -> bool:
    if style != "single":
        return True
    before = text[i - 1] if i else " "
    after = text[i + 1] if i + 1 < len(text) else ""
    return before in _OPENING_CONTEXT and after.strip() != ""


def _can_close(text: str, i: int, style: QuoteStyle) -> bool:
    if style != "single":
        return True
    before = text[i - 1] if i else ""
    after = text[i + 1] if i + 1 < len(text) else " "
    return before.strip() != "" and after in _CLOSING_CONTEXT


def _has_words(text: str) -> bool:
    return any(ch.isalnum() for ch in text)


def _split_dash(text: str) -> list[Span]:
    stripped = text.strip()
    if stripped[:1] not in _DASHES:
        return [Span("narration", stripped)]
    parts = [p.strip(" ,") for p in re.split(rf"\s*[{_DASHES}]\s*", stripped)]
    spans: list[Span] = []
    for n, part in enumerate(parts[1:]):
        part = part.lstrip(".,;: ")
        if _has_words(part):
            spans.append(Span("dialogue" if n % 2 == 0 else "narration", part))
    return spans


def split_dialogue(text: str, style: QuoteStyle = "double", continues: bool = False) -> list[Span]:
    """Narration and dialogue spans in reading order. `continues`: the paragraph starts inside an
    open quote (some books don't repeat the opening mark)."""
    if style == "dash":
        return _split_dash(text)
    pairs = _PAIRS[style]
    spans: list[Span] = []
    buffer: list[str] = []
    closers = next(iter(pairs.values())) if continues else ""
    inside = continues

    def flush(kind: Literal["narration", "dialogue"], open_end: bool = False) -> None:
        chunk = "".join(buffer).strip()
        buffer.clear()
        if _has_words(chunk):
            spans.append(Span(kind, chunk, open_end))

    for i, ch in enumerate(text):
        if not inside:
            if ch in pairs and _can_open(text, i, style):
                flush("narration")
                inside, closers = True, pairs[ch]
            buffer.append(ch)
        else:
            buffer.append(ch)
            if ch in closers and _can_close(text, i, style):
                flush("dialogue")
                inside = False
    flush("dialogue" if inside else "narration", open_end=inside)
    return spans


_SPEECH_VERBS = (
    "said|says|asked|asks|replied|replies|answered|cried|shouted|yelled|whispered|murmured|muttered|called|"
    "exclaimed|added|continued|began|repeated|insisted|demanded|snapped|snarled|growled|hissed|sighed|laughed|"
    "sobbed|gasped|breathed|told|warned|agreed|admitted|announced|declared|protested|pleaded|begged|suggested|"
    "observed|remarked|retorted|responded|interrupted|stammered|stuttered|groaned|moaned|screamed|roared|"
    "bellowed|chuckled|giggled|teased|explained|offered|went on|whimpered|wailed|grumbled|mumbled|purred|"
    "returned|rejoined|resumed|persisted|urged|inquired|enquired|ventured|concluded|corrected|conceded"
)
SPEECH_VERB_ALTERNATION = _SPEECH_VERBS
SPEECH_VERBS = re.compile(rf"\b(?:{_SPEECH_VERBS})\b", re.IGNORECASE)
_NAME = r"(?:the\s+)?(?:[A-Z][\w'’.-]*)(?:\s+[A-Z][\w'’.-]*){0,2}"
_SUBJECT = rf"(?:he|she|they|i|we|it|{_NAME})"
_TAG_ONLY = re.compile(
    rf"^[\s,;:—–-]*(?:{_SUBJECT}\s+(?:\w+ly\s+)?(?:{_SPEECH_VERBS})|(?:{_SPEECH_VERBS})\s+{_SUBJECT})"
    rf"(?:\s+\w+ly)?(?:\s+(?:softly|quietly|loudly|again|at last|finally))?[\s,.;:!?—–-]*$",
    re.IGNORECASE,
)


def is_speech_tag(narration: str) -> bool:
    """True for bare dialogue tags like "she said." / ", said Mr. Darcy," that drama mode can drop."""
    text = narration.strip()
    return bool(text) and len(text.split()) <= 7 and bool(_TAG_ONLY.match(text))
