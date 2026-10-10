"""What the voice should *say*: quote marks stripped, numbers and abbreviations spelled out,
chapter numerals read as words, shouted ALL-CAPS sentences calmed so they aren't spelled letter by letter."""

from __future__ import annotations

import re

from num2words import num2words

_DOUBLE_QUOTES = re.compile(r"[“”\"«»„「」『』‹›]")
_EDGE_SINGLE = re.compile(r"^[‘']|[’']$")
# "[12]" or a note number glued to a word: "carriage.3 The" (but not "5.50" or "B12").
_FOOTNOTE = re.compile(r"\[\d{1,3}\]|(?<=[a-z][.,;!?])\d{1,2}(?=\s|$)|(?<=[a-z]{2})\d{1,2}(?=[\s.,;!?]|$)")
_URL = re.compile(r"https?://\S+|www\.\S+")
_MARKUP = re.compile(r"[*_#~`|<>]+")
_ROMAN = re.compile(r"^(?=[IVXLCDM]+$)M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")

_ABBREVIATIONS_EN = [
    (re.compile(r"\bMr\.(?=\s)"), "Mister"),
    (re.compile(r"\bMrs\.(?=\s)"), "Missus"),
    (re.compile(r"\bMs\.(?=\s)"), "Miz"),
    (re.compile(r"\bDr\.(?=\s+[A-Z])"), "Doctor"),
    (re.compile(r"\bSt\.(?=\s+[A-Z])"), "Saint"),
    (re.compile(r"\bProf\.(?=\s+[A-Z])"), "Professor"),
    (re.compile(r"\be\.g\.", re.IGNORECASE), "for example"),
    (re.compile(r"\bi\.e\.", re.IGNORECASE), "that is"),
    (re.compile(r"\betc\.", re.IGNORECASE), "et cetera"),
    (re.compile(r"\bvs\.", re.IGNORECASE), "versus"),
    (re.compile(r"\bNo\.(?=\s*\d)"), "number"),
    (re.compile(r"&"), " and "),
]
_CURRENCY = {"$": ("dollar", "dollars", "cent", "cents"), "£": ("pound", "pounds", "penny", "pence"),
             "€": ("euro", "euros", "cent", "cents")}  # fmt: skip


def roman_to_int(numeral: str) -> int | None:
    numeral = numeral.upper()
    if not numeral or not _ROMAN.match(numeral):
        return None
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    for i, ch in enumerate(numeral):
        value = values[ch]
        total += -value if i + 1 < len(numeral) and values[numeral[i + 1]] > value else value
    return total


def number_words(n: int | float, language: str = "en", to: str = "cardinal") -> str:
    try:
        return num2words(n, lang=language, to=to).replace(",", "")
    except (NotImplementedError, OverflowError, ValueError, TypeError):
        return str(n)


def _money(match: re.Match[str]) -> str:
    one, many, sub_one, sub_many = _CURRENCY[match["sym"]]
    whole = int(match["whole"].replace(",", ""))
    words = f"{number_words(whole)} {one if whole == 1 else many}"
    if match["cents"] and int(match["cents"]):
        cents = int(match["cents"])
        words += f" and {number_words(cents)} {sub_one if cents == 1 else sub_many}"
    return words


def _time(match: re.Match[str]) -> str:
    hour, minute = int(match[1]), int(match[2])
    if hour > 24 or minute > 59:
        return match[0]
    if minute == 0:
        return f"{number_words(hour)} o'clock"
    minute_words = f"oh {number_words(minute)}" if minute < 10 else number_words(minute)
    return f"{number_words(hour)} {minute_words}"


def _decade(match: re.Match[str]) -> str:
    """ "1920s" -> "nineteen twenties"."""
    words = number_words(int(match[1]), to="year")
    return words[:-1] + "ies" if words.endswith("y") else words + "s"


def _numbers_en(text: str) -> str:
    text = re.sub(r"(?P<sym>[$£€])(?P<whole>\d[\d,]*)(?:\.(?P<cents>\d{2}))?\b", _money, text)
    text = re.sub(
        r"\b(\d+(?:\.\d+)?)\s?%",
        lambda m: f"{number_words(float(m[1]) if '.' in m[1] else int(m[1]))} percent",
        text,
    )
    text = re.sub(r"\b(\d{1,2}):(\d{2})\b", _time, text)
    text = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", lambda m: number_words(int(m[1]), to="ordinal"), text)
    text = re.sub(r"\b\d{1,3}(?:,\d{3})+\b", lambda m: number_words(int(m[0].replace(",", ""))), text)
    text = re.sub(r"\b\d+\.\d+\b", lambda m: number_words(float(m[0])), text)
    text = re.sub(r"\b(1[1-9]\d\d|20\d\d)s\b", _decade, text)
    text = re.sub(r"\b(1[1-9]\d\d|20\d\d)\b", lambda m: number_words(int(m[1]), to="year"), text)
    return re.sub(r"\b\d+\b", lambda m: number_words(int(m[0])), text)


# Thousands separators by language ("1.500" in German, "5 000" in French, "1,00,000" in Hindi).
_DOT_GROUPS = {"de", "es", "it", "pt", "nl", "da", "tr", "id", "el", "ro", "hr", "sl", "sr", "is", "vi"}
_SPACE_GROUPS = {"fr", "ru", "pl", "sv", "no", "nb", "fi", "cs", "sk", "uk", "bg", "hu", "et", "lv", "lt"}


def _numbers_other(text: str, language: str) -> str:
    if language in _DOT_GROUPS:
        grouped = r"\b\d{1,3}(?:\.\d{3})+\b"
    elif language in _SPACE_GROUPS:
        grouped = r"\b\d{1,3}(?:[ \u00a0\u202f]\d{3})+\b"
    else:
        grouped = r"\b\d{1,3}(?:,\d{2,3})+\b"
    text = re.sub(grouped, lambda m: number_words(int(re.sub(r"\D", "", m[0])), language), text)
    # Plain integers only: decimals ("3,5", "2.75") are left for the TTS model to read.
    return re.sub(r"(?<![\d.,])\d+(?![.,]?\d)", lambda m: number_words(int(m[0]), language), text)


def _calm_caps(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    if len(text.split()) < 3 or not letters or sum(c.isupper() for c in letters) / len(letters) < 0.8:
        return text
    lowered = text.lower()
    lowered = re.sub(r"(^|[.!?…]\s+)(\w)", lambda m: m[1] + m[2].upper(), lowered)
    return re.sub(r"\bi\b", "I", lowered)


def speak_heading(text: str, language: str = "en") -> str:
    """ "CHAPTER IV" -> "Chapter Four."; "12" -> "Chapter Twelve."; "Chapter One: The Letter" -> "Chapter One. The Letter." """
    text = re.sub(r"\s+", " ", text).strip()
    letters = [c for c in text if c.isalpha()]
    if letters and all(c.isupper() for c in letters):
        text = " ".join(w if _ROMAN.match(w.strip(".:")) else w.capitalize() for w in text.split(" "))

    def numeral(token: str) -> str | None:
        token = token.strip(".:")
        value = int(token) if token.isdigit() else roman_to_int(token)
        return number_words(value, language).title() if value is not None else None

    words = text.split(" ")
    if len(words) == 1 and numeral(words[0]):
        text = f"Chapter {numeral(words[0])}" if language == "en" else numeral(words[0]) or text
    elif len(words) >= 2 and numeral(words[1]) and re.match(r"(?i)^(chapter|part|book|volume)$", words[0]):
        text = " ".join([words[0].capitalize(), numeral(words[1]) or words[1], *words[2:]])
    text = text.replace(":", ".")
    return text if text.endswith((".", "!", "?")) else f"{text}."


def normalize_for_tts(text: str, language: str = "en") -> str:
    text = _URL.sub("a link", text)
    text = _FOOTNOTE.sub("", text)
    text = _DOUBLE_QUOTES.sub("", text)
    text = _EDGE_SINGLE.sub("", text.strip())
    text = _MARKUP.sub(" ", text)
    text = text.replace("--", "—")
    if language == "en":
        for pattern, replacement in _ABBREVIATIONS_EN:
            text = pattern.sub(replacement, text)
        text = _numbers_en(text)
    else:
        text = _numbers_other(text, language)
    text = _calm_caps(text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()
