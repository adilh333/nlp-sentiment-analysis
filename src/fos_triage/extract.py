"""Build an intake-style text from an ombudsman decision.

A published decision is written after the outcome is known. The sections that
state the decision ("What I've decided", "My final decision", "Putting things
right") are dropped. Within the complaint and background, sentences that give
away the ombudsman's or investigator's conclusion are dropped too.

What remains is a proxy for the narrative a firm's complaints team has at
intake: what the customer says went wrong, and the factual background. It is
still the ombudsman's paraphrase, not the customer's original email.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_QUOTE_MAP = str.maketrans({"’": "'", "‘": "'", "–": "-", "—": "-", "\u00a0": " "})

_DECISION_HEADINGS = (
    "what i've decided",
    "what i have decided",
    "my final decision",
    "my decision",
    "putting things right",
    "my findings",
    "what i provisionally decided",
    "my provisional decision",
    "provisional decision",
    "how to put things right",
    "fair compensation to put things right",
    "my provisional conclusions",
    "provisional conclusions",
)

_BACKGROUND_HEADINGS = {
    "what happened": "background",
    "what happened next": "background",
    "background": "background",
    "the background": "background",
    "background to the complaint": "background",
    "the background to the complaint": "background",
    "circumstances": "background",
    "the circumstances": "background",
}

_COMPLAINT_HEADINGS = {
    "the complaint": "complaint",
    "complaint": "complaint",
    "what the complaint is about": "complaint",
    "summary of the complaint": "complaint",
}

# Sentences that state a view on the merits. "Uphold" is included because it
# is the label. A customer's request for a refund does not need that word.
_LEAK_SENTENCE = re.compile(
    r"(?i)("
    r"\binvestigators?\b"
    r"|\bour service\b"
    r"|\bthis service\b"
    r"|referred(?: the complaint)? to me"
    r"|i(?:'ve| have) decided"
    r"|\b(?:upheld|upholding|upholds|uphold)\b"
    r"|\badjudicators?\b"
    r"|provisional decision"
    r"|put things right"
    r"|fair and reasonable"
    r"|\bfinal decision\b"
    r"|putting things right"
    r")"
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_DRN_LINE = re.compile(r"^DRN-?\d+$", re.I)
_PAGE_LINE = re.compile(r"^(?:page\s+)?\d+(\s+of\s+\d+)?$", re.I)


@dataclass
class IntakeText:
    complaint: str
    background: str
    model_text: str
    sentences_removed: int = 0
    used_headings: bool = False
    leak_flags: list[str] = field(default_factory=list)

    @property
    def n_chars(self) -> int:
        return len(self.model_text)


def _canon(line: str) -> str:
    text = line.translate(_QUOTE_MAP).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .:-")
    return text


def heading_kind(line: str) -> str | None:
    key = _canon(line)
    if not key or len(key) > 80:
        return None
    if key in _COMPLAINT_HEADINGS:
        return "complaint"
    if key in _BACKGROUND_HEADINGS:
        return "background"
    for prefix in _DECISION_HEADINGS:
        if key == prefix or key.startswith(prefix):
            return "decision"
    return None


# Only break a heading out when it follows a sentence boundary. Mid-sentence
# phrases such as "referred the complaint" must stay in the body.
_INLINE_HEADING = re.compile(
    r"(?i)(?<=[.!?])\s*(?=(?:"
    r"the complaint|what the complaint is about|summary of the complaint|"
    r"what happened|background to the complaint|"
    r"what i(?:'ve| have) decided|my final decision|putting things right|"
    r"my findings|my provisional decision|my provisional conclusions"
    r")\b)"
)


def _prepare_lines(raw: str) -> list[str]:
    text = raw.translate(_QUOTE_MAP).replace("\r\n", "\n").replace("\r", "\n")
    text = _INLINE_HEADING.sub("\n", text)
    lines: list[str] = []
    for line in text.split("\n"):
        cleaned = re.sub(r"\s+", " ", line).strip()
        if not cleaned:
            lines.append("")
            continue
        if _DRN_LINE.match(cleaned) or _PAGE_LINE.match(cleaned):
            continue
        if cleaned.lower() in {"financial ombudsman service", "ombudsman decision"}:
            continue
        lines.append(cleaned)
    return lines


def _paragraphs(lines: list[str]) -> list[tuple[str, str]]:
    """Return (kind, text) where kind is 'heading' or 'body'."""
    blocks: list[tuple[str, str]] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            blocks.append(("body", " ".join(buffer)))
            buffer.clear()

    for line in lines:
        if line == "":
            flush()
            continue
        kind = heading_kind(line)
        if kind:
            flush()
            blocks.append(("heading", kind))
            continue
        buffer.append(line)
    flush()
    return blocks


def _filter_sentences(text: str) -> tuple[str, int]:
    if not text.strip():
        return "", 0
    parts = [part.strip() for part in _SENTENCE_SPLIT.split(text.strip()) if part.strip()]
    kept: list[str] = []
    for index, sentence in enumerate(parts):
        if _LEAK_SENTENCE.search(sentence):
            return " ".join(kept), len(parts) - index
        kept.append(sentence)
    return " ".join(kept), 0


def _sections_from_blocks(blocks: list[tuple[str, str]]) -> tuple[str, str, int, bool]:
    section = None
    complaint: list[str] = []
    background: list[str] = []
    removed = 0
    saw_heading = False
    # A sentence that states the investigator's view ends that section.
    # Later paragraphs in the same section are part of the same discussion.
    section_closed = False
    for kind, value in blocks:
        if kind == "heading":
            saw_heading = True
            if value == "decision":
                break
            section = value
            section_closed = False
            continue
        if section is None or section_closed:
            continue
        filtered, n_removed = _filter_sentences(value)
        removed += n_removed
        if n_removed:
            section_closed = True
        if not filtered:
            continue
        if section == "complaint":
            complaint.append(filtered)
        elif section == "background":
            background.append(filtered)
    return " ".join(complaint).strip(), " ".join(background).strip(), removed, saw_heading


def _fallback_cut(raw: str) -> tuple[str, str, int]:
    """Used when the PDF text has no recoverable headings."""
    flat = re.sub(r"\s+", " ", raw.translate(_QUOTE_MAP)).strip()
    cut = re.split(
        r"(?i)\b(?:what i(?:'ve| have) decided|my final decision|putting things right)\b",
        flat,
        maxsplit=1,
    )[0]
    # Prefer the span after "the complaint" if that phrase exists.
    match = re.search(r"(?i)\bthe complaint\b", cut)
    if match:
        cut = cut[match.end():]
    background = ""
    complaint = cut
    bg = re.split(r"(?i)\bwhat happened\b|\bbackground to the complaint\b", complaint, maxsplit=1)
    if len(bg) == 2:
        complaint, background = bg
    complaint, n1 = _filter_sentences(complaint)
    background, n2 = _filter_sentences(background)
    return complaint.strip(), background.strip(), n1 + n2


def residual_leak_flags(text: str) -> list[str]:
    flags = []
    checks = {
        "uphold": r"(?i)\b(?:upheld|upholding|upholds|uphold)\b",
        "adjudicator": r"(?i)\badjudicators?\b",
        "provisional_decision": r"(?i)provisional decision",
        "put_things_right": r"(?i)put things right",
        "investigator": r"(?i)\binvestigators?\b",
        "decided": r"(?i)\bi(?:'ve| have) decided\b",
        "final_decision": r"(?i)\bfinal decision\b",
    }
    for name, pattern in checks.items():
        if re.search(pattern, text):
            flags.append(name)
    return flags


def intake_from_text(raw: str) -> IntakeText:
    if not raw or not raw.strip():
        return IntakeText("", "", "")
    lines = _prepare_lines(raw)
    blocks = _paragraphs(lines)
    complaint, background, removed, saw = _sections_from_blocks(blocks)
    if len(complaint) < 40:
        fb_complaint, fb_background, fb_removed = _fallback_cut(raw)
        if len(fb_complaint) > len(complaint):
            complaint, background, removed, saw = fb_complaint, fb_background, fb_removed, False
    model = complaint
    if background:
        model = f"{complaint}\n\n{background}".strip()
    flags = residual_leak_flags(model)
    return IntakeText(
        complaint=complaint,
        background=background,
        model_text=model,
        sentences_removed=removed,
        used_headings=saw,
        leak_flags=flags,
    )
