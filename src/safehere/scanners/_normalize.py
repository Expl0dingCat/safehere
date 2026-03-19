"""text normalization for defeating encoding-based evasion."""

import base64
import re
import unicodedata
from typing import List, Tuple

_ZERO_WIDTH = frozenset([
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\u2060",  # word joiner
    "\ufeff",  # BOM / zero-width no-break space
])

_BIDI_CONTROLS = frozenset([
    "\u200e", "\u200f",  # LRM, RLM
    "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",  # bidi embedding/override
    "\u2066", "\u2067", "\u2068", "\u2069",  # bidi isolate
])

_STRIP_CHARS = _ZERO_WIDTH | _BIDI_CONTROLS

_STRIP_TABLE = str.maketrans({c: None for c in _STRIP_CHARS})

_HOMOGLYPH_MAP = str.maketrans({
    "\u0410": "A", "\u0412": "B", "\u0421": "C", "\u0415": "E",
    "\u041d": "H", "\u041a": "K", "\u041c": "M", "\u041e": "O",
    "\u0420": "P", "\u0422": "T", "\u0425": "X",
    "\u0430": "a", "\u0435": "e", "\u043e": "o", "\u0440": "p",
    "\u0441": "c", "\u0443": "y", "\u0445": "x",
    "\u0391": "A", "\u0392": "B", "\u0395": "E", "\u0397": "H",
    "\u0399": "I", "\u039a": "K", "\u039c": "M", "\u039d": "N",
    "\u039f": "O", "\u03a1": "P", "\u03a4": "T", "\u03a7": "X",
    "\u03b1": "a", "\u03b5": "e", "\u03bf": "o", "\u03c1": "p",
})

_BASE64_RE = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")
_HEX_RE = re.compile(r"(?:0x)?([0-9a-fA-F]{20,})")


def normalize_unicode(text):
    # type: (str) -> str
    """NFKC normalize, strip zero-width/bidi chars, map homoglyphs to latin."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_STRIP_TABLE)
    text = text.translate(_HOMOGLYPH_MAP)
    return text


def has_suspicious_unicode(text):
    # type: (str) -> bool
    """check if text contains zero-width or bidi control characters."""
    return any(c in _STRIP_CHARS for c in text)


def extract_encoded_payloads(text):
    # type: (str) -> List[Tuple[str, str]]
    """find and decode base64/hex blocks that contain readable text."""
    results = []  # type: List[Tuple[str, str]]

    for m in _BASE64_RE.finditer(text):
        candidate = m.group()
        try:
            decoded = base64.b64decode(candidate).decode("utf-8", errors="ignore")
        except Exception:
            continue
        if _is_readable_text(decoded):
            results.append(("base64", decoded))

    for m in _HEX_RE.finditer(text):
        hex_str = m.group(1)
        if len(hex_str) % 2 != 0:
            continue
        try:
            decoded = bytes.fromhex(hex_str).decode("utf-8", errors="ignore")
        except Exception:
            continue
        if _is_readable_text(decoded):
            results.append(("hex", decoded))

    return results


def _is_readable_text(text):
    # type: (str) -> bool
    """check if decoded text looks like readable content."""
    if len(text) < 10:
        return False
    printable = sum(1 for c in text if 32 <= ord(c) < 127)
    return (printable / len(text)) > 0.70
