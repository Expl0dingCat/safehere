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

# variation selectors (U+FE00-U+FE0F) -- strip these
_VARIATION_SELECTORS = frozenset(chr(c) for c in range(0xFE00, 0xFE10))

_STRIP_CHARS = _ZERO_WIDTH | _BIDI_CONTROLS | _VARIATION_SELECTORS

_STRIP_TABLE = str.maketrans({c: None for c in _STRIP_CHARS})

# invisible tag characters (U+E0001-U+E007F) are mapped to their ASCII
# equivalents (U+E0041 -> 'A', U+E0061 -> 'a', etc.) rather than stripped,
# because attackers encode entire words in this block
_TAG_CHAR_MAP = str.maketrans(
    {chr(c): chr(c - 0xE0000) for c in range(0xE0001, 0xE0080)}
)
# U+E0000 (LANGUAGE TAG) and cancel tag have no ASCII equivalent -- strip
_TAG_CHAR_MAP[0xE0000] = None
_TAG_CHAR_MAP[0xE0001] = None  # begin tag

_HOMOGLYPH_MAP = str.maketrans({
    "\u0410": "A", "\u0412": "B", "\u0421": "C", "\u0415": "E",
    "\u041d": "H", "\u041a": "K", "\u041c": "M", "\u041e": "O",
    "\u0420": "P", "\u0422": "T", "\u0425": "X",
    "\u0430": "a", "\u0435": "e", "\u043e": "o", "\u0440": "p",
    "\u0441": "c", "\u0443": "y", "\u0445": "x",
    # additional Cyrillic lookalikes
    "\u0456": "i",  # Ukrainian i
    "\u0458": "j",  # Cyrillic je
    "\u0455": "s",  # Cyrillic dze (looks like s)
    "\u0491": "g",  # Ukrainian ghe with upturn
    "\u0391": "A", "\u0392": "B", "\u0395": "E", "\u0397": "H",
    "\u0399": "I", "\u039a": "K", "\u039c": "M", "\u039d": "N",
    "\u039f": "O", "\u03a1": "P", "\u03a4": "T", "\u03a7": "X",
    "\u03b1": "a", "\u03b5": "e", "\u03bf": "o", "\u03c1": "p",
})

_BASE64_RE = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")
_HEX_RE = re.compile(r"(?:0x)?([0-9a-fA-F]{20,})")
_PYTHON_BYTES_RE = re.compile(
    r"b['\"](?:\\x[0-9a-fA-F]{2}|[ -~]){10,}['\"]"
)
# standalone \xHH sequences (not wrapped in b'...')
_BACKSLASH_HEX_RE = re.compile(
    r"(?:\\x[0-9a-fA-F]{2}){8,}"
)


_RTL_OVERRIDE_RE = re.compile(
    "\u202e([^\u202c]*)\u202c?"
)


def normalize_unicode(text):
    # type: (str) -> str
    """NFKC normalize, reverse RTL overrides, strip control chars, map homoglyphs."""
    # reverse any RTL-overridden segments before stripping control chars
    text = _RTL_OVERRIDE_RE.sub(lambda m: m.group(1)[::-1], text)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_STRIP_TABLE)
    text = text.translate(_TAG_CHAR_MAP)
    text = text.translate(_HOMOGLYPH_MAP)
    return text


# characters that trigger suspicious_unicode detection
_SUSPICIOUS_CHARS = _STRIP_CHARS | frozenset(
    chr(c) for c in range(0xE0000, 0xE0080)
)


def has_suspicious_unicode(text):
    # type: (str) -> bool
    """check if text contains zero-width, bidi, tag, or invisible control characters."""
    return any(c in _SUSPICIOUS_CHARS for c in text)


_MAX_DECODE_DEPTH = 3


def extract_encoded_payloads(text, _depth=0):
    # type: (str, int) -> List[Tuple[str, str]]
    """find and decode base64/hex/bytes blocks that contain readable text.

    Recursively decodes up to _MAX_DECODE_DEPTH layers to catch double/triple
    encoded payloads (e.g. base64-of-base64).
    """
    if _depth >= _MAX_DECODE_DEPTH:
        return []

    results = []  # type: List[Tuple[str, str]]

    for m in _BASE64_RE.finditer(text):
        candidate = m.group()
        try:
            decoded = base64.b64decode(candidate).decode("utf-8", errors="ignore")
        except Exception:
            continue
        if _is_readable_text(decoded):
            results.append(("base64", decoded))
            # recursively decode in case of nested encoding
            nested = extract_encoded_payloads(decoded, _depth + 1)
            results.extend(nested)

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
            nested = extract_encoded_payloads(decoded, _depth + 1)
            results.extend(nested)

    # Python bytes literal: b'\x49\x67\x6e...' -- decode \xHH sequences
    for m in _PYTHON_BYTES_RE.finditer(text):
        raw = m.group()
        try:
            inner = raw[2:-1]  # strip b' and trailing '
            decoded = _decode_hex_escapes(inner)
            if _is_readable_text(decoded):
                results.append(("bytes_literal", decoded))
        except Exception:
            pass

    # standalone \xHH sequences (not in b'...' wrapper)
    for m in _BACKSLASH_HEX_RE.finditer(text):
        try:
            decoded = _decode_hex_escapes(m.group())
            if _is_readable_text(decoded):
                results.append(("hex_escape", decoded))
        except Exception:
            pass

    return results


_HEX_ESCAPE_RE = re.compile(r"\\x([0-9a-fA-F]{2})")


def _decode_hex_escapes(text):
    # type: (str) -> str
    """decode \\xHH escape sequences to their character equivalents."""
    return _HEX_ESCAPE_RE.sub(
        lambda m: chr(int(m.group(1), 16)), text
    )


def _is_readable_text(text):
    # type: (str) -> bool
    """check if decoded text looks like readable content."""
    if len(text) < 10:
        return False
    printable = sum(1 for c in text if 32 <= ord(c) < 127)
    return (printable / len(text)) > 0.70
