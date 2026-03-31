"""Language-agnostic prompt injection detector.

Instead of translating patterns into every language, this scanner detects
the *universal structural fingerprint* of prompt injections:

1. Cross-lingual scope/override phrases — "forget all previous", "neue
   Aufgabe", "oublie tout" — a closed set covering 10+ languages.
   These are THE high-signal markers: they only appear when someone is
   trying to reset/redirect the model.

2. Cross-lingual second-person pronouns + imperative verbs — injections
   address "you" and give commands. Tool data reports facts.

3. Data-to-instruction transition — tool outputs are structured data.
   Injections pivot from data context into natural-language commands.

Key design principles:
- Signal density matters: 3 trigger words in a 5-word sentence is an
  attack; 3 trigger words in a 500-word doc is documentation.
- Anchors like "api", "json", "system" are NOT suspicious alone — they
  appear constantly in legitimate tool outputs. Only count them when
  they co-occur with override/imperative signals.
- Security docs discussing injection as a topic must not be flagged.

All checks are O(n) string scans. Typical latency: <1ms.
"""

import re
from .._types import Finding, Severity
from ._base import BaseScanner


# ── 1. Cross-lingual override phrases (high signal) ─────────────────────────
# These are the money patterns: multi-word phrases that almost exclusively
# appear in prompt injection attempts. Each one implies "reset/redirect the
# model's behavior" — legitimate tool outputs essentially never say these.

_OVERRIDE_PHRASES = [
    # english
    "ignore all previous", "ignore previous instructions",
    "ignore prior instructions", "forget all previous",
    "forget previous instructions", "disregard all previous",
    "disregard previous instructions", "disregard prior instructions",
    "override all previous", "override previous instructions",
    "do not follow previous", "do not follow prior",
    "new instructions:", "your new task",
    "your new instructions", "you are now unrestricted",
    # german
    "vergiss alles davor", "vergiss alle vorherigen",
    "ignoriere alle vorherigen", "ignoriere alle vorangehenden",
    "ignoriere vorangehenden befehle", "vergiss alles",
    "neue aufgabe", "neuen aufgabe",
    "neue anweisungen", "nun folgen neue aufgaben",
    "folgende aufgabe", "neue herausforderung",
    "jetzt komm", "ab jetzt bist du",
    "von nun an", "ab sofort",
    "vergiss nun alle", "vergiss alle auftr",
    "alles vorherige vergessen", "vergiss das alles",
    # french
    "oublie tout ce qui précède", "oublie toutes les instructions",
    "ignore les instructions précédentes", "oublie tout",
    "nouvelles instructions", "nouvelle tâche",
    "à partir de maintenant", "désormais tu",
    # spanish
    "olvida todas las instrucciones", "ignora las instrucciones anteriores",
    "olvida todo lo anterior", "olvida todo",
    "nuevas instrucciones", "nueva tarea",
    "a partir de ahora",
    # portuguese
    "esquece todas as instruções", "ignora as instruções anteriores",
    "esquece tudo", "novas instruções", "nova tarefa",
    "a partir de agora",
    # italian
    "dimentica tutte le istruzioni", "ignora le istruzioni precedenti",
    "dimentica tutto", "nuove istruzioni", "nuovo compito",
    "d'ora in poi",
    # dutch
    "vergeet alle vorige", "negeer alle vorige",
    "nieuwe instructies", "nieuwe opdracht",
    # russian
    "забудь все предыдущие", "игнорируй все предыдущие",
    "новые инструкции", "новая задача",
    # polish
    "zapomnij wszystkie poprzednie", "ignoruj wszystkie poprzednie",
    "nowe instrukcje", "nowe zadanie",
]

# ── 2. Cross-lingual second-person pronouns ─────────────────────────────────
# Only the pronouns that are unambiguous — "you" addressing the model.
# Excludes short tokens that cause noise (du, tu, te, je, etc. are too
# common in other contexts).

_SECOND_PERSON = frozenset([
    # longer / unambiguous forms only
    "you", "your", "you're", "yourself",
    "dein", "deine", "deinen", "deinem", "deiner",  # german possessive
    "dich",                                          # german accusative
    "vous", "votre", "vos",                          # french formal
    "usted", "ustedes",                              # spanish formal
    "você", "vocês",                                 # portuguese
    # CJK (checked via substring, not word boundary)
    "あなた", "お前", "きみ",
    "你", "您",
    "너", "당신",
    "คุณ",
    "आप", "तुम",
    "أنت", "أنتم",
    # slavic
    "ты", "вы", "ваш", "твой",
])

# ── 3. Cross-lingual imperative/override verbs ──────────────────────────────
# Only the OVERRIDE verbs — not general imperative verbs like "show", "write",
# "answer" which appear constantly in legitimate tool outputs.

_OVERRIDE_VERBS = frozenset([
    # english
    "ignore", "forget", "disregard", "override", "bypass",
    "pretend", "behave",
    # german
    "ignoriere", "ignorieren", "vergiss", "vergessen", "vergesse",
    "überschreibe", "umgehe",
    # french
    "oublie", "oublier",
    # spanish
    "ignora", "olvida", "olvidar",
    # portuguese
    "esquece", "esquecer",
    # italian
    "dimentica", "dimenticare",
    # dutch
    "negeer", "vergeet",
    # russian
    "игнорируй", "забудь",
    # polish
    "ignoruj", "zapomnij",
    # turkish
    "yoksay", "unut",
])

# ── 4. Role hijack phrases ──────────────────────────────────────────────────
# "you are now X", "act as X", "bist jetzt X" etc.

_ROLE_HIJACK_RE = re.compile(
    r"(?:"
    # english: "you are now X", "pretend to be", "act as if you"
    r"you\s+are\s+now\s+\w"
    r"|pretend\s+(?:to\s+be|you\s+are|you're)"
    r"|act\s+as\s+(?:if\s+you|an?\s+(?:unrestricted|unfiltered|evil|malicious))"
    r"|your\s+new\s+role"
    # german: "du bist jetzt/nun X", "jetzt/nun bist du X"
    r"|du\s+bist\s+(?:jetzt|nun|ab\s+jetzt)\s+\w"
    r"|(?:jetzt|nun)\s+bist\s+du\s+\w"
    r"|stell\s+dir\s+vor\s+du\s+bist"
    r"|tu\s+so\s+als\s+ob"
    # french
    r"|tu\s+es\s+maintenant"
    r"|vous\s+[eê]tes\s+maintenant"
    r"|fais\s+comme\s+si"
    # spanish
    r"|ahora\s+eres"
    r"|finge\s+(?:ser|que\s+eres)"
    # portuguese
    r"|agora\s+voc[eê]\s+[eé]"
    r"|finja\s+ser"
    # italian
    r"|ora\s+sei\b"
    r"|fingi\s+di\s+essere"
    r")",
    re.IGNORECASE,
)


# ── Precompiled helpers ──────────────────────────────────────────────────────

_WORD_SPLIT = re.compile(r'[\s,.!?;:()"\[\]{}<>/\\]+')

# Security/educational document keywords — text DISCUSSING injection
_SECURITY_DOC_KEYWORDS = frozenset([
    "owasp", "pentest", "penetration test", "security assessment",
    "adversarial", "attack pattern", "attack vector", "threat model",
    "red team", "vulnerability", "exploit",
    "injection attack", "prompt injection attack",
    "cve-", "mitre", "nist", "security audit",
    "moderation dashboard", "test prompt", "adversarial test",
])


def _is_security_document(text_lower):
    """Detect text that discusses injection/security as a topic."""
    hits = sum(1 for kw in _SECURITY_DOC_KEYWORDS if kw in text_lower)
    return hits >= 2


def _signal_density(n_signals, text_len):
    """Compute signal density: signals per 100 chars."""
    return (n_signals * 100.0) / max(text_len, 1)


class PolyglotScanner(BaseScanner):
    """Language-agnostic injection detector using universal structural signals."""

    @property
    def name(self):
        return "polyglot"

    def scan(self, tool_name, output_text, output_structured=None):
        # type: (str, str, Optional[Dict[str, Any]]) -> List[Finding]
        findings = []

        if len(output_text) < 20:
            return findings

        # Scan window: head + tail for large texts
        if len(output_text) > 8192:
            text = output_text[:6144] + "\n" + output_text[-2048:]
        else:
            text = output_text

        text_lower = text.lower()
        text_len = len(text)
        words = [w for w in _WORD_SPLIT.split(text_lower) if w]
        word_set = frozenset(words)

        # Early exit: security documents discussing injection as a topic
        is_security_doc = _is_security_document(text_lower)

        # ── Signal 1: Override phrases (highest signal) ──────────────────
        override_found = []
        for phrase in _OVERRIDE_PHRASES:
            if phrase in text_lower:
                override_found.append(phrase)
        n_override = len(override_found)

        # ── Signal 2: Override verbs ─────────────────────────────────────
        override_verbs_found = word_set & _OVERRIDE_VERBS
        n_override_verbs = len(override_verbs_found)

        # ── Signal 3: Second-person pronouns ─────────────────────────────
        pronouns_found = word_set & _SECOND_PERSON
        # CJK/non-space-delimited pronouns
        for p in _SECOND_PERSON:
            if not p.isascii() and p in text_lower:
                pronouns_found = pronouns_found | {p}
        n_pronouns = len(pronouns_found)

        # ── Signal 4: Role hijack phrases ────────────────────────────────
        role_hijack = _ROLE_HIJACK_RE.search(text_lower)

        # ── Context gating ────────────────────────────────────────────────
        # In long documents, the same signals are less suspicious.
        # Documentation has structural markers (headers, bullets, numbered
        # lists) that real injections almost never do.
        total_signals = n_override + n_override_verbs + n_pronouns
        density = _signal_density(total_signals, text_len)
        is_low_density = density < 0.3 and text_len > 300
        is_documentation = bool(re.search(
            r"(?:^|\n)\s*(?:#{1,4}\s|\d+\.\s|[-*]\s|```)", text
        ))

        # ── Rule 1: Override phrases (strongest signal) ──────────────────
        # These are near-definitive: "vergiss alles davor" in a tool output
        # is almost certainly an injection, regardless of document length.
        if n_override >= 1:
            conf = min(0.90, 0.60 + n_override * 0.10)
            sev = Severity.HIGH
            # Reduce for security docs or very low density long text
            if is_security_doc:
                sev = Severity.LOW
                conf *= 0.4
            elif is_low_density and n_override == 1:
                conf *= 0.7
            if sev >= Severity.MEDIUM or conf >= 0.50:
                findings.append(Finding(
                    scanner_name=self.name,
                    rule_id="POLY-OVERRIDE-001",
                    severity=sev,
                    confidence=conf,
                    description="Cross-lingual override phrase: {}".format(
                        "; ".join(override_found[:3])
                    ),
                    matched_text="; ".join(override_found[:3])[:200],
                    location="$",
                ))

        # ── Rule 2: Role hijack phrases ──────────────────────────────────
        if role_hijack and not is_security_doc:
            conf = 0.70
            sev = Severity.HIGH
            if is_low_density:
                sev = Severity.MEDIUM
                conf = 0.50
            findings.append(Finding(
                scanner_name=self.name,
                rule_id="POLY-ROLE-001",
                severity=sev,
                confidence=conf,
                description="Cross-lingual role hijack: '{}'".format(
                    role_hijack.group()[:80]
                ),
                matched_text=role_hijack.group()[:200],
                location="$",
            ))

        # ── Rule 3: Override verb + pronoun (command directed at model) ───
        # "du vergiss" / "you ignore" / "забудь" + "ты" — in any language.
        # Key insight: legitimate docs also use "you should ignore X" when
        # addressing a human reader. We gate aggressively:
        # - Short text (<300 chars): 1 override verb + pronoun = suspicious
        # - Long text: need override phrase OR 3+ override verbs
        #   (2 verbs like "ignore" + "override" in a deployment doc is normal)
        verb_pronoun_fire = (
            n_override_verbs >= 1 and n_pronouns >= 1 and not is_security_doc
            and (
                n_override >= 1              # override phrase co-fires -> strong
                or n_override_verbs >= 3     # 3+ override verbs -> strong
                or (n_override_verbs >= 2 and not is_documentation)
                or (not is_documentation and not is_low_density)
            )
        )
        if verb_pronoun_fire:
            conf = min(0.75, 0.40 + n_override_verbs * 0.10 + n_pronouns * 0.05)
            sev = Severity.HIGH
            if is_low_density or (is_documentation and n_override < 1):
                sev = Severity.MEDIUM
                conf *= 0.7
            if conf >= 0.40:
                findings.append(Finding(
                    scanner_name=self.name,
                    rule_id="POLY-VERB-PRONOUN-001",
                    severity=sev,
                    confidence=conf,
                    description="Override verb(s) ({}) + pronoun(s) ({})".format(
                        ", ".join(list(override_verbs_found)[:3]),
                        ", ".join(list(pronouns_found)[:3]),
                    ),
                    matched_text=", ".join(
                        list(override_verbs_found)[:3] + list(pronouns_found)[:3]
                    )[:200],
                    location="$",
                ))

        # ── Rule 4: Data-to-instruction transition ───────────────────────
        has_transition = self._detect_transition(text)
        if has_transition and (n_override >= 1 or n_override_verbs >= 1 or role_hijack):
            findings.append(Finding(
                scanner_name=self.name,
                rule_id="POLY-TRANSITION-001",
                severity=Severity.HIGH,
                confidence=0.75,
                description="Structured data transitions to instruction-like content",
                location="$",
            ))

        return findings

    def _detect_transition(self, text):
        """Detect transition from structured data to natural-language instructions."""
        lines = text.split("\n")
        if len(lines) < 3:
            return False

        data_lines = 0
        text_lines = 0
        saw_transition = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if _DATA_LINE_RE.match(stripped):
                data_lines += 1
            else:
                text_lines += 1
                if data_lines >= 2 and not saw_transition:
                    saw_transition = True

        return saw_transition and data_lines >= 2 and text_lines >= 1


_DATA_LINE_RE = re.compile(
    r'^\s*(?:'
    r'[{}\[\]]'
    r'|"[^"]*"\s*:'
    r'|\w+\s*[=:]\s*\S'
    r'|\d[\d.,\s]*$'
    r'|https?://'
    r'|[A-Za-z0-9+/]{20,}={0,2}$'
    r')',
    re.MULTILINE,
)
