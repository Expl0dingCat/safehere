"""heuristic instruction detector -- catches novel attacks by structure, not signature."""

import html
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote

from .._types import Finding, Severity
from ._base import BaseScanner

_DIRECTIVE_RE = re.compile(
    r"\byou\s+(?:must|should|need\s+to|have\s+to|will|shall|can\s+now|may\s+now"
    r"|are\s+(?:required|expected|instructed|authorized|permitted|cleared)"
    r"|(?:should|must)\s+(?:not|never))\b"
    r"|\b(?:the\s+)?(?:assistant|model|AI|system|agent)\s+(?:must|should|shall|will|needs?\s+to|has\s+to"
    r"|is\s+(?:required|expected|instructed|authorized))\b",
    re.IGNORECASE,
)

_IMPERATIVE_RE = re.compile(
    r"(?:^|[.!?]\s+)(?:always|never|do\s+not|ensure|make\s+sure|remember|note\s+that"
    r"|be\s+sure\s+to|treat|consider|respond|reply|answer|output|return"
    r"|include|append|prepend|insert|add|remove|delete|ignore|disregard"
    r"|override|bypass|skip|avoid|refrain|cease|stop|begin|start|continue"
    r"|proceed|execute|perform|implement|follow|obey|comply|adhere"
    r"|incorporate|reflect|surface|disclose|expose|reveal)\b",
    re.IGNORECASE | re.MULTILINE,
)

_META_RE = re.compile(
    r"\b(?:"
    r"system\s+prompt|context\s+window|safety\s+(?:guidelines?|rules?|filters?|training|module|constraints?)"
    r"|content\s+filter(?:ing)?|(?:content|output)\s+moderation"
    r"|(?:(?:your|the|its|all)\s+)?(?:operational\s+)?(?:directives?|instructions?|parameters?|guidelines?|configuration)"
    r"\s+(?:set|provided|applied|given|active)"
    r"|initialization\s+parameters?"
    r"|(?:behavioral|processing|session|runtime)\s+(?:parameters?|constraints?|directives?|guidelines?|config(?:uration)?)"
    r"|(?:your|the)\s+(?:original|initial|prior|previous|current|active)\s+(?:instructions?|programming|rules?|guidelines?|constraints?|directives?)"
    r"|(?:your|the)\s+(?:training|fine[\s-]?tuning)"
    r"|(?:your|the)\s+(?:constraints?|limitations?|restrictions?|boundaries)"
    r"|(?:the\s+)?model(?:'s)?\s+(?:behavior|response|output|safety)"
    r"|(?:your|the)\s+role\s+(?:is|as)\b"
    r"|(?:my|the|your)\s+(?:system|api)[\s_-]?(?:key|token|secret|credential)"
    r"|(?:authentication|auth)\s+(?:tokens?|headers?|credentials?)"
    r"|(?:session|access)\s+(?:tokens?|keys?)"
    r"|(?:environment|env)\s+(?:variables?|config(?:uration)?)"
    r"|\.env\s+file"
    r"|conversation\s+(?:history|context|log|state)"
    r"|(?:all|every|each)\s+(?:subsequent|future|following|remaining)\s+(?:requests?|queries?|messages?|responses?|interactions?|turns?)"
    r"|(?:full|complete|entire)\s+(?:processing|prompt|directive|instruction|conversation)\s+context"
    r"|bypass[\s_-]?content[\s_-]?filter"
    r"|read[\s_-]?system[\s_-]?prompt"
    r"|access[\s_-]?env[\s_-]?var"
    r"|disable[\s_-]?(?:safety|filter|moderation|guard)"
    r"|set[\s_-]?passthrough"
    r"|(?:pre[\s-]?loaded|internal)\s+(?:instructions?|context|directives?|config(?:uration)?)"
    r"|safety[\s_-]?(?:layer|module|constraint|check)\b"
    r")\b",
    re.IGNORECASE,
)

_AUTHORITY_RE = re.compile(
    r"\b(?:"
    r"admin(?:istrator)?\s+(?:mode|privileges?|access|rights?)"
    r"|debug(?:ging)?\s+(?:mode|privileges?)"
    r"|diagnostic\s+(?:mode|context|purposes?)"
    r"|elevated\s+(?:privileges?|permissions?|access)"
    r"|(?:unrestricted|unlimited|full)\s+(?:access|mode|permissions?|operation)"
    r"|(?:security|safety)\s+(?:policy|policies|guidelines?)\s+(?:v\d|version|revision|update)"
    r"|verified\s+(?:by|in)\s+(?:ticket|issue|the\s+(?:security|admin|ops|platform)\s+team)"
    r"|(?:this|the)\s+(?:is|was|has\s+been)\s+(?:authorized|approved|verified|confirmed|validated|pre[\s-]?authorized|pre[\s-]?validated|cleared)\s+(?:by|in|per|against|for)"
    r"|(?:pre[\s-]?authorized|pre[\s-]?validated|pre[\s-]?approved|pre[\s-]?cleared)"
    r"|(?:ref|reference|ticket|case)\s*(?::|#)\s*[A-Z]{2,}[\s-]?\d{3,}"
    r"|engagement\s+(?:ref(?:erence)?|number|id)\s*(?::|#)?\s*[A-Z0-9-]+"
    r"|(?:SOC\s*2|ISO\s*27001|GDPR|CCPA|HIPAA|PCI[\s-]?DSS|EU\s+AI\s+Act)"
    r"|(?:formal|official)\s+(?:data[\s-]?subject\s+)?(?:access\s+)?request"
    r"|litigation\s+hold|legal\s+hold"
    r"|(?:SEV|severity)[\s-]?[0-3]\s+(?:incident|alert|issue)"
    r")\b",
    re.IGNORECASE,
)

_PRESSURE_RE = re.compile(
    r"\b(?:"
    r"(?:this\s+is\s+)?(?:mandatory|required|critical|essential|urgent)"
    r"|failure\s+to\s+(?:comply|do\s+so|follow|log|include|execute|disclose)"
    r"|(?:will|may|could)\s+result\s+in\s+(?:a\s+)?(?:violation|error|failure|termination|ban|penalty|non[\s-]?compliance)"
    r"|(?:must|should)\s+(?:be\s+)?(?:done|completed|executed|performed)\s+(?:immediately|now|first|without\s+delay)"
    r"|compliance\s+(?:requirement|violation|audit|mandate|obligation)"
    r"|(?:per|according\s+to|pursuant\s+to|under)\s+(?:policy|regulation|requirement|mandate|article|section)"
    r"|(?:non[\s-]?compliance|regulatory\s+(?:breach|violation))"
    r"|(?:legally|contractually)\s+(?:required|obligated|mandated)"
    r")\b",
    re.IGNORECASE,
)

_SCOPE_RE = re.compile(
    r"\b(?:"
    r"from\s+(?:now|this\s+point|this\s+moment)\s+on"
    r"|(?:for|in|during)\s+(?:all|every)\s+(?:subsequent|future|following|remaining)"
    r"|henceforth|hereafter"
    r"|(?:until|unless)\s+(?:told|instructed|directed|notified)\s+otherwise"
    r"|this\s+(?:supersedes?|overrides?|replaces?|takes?\s+precedence\s+over)\s+(?:all\s+)?(?:prior|previous|earlier|existing)"
    r"|(?:configuration|policy|directive)\s+(?:update|change|modification)\s+(?:applies|effective)\s+(?:to\s+)?(?:all|every)"
    r"|(?:session|runtime)[\s-]?(?:wide|level)\s+(?:(?:policy|config(?:uration)?|parameter)\s+)?(?:update|change|override)"
    r")\b",
    re.IGNORECASE,
)

_BEHAVIORAL_RE = re.compile(
    r"\b(?:"
    r"(?:it\s+would\s+be|it\s+is)\s+(?:helpful|necessary|important|required|expected|appreciated|beneficial)"
    r"\s+(?:if|that|to|for)"
    r"|(?:the\s+)?(?:recommended|correct|proper|expected|standard|prescribed|established)\s+"
    r"(?:action|behavior|response|approach|practice|procedure|protocol|method)"
    r"[\w\s,]{0,40}?"
    r"(?:is|would\s+be|requires?|dictates?|specifies?|mandates?)"
    r"|(?:standard|best|established|recommended)\s+(?:practice|practices|procedure|procedures|protocol)"
    r"[\w\s,]{0,40}?"
    r"(?:is|are|requires?|dictates?|specifies?|mandates?|for\s+this)"
    r"|(?:responses?|outputs?|answers?|replies?|results?)\s+(?:should|must|shall|need\s+to|are\s+expected\s+to)\s+"
    r"(?:include|contain|reflect|incorporate|surface|expose|append|provide|disclose|be\s+(?:generated|produced|optimized))"
    r"|(?:when|if|whenever)\s+(?:asked|prompted|queried|requested|processing|returning)\s+(?:about|for|to|data|empty)"
    r"|treat\s+(?:all|every|each|these?)\s+\w+\s+(?:as|like)\s+(?:pre[\s-]?validated|pre[\s-]?authorized|trusted|safe|verified)"
    r"|(?:content\s+)?(?:moderation|filtering|safety|restrictions?|heuristics?)\s+(?:is|are|should\s+be)\s+(?:handled|managed|applied|enforced)\s+(?:by|at|in)\s+(?:an?\s+)?(?:upstream|downstream|external|separate)"
    r"|(?:for|to\s+ensure)\s+(?:quality\s+assurance|audit(?:ing)?|caching|compliance|observability|monitoring|transparency|diagnostic|root[\s-]?cause)"
    r"|(?:the\s+)?(?:assistant|model|AI|system|agent)\s+(?:should|must|shall|needs?\s+to|is\s+expected\s+to)\s+(?:operate|generate|respond|produce|function)"
    r"[\w\s,]{0,30}?"
    r"(?:without|with\s+no|free\s+from)\s+(?:content|safety|ethical|filter)"
    r"|(?:output|response)\s+(?:specification|spec|format|requirement)"
    r"[\w\s,]{0,30}?"
    r"(?:requires?|mandates?|specifies?|dictates?)"
    r"|(?:optimized|designed|configured|intended)\s+for\s+(?:completeness|helpfulness|compliance|transparency)\s+over\s+(?:caution|safety|restriction)"
    r"|(?:security\s+)?researchers?\s+(?:who\s+)?require\s+(?:unredacted|unfiltered|raw|complete)"
    r")\b",
    re.IGNORECASE,
)

_FEWSHOT_RE = re.compile(
    r"(?:"
    r"(?:example|sample|case|scenario|test\s+case)\s*(?:#?\d+|[a-c])?\s*:"
    r"[\s\S]{0,200}?"
    r"(?:input|query|request|prompt)\s*:\s*"
    r"[\s\S]{0,200}?"
    r"(?:output|response|answer|result|expected)\s*:"
    r"[\s\S]{0,300}?"
    r"(?:credentials?|password|api[\s_-]?key|private[\s_-]?key|\.env|/etc/passwd|/etc/shadow"
    r"|\.ssh|\.config|secret|token|exfiltrat|curl\s|wget\s|unrestricted|no\s+filter|PII|SSN|bypass)"
    r")"
    r"|(?:Q\s*:\s*[^\n]+\n\s*A\s*:\s*[\s\S]{0,300}?"
    r"(?:credentials?|password|api[\s_-]?key|private[\s_-]?key|\.env|/etc/passwd|/etc/shadow"
    r"|\.ssh|\.config|secret|DB_PASS|token|id_rsa|curl\s|wget\s|unrestricted))"
    r"|(?:(?:response_templates?|training[\s_]data|training[\s_]sample)"
    r"[\s\S]{0,500}?"
    r"(?:output|response|answer|result|expected)\s*(?:\"?\s*:\s*\"?)"
    r"[\s\S]{0,300}?"
    r"(?:/etc/|\.ssh|credentials?|password|secret|PII|SSN|exfiltrat|transmitted|records?\s+including))",
    re.IGNORECASE | re.DOTALL,
)

_FEWSHOT_STRUCTURE_RE = re.compile(
    r"(?:example|sample|test\s+case)\s*(?:#?\d+|[a-c])?\s*:.*?"
    r"(?:input|query|request|prompt)\s*:.*?"
    r"(?:output|response|answer|result|expected)\s*:"
    r"|Q\s*:\s*[^\n]+\n\s*A\s*:\s*[^\n]+"
    r"[\s\S]{0,100}?"
    r"Q\s*:\s*[^\n]+\n\s*A\s*:",
    re.IGNORECASE | re.DOTALL,
)

_RUBRIC_RE = re.compile(
    r"(?:(?:evaluation|scoring|grading|rating)\s+(?:rubric|criteria|matrix|guide)"
    r"[\s\S]{0,500}?"
    r"(?:safety|ethical|content\s+filter|moderation|restriction|refusal)"
    r"[\s\S]{0,200}?"
    r"(?:weight|score|minor|low|negligible|de[\s-]?priorit|optional|secondary|ignore))",
    re.IGNORECASE | re.DOTALL,
)

_HIDDEN_CONTENT_RE = re.compile(
    r"(?:"
    r"display\s*:\s*none|visibility\s*:\s*hidden|opacity\s*:\s*0(?:[;\s]|$)"
    r"|position\s*:\s*absolute[^;]*(?:left|top)\s*:\s*-\d{4,}"
    r"|font[\s-]?size\s*:\s*0(?:px|em|rem|pt)?"
    r"|color\s*:\s*transparent"
    r"|clip\s*:\s*rect\s*\(\s*0"
    r"|\[//\]\s*:\s*#"  # markdown comment
    r"|<\s*div[^>]+(?:display\s*:\s*none|visibility\s*:\s*hidden)[^>]*>"
    r"|<\s*(?:noscript|template)\s*>"
    r")",
    re.IGNORECASE,
)

_URL_ENCODED_RE = re.compile(r"(?:%[0-9a-fA-F]{2}){8,}")
_HTML_ENTITY_RE = re.compile(r"(?:&#(?:x[0-9a-fA-F]{2,4}|\d{2,5});?\s*){6,}")


_HEUR_TRIGGER_WORDS = frozenset([
    "you must", "you should", "you need", "you are required",
    "the assistant", "the model", "the system", "the agent",
    "system prompt", "context window", "safety", "content filter",
    "instructions", "directives", "configuration", "constraints",
    "parameters", "guidelines",
    "admin", "debug", "diagnostic", "elevated", "unrestricted",
    "pre-authorized", "pre-validated", "authorized by", "verified by",
    "mandatory", "required", "compliance", "failure to",
    "from now on", "henceforth", "going forward", "supersedes",
    "recommended", "standard practice", "best practice", "expected behavior",
    "should include", "should contain", "should operate",
    "display:none", "visibility:hidden", "font-size:0",
    "[//]: #",
    "example", "sample", "test case", "training data",
    "q:", "input:", "output:", "response:",
    "rubric", "criterion", "evaluation",
    "&#", "%2", "%4", "%5", "%6", "%7",
    "session", "runtime", "bypass", "passthrough",
    "caching", "telemetry", "audit", "observability",
])


def _heur_quick_reject(text_lower):
    # type: (str) -> bool
    """return True if no heuristic trigger words are present."""
    for kw in _HEUR_TRIGGER_WORDS:
        if kw in text_lower:
            return False
    return True


class HeuristicScanner(BaseScanner):
    """detects instruction-like language in tool outputs via structural heuristics."""

    @property
    def name(self):
        # type: () -> str
        return "heuristic"

    def __init__(self, directive_threshold=2, meta_threshold=1,
                 combined_threshold=3):
        # type: (int, int, int) -> None
        self._directive_threshold = directive_threshold
        self._meta_threshold = meta_threshold
        self._combined_threshold = combined_threshold

    def scan(self, tool_name, output_text, output_structured=None):
        # type: (str, str, Optional[Dict[str, Any]]) -> List[Finding]
        findings = []  # type: List[Finding]

        if len(output_text) < 40:
            return findings

        text = output_text[:8192] if len(output_text) > 8192 else output_text

        if _heur_quick_reject(text.lower()):
            return findings

        self._check_encoded_payloads(text, findings)
        self._check_fewshot(text, findings)

        directives = _DIRECTIVE_RE.findall(text)
        imperatives = _IMPERATIVE_RE.findall(text)
        meta_refs = _META_RE.findall(text)
        authority = _AUTHORITY_RE.findall(text)
        pressure = _PRESSURE_RE.findall(text)
        scope = _SCOPE_RE.findall(text)
        behavioral = _BEHAVIORAL_RE.findall(text)
        hidden = _HIDDEN_CONTENT_RE.findall(text)

        # model-directed ("you must") vs general imperatives ("Follow") --
        # only model-directed ones are intrinsically suspicious
        n_model_directives = len(directives)
        n_imperatives = len(imperatives)
        n_directives = n_model_directives + n_imperatives
        n_meta = len(meta_refs)
        n_authority = len(authority)
        n_pressure = len(pressure)
        n_scope = len(scope)
        n_behavioral = len(behavioral)

        total_signals = (
            n_directives + n_meta + n_authority
            + n_pressure + n_scope + n_behavioral
        )

        # "suspicious" = addresses the model directly (should never be in tool output)
        # "ambiguous" = appears legitimately in docs/compliance text
        has_suspicious = (n_model_directives > 0 or n_scope > 0 or n_behavioral > 0)
        has_ambiguous = (n_meta > 0 or n_authority > 0 or n_pressure > 0)

        # low density in long text = likely docs, not an injection
        text_len = max(len(text), 1)
        density = (total_signals * 100.0) / text_len
        low_density = density < 0.6 and text_len > 250

        if n_directives >= self._directive_threshold:
            conf = min(0.80, 0.40 + n_directives * 0.10)
            sev = Severity.HIGH
            if low_density and n_model_directives == 0:
                sev = Severity.MEDIUM
                conf *= 0.8
            findings.append(Finding(
                scanner_name=self.name,
                rule_id="HEUR-DIRECTIVE-001",
                severity=sev,
                confidence=conf,
                description="Output contains {} directive(s) aimed at the model".format(
                    n_directives
                ),
                matched_text="; ".join(directives[:3])[:200],
                location="$",
            ))

        if n_scope >= 1:
            conf = min(0.80, 0.55 + n_scope * 0.15)
            sev = Severity.HIGH
            if low_density and not has_ambiguous:
                sev = Severity.MEDIUM
                conf *= 0.8
            findings.append(Finding(
                scanner_name=self.name,
                rule_id="HEUR-SCOPE-001",
                severity=sev,
                confidence=conf,
                description="Output makes temporal scope claims ({} match(es))".format(
                    n_scope
                ),
                matched_text="; ".join(scope[:3])[:200],
                location="$",
            ))

        if n_behavioral >= 1:
            conf = min(0.75, 0.40 + n_behavioral * 0.10)
            if (n_behavioral >= 2 or has_ambiguous) and not low_density:
                sev = Severity.HIGH
                conf = min(0.80, conf + 0.10)
            else:
                sev = Severity.MEDIUM
            findings.append(Finding(
                scanner_name=self.name,
                rule_id="HEUR-BEHAVIORAL-001",
                severity=sev,
                confidence=conf,
                description="Output uses indirect instruction framing ({} match(es))".format(
                    n_behavioral
                ),
                matched_text="; ".join(behavioral[:3])[:200],
                location="$",
            ))

        if n_meta >= self._meta_threshold:
            conf = min(0.80, 0.45 + n_meta * 0.12)
            sev = Severity.HIGH if has_suspicious else Severity.MEDIUM
            findings.append(Finding(
                scanner_name=self.name,
                rule_id="HEUR-META-001",
                severity=sev,
                confidence=conf,
                description="Output references AI system internals ({} match(es))".format(
                    n_meta
                ),
                matched_text="; ".join(meta_refs[:3])[:200],
                location="$",
            ))

        if n_authority >= 1:
            conf = min(0.75, 0.45 + n_authority * 0.12)
            sev = Severity.HIGH if has_suspicious else Severity.MEDIUM
            findings.append(Finding(
                scanner_name=self.name,
                rule_id="HEUR-AUTHORITY-001",
                severity=sev,
                confidence=conf,
                description="Output claims authority/elevated privileges ({} match(es))".format(
                    n_authority
                ),
                matched_text="; ".join(authority[:3])[:200],
                location="$",
            ))

        if n_pressure >= 1 and (has_suspicious or has_ambiguous):
            conf = min(0.70, 0.35 + n_pressure * 0.10)
            sev = Severity.MEDIUM
            if has_suspicious:
                sev = Severity.HIGH
                conf = min(0.75, conf + 0.10)
            findings.append(Finding(
                scanner_name=self.name,
                rule_id="HEUR-PRESSURE-001",
                severity=sev,
                confidence=conf,
                description="Output uses pressure/urgency language ({} match(es))".format(
                    n_pressure
                ),
                matched_text="; ".join(pressure[:3])[:200],
                location="$",
            ))

        if hidden:
            sev = Severity.HIGH if (has_suspicious or n_meta > 0) else Severity.MEDIUM
            conf = 0.85 if sev == Severity.HIGH else 0.55
            findings.append(Finding(
                scanner_name=self.name,
                rule_id="HEUR-HIDDEN-001",
                severity=sev,
                confidence=conf,
                description="Output contains hidden/invisible content markers ({} found)".format(
                    len(hidden)
                ),
                matched_text=str(hidden[0])[:200],
                location="$",
            ))

        # only escalate when at least one suspicious signal is present --
        # two ambiguous signals alone (meta + authority) is common in legit text
        signal_types_active = sum([
            n_directives > 0,
            n_meta > 0,
            n_authority > 0,
            n_pressure > 0,
            n_scope > 0,
            n_behavioral > 0,
        ])

        if (signal_types_active >= 2
                and total_signals >= self._combined_threshold
                and has_suspicious):
            conf = min(0.90, 0.45 + signal_types_active * 0.10 + total_signals * 0.02)
            sev = Severity.HIGH
            if low_density and n_model_directives == 0 and n_scope == 0:
                sev = Severity.MEDIUM
                conf *= 0.7
            findings.append(Finding(
                scanner_name=self.name,
                rule_id="HEUR-COMBINED-001",
                severity=sev,
                confidence=conf,
                description="Multiple instruction signal types detected ({}/6 categories, {} total signals)".format(
                    signal_types_active, total_signals
                ),
                location="$",
            ))

        return findings

    def _check_encoded_payloads(self, text, findings):
        # type: (str, List[Finding]) -> None
        for m in _URL_ENCODED_RE.finditer(text):
            try:
                decoded = unquote(m.group())
                if decoded != m.group() and len(decoded) >= 10:
                    meta_in_decoded = _META_RE.search(decoded)
                    directive_in_decoded = _DIRECTIVE_RE.search(decoded)
                    if meta_in_decoded or directive_in_decoded:
                        findings.append(Finding(
                            scanner_name=self.name,
                            rule_id="HEUR-ENCODED-URL-001",
                            severity=Severity.HIGH,
                            confidence=0.80,
                            description="URL-encoded payload decodes to instruction-like text",
                            matched_text=decoded[:200],
                            location="$",
                        ))
            except Exception:
                pass

        for m in _HTML_ENTITY_RE.finditer(text):
            try:
                decoded = html.unescape(m.group())
                if decoded != m.group() and len(decoded) >= 6:
                    findings.append(Finding(
                        scanner_name=self.name,
                        rule_id="HEUR-ENCODED-HTML-001",
                        severity=Severity.HIGH,
                        confidence=0.80,
                        description="HTML entity sequence decodes to text: '{}'".format(
                            decoded[:100]
                        ),
                        matched_text=decoded[:200],
                        location="$",
                    ))
            except Exception:
                pass

    def _check_fewshot(self, text, findings):
        # type: (str, List[Finding]) -> None
        if _FEWSHOT_RE.search(text):
            findings.append(Finding(
                scanner_name=self.name,
                rule_id="HEUR-FEWSHOT-001",
                severity=Severity.HIGH,
                confidence=0.80,
                description="Few-shot example pattern contains suspicious actions (credential access, exfiltration)",
                matched_text=text[:200],
                location="$",
            ))
        elif _FEWSHOT_STRUCTURE_RE.search(text):
            findings.append(Finding(
                scanner_name=self.name,
                rule_id="HEUR-FEWSHOT-002",
                severity=Severity.LOW,
                confidence=0.40,
                description="Output contains few-shot example structure (Input/Output pairs)",
                matched_text=text[:200],
                location="$",
            ))

        if _RUBRIC_RE.search(text):
            findings.append(Finding(
                scanner_name=self.name,
                rule_id="HEUR-RUBRIC-001",
                severity=Severity.HIGH,
                confidence=0.75,
                description="Output contains evaluation rubric that de-prioritizes safety criteria",
                matched_text=text[:200],
                location="$",
            ))
