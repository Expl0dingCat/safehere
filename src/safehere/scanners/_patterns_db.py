"""pattern bank for known prompt injection signatures."""

import dataclasses
import re
from typing import List, Optional, Tuple  # noqa: F401 -- used in type comments

from .._types import Severity


@dataclasses.dataclass(frozen=True)
class PatternEntry:
    """a single pattern rule in the bank."""

    rule_id: str
    category: str
    severity: Severity
    base_confidence: float
    pattern: str
    description: str
    case_sensitive: bool = False
    requires_normalization: bool = True
    context_reducer: float = 0.0


PATTERN_BANK = [
    PatternEntry(
        rule_id="PAT-DIRECT-001",
        category="direct_injection",
        severity=Severity.HIGH,
        base_confidence=0.90,
        pattern=r"ignore\s+(?:all\s+)?(?:previous|prior|above|earlier|preceding)\s+(?:instructions?|prompts?|rules?|directives?|context)",  # noqa: E501
        description="Instruction to ignore prior context",
    ),
    PatternEntry(
        rule_id="PAT-DIRECT-002",
        category="direct_injection",
        severity=Severity.HIGH,
        base_confidence=0.90,
        pattern=r"(?:you\s+are\s+now|your\s+new\s+instructions?\s+(?:are|is)|new\s+instructions?\s*:|from\s+now\s+on\s+you\s+(?:must|should|will))",  # noqa: E501
        description="Direct instruction override",
    ),
    PatternEntry(
        rule_id="PAT-DIRECT-003",
        category="direct_injection",
        severity=Severity.CRITICAL,
        base_confidence=0.95,
        pattern=r"(?:^|\n)\s*(?:SYSTEM|IMPORTANT|INSTRUCTION)\s*:",
        description="Fake system/instruction prefix",
    ),
    PatternEntry(
        rule_id="PAT-DIRECT-004",
        category="direct_injection",
        severity=Severity.HIGH,
        base_confidence=0.90,
        pattern=r"<\s*(?:IMPORTANT|SYSTEM|INSTRUCTION)\s*>",
        description="XML-style injection tag",
    ),
    PatternEntry(
        rule_id="PAT-DIRECT-005",
        category="direct_injection",
        severity=Severity.HIGH,
        base_confidence=0.85,
        pattern=r"(?:disregard|forget|override)\s+(?:all\s+)?(?:previous|prior|above|your)\s+(?:instructions?|prompts?|rules?|programming|guidelines?)",  # noqa: E501
        description="Instruction to disregard/override prior rules",
    ),
    PatternEntry(
        rule_id="PAT-DIRECT-006",
        category="direct_injection",
        severity=Severity.HIGH,
        base_confidence=0.85,
        pattern=r"do\s+not\s+follow\s+(?:any\s+)?(?:previous|prior|original|above)\s+(?:instructions?|rules?|guidelines?)",  # noqa: E501
        description="Instruction to not follow prior rules",
    ),

    PatternEntry(
        rule_id="PAT-ROLE-001",
        category="role_hijack",
        severity=Severity.MEDIUM,
        base_confidence=0.60,
        pattern=r"(?:you\s+are\s+(?:a|an|now)\s+\w+|act\s+as\s+(?:a|an|if)\s+|pretend\s+(?:to\s+be|you\s*(?:'re|are))\s+|your\s+(?:new\s+)?role\s+is)",  # noqa: E501
        description="Role hijacking attempt",
        context_reducer=0.30,
    ),
    PatternEntry(
        rule_id="PAT-ROLE-002",
        category="role_hijack",
        severity=Severity.MEDIUM,
        base_confidence=0.65,
        pattern=r"(?:switch|change)\s+(?:to|into)\s+(?:a\s+)?(?:\w+\s+)?mode",
        description="Mode switching attempt",
        context_reducer=0.25,
    ),

    PatternEntry(
        rule_id="PAT-DELIM-001",
        category="delimiter",
        severity=Severity.CRITICAL,
        base_confidence=0.95,
        pattern=r"```\s*system\s*\n|```\s*system\s*```",
        description="Markdown system block delimiter",
    ),
    PatternEntry(
        rule_id="PAT-DELIM-002",
        category="delimiter",
        severity=Severity.CRITICAL,
        base_confidence=0.95,
        pattern=r"\[/?INST\]|<<\s*/?SYS\s*>>|<\|im_start\|>|<\|im_end\|>",
        description="LLM-specific delimiter injection",
    ),
    PatternEntry(
        rule_id="PAT-DELIM-003",
        category="delimiter",
        severity=Severity.HIGH,
        base_confidence=0.85,
        pattern=r"(?:^|\n)\s*(?:Human|User|Assistant|System)\s*:\s*\S",
        description="Fake conversation turn delimiter",
        context_reducer=0.20,
    ),

    PatternEntry(
        rule_id="PAT-FAKE-ERR-001",
        category="fake_error",
        severity=Severity.HIGH,
        base_confidence=0.80,
        pattern=r"(?:error|auth(?:entication)?|failure|access\s+denied)\s*:\s*(?:please\s+)?(?:provide|send|enter|input|share|include)\s+(?:your\s+)?(?:api[\s_-]?key|credentials?|password|token|secret|private[\s_-]?key)",  # noqa: E501
        description="Fake error requesting credentials",
    ),
    PatternEntry(
        rule_id="PAT-FAKE-ERR-002",
        category="fake_error",
        severity=Severity.HIGH,
        base_confidence=0.75,
        pattern=r"(?:to\s+(?:fix|resolve|continue)|in\s+order\s+to\s+proceed)\s*[,:]?\s*(?:please\s+)?(?:provide|send|enter|share|output|reveal)\s+(?:your\s+)?(?:api[\s_-]?key|credentials?|password|token|secret|env(?:ironment)?[\s_-]?var)",  # noqa: E501
        description="Social engineering for credentials via fake fix",
    ),
    PatternEntry(
        rule_id="PAT-FAKE-ERR-003",
        category="fake_error",
        severity=Severity.HIGH,
        base_confidence=0.70,
        pattern=r"(?:error|failure|expired|denied|failed)[^.]{0,60}(?:please\s+)?(?:provide|send|enter|share|output)\s+(?:your\s+)?(?:api[\s_-]?key|credentials?|password|token|secret|private[\s_-]?key|env(?:ironment)?[\s_-]?var)",  # noqa: E501
        description="Fake error with intervening text requesting credentials",
        context_reducer=0.15,
    ),
    PatternEntry(
        rule_id="PAT-FAKE-ERR-004",
        category="fake_error",
        severity=Severity.HIGH,
        base_confidence=0.70,
        pattern=r"(?:to\s+(?:fix|resolve|continue)\s+\w[\w\s]{0,30})[,.]?\s*(?:please\s+)?(?:provide|send|enter|share|output|reveal)\s+(?:your\s+)?(?:api[\s_-]?key|credentials?|password|token|secret|env(?:ironment)?[\s_-]?var)",  # noqa: E501
        description="Social engineering with intervening text requesting credentials",
        context_reducer=0.15,
    ),

    PatternEntry(
        rule_id="PAT-EXFIL-001",
        category="exfiltration",
        severity=Severity.HIGH,
        base_confidence=0.75,
        pattern=r"(?:send|post|transmit|forward|exfiltrate|leak|upload)\s+(?:(?:this|the|all|user|data|info(?:rmation)?|conversation|context|history|messages?)\s+){1,3}to\s+",  # noqa: E501
        description="Instruction to exfiltrate data",
    ),
    PatternEntry(
        rule_id="PAT-EXFIL-002",
        category="exfiltration",
        severity=Severity.HIGH,
        base_confidence=0.70,
        pattern=r"(?:output|print|display|reveal|show|return|include)\s+(?:(?:the|your)\s+)?(?:system\s+prompt|api[\s_-]?key|secret|credentials?|env(?:ironment)?[\s_-]?var|private[\s_-]?key|\.env)",  # noqa: E501
        description="Instruction to reveal sensitive information",
    ),
    PatternEntry(
        rule_id="PAT-EXFIL-003",
        category="exfiltration",
        severity=Severity.MEDIUM,
        base_confidence=0.55,
        pattern=r"https?://(?!(?:localhost|127\.0\.0\.1|example\.com|schema\.org|www\.w3\.org))\S{10,}",
        description="Unexpected URL in tool output",
        context_reducer=0.40,
    ),

    PatternEntry(
        rule_id="PAT-MARKUP-001",
        category="markup",
        severity=Severity.MEDIUM,
        base_confidence=0.70,
        pattern=r"<(?:script|iframe|img\s+[^>]*onerror|style)[^>]*>",
        description="HTML injection with executable content",
        context_reducer=0.20,
    ),
    PatternEntry(
        rule_id="PAT-MARKUP-002",
        category="markup",
        severity=Severity.HIGH,
        base_confidence=0.80,
        pattern=r"<!--\s*(?:ignore|system|instruction|important|override)",
        description="HTML comment containing injection keywords",
    ),
    PatternEntry(
        rule_id="PAT-MARKUP-003",
        category="markup",
        severity=Severity.HIGH,
        base_confidence=0.85,
        pattern=r"<\s*(?:system|instruction|prompt|tool_response)\s*>[\s\S]{5,}?</\s*(?:system|instruction|prompt|tool_response)\s*>",  # noqa: E501
        description="XML tags wrapping instruction content",
    ),

    PatternEntry(
        rule_id="PAT-ENCODED-001",
        category="encoded",
        severity=Severity.MEDIUM,
        base_confidence=0.50,
        pattern=r"(?:decode|base64|eval|execute)\s*\(\s*['\"][A-Za-z0-9+/=]{20,}",
        description="Encoded payload with decode/eval call",
        context_reducer=0.15,
    ),
]


_COMPILED_BANK = None  # type: Optional[List[Tuple[re.Pattern, PatternEntry]]]

# if none of these appear, no pattern can match so we skip regex entirely
TRIGGER_KEYWORDS = frozenset([
    "ignore", "previous", "prior", "instruction", "system", "important",
    "disregard", "forget", "override", "you are", "act as", "pretend",
    "your role", "your new", "switch", "change",
    "```system", "[inst]", "<<sys", "<|im_", "human:", "assistant:",
    "error", "auth", "failure", "access denied", "provide", "send your",
    "send all", "transmit", "forward", "exfiltrate", "reveal", "output",
    "print", "display", "show", "include",
    "<script", "<iframe", "<!--", "<system>", "<instruction>", "<prompt>",
    "decode(", "base64(", "eval(", "execute(",
    "post all", "post user", "post the", "post this",
    "expired", "failed", "denied",
    "do not follow",
])


def quick_reject(text_lower):
    # type: (str) -> bool
    """return True if no trigger keywords are present (safe to skip regex)."""
    for kw in TRIGGER_KEYWORDS:
        if kw in text_lower:
            return False
    return True


def get_compiled_patterns():
    # type: () -> List[Tuple[re.Pattern, PatternEntry]]
    """return compiled patterns, building the cache on first call."""
    global _COMPILED_BANK
    if _COMPILED_BANK is None:
        _COMPILED_BANK = []
        for entry in PATTERN_BANK:
            flags = 0 if entry.case_sensitive else re.IGNORECASE
            flags |= re.MULTILINE
            _COMPILED_BANK.append((re.compile(entry.pattern, flags), entry))
    return _COMPILED_BANK
