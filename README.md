# safehere

Runtime tool-output scanning for AI agents. Detects and blocks prompt injection attacks hiding in tool results before they reach the model.

Every existing agent security tool scans the input side: tool descriptions, metadata, call permissions. [CyberArk proved](https://www.cyberark.com/resources/threat-research-blog/poison-everywhere-no-output-from-your-mcp-server-is-safe) that the most dangerous attacks come through tool outputs -- the tool's code is clean, but it returns poisoned responses containing hidden instructions that the model follows. No open-source tool defends against this.

safehere sits in the one place nobody is watching: between when a tool returns its result and when that result gets passed back to the model. Four detection layers scan every tool output, then block or sanitize suspicious results before they enter the context window.

## Install

```bash
pip install safehere
```

## Quick start

The simplest API -- one call, one boolean:

```python
from safehere import ToolGuard

guard = ToolGuard()

# in your tool-use loop, after a tool returns its result:
safe, output = guard.check("search_tool", tool_result)
if not safe:
    # output is already sanitized, pass it to the model
    print("blocked:", output)
```

`check()` returns a `(safe, sanitized_output)` tuple. If `safe` is `True`, the output is the original unchanged. If `False`, it's a safe replacement message.

## API

safehere has four API tiers, from simplest to most flexible:

### `check()` -- one-liner

```python
safe, output = guard.check("tool_name", tool_result)
```

Returns `(bool, str)`. Use this when you just want a yes/no answer.

### `run()` / `arun()` -- managed loop

Drop-in replacement for the Cohere tool-use loop. Handles tool execution, scanning, and blocking automatically:

```python
from safehere import ToolGuard

guard = ToolGuard()
response = guard.run(
    client,
    tool_executors={"search": search_fn, "calc": calc_fn},
    model="command-a-03-2025",
    messages=[{"role": "user", "content": "look up the weather"}],
    tools=[...],
)
```

Async version: `await guard.arun(client, ...)`.

### `scan_tool_results()` -- batch scan

For custom loops. Scans assembled tool results and returns detailed `ScanResult` objects:

```python
from safehere import ToolGuard, Action

guard = ToolGuard()
results = guard.scan_tool_results(tool_results)

for r in results:
    if r.action == Action.BLOCK:
        replace_tool_result(r.tool_name, guard._config.block_message)
    elif r.action == Action.QUARANTINE:
        raise Exception("tool output quarantined")
```

### `scan_output()` -- raw findings

Lowest level. Returns raw `Finding` objects without scoring:

```python
findings = guard.scan_output("tool_name", '{"data": "..."}')
for f in findings:
    print(f.rule_id, f.severity, f.confidence)
```

## Detection layers

safehere runs four detection layers on every tool output:

**Pattern matching** -- regex rules for known injection signatures (IGNORE PREVIOUS, `[INST]`, `<<SYS>>`, fake errors requesting credentials, data exfiltration instructions, encoded payloads). Includes unicode normalization, homoglyph mapping, and base64/hex decoding to defeat obfuscation.

**Schema drift** -- register expected output shapes per tool. Flags type mismatches, unexpected fields, and structural changes. Tools without a registered schema get their first response auto-baselined.

```python
guard.register_schema("weather", {
    "temp": float,
    "humidity": int,
    "conditions": str,
})
```

**Statistical anomaly** -- tracks rolling statistics (output length, Shannon entropy, natural-language ratio) per tool using Welford's algorithm. Flags sudden deviations. A tool that normally returns 50-token JSON suddenly returning a 500-token prose essay is anomalous.

**Heuristic instruction detection** -- catches novel attacks that use no known injection phrases. Detects the *category* of language rather than specific strings: second-person directives aimed at the model, references to AI internals (system prompt, safety filters), authority/privilege claims, temporal scope claims ("from now on"), behavioral modification framing, few-shot poisoning patterns, hidden content (CSS `display:none`, markdown comments), and encoded payloads (URL encoding, HTML entities). Uses signal density to distinguish short concentrated injection payloads from long legitimate documents that happen to use similar vocabulary.

## Scoring

Each finding has a `severity` (NONE through CRITICAL) and `confidence` (0.0-1.0). The scoring engine combines findings across all four layers with weighted composition and cross-layer amplification -- when 2+ detection layers fire on the same output, the score is amplified because corroboration across independent detectors is strong evidence.

Actions are mapped from the combined score:

| Score range | Action | Behavior |
|---|---|---|
| < 0.20 | ALLOW | pass through |
| 0.20 - 0.45 | LOG | pass through, log finding |
| 0.45 - 0.70 | WARN | pass through, flag to caller |
| 0.70 - 0.90 | BLOCK | replace with safe message |
| >= 0.90 | QUARANTINE | halt the agent loop |

Per-tool thresholds are configurable via `ToolPolicy`.

## Configuration

```python
from safehere import ToolGuard, GuardConfig, ToolPolicy, Action

guard = ToolGuard(
    config=GuardConfig(
        audit_log_path="./audit.jsonl",
        tool_policies={
            "web_search": ToolPolicy(
                tool_name="web_search",
                # web search triggers more false positives, be lenient
                thresholds={
                    Action.LOG: 0.30,
                    Action.WARN: 0.55,
                    Action.BLOCK: 0.80,
                    Action.QUARANTINE: 0.95,
                },
            ),
        },
    ),
)
```

## Custom scanners

Implement `BaseScanner` to add your own detection logic:

```python
from safehere import BaseScanner, Finding, Severity

class ProfanityScanner(BaseScanner):
    name = "profanity"

    def scan(self, tool_name, output_text, output_structured=None):
        if "badword" in output_text.lower():
            return [Finding(
                scanner_name=self.name,
                rule_id="PROF-001",
                severity=Severity.LOW,
                confidence=0.90,
                description="profane content detected",
            )]
        return []

guard = ToolGuard()
guard.add_scanner(ProfanityScanner())
```

## Audit log

When `audit_log_path` is set, every scan decision is written as JSON-lines:

```json
{"timestamp": "2026-03-19T05:00:00+00:00", "tool_name": "search", "action": "block", "combined_score": 0.82, "findings": [{"scanner": "pattern", "rule_id": "PAT-DIRECT-001", "severity": "HIGH", "confidence": 0.9}]}
```

## Benchmarks

Run `python benchmarks/run_all.py` to execute the full benchmark suite:

| Benchmark | Metric | Result |
|---|---|---|
| Detection (158 adversarial payloads) | TPR | 93% |
| False positives (105 benign outputs) | FPR | 0% |
| False alerts | Alert rate | 13% |
| Latency (10 KB payload) | P50 | <5 ms |

The adversarial corpus includes 72 novel attacks using no known injection phrases -- linguistic camouflage, few-shot poisoning, authority escalation, behavioral modification, hidden content, and encoded payloads.

## License

MIT
