# safehere

**Runtime tool-output scanning for Cohere agents.**

Every existing MCP/agent security tool scans the input side: tool descriptions, metadata, call permissions. But CyberArk proved that the most dangerous attacks come through tool outputs — the tool's description and code are clean, but it returns poisoned responses containing hidden instructions that the model follows.

`safehere` is a Python middleware that sits between when a tool returns its result and when that result gets passed back to Cohere's model. It scans every tool output for injected instructions, schema anomalies, and behavioral drift, then blocks or sanitizes suspicious results before they ever reach the model's context window.

## Installation

```bash
pip install safehere
```

## Usage

```python
from cohere import Client
from safehere import ToolGuard

client = Client(api_key="...")
guard = ToolGuard(client=client, tools=[...])

# Use in your tool-use loop
response = guard.run_with_protection(model="command-r", ...)
```

## Documentation

Full documentation coming soon.

## License

MIT
