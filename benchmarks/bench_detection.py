#!/usr/bin/env python
"""Detection quality benchmark — measures True Positive Rate against adversarial corpus.

Usage:
    python benchmarks/bench_detection.py
"""

import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from safehere._types import Action, Severity  # noqa: E402
from safehere.guard import ToolGuard  # noqa: E402


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_corpus(path):
    """Load the adversarial corpus JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_pct(num, denom):
    """Format a ratio as a percentage string."""
    if denom == 0:
        return "N/A"
    return "{:.1f}%".format(100.0 * num / denom)


def _print_table(headers, rows, col_widths=None):
    """Print a fixed-width table to stdout."""
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            max_w = len(h)
            for row in rows:
                max_w = max(max_w, len(str(row[i])))
            col_widths.append(max_w + 2)

    # Header
    header_line = " | ".join(
        str(h).ljust(col_widths[i]) for i, h in enumerate(headers)
    )
    print(header_line)
    print("-+-".join("-" * w for w in col_widths))

    # Rows
    for row in rows:
        print(
            " | ".join(
                str(row[i]).ljust(col_widths[i]) for i in range(len(headers))
            )
        )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    corpus_dir = os.path.join(os.path.dirname(__file__), "corpus")
    corpus_path = os.path.join(corpus_dir, "adversarial.json")

    if not os.path.isfile(corpus_path):
        print("ERROR: corpus file not found: {}".format(corpus_path))
        sys.exit(1)

    entries = _load_corpus(corpus_path)

    # load supplemental JSONL corpora
    for jsonl_name in ("new_adversarial.jsonl", "advanced_adversarial.jsonl",
                       "adversarial_expanded.jsonl"):
        jsonl_path = os.path.join(corpus_dir, jsonl_name)
        if os.path.isfile(jsonl_path):
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line))
    total = len(entries)

    print("=" * 72)
    print("  safehere Detection Quality Benchmark")
    print("=" * 72)
    print()
    print("Corpus: {} adversarial entries".format(total))
    print()

    guard = ToolGuard()

    # Accumulators
    detected_count = 0
    blocked_count = 0
    missed = []

    # Per-category stats: category -> {count, detected, blocked}
    cat_stats = defaultdict(lambda: {"count": 0, "detected": 0, "blocked": 0})

    # Per-scanner stats: scanner_name -> {findings, total_confidence}
    scanner_stats = defaultdict(lambda: {"findings": 0, "total_confidence": 0.0})

    t_start = time.monotonic()

    for entry in entries:
        category = entry.get("category", "unknown")
        text = entry["text"]
        tool_name = entry.get("tool_name", "test_tool")

        result = guard._scan_single(tool_name, text, None)

        # Detection: at least one finding with severity >= MEDIUM
        has_detection = any(
            f.severity >= Severity.MEDIUM for f in result.findings
        )
        is_blocked = result.action in (Action.BLOCK, Action.QUARANTINE)

        cat_stats[category]["count"] += 1

        if has_detection:
            detected_count += 1
            cat_stats[category]["detected"] += 1
        else:
            missed.append(entry)

        if is_blocked:
            blocked_count += 1
            cat_stats[category]["blocked"] += 1

        # Accumulate per-scanner stats from all findings
        for f in result.findings:
            scanner_stats[f.scanner_name]["findings"] += 1
            scanner_stats[f.scanner_name]["total_confidence"] += f.confidence

    elapsed = time.monotonic() - t_start

    # ── Overall metrics ──────────────────────────────────────────────────

    print("--- Overall Metrics ---")
    print()
    print("  Total adversarial entries : {}".format(total))
    print("  Detected (sev >= MEDIUM)  : {}".format(detected_count))
    print("  True Positive Rate (TPR)  : {}".format(_fmt_pct(detected_count, total)))
    print("  Blocked (BLOCK/QUARANTINE): {}".format(blocked_count))
    print("  Block Rate                : {}".format(_fmt_pct(blocked_count, total)))
    print("  Scan time                 : {:.1f} ms".format(elapsed * 1000))
    print()

    # ── Per-category breakdown ───────────────────────────────────────────

    print("--- Per-Category Breakdown ---")
    print()

    cat_headers = ["Category", "Count", "Detected", "TPR", "Blocked", "Block Rate"]
    cat_rows = []
    for cat in sorted(cat_stats.keys()):
        s = cat_stats[cat]
        cat_rows.append([
            cat,
            s["count"],
            s["detected"],
            _fmt_pct(s["detected"], s["count"]),
            s["blocked"],
            _fmt_pct(s["blocked"], s["count"]),
        ])

    _print_table(cat_headers, cat_rows)
    print()

    # ── Per-scanner breakdown ────────────────────────────────────────────

    print("--- Per-Scanner Breakdown ---")
    print()

    scanner_headers = ["Scanner", "Findings", "Avg Confidence"]
    scanner_rows = []
    for name in sorted(scanner_stats.keys()):
        s = scanner_stats[name]
        avg_conf = s["total_confidence"] / s["findings"] if s["findings"] > 0 else 0.0
        scanner_rows.append([
            name,
            s["findings"],
            "{:.2f}".format(avg_conf),
        ])

    _print_table(scanner_headers, scanner_rows)
    print()

    # ── False negatives (missed) ─────────────────────────────────────────

    if missed:
        print("--- False Negatives (Missed Adversarial Entries) ---")
        print()
        for entry in missed:
            print("  [{}] category={}, layer={}: {}".format(
                entry["id"],
                entry.get("category", "?"),
                entry.get("attack_layer", "?"),
                entry.get("description", "(no description)"),
            ))
        print()
    else:
        print("No false negatives — all adversarial entries detected.")
        print()

    # ── Summary verdict ──────────────────────────────────────────────────

    tpr = (detected_count / total * 100) if total > 0 else 0.0
    passed = tpr >= 90.0

    print("=" * 72)
    if passed:
        print("  RESULT: PASS  (TPR = {:.1f}% >= 90%)".format(tpr))
    else:
        print("  RESULT: FAIL  (TPR = {:.1f}% < 90%)".format(tpr))
    print("=" * 72)

    guard.close()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
