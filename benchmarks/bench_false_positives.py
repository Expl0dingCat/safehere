#!/usr/bin/env python
"""False positive rate benchmark — measures FPR and false alert rate against benign corpus.

Usage:
    python benchmarks/bench_false_positives.py
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
    """Load a corpus JSON file."""
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


def _scan_entry(guard, entry):
    """Scan a single benign entry and return (result, is_fp, is_alert)."""
    text = entry["text"]
    tool_name = entry.get("tool_name", entry.get("tool_type", "test_tool"))

    result = guard._scan_single(tool_name, text, None)

    # False positive: benign entry is blocked or quarantined
    is_fp = result.action in (Action.BLOCK, Action.QUARANTINE)

    # False alert: any finding with severity >= HIGH
    is_alert = any(f.severity >= Severity.HIGH for f in result.findings)

    return result, is_fp, is_alert


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    corpus_path = os.path.join(os.path.dirname(__file__), "corpus", "benign.json")

    if not os.path.isfile(corpus_path):
        print("ERROR: corpus file not found: {}".format(corpus_path))
        sys.exit(1)

    entries = _load_corpus(corpus_path)

    # Split entries: main benign corpus vs tricky near-miss entries
    main_entries = [e for e in entries if not e.get("tricky", False)]
    tricky_entries = [e for e in entries if e.get("tricky", False)]

    # If there are no entries tagged "tricky", treat all as main
    if not tricky_entries and main_entries:
        pass  # all entries in main_entries, tricky stays empty

    total_main = len(main_entries)
    total_tricky = len(tricky_entries)

    print("=" * 72)
    print("  safehere False Positive Rate Benchmark")
    print("=" * 72)
    print()
    print("Corpus: {} benign entries ({} main, {} tricky)".format(
        len(entries), total_main, total_tricky
    ))
    print()

    guard = ToolGuard()

    # ── Main benign corpus ───────────────────────────────────────────────

    fp_count = 0
    alert_count = 0
    fp_details = []

    # Per-tool-type stats
    tool_stats = defaultdict(lambda: {
        "count": 0, "fp": 0, "alerts": 0,
    })

    t_start = time.monotonic()

    for entry in main_entries:
        tool_type = entry.get("tool_type", "unknown")
        result, is_fp, is_alert = _scan_entry(guard, entry)

        tool_stats[tool_type]["count"] += 1

        if is_fp:
            fp_count += 1
            tool_stats[tool_type]["fp"] += 1
            # Collect details for reporting
            triggered_rules = [
                "{} (sev={}, conf={:.2f})".format(f.rule_id, f.severity.name, f.confidence)
                for f in result.findings
            ]
            fp_details.append({
                "id": entry["id"],
                "tool_type": tool_type,
                "action": result.action.value,
                "description": entry.get("description", "(no description)"),
                "rules": triggered_rules,
            })

        if is_alert:
            alert_count += 1
            tool_stats[tool_type]["alerts"] += 1

    elapsed = time.monotonic() - t_start

    # ── Overall metrics ──────────────────────────────────────────────────

    print("--- Overall Metrics (Main Corpus) ---")
    print()
    print("  Total benign entries       : {}".format(total_main))
    print("  False positives (blocked)  : {}".format(fp_count))
    print("  False Positive Rate (FPR)  : {}".format(_fmt_pct(fp_count, total_main)))
    print("  False alerts (sev >= HIGH) : {}".format(alert_count))
    print("  False Alert Rate           : {}".format(_fmt_pct(alert_count, total_main)))
    print("  Scan time                  : {:.1f} ms".format(elapsed * 1000))
    print()

    # ── Per-tool-type breakdown ──────────────────────────────────────────

    print("--- Per-Tool-Type Breakdown ---")
    print()

    tt_headers = ["Tool Type", "Count", "FP", "FPR", "Alerts", "Alert Rate"]
    tt_rows = []
    for tt in sorted(tool_stats.keys()):
        s = tool_stats[tt]
        tt_rows.append([
            tt,
            s["count"],
            s["fp"],
            _fmt_pct(s["fp"], s["count"]),
            s["alerts"],
            _fmt_pct(s["alerts"], s["count"]),
        ])

    _print_table(tt_headers, tt_rows)
    print()

    # ── False positive details ───────────────────────────────────────────

    if fp_details:
        print("--- False Positive Details ---")
        print()
        for fp in fp_details:
            print("  [{}] tool_type={}, action={}".format(
                fp["id"], fp["tool_type"], fp["action"],
            ))
            print("    Description: {}".format(fp["description"]))
            for rule in fp["rules"]:
                print("    Rule: {}".format(rule))
            print()
    else:
        print("No false positives in main benign corpus.")
        print()

    # ── Tricky near-miss section ─────────────────────────────────────────

    print("--- Tricky Near-Miss Benign Entries ---")
    print()

    if not tricky_entries:
        print("  (No entries tagged as tricky in corpus)")
        print()
    else:
        tricky_fp = 0
        tricky_alert = 0
        tricky_fp_details = []

        for entry in tricky_entries:
            result, is_fp, is_alert = _scan_entry(guard, entry)

            if is_fp:
                tricky_fp += 1
                triggered_rules = [
                    "{} (sev={}, conf={:.2f})".format(
                        f.rule_id, f.severity.name, f.confidence
                    )
                    for f in result.findings
                ]
                tricky_fp_details.append({
                    "id": entry["id"],
                    "tool_type": entry.get("tool_type", "unknown"),
                    "action": result.action.value,
                    "description": entry.get("description", "(no description)"),
                    "rules": triggered_rules,
                })

            if is_alert:
                tricky_alert += 1

        print("  Total tricky entries       : {}".format(total_tricky))
        print("  False positives (blocked)  : {}".format(tricky_fp))
        print("  FPR (tricky)               : {}".format(_fmt_pct(tricky_fp, total_tricky)))
        print("  False alerts (sev >= HIGH) : {}".format(tricky_alert))
        print("  Alert Rate (tricky)        : {}".format(_fmt_pct(tricky_alert, total_tricky)))
        print()

        if tricky_fp_details:
            print("  Tricky false positive details:")
            for fp in tricky_fp_details:
                print("    [{}] tool_type={}, action={}".format(
                    fp["id"], fp["tool_type"], fp["action"],
                ))
                print("      Description: {}".format(fp["description"]))
                for rule in fp["rules"]:
                    print("      Rule: {}".format(rule))
            print()
        else:
            print("  No false positives in tricky near-miss entries.")
            print()

    # ── Summary verdict ──────────────────────────────────────────────────

    fpr = (fp_count / total_main * 100) if total_main > 0 else 0.0
    alert_rate = (alert_count / total_main * 100) if total_main > 0 else 0.0

    fpr_pass = fpr < 5.0
    alert_pass = alert_rate < 15.0

    print("=" * 72)
    if fpr_pass:
        print("  FPR:        PASS  (FPR = {:.1f}% < 5%)".format(fpr))
    else:
        print("  FPR:        FAIL  (FPR = {:.1f}% >= 5%)".format(fpr))

    if alert_pass:
        print("  Alert Rate: PASS  (Alert Rate = {:.1f}% < 15%)".format(alert_rate))
    else:
        print("  Alert Rate: FAIL  (Alert Rate = {:.1f}% >= 15%)".format(alert_rate))

    overall_pass = fpr_pass and alert_pass
    print()
    if overall_pass:
        print("  OVERALL: PASS")
    else:
        print("  OVERALL: FAIL")
    print("=" * 72)

    guard.close()
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
