#!/usr/bin/env python
"""Latency benchmark — measures scan performance across payload sizes.

Usage:
    python benchmarks/bench_latency.py
"""

import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from safehere._types import Action
from safehere.guard import ToolGuard
from safehere.scanners.anomaly import AnomalyScanner
from safehere.scanners.pattern import PatternScanner
from safehere.scanners.schema import SchemaDriftScanner


# ── Payload generators ───────────────────────────────────────────────────────

def _make_tiny():
    """~100 bytes: simple JSON object."""
    return '{"status": "ok", "value": 42}'


def _make_small():
    """~1 KB: repeated JSON key-value pairs."""
    pairs = {
        "field_{:03d}".format(i): "value_{:03d}".format(i)
        for i in range(30)
    }
    return json.dumps(pairs)


def _make_medium():
    """~10 KB: nested JSON with arrays."""
    data = {
        "metadata": {"version": "1.0", "count": 50, "page": 1},
        "results": [
            {
                "id": i,
                "title": "Result item number {}".format(i),
                "description": "This is a longer description for item {} with more text.".format(i),
                "tags": ["tag_a", "tag_b", "tag_c"],
                "score": 0.95 - i * 0.01,
                "nested": {"alpha": i * 10, "beta": "nested_value_{}".format(i)},
            }
            for i in range(40)
        ],
    }
    return json.dumps(data)


def _make_large():
    """~100 KB: large JSON array of objects."""
    data = [
        {
            "id": i,
            "name": "Entity {} with a moderately long name".format(i),
            "description": "A detailed description for entity {}. ".format(i) * 5,
            "attributes": {
                "attr_{}".format(j): "value_{}_{}".format(i, j)
                for j in range(10)
            },
            "measurements": [float(i + j * 0.1) for j in range(10)],
            "active": i % 2 == 0,
        }
        for i in range(200)
    ]
    return json.dumps(data)


def _make_xlarge():
    """~500 KB: very large payload."""
    data = [
        {
            "id": i,
            "name": "Large entity {} with an extended name field".format(i),
            "description": "Extended description for entity {}. ".format(i) * 15,
            "tags": ["category_{}".format(j) for j in range(20)],
            "data_block": {
                "field_{:03d}".format(j): "payload_data_{}_{}".format(i, j)
                for j in range(20)
            },
            "history": [
                {"ts": 1700000000 + k, "val": float(k)}
                for k in range(10)
            ],
        }
        for i in range(250)
    ]
    return json.dumps(data)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _fmt_size(nbytes):
    """Format byte count as human-readable string."""
    if nbytes < 1024:
        return "{} B".format(nbytes)
    elif nbytes < 1024 * 1024:
        return "{:.0f} KB".format(nbytes / 1024)
    else:
        return "{:.1f} MB".format(nbytes / (1024 * 1024))


def _print_table(headers, rows, col_widths=None):
    """Print a fixed-width table to stdout."""
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            max_w = len(h)
            for row in rows:
                max_w = max(max_w, len(str(row[i])))
            col_widths.append(max_w + 2)

    header_line = " | ".join(
        str(h).ljust(col_widths[i]) for i, h in enumerate(headers)
    )
    print(header_line)
    print("-+-".join("-" * w for w in col_widths))

    for row in rows:
        print(
            " | ".join(
                str(row[i]).ljust(col_widths[i]) for i in range(len(headers))
            )
        )


def _measure_latencies(guard, tool_name, payload, iterations):
    """Run scan iterations and return list of latency values in ms."""
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        guard._scan_single(tool_name, payload, None)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
    return latencies


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    iterations = 100

    payloads = [
        ("Tiny", _make_tiny()),
        ("Small", _make_small()),
        ("Medium", _make_medium()),
        ("Large", _make_large()),
        ("XLarge", _make_xlarge()),
    ]

    print("=" * 78)
    print("  safehere Latency Benchmark")
    print("=" * 78)
    print()
    print("Iterations per payload size: {}".format(iterations))
    print()

    # ── Section 1: Payload size vs latency ────────────────────────────────

    print("--- Payload Size vs Latency ---")
    print()

    size_headers = ["Payload Size", "Mean (ms)", "P50 (ms)", "P95 (ms)", "P99 (ms)", "Scans/sec"]
    size_rows = []
    size_results = {}  # label -> mean_ms, for pass/fail check

    for label, payload in payloads:
        guard = ToolGuard()
        size_str = _fmt_size(len(payload))

        # Warmup: run cold_start_count + 1 calls to populate anomaly stats
        for _ in range(6):
            guard._scan_single("latency_bench", payload, None)

        latencies = _measure_latencies(guard, "latency_bench", payload, iterations)
        guard.close()

        mean_ms = statistics.mean(latencies)
        p50 = statistics.median(latencies)
        sorted_lat = sorted(latencies)
        p95_idx = int(len(sorted_lat) * 0.95)
        p99_idx = int(len(sorted_lat) * 0.99)
        p95 = sorted_lat[min(p95_idx, len(sorted_lat) - 1)]
        p99 = sorted_lat[min(p99_idx, len(sorted_lat) - 1)]
        scans_sec = int(1000.0 / mean_ms) if mean_ms > 0 else 0

        size_results[label] = p50  # use P50 for pass/fail (less sensitive to GC spikes)

        size_rows.append([
            "{} ({})".format(label, size_str),
            "{:.2f}".format(mean_ms),
            "{:.2f}".format(p50),
            "{:.2f}".format(p95),
            "{:.2f}".format(p99),
            str(scans_sec),
        ])

    _print_table(size_headers, size_rows)
    print()

    # ── Section 2: Warmup cost ────────────────────────────────────────────

    print("--- Warmup Cost (Cold Start) ---")
    print()

    cold_start_count = AnomalyScanner.DEFAULT_COLD_START_COUNT
    warmup_calls = 10
    medium_payload = payloads[2][1]  # Medium payload

    guard = ToolGuard()
    warmup_latencies = []
    for i in range(warmup_calls):
        t0 = time.perf_counter()
        guard._scan_single("warmup_bench", medium_payload, None)
        t1 = time.perf_counter()
        warmup_latencies.append((t1 - t0) * 1000)

    # Steady-state: measure after warmup
    steady_latencies = _measure_latencies(guard, "warmup_bench", medium_payload, iterations)
    guard.close()

    cold_phase = warmup_latencies[:cold_start_count]
    warm_phase = warmup_latencies[cold_start_count:]

    print("  Cold start count (anomaly detector): {}".format(cold_start_count))
    print("  Mean latency (first {} calls, cold) : {:.2f} ms".format(
        cold_start_count, statistics.mean(cold_phase)))
    print("  Mean latency (calls {}-{}, warm)     : {:.2f} ms".format(
        cold_start_count + 1, warmup_calls, statistics.mean(warm_phase)))
    print("  Mean latency (steady state, n={})  : {:.2f} ms".format(
        iterations, statistics.mean(steady_latencies)))
    print()

    # ── Section 3: Per-scanner overhead ───────────────────────────────────

    print("--- Per-Scanner Overhead ---")
    print()

    scanner_configs = {
        "pattern": [PatternScanner()],
        "schema_drift": [SchemaDriftScanner()],
        "anomaly": [AnomalyScanner()],
    }

    # Full pipeline baseline
    full_guard = ToolGuard()
    for _ in range(6):
        full_guard._scan_single("scanner_bench", medium_payload, None)
    full_latencies = _measure_latencies(full_guard, "scanner_bench", medium_payload, iterations)
    full_mean = statistics.mean(full_latencies)
    full_guard.close()

    scanner_headers = ["Scanner", "Mean (ms)", "% of Total"]
    scanner_rows = []
    scanner_total = 0.0

    for scanner_name, scanner_list in scanner_configs.items():
        guard = ToolGuard(scanners=scanner_list)
        # Warmup for anomaly scanner
        for _ in range(6):
            guard._scan_single("scanner_bench", medium_payload, None)
        lats = _measure_latencies(guard, "scanner_bench", medium_payload, iterations)
        guard.close()

        mean_ms = statistics.mean(lats)
        scanner_total += mean_ms
        pct = (mean_ms / full_mean * 100) if full_mean > 0 else 0
        scanner_rows.append([
            scanner_name,
            "{:.2f}".format(mean_ms),
            "{:.0f}%".format(pct),
        ])

    # Pipeline overhead = full - sum of individual scanners
    overhead = max(0.0, full_mean - scanner_total)
    overhead_pct = (overhead / full_mean * 100) if full_mean > 0 else 0
    scanner_rows.append([
        "pipeline_overhead",
        "{:.2f}".format(overhead),
        "{:.0f}%".format(overhead_pct),
    ])

    scanner_rows.append([
        "TOTAL (full pipeline)",
        "{:.2f}".format(full_mean),
        "100%",
    ])

    _print_table(scanner_headers, scanner_rows)
    print()

    # ── Summary ───────────────────────────────────────────────────────────

    medium_ok = size_results.get("Medium", 999) < 5.0
    large_ok = size_results.get("Large", 999) < 20.0

    print("=" * 78)
    print("  Pass Criteria (P50 median):")
    print("    Medium (10 KB) P50 < 5 ms   : {} ({:.2f} ms)".format(
        "PASS" if medium_ok else "FAIL", size_results.get("Medium", 0)))
    print("    Large (100 KB) P50 < 20 ms  : {} ({:.2f} ms)".format(
        "PASS" if large_ok else "FAIL", size_results.get("Large", 0)))
    print()

    passed = medium_ok and large_ok
    if passed:
        print("  RESULT: PASS")
    else:
        print("  RESULT: FAIL")
    print("=" * 78)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
