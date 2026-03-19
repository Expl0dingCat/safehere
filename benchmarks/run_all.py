#!/usr/bin/env python
"""Unified benchmark runner — runs all benchmark scripts and reports results.

Usage:
    python benchmarks/run_all.py
"""

import os
import subprocess
import sys
import time


# ── Configuration ────────────────────────────────────────────────────────────

BENCHMARKS = [
    ("bench_detection", "bench_detection.py"),
    ("bench_false_positives", "bench_false_positives.py"),
    ("bench_latency", "bench_latency.py"),
    ("bench_comparison", "bench_comparison.py"),
]

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Helpers ──────────────────────────────────────────────────────────────────

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


def _run_benchmark(name, script_path):
    """Run a single benchmark script and return (status, duration_s, output)."""
    if not os.path.isfile(script_path):
        return "SKIP", 0.0, "Script not found: {}".format(script_path)

    t0 = time.monotonic()
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout per benchmark
            cwd=os.path.dirname(script_path),
        )
        elapsed = time.monotonic() - t0
        output = result.stdout + result.stderr

        # Determine pass/fail from output and exit code
        if result.returncode == 0:
            status = "PASS"
        else:
            # Check if output explicitly says PASS (shouldn't happen with nonzero exit)
            status = "FAIL"

        # Double-check: look for explicit PASS/FAIL in output
        output_upper = output.upper()
        if "RESULT: PASS" in output_upper:
            status = "PASS"
        elif "RESULT: FAIL" in output_upper:
            status = "FAIL"

        return status, elapsed, output

    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        return "TIMEOUT", elapsed, "Benchmark timed out after 300 seconds"
    except Exception as e:
        elapsed = time.monotonic() - t0
        return "ERROR", elapsed, "Failed to run benchmark: {}".format(e)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  safehere Benchmark Suite")
    print("=" * 72)
    print()

    results = []
    all_passed = True
    any_ran = False

    for name, filename in BENCHMARKS:
        script_path = os.path.join(BENCH_DIR, filename)

        print("-" * 72)
        print("  Running: {} ({})".format(name, filename))
        print("-" * 72)

        status, duration, output = _run_benchmark(name, script_path)

        if status == "SKIP":
            print("  SKIPPED: {}".format(output))
            print()
        else:
            any_ran = True
            # Print benchmark output indented
            for line in output.rstrip().split("\n"):
                print("  | {}".format(line))
            print()

        if status not in ("PASS", "SKIP"):
            all_passed = False

        results.append((name, status, duration))

    # ── Summary table ─────────────────────────────────────────────────────

    print()
    print("=" * 72)
    print("  Summary")
    print("=" * 72)
    print()

    summary_headers = ["Benchmark", "Status", "Duration"]
    summary_rows = []
    for name, status, duration in results:
        if status == "SKIP":
            dur_str = "-"
        else:
            dur_str = "{:.1f}s".format(duration)
        summary_rows.append([name, status, dur_str])

    _print_table(summary_headers, summary_rows)
    print()

    # Count stats
    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = sum(1 for _, s, _ in results if s == "FAIL")
    skipped = sum(1 for _, s, _ in results if s == "SKIP")
    errored = sum(1 for _, s, _ in results if s in ("ERROR", "TIMEOUT"))

    print("  Passed: {}  Failed: {}  Skipped: {}  Errors: {}".format(
        passed, failed, skipped, errored))
    print()

    if not any_ran:
        print("  No benchmarks were executed.")
        print("=" * 72)
        sys.exit(1)

    if all_passed:
        print("  OVERALL: PASS")
    else:
        print("  OVERALL: FAIL")
    print("=" * 72)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
