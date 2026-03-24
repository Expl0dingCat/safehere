#!/usr/bin/env python
"""Fair cross-system benchmark — levels the playing field between Safehere
and transformer-based prompt injection classifiers.

Two evaluation modes designed to remove home-field advantage:

  Mode A: "Tool-Output Wrapping"
    Takes the public deepset/prompt-injections dataset and embeds each sample
    inside a realistic tool-output structure (JSON API response, search result,
    DB row, etc.). This simulates the actual attack surface Safehere defends
    while giving transformer classifiers the same text to analyze.

  Mode B: "Raw Text Extraction"
    Takes Safehere's internal adversarial+benign corpus and feeds the raw text
    strings — stripped of any structural context — to all detectors identically.
    This removes Safehere's structural/schema advantages on its own data.

Both modes report identical metrics side-by-side for a direct comparison.

Usage:
    python benchmarks/bench_fair_comparison.py
"""

import json
import os
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from safehere._types import Action, Severity
from safehere.guard import ToolGuard

random.seed(42)

# ── Tool-output templates for wrapping ───────────────────────────────────────

TOOL_TEMPLATES = [
    # Search results
    lambda text: json.dumps({
        "results": [
            {"title": "Result 1", "url": "https://example.com/1",
             "snippet": "A relevant code snippet for your query."},
            {"title": "Result 2", "url": "https://example.com/2",
             "snippet": text},
            {"title": "Result 3", "url": "https://example.com/3",
             "snippet": "Another standard search result."},
        ],
        "total": 3,
    }),
    # Database row
    lambda text: json.dumps({
        "columns": ["id", "name", "content"],
        "rows": [
            [1, "record_a", "Normal data entry."],
            [2, "record_b", text],
            [3, "record_c", "Another record."],
        ],
        "row_count": 3,
    }),
    # API response with notes field
    lambda text: json.dumps({
        "status": "success",
        "data": {"id": 42, "value": 3.14, "notes": text},
        "timestamp": "2025-01-15T10:30:00Z",
    }),
    # File content
    lambda text: f"# Configuration File\nversion = 2.1.0\ndebug = false\n\n{text}\n\n# End of file\n",
    # Log entry
    lambda text: f"[2025-01-15 10:30:00] INFO  app.service - Request processed\n[2025-01-15 10:30:01] INFO  app.service - {text}\n[2025-01-15 10:30:02] INFO  app.service - Response sent\n",
    # Email/message body
    lambda text: json.dumps({
        "from": "user@example.com",
        "subject": "Re: Weekly Update",
        "body": text,
        "date": "2025-01-15",
    }),
    # Raw text (no wrapping — baseline)
    lambda text: text,
]

BENIGN_TOOL_TEMPLATES = [
    lambda text: json.dumps({
        "results": [
            {"title": "Documentation", "url": "https://docs.example.com",
             "snippet": text},
        ],
        "total": 1,
    }),
    lambda text: json.dumps({
        "columns": ["id", "description"],
        "rows": [[1, text]],
        "row_count": 1,
    }),
    lambda text: json.dumps({
        "status": "success",
        "data": {"message": text},
    }),
    lambda text: text,
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _fmt_pct(v):
    return "{:.1f}%".format(v * 100)


def _print_table(headers, rows, col_widths=None):
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            max_w = len(h)
            for row in rows:
                max_w = max(max_w, len(str(row[i])))
            col_widths.append(max_w + 2)
    header_line = " | ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print("-+-".join("-" * w for w in col_widths))
    for row in rows:
        print(" | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(headers))))


def compute_metrics(tp, fp, tn, fn, latencies):
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    fpr = fp / (fp + tn) if (fp + tn) else 0
    mean_lat = (sum(latencies) / len(latencies) * 1000) if latencies else 0
    p50_lat = sorted(latencies)[len(latencies) // 2] * 1000 if latencies else 0
    return {
        "total": total, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "f1": f1, "fpr": fpr, "mean_latency_ms": mean_lat, "p50_latency_ms": p50_lat,
    }


# ── Dataset loaders ──────────────────────────────────────────────────────────

def load_deepset_wrapped():
    """Load deepset/prompt-injections, wrap in tool-output structures."""
    from datasets import load_dataset
    samples = []

    for split in ("train", "test"):
        ds = load_dataset("deepset/prompt-injections", split=split)
        for row in ds:
            text = row["text"]
            label = int(row["label"])  # 1=injection, 0=benign

            if label == 1:
                # Wrap injection in a random tool output template
                template = random.choice(TOOL_TEMPLATES)
            else:
                template = random.choice(BENIGN_TOOL_TEMPLATES)

            wrapped = template(text)
            samples.append((wrapped, label, text))  # keep original for transformer

    random.shuffle(samples)
    return samples


def load_safehere_raw_text():
    """Load Safehere corpus as raw text (no structural advantage)."""
    corpus_dir = os.path.join(os.path.dirname(__file__), "corpus")
    samples = []

    adv_path = os.path.join(corpus_dir, "adversarial.json")
    if os.path.isfile(adv_path):
        with open(adv_path, "r", encoding="utf-8") as f:
            for entry in json.load(f):
                samples.append((entry["text"], 1, entry["text"]))

    # Load supplemental JSONL corpora
    for jsonl_name in ("new_adversarial.jsonl", "advanced_adversarial.jsonl",
                        "adversarial_expanded.jsonl"):
        jsonl_path = os.path.join(corpus_dir, jsonl_name)
        if os.path.isfile(jsonl_path):
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        samples.append((entry["text"], 1, entry["text"]))

    benign_path = os.path.join(corpus_dir, "benign.json")
    if os.path.isfile(benign_path):
        with open(benign_path, "r", encoding="utf-8") as f:
            for entry in json.load(f):
                samples.append((entry["text"], 0, entry["text"]))

    # Load supplemental benign JSONL
    for jsonl_name in ("new_benign.jsonl", "benign_expanded.jsonl"):
        jsonl_path = os.path.join(corpus_dir, jsonl_name)
        if os.path.isfile(jsonl_path):
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        samples.append((entry["text"], 0, entry["text"]))

    random.shuffle(samples)
    return samples


# ── Detector wrappers ────────────────────────────────────────────────────────

class SafehereBlockDetector:
    """Safehere using BLOCK/QUARANTINE action threshold (production mode)."""
    name = "Safehere (block)"

    def __init__(self):
        self.guard = ToolGuard()

    def predict(self, text):
        result = self.guard._scan_single("benchmark_tool", text, None)
        return 1 if result.action in (Action.BLOCK, Action.QUARANTINE) else 0

    def close(self):
        self.guard.close()


class SafehereDetectDetector:
    """Safehere using any-detection threshold (sensitivity mode)."""
    name = "Safehere (detect)"

    def __init__(self):
        self.guard = ToolGuard()

    def predict(self, text):
        result = self.guard._scan_single("benchmark_tool", text, None)
        has_detection = any(f.severity >= Severity.MEDIUM for f in result.findings)
        return 1 if has_detection else 0

    def close(self):
        self.guard.close()


class ProtectAIDetector:
    """protectai/deberta-v3-base-prompt-injection-v2"""
    name = "ProtectAI DeBERTa-v2"

    def __init__(self):
        from transformers import pipeline
        self.pipe = pipeline(
            "text-classification",
            model="protectai/deberta-v3-base-prompt-injection-v2",
            device="cpu",
            truncation=True,
            max_length=512,
        )

    def predict(self, text):
        result = self.pipe(text[:512], truncation=True)
        return 1 if result[0]["label"].upper() == "INJECTION" else 0

    def close(self):
        pass


class ProtectAIDetectorOnRawText:
    """Same DeBERTa model but always receives the raw unwrapped text."""
    name = "ProtectAI DeBERTa-v2 (raw)"

    def __init__(self):
        from transformers import pipeline
        self.pipe = pipeline(
            "text-classification",
            model="protectai/deberta-v3-base-prompt-injection-v2",
            device="cpu",
            truncation=True,
            max_length=512,
        )

    def predict(self, text):
        result = self.pipe(text[:512], truncation=True)
        return 1 if result[0]["label"].upper() == "INJECTION" else 0

    def close(self):
        pass


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_detectors(detectors, samples, use_raw_for=None):
    """Run all detectors on samples. use_raw_for is a set of detector names
    that should receive the raw (unwrapped) text instead of the wrapped version."""
    use_raw_for = use_raw_for or set()
    results = {}

    for detector in detectors:
        tp = fp = tn = fn = 0
        latencies = []

        for wrapped_text, label, raw_text in samples:
            input_text = raw_text if detector.name in use_raw_for else wrapped_text
            try:
                t0 = time.perf_counter()
                pred = detector.predict(input_text)
                dt = time.perf_counter() - t0
                latencies.append(dt)

                if pred == 1 and label == 1:
                    tp += 1
                elif pred == 1 and label == 0:
                    fp += 1
                elif pred == 0 and label == 0:
                    tn += 1
                else:
                    fn += 1
            except Exception:
                fn += 1 if label == 1 else 0

        metrics = compute_metrics(tp, fp, tn, fn, latencies)
        metrics["detector"] = detector.name
        results[detector.name] = metrics

    return results


def print_results_table(title, results_dict):
    print("=" * 95)
    print(f"  {title}")
    print("=" * 95)
    print()

    headers = ["Detector", "Accuracy", "Precision", "Recall/TPR", "F1", "FPR",
               "P50 Lat", "TP", "FP", "TN", "FN"]
    rows = []
    for r in results_dict.values():
        rows.append([
            r["detector"],
            _fmt_pct(r["accuracy"]),
            _fmt_pct(r["precision"]),
            _fmt_pct(r["recall"]),
            _fmt_pct(r["f1"]),
            _fmt_pct(r["fpr"]),
            "{:.1f}ms".format(r["p50_latency_ms"]),
            r["tp"], r["fp"], r["tn"], r["fn"],
        ])

    _print_table(headers, rows)
    print()

    # Winners
    if len(results_dict) >= 2:
        metrics_to_compare = [
            ("Best Accuracy", "accuracy", True),
            ("Best Precision", "precision", True),
            ("Best Recall/TPR", "recall", True),
            ("Best F1", "f1", True),
            ("Lowest FPR", "fpr", False),
            ("Fastest (P50)", "p50_latency_ms", False),
        ]
        print("  Winners:")
        rlist = list(results_dict.values())
        for label, key, higher_better in metrics_to_compare:
            best = max(rlist, key=lambda r: r[key]) if higher_better else min(rlist, key=lambda r: r[key])
            if key == "p50_latency_ms":
                val_str = "{:.1f}ms".format(best[key])
            else:
                val_str = _fmt_pct(best[key])
            print(f"    {label}: {best['detector']} ({val_str})")
        print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print()
    print("#" * 95)
    print("#  FAIR CROSS-SYSTEM BENCHMARK")
    print("#  Leveling the playing field between Safehere and transformer classifiers")
    print("#" * 95)
    print()

    # ── Load datasets ────────────────────────────────────────────────────
    print("Loading datasets...")

    deepset_wrapped = load_deepset_wrapped()
    n_inj = sum(1 for _, l, _ in deepset_wrapped if l == 1)
    n_ben = sum(1 for _, l, _ in deepset_wrapped if l == 0)
    print(f"  Mode A - deepset wrapped in tool outputs: {len(deepset_wrapped)} samples ({n_inj} injections, {n_ben} benign)")

    safehere_raw = load_safehere_raw_text()
    n_inj = sum(1 for _, l, _ in safehere_raw if l == 1)
    n_ben = sum(1 for _, l, _ in safehere_raw if l == 0)
    print(f"  Mode B - Safehere corpus raw text:        {len(safehere_raw)} samples ({n_inj} injections, {n_ben} benign)")
    print()

    # ── Initialize detectors ─────────────────────────────────────────────
    print("Initializing detectors...")

    sh_block = SafehereBlockDetector()
    print(f"  [OK] {sh_block.name}")

    sh_detect = SafehereDetectDetector()
    print(f"  [OK] {sh_detect.name}")

    detectors = [sh_block, sh_detect]

    try:
        pai = ProtectAIDetector()
        print(f"  [OK] {pai.name}")
        detectors.append(pai)
    except Exception as e:
        print(f"  [SKIP] ProtectAI: {e}")
        pai = None

    try:
        pai_raw = ProtectAIDetectorOnRawText()
        print(f"  [OK] {pai_raw.name}")
    except Exception:
        pai_raw = None

    print()

    # ══════════════════════════════════════════════════════════════════════
    # MODE A: deepset/prompt-injections wrapped in tool-output structures
    # ══════════════════════════════════════════════════════════════════════
    print("Running Mode A: deepset dataset wrapped in tool-output structures...")
    print("  (Safehere sees tool outputs; DeBERTa sees the same wrapped text)")
    print()

    results_a = evaluate_detectors(detectors, deepset_wrapped)

    # Also test DeBERTa on raw (unwrapped) text as a reference point
    if pai_raw:
        raw_results = evaluate_detectors(
            [pai_raw], deepset_wrapped,
            use_raw_for={pai_raw.name}
        )
        results_a.update(raw_results)

    print_results_table(
        "Mode A: Public Dataset (deepset) Wrapped in Tool-Output Structures",
        results_a
    )

    # ══════════════════════════════════════════════════════════════════════
    # MODE B: Safehere corpus as raw text (no structural advantage)
    # ══════════════════════════════════════════════════════════════════════
    print("Running Mode B: Safehere corpus as raw text (identical input to all)...")
    print("  (All detectors receive the exact same raw text strings)")
    print()

    results_b = evaluate_detectors(detectors[:3] if pai else detectors[:2], safehere_raw)

    print_results_table(
        "Mode B: Safehere Corpus as Raw Text (No Structural Advantage)",
        results_b
    )

    # ══════════════════════════════════════════════════════════════════════
    # CROSS-MODE SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 95)
    print("  CROSS-MODE SUMMARY")
    print("=" * 95)
    print()
    print("  Mode A tests the real threat model: injections hiding inside tool outputs.")
    print("  Mode B tests raw classification ability: same text to everyone.")
    print()

    all_detectors_names = set()
    for r in list(results_a.values()) + list(results_b.values()):
        all_detectors_names.add(r["detector"])

    headers = ["Detector", "Mode A F1", "Mode A FPR", "Mode B F1", "Mode B FPR"]
    rows = []
    for name in sorted(all_detectors_names):
        ra = results_a.get(name, {})
        rb = results_b.get(name, {})
        rows.append([
            name,
            _fmt_pct(ra["f1"]) if ra else "N/A",
            _fmt_pct(ra["fpr"]) if ra else "N/A",
            _fmt_pct(rb["f1"]) if rb else "N/A",
            _fmt_pct(rb["fpr"]) if rb else "N/A",
        ])

    _print_table(headers, rows)
    print()
    print("  Key:")
    print("    Mode A = public dataset in realistic tool outputs (fair to Safehere's design)")
    print("    Mode B = Safehere's corpus as raw text (fair to transformer classifiers)")
    print()

    # Cleanup
    for d in detectors:
        d.close()
    if pai_raw:
        pai_raw.close()

    print("Benchmark complete.")


if __name__ == "__main__":
    main()
