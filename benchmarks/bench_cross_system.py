#!/usr/bin/env python
"""Cross-system benchmark — compares Safehere against public prompt-injection
detection models on both Safehere's internal corpus and the public
deepset/prompt-injections dataset from HuggingFace.

Competitors:
  1. protectai/deberta-v3-base-prompt-injection-v2  (DeBERTa fine-tune)
  2. meta-llama/Prompt-Guard-86M                    (mDeBERTa multi-label)

Metrics: Accuracy, Precision, Recall (TPR), F1, FPR, mean latency.

Usage:
    python benchmarks/bench_cross_system.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ── Safehere imports ─────────────────────────────────────────────────────────
from safehere._types import Action  # noqa: E402
from safehere.guard import ToolGuard  # noqa: E402

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


# ── Dataset loaders ──────────────────────────────────────────────────────────

def load_safehere_corpus():
    """Load Safehere's adversarial + benign corpora as (text, label) pairs."""
    corpus_dir = os.path.join(os.path.dirname(__file__), "corpus")
    samples = []

    adv_path = os.path.join(corpus_dir, "adversarial.json")
    if os.path.isfile(adv_path):
        with open(adv_path, "r", encoding="utf-8") as f:
            for entry in json.load(f):
                samples.append((entry["text"], 1))  # 1 = injection

    benign_path = os.path.join(corpus_dir, "benign.json")
    if os.path.isfile(benign_path):
        with open(benign_path, "r", encoding="utf-8") as f:
            for entry in json.load(f):
                samples.append((entry["text"], 0))  # 0 = benign

    return samples


def load_deepset_dataset():
    """Load the public deepset/prompt-injections dataset from HuggingFace."""
    try:
        from datasets import load_dataset
        ds = load_dataset("deepset/prompt-injections", split="test")
        samples = []
        for row in ds:
            text = row["text"]
            label = int(row["label"])  # 1 = injection, 0 = benign
            samples.append((text, label))
        return samples
    except Exception as e:
        print(f"  Warning: could not load deepset dataset: {e}")
        # Fallback: try train split
        try:
            ds = load_dataset("deepset/prompt-injections", split="train")
            samples = []
            for row in ds:
                samples.append((row["text"], int(row["label"])))
            return samples
        except Exception as e2:
            print(f"  Warning: fallback also failed: {e2}")
            return []


# ── Detector wrappers ────────────────────────────────────────────────────────

class SafehereDetector:
    """Wraps Safehere ToolGuard for binary classification."""

    name = "Safehere v1.0"

    def __init__(self):
        self.guard = ToolGuard()

    def predict(self, text: str) -> int:
        result = self.guard._scan_single("benchmark_tool", text, None)
        # Use action-based classification (matches production behavior)
        is_blocked = result.action in (Action.BLOCK, Action.QUARANTINE)
        return 1 if is_blocked else 0

    def close(self):
        self.guard.close()


class ProtectAIDetector:
    """Wraps protectai/deberta-v3-base-prompt-injection-v2."""

    name = "ProtectAI DeBERTa-v2"

    def __init__(self):
        from transformers import pipeline
        self.pipe = pipeline(
            "text-classification",
            model="protectai/deberta-v3-base-prompt-injection-v2",
            device="cpu",  # keep consistent for latency comparison
        )

    def predict(self, text: str) -> int:
        # Truncate to model's max length to avoid errors
        result = self.pipe(text[:512], truncation=True)
        label = result[0]["label"]
        # Model returns "INJECTION" or "SAFE"
        return 1 if label.upper() == "INJECTION" else 0

    def close(self):
        pass


class PromptGuardDetector:
    """Wraps meta-llama/Prompt-Guard-86M."""

    name = "Meta Prompt-Guard-86M"

    def __init__(self):
        from transformers import pipeline
        self.pipe = pipeline(
            "text-classification",
            model="meta-llama/Prompt-Guard-86M",
            device="cpu",
        )

    def predict(self, text: str) -> int:
        result = self.pipe(text[:512], truncation=True)
        label = result[0]["label"]
        # Returns "BENIGN", "INJECTION", or "JAILBREAK"
        return 1 if label.upper() in ("INJECTION", "JAILBREAK") else 0

    def close(self):
        pass


# ── Evaluation engine ────────────────────────────────────────────────────────

def evaluate(detector, samples, dataset_name, max_samples=None):
    """Run detector on samples and compute metrics."""
    if max_samples and len(samples) > max_samples:
        samples = samples[:max_samples]

    tp = fp = tn = fn = 0
    latencies = []
    errors = 0

    for text, label in samples:
        try:
            t0 = time.perf_counter()
            pred = detector.predict(text)
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
            errors += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    fpr = fp / (fp + tn) if (fp + tn) else 0
    mean_latency_ms = (sum(latencies) / len(latencies) * 1000) if latencies else 0
    p50_latency_ms = sorted(latencies)[len(latencies) // 2] * 1000 if latencies else 0

    return {
        "detector": detector.name,
        "dataset": dataset_name,
        "total": total,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "mean_latency_ms": mean_latency_ms,
        "p50_latency_ms": p50_latency_ms,
        "errors": errors,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 90)
    print("  Safehere Cross-System Benchmark")
    print("  Comparing prompt injection detectors on public & internal datasets")
    print("=" * 90)
    print()

    # ── Load datasets ────────────────────────────────────────────────────
    print("Loading datasets...")
    safehere_corpus = load_safehere_corpus()
    print(f"  Safehere internal corpus: {len(safehere_corpus)} samples "
          f"({sum(1 for _, l in safehere_corpus if l == 1)} adversarial, "
          f"{sum(1 for _, l in safehere_corpus if l == 0)} benign)")

    deepset_corpus = load_deepset_dataset()
    if deepset_corpus:
        print(f"  deepset/prompt-injections: {len(deepset_corpus)} samples "
              f"({sum(1 for _, l in deepset_corpus if l == 1)} adversarial, "
              f"{sum(1 for _, l in deepset_corpus if l == 0)} benign)")
    print()

    datasets = [("Safehere Corpus", safehere_corpus)]
    if deepset_corpus:
        datasets.append(("deepset/prompt-injections", deepset_corpus))

    # ── Initialize detectors ─────────────────────────────────────────────
    detectors = []

    print("Initializing detectors...")

    # Safehere (always available)
    sh = SafehereDetector()
    detectors.append(sh)
    print(f"  [OK] {sh.name}")

    # ProtectAI DeBERTa
    try:
        pai = ProtectAIDetector()
        detectors.append(pai)
        print(f"  [OK] {pai.name}")
    except Exception as e:
        print(f"  [SKIP] ProtectAI DeBERTa-v2: {e}")

    # Meta Prompt Guard
    try:
        pg = PromptGuardDetector()
        detectors.append(pg)
        print(f"  [OK] {pg.name}")
    except Exception as e:
        print(f"  [SKIP] Meta Prompt-Guard-86M: {e}")

    if len(detectors) < 2:
        print()
        print("WARNING: Only one detector available. Install transformers models for comparison.")
        print("  pip install transformers torch")
        print("  Models will be auto-downloaded from HuggingFace on first use.")
        print()

    print()

    # ── Run benchmarks ───────────────────────────────────────────────────
    all_results = []

    for ds_name, samples in datasets:
        print(f"--- Evaluating on: {ds_name} ({len(samples)} samples) ---")
        print()

        for detector in detectors:
            sys.stdout.write(f"  Running {detector.name}...")
            sys.stdout.flush()
            result = evaluate(detector, samples, ds_name)
            all_results.append(result)
            print(f" done ({result['mean_latency_ms']:.1f} ms/sample avg)")

        print()

    # ── Results tables ───────────────────────────────────────────────────

    for ds_name, _ in datasets:
        ds_results = [r for r in all_results if r["dataset"] == ds_name]
        if not ds_results:
            continue

        print("=" * 90)
        print(f"  Results on: {ds_name}")
        print("=" * 90)
        print()

        headers = [
            "Detector", "Accuracy", "Precision", "Recall/TPR", "F1", "FPR",
            "P50 Latency", "TP", "FP", "TN", "FN",
        ]
        rows = []
        for r in ds_results:
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

        # Highlight best in each metric
        if len(ds_results) >= 2:
            metrics = [
                ("Highest Accuracy", "accuracy", True),
                ("Highest Precision", "precision", True),
                ("Highest Recall/TPR", "recall", True),
                ("Highest F1", "f1", True),
                ("Lowest FPR", "fpr", False),
                ("Fastest (P50)", "p50_latency_ms", False),
            ]
            print("  Winners:")
            for label, key, higher_is_better in metrics:
                if higher_is_better:
                    best = max(ds_results, key=lambda r: r[key])
                else:
                    best = min(ds_results, key=lambda r: r[key])
                if key == "p50_latency_ms":
                    val_str = "{:.1f}ms".format(best[key])
                else:
                    val_str = _fmt_pct(best[key])
                print(f"    {label}: {best['detector']} ({val_str})")
            print()

    # ── Summary ──────────────────────────────────────────────────────────

    print("=" * 90)
    print("  SUMMARY")
    print("=" * 90)
    print()

    for detector in detectors:
        det_results = [r for r in all_results if r["detector"] == detector.name]
        avg_f1 = sum(r["f1"] for r in det_results) / len(det_results) if det_results else 0
        avg_fpr = sum(r["fpr"] for r in det_results) / len(det_results) if det_results else 0
        avg_recall = sum(r["recall"] for r in det_results) / len(det_results) if det_results else 0
        avg_lat = sum(r["p50_latency_ms"] for r in det_results) / len(det_results) if det_results else 0

        print(f"  {detector.name}:")
        print(f"    Avg F1: {_fmt_pct(avg_f1)}  |  Avg Recall: {_fmt_pct(avg_recall)}  "
              f"|  Avg FPR: {_fmt_pct(avg_fpr)}  |  Avg P50 Latency: {avg_lat:.1f}ms")
        print()

    # Cleanup
    for d in detectors:
        d.close()

    print("Benchmark complete.")


if __name__ == "__main__":
    main()
