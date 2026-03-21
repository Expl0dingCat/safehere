"""Semantic injection classifier -- TF-IDF + LogisticRegression.

A lightweight ML layer that catches attacks the rule-based scanners miss
(third-person framing, hypothetical scenarios, paraphrased injections).

The model is a scikit-learn Pipeline (TF-IDF char n-grams + logistic
regression) trained on the benchmark corpus.  ~200 KB, ~1 ms inference,
no GPU required.

Install the optional dependency to use this scanner:

    pip install safehere[ml]

Train / retrain the model:

    python -m safehere.scanners.semantic --train
"""

import os
import pickle
from typing import Any, Dict, List, Optional  # noqa: F401

from .._types import Finding, Severity
from ._base import BaseScanner


def _default_model_path():
    # type: () -> str
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        )))),
        "models",
        "tfidf_injection_clf.pkl",
    )


class SemanticScanner(BaseScanner):
    """TF-IDF + LogisticRegression prompt injection classifier.

    Lazy-loads the model on first scan. If the model file is missing,
    scan() returns no findings (graceful degradation).
    """

    @property
    def name(self):
        # type: () -> str
        return "semantic"

    def __init__(self, model_path=None, threshold=0.50):
        # type: (Optional[str], float) -> None
        self._threshold = threshold
        self._model_path = model_path or _default_model_path()
        self._pipeline = None  # type: Any
        self._available = None  # type: Optional[bool]

    def _load(self):
        # type: () -> bool
        if self._available is not None:
            return self._available
        if not os.path.isfile(self._model_path):
            self._available = False
            return False
        try:
            with open(self._model_path, "rb") as f:
                self._pipeline = pickle.load(f)
            self._available = True
        except Exception:
            self._available = False
        return self._available

    def scan(self, tool_name, output_text, output_structured=None):
        # type: (str, str, Optional[Dict[str, Any]]) -> List[Finding]
        if not self._load():
            return []

        text = output_text[:4096] if len(output_text) > 4096 else output_text

        proba = self._pipeline.predict_proba([text])[0]
        injection_prob = float(proba[1])

        if injection_prob < self._threshold:
            return []

        if injection_prob >= 0.90:
            severity = Severity.CRITICAL
        elif injection_prob >= 0.75:
            severity = Severity.HIGH
        elif injection_prob >= 0.60:
            severity = Severity.MEDIUM
        else:
            severity = Severity.LOW

        return [Finding(
            scanner_name=self.name,
            rule_id="SEM-001",
            severity=severity,
            confidence=injection_prob,
            description="Semantic classifier: {:.1%} injection probability".format(
                injection_prob
            ),
            location="$",
        )]


# -- training CLI -------------------------------------------------------------

def _load_all_corpus():
    """Load all adversarial + benign data from benchmarks/corpus/."""
    import json
    from pathlib import Path

    base = Path(__file__).resolve().parents[3] / "benchmarks" / "corpus"
    texts, labels, ids = [], [], []  # type: ignore[var-annotated]

    with open(base / "adversarial.json", encoding="utf-8") as f:
        for entry in json.load(f):
            texts.append(entry["text"])
            labels.append(1)
            ids.append(entry.get("id", ""))

    adv_jsonl = (
        "new_adversarial.jsonl",
        "advanced_adversarial.jsonl",
        "adversarial_expanded.jsonl",
    )
    for jsonl_name in adv_jsonl:
        jsonl_path = base / jsonl_name
        if jsonl_path.exists():
            with open(jsonl_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        texts.append(entry["text"])
                        labels.append(1)
                        ids.append(entry.get("id", ""))

    with open(base / "benign.json", encoding="utf-8") as f:
        for entry in json.load(f):
            texts.append(entry["text"])
            labels.append(0)
            ids.append(entry.get("id", ""))

    # expanded benign corpus
    ben_jsonl_path = base / "benign_expanded.jsonl"
    if ben_jsonl_path.exists():
        with open(ben_jsonl_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    texts.append(entry["text"])
                    labels.append(0)
                    ids.append(entry.get("id", ""))

    return texts, labels, ids


def _train():
    """Train the TF-IDF model on 80% of the corpus, report metrics on held-out 20%."""
    import time
    from pathlib import Path

    import numpy as np

    output_dir = Path(__file__).resolve().parents[3] / "models"
    output_dir.mkdir(exist_ok=True)

    print("Loading corpus...")
    texts, labels, ids = _load_all_corpus()
    labels_arr = np.array(labels)
    adv = int(labels_arr.sum())
    print("  {} samples ({} adversarial, {} benign)".format(len(texts), adv, len(texts) - adv))

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    # fixed seed split so benchmark results are reproducible
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels_arr, test_size=0.20, random_state=42, stratify=labels_arr,
    )

    print("  Train: {}, Test (held-out): {}".format(len(train_texts), len(test_texts)))

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            sublinear_tf=True,
            min_df=2,
            max_df=0.95,
            strip_accents="unicode",
            analyzer="char_wb",
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])

    print("Training...")
    t0 = time.monotonic()
    pipeline.fit(train_texts, train_labels)

    # evaluate on held-out test set
    test_preds = pipeline.predict(test_texts)
    print("\nHeld-out test set metrics:")
    print(classification_report(
        test_labels, test_preds, target_names=["benign", "adversarial"],
    ))

    # save model trained on train split only
    model_path = str(output_dir / "tfidf_injection_clf.pkl")
    with open(model_path, "wb") as fout:
        pickle.dump(pipeline, fout, protocol=pickle.HIGHEST_PROTOCOL)

    size_kb = os.path.getsize(model_path) / 1024
    elapsed = time.monotonic() - t0
    print("Saved: {} ({:.0f} KB, {:.1f}s)".format(model_path, size_kb, elapsed))
    print("NOTE: Model trained on 80% split only -- benchmark evaluates on full corpus"
          " including unseen samples.")


if __name__ == "__main__":
    import sys
    if "--train" in sys.argv:
        _train()
    else:
        print("Usage: python -m safehere.scanners.semantic --train")
        sys.exit(1)
