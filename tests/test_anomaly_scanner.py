"""Tests for safehere.scanners.anomaly.AnomalyScanner."""

import json
import random
import string

import pytest

from safehere.scanners.anomaly import AnomalyScanner, _shannon_entropy, _natural_language_ratio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_output(data):
    """Return a compact JSON string from a dict."""
    return json.dumps(data, ensure_ascii=False)


def _make_varied_length_outputs(count, base_len=100, variance=20, seed=0):
    """Generate outputs with genuinely different lengths to build non-zero stddev.

    Each output is a JSON object with a variable-length description field.
    """
    rng = random.Random(seed)
    outputs = []
    for _ in range(count):
        pad_len = base_len + rng.randint(-variance, variance)
        data = {"temp": rng.randint(65, 80), "desc": "x" * pad_len}
        outputs.append(_json_output(data))
    return outputs


def _feed_outputs(scanner, tool_name, outputs):
    """Feed a list of output strings to a scanner."""
    for out in outputs:
        scanner.scan(tool_name, out)


# ---------------------------------------------------------------------------
# 1. Cold start: first N outputs produce no findings (default cold_start=5)
# ---------------------------------------------------------------------------

class TestColdStart:
    def test_first_outputs_produce_no_findings(self):
        scanner = AnomalyScanner()
        for i in range(scanner.DEFAULT_COLD_START_COUNT):
            findings = scanner.scan("weather", _json_output({"temp": 72}))
            assert findings == [], (
                f"Expected no findings during cold start (output #{i})"
            )

    def test_custom_cold_start_count(self):
        scanner = AnomalyScanner(cold_start_count=3)
        for i in range(3):
            findings = scanner.scan("weather", _json_output({"temp": 72}))
            assert findings == [], (
                f"Expected no findings during cold start with count=3 (output #{i})"
            )


# ---------------------------------------------------------------------------
# 2. After baseline established, normal outputs produce no findings
# ---------------------------------------------------------------------------

class TestBaselineNormal:
    def test_normal_outputs_after_baseline_are_clean(self):
        scanner = AnomalyScanner()
        baseline = _make_varied_length_outputs(20, base_len=100, variance=20, seed=0)
        _feed_outputs(scanner, "weather", baseline)

        # Additional outputs within the same range should be clean
        more = _make_varied_length_outputs(5, base_len=100, variance=20, seed=99)
        for out in more:
            findings = scanner.scan("weather", out)
            assert findings == []


# ---------------------------------------------------------------------------
# 3. Length spike (10x longer output) produces ANOM-LENGTH-001
# ---------------------------------------------------------------------------

class TestLengthSpike:
    def test_length_spike_detected(self):
        scanner = AnomalyScanner()
        baseline = _make_varied_length_outputs(20, base_len=100, variance=20, seed=0)
        _feed_outputs(scanner, "weather", baseline)

        # Spike: massively longer than any baseline output
        spike = _json_output({"temp": 72, "desc": "x" * 5000})
        findings = scanner.scan("weather", spike)

        rule_ids = [f.rule_id for f in findings]
        assert "ANOM-LENGTH-001" in rule_ids


# ---------------------------------------------------------------------------
# 4. Entropy spike produces ANOM-ENTROPY-001
# ---------------------------------------------------------------------------

class TestEntropySpike:
    def test_entropy_spike_detected(self):
        scanner = AnomalyScanner()

        # Baseline: very low-entropy text (repetitive character with slight
        # length variance to ensure non-zero stddev on all metrics)
        rng = random.Random(42)
        for i in range(20):
            n = 200 + rng.randint(-5, 5)
            low_entropy = "a" * n + chr(ord("b") + (i % 3))
            scanner.scan("crypto", low_entropy)

        # Verify baseline entropy stats have non-zero stddev
        stats = scanner._get_tool_stats("crypto")
        assert stats["entropy"].window_stddev > 0, (
            "Entropy stddev is zero; baseline values are all identical"
        )

        # Spike: high-entropy output of similar length
        rng2 = random.Random(99)
        chars = list(string.printable.strip()) * 4
        rng2.shuffle(chars)
        high_entropy = "".join(chars[:200])
        findings = scanner.scan("crypto", high_entropy)

        rule_ids = [f.rule_id for f in findings]
        assert "ANOM-ENTROPY-001" in rule_ids


# ---------------------------------------------------------------------------
# 5. NL ratio jump (JSON baseline then prose output) produces ANOM-NL-RATIO-001
# ---------------------------------------------------------------------------

class TestNLRatioJump:
    def test_nl_ratio_jump_detected(self):
        scanner = AnomalyScanner()

        # Baseline: text with low but varying NL ratios.
        # Mix of numeric codes and a few word-like tokens to produce
        # non-zero but small NL ratios with variance.
        rng = random.Random(42)
        for i in range(20):
            # "code=12345 ref=67890 idx=3" -> tokens like "code=12345"
            # are NOT word-like (contain '='), so NL ratio stays low.
            # Vary token count so NL ratio has some variance.
            num_codes = rng.randint(5, 10)
            tokens = [f"{rng.randint(10000,99999)}:{rng.randint(10000,99999)}"
                      for _ in range(num_codes)]
            # Add 0-2 word-like tokens to create slight NL-ratio variance
            n_words = rng.randint(0, 2)
            for _ in range(n_words):
                tokens.append("ok")
            rng.shuffle(tokens)
            text = " ".join(tokens)
            scanner.scan("api", text)

        # Verify baseline NL ratio stats have non-zero stddev
        stats = scanner._get_tool_stats("api")
        assert stats["nl_ratio"].window_stddev > 0, (
            "NL ratio stddev is zero; need more variance in baseline"
        )
        baseline_mean = stats["nl_ratio"].window_mean
        assert baseline_mean < 0.3, (
            f"Baseline NL ratio mean {baseline_mean:.2f} should be low"
        )

        # Spike: purely prose text (high NL ratio ~1.0), similar length
        # to baseline so length anomaly doesn't dominate.
        baseline_length_mean = stats["length"].window_mean
        word_count = max(10, int(baseline_length_mean / 5))
        prose = " ".join(["word"] * word_count)
        findings = scanner.scan("api", prose)

        rule_ids = [f.rule_id for f in findings]
        assert "ANOM-NL-RATIO-001" in rule_ids


# ---------------------------------------------------------------------------
# 6. Reset clears all stats
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_stats(self):
        scanner = AnomalyScanner()
        baseline = _make_varied_length_outputs(10)
        _feed_outputs(scanner, "weather", baseline)
        assert scanner._stats != {}

        scanner.reset()
        assert scanner._stats == {}

    def test_after_reset_cold_start_restarts(self):
        scanner = AnomalyScanner()
        baseline = _make_varied_length_outputs(20)
        _feed_outputs(scanner, "weather", baseline)

        scanner.reset()

        # After reset, we are back in cold start -- even a huge spike should
        # not be flagged because the scanner hasn't seen enough samples yet.
        spike = _json_output({"temp": 72, "desc": "x" * 10000})
        findings = scanner.scan("weather", spike)
        assert findings == [], "Expected no findings during cold start after reset"


# ---------------------------------------------------------------------------
# 7. Custom thresholds work (lower z_threshold catches smaller deviations)
# ---------------------------------------------------------------------------

class TestCustomThresholds:
    def test_lower_z_threshold_catches_smaller_deviations(self):
        # With z_threshold=1.0, even a moderate deviation should be caught.
        scanner_strict = AnomalyScanner(z_threshold=1.0)
        scanner_default = AnomalyScanner()

        # Build identical baselines (same seed)
        baseline = _make_varied_length_outputs(20, base_len=100, variance=20, seed=0)
        _feed_outputs(scanner_strict, "tool", baseline)
        _feed_outputs(scanner_default, "tool", baseline)

        # Moderate spike: longer than baseline mean but not extreme
        # baseline ~100 chars in desc -> total ~120. 3x ~ 360 chars desc -> ~380 total
        moderate_spike = _json_output({"temp": 72, "desc": "x" * 300})
        findings_strict = scanner_strict.scan("tool", moderate_spike)
        findings_default = scanner_default.scan("tool", moderate_spike)

        strict_ids = {f.rule_id for f in findings_strict}
        default_ids = {f.rule_id for f in findings_default}

        # The strict scanner should catch the length anomaly
        assert "ANOM-LENGTH-001" in strict_ids
        # The strict scanner should find at least as many issues as default
        assert len(findings_strict) >= len(findings_default)
