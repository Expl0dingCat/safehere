"""exception hierarchy for safehere."""


class SafeHereError(Exception):
    """base exception for all safehere errors."""


class QuarantineError(SafeHereError):
    """raised when a tool output triggers the quarantine policy."""

    def __init__(self, scan_result, tool_name):
        # type: (Any, str) -> None
        self.scan_result = scan_result
        self.tool_name = tool_name
        super().__init__(
            "Tool '{}' output quarantined: {}".format(
                tool_name, scan_result.combined_score
            )
        )


class ConfigurationError(SafeHereError):
    """raised for invalid configuration."""


class ScanError(SafeHereError):
    """raised when a scanner itself fails."""
