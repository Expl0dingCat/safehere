"""abstract base class for all scanners."""

import abc
from typing import Any, Dict, List, Optional

from .._types import Finding


class BaseScanner(abc.ABC):
    """abstract base class for safehere scanners."""

    @property
    @abc.abstractmethod
    def name(self):
        # type: () -> str
        """unique identifier for this scanner."""
        ...

    @abc.abstractmethod
    def scan(self, tool_name, output_text, output_structured=None):
        # type: (str, str, Optional[Dict[str, Any]]) -> List[Finding]
        """scan a single tool output and return a list of findings."""
        ...

    def reset(self):
        # type: () -> None
        """reset any accumulated state."""
