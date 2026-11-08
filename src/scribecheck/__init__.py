"""scribecheck: check AI-written clinical notes against the consultation transcript."""

__version__ = "0.0.1.dev0"

from scribecheck.checker import check, check_files  # noqa: E402
from scribecheck.config import Config  # noqa: E402
from scribecheck.report import Report  # noqa: E402

__all__ = ["Config", "Report", "__version__", "check", "check_files"]
