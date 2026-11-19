import logging
import json

class _JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        return json.dumps(payload)


def configure_logging(json_format: bool = False) -> None:
    """Send app logs (INFO+) to stdout once; safe to call more than once."""
    root = logging.getLogger()
    if getattr(root, "_safeear_configured", False):
        return
    root._safeear_configured = True  # type: ignore[attr-defined]
    if not root.handlers:
        root.addHandler(logging.StreamHandler())
    handler = root.handlers[0]
    if json_format:
        handler.setFormatter(_JsonLogFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] %(message)s"))
    root.setLevel(logging.INFO)
