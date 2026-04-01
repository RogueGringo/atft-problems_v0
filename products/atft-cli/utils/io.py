"""
utils/io.py — JSON I/O utilities for the atft CLI.

All commands output a JSON envelope to stdout and a human summary to stderr.
Pipeline stages can read a previous stage's envelope from stdin and merge results.
"""

import json
import sys
from datetime import datetime, timezone
from typing import Any


def make_result(
    command: str,
    result: Any,
    meta: dict | None = None,
    transducer: str | None = None,
    input_path: str | None = None,
) -> dict:
    """Build a standard output envelope with timestamp."""
    envelope: dict = {
        "command": command,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }
    if meta is not None:
        envelope["meta"] = meta
    if transducer is not None:
        envelope["transducer"] = transducer
    if input_path is not None:
        envelope["input_path"] = input_path
    return envelope


def emit(envelope: dict) -> None:
    """Write a JSON envelope to stdout."""
    json.dump(envelope, sys.stdout, indent=2)
    sys.stdout.write("\n")
    sys.stdout.flush()


def summary(message: str) -> None:
    """Write a human-readable summary line to stderr."""
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


def read_stdin_json() -> dict | None:
    """
    Read a JSON envelope from stdin if stdin is piped (not a TTY).

    Returns the parsed dict, or None if stdin is a TTY or empty.
    """
    if sys.stdin.isatty():
        return None
    raw = sys.stdin.read().strip()
    if not raw:
        return None
    return json.loads(raw)


def merge_results(
    previous: dict,
    command: str,
    result: Any,
    meta: dict | None = None,
) -> dict:
    """
    Merge new results into a piped previous envelope.

    The combined command name is the previous command and the new command
    joined by "+".  The result is stored under the new command's name so
    each stage's output remains accessible.
    """
    combined_command = previous["command"] + "+" + command
    merged: dict = dict(previous)
    merged["command"] = combined_command
    merged["timestamp"] = datetime.now(timezone.utc).isoformat()
    # Preserve each stage's result under its own key
    merged.setdefault("stages", {})[previous["command"]] = previous.get("result")
    merged["stages"][command] = result
    # Top-level result is the latest stage
    merged["result"] = result
    if meta is not None:
        existing_meta = merged.get("meta", {})
        existing_meta.update(meta)
        merged["meta"] = existing_meta
    return merged
