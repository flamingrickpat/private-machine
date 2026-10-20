"""Test package for unit and contract suites.

Importing this package redirects the prompt log away from the operator's real
private-machine home. The LLM HTTP client writes one log file per inference
attempt, including the attempts that unit tests make against deliberately
unavailable endpoints. Without this redirect, those files mix with the real
prompt evidence in ``~/.pm/private-machine``.

The redirect sets the environment variable only when it is absent, so an
operator or a test can still select its own log directory, and
``PM_PROMPT_LOG=0`` still disables the log completely. The temporary directory
lives under the system temp root, and the operating system owns its cleanup.
"""
import os
import tempfile

os.environ.setdefault(
    "PM_PROMPT_LOG_DIR",
    tempfile.mkdtemp(prefix="pm-test-prompt-log-"),
)
os.environ.setdefault("PM_CACHE_DIR", tempfile.mkdtemp(prefix="pm-test-cache-"))
