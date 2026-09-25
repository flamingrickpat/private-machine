"""Compatibility imports. New code uses bus/ and runtime/shell.py."""
from bus.bus import BusSystem, Mailbox
from bus.envelopes import Envelope, _new_id
from runtime.shell import Application, ShellModule

__all__ = ["Envelope", "Mailbox", "BusSystem", "ShellModule", "Application", "_new_id"]
