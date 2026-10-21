"""A tiny real MCP server for the tool_call integration check. Development tool, not product code.

It serves two tools over streamable HTTP at ``http://127.0.0.1:<port>/mcp``:

- ``roll_dice(sides)`` rolls one die and returns the number.
- ``make_test_image(color, save_path="")`` makes a small solid-color PNG and
  returns it as an MCP image result. With ``save_path`` (an absolute path on
  this host), it also writes the PNG there.

Every call is appended to the call log (``--log``), so a check can compare the
real server-side calls with the session record. Run::

    uv run --locked python dev/tools/demo_mcp.py --port 9881 --log .runtime/toolcall-test/demo-mcp-calls.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import struct
import time
import zlib
from pathlib import Path

from fastmcp import FastMCP
from fastmcp.utilities.types import Image

COLORS = {"red": (220, 30, 30), "green": (30, 180, 60), "blue": (30, 80, 220), "yellow": (240, 210, 40),
          "black": (0, 0, 0), "white": (255, 255, 255), "purple": (140, 50, 180), "orange": (250, 140, 20)}
SIZE = 32


def solid_png(rgb: tuple[int, int, int], size: int = SIZE) -> bytes:
    """Return a valid size x size RGB PNG of one color, without an image library."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    row = b"\x00" + bytes(rgb) * size
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(row * size))
            + chunk(b"IEND", b""))


def build_server(log: Path) -> FastMCP:
    """Return the server. Each tool appends one JSON line to ``log``."""
    server = FastMCP("demo")

    def record(tool: str, arguments: dict, result: object) -> None:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time": time.time(), "tool": tool, "arguments": arguments,
                                     "result": result}) + "\n")

    @server.tool
    def roll_dice(sides: int) -> int:
        """Roll one fair die with the given number of sides and return the result."""
        if sides < 2:
            raise ValueError("A die needs at least 2 sides.")
        value = random.randint(1, sides)
        record("roll_dice", {"sides": sides}, value)
        return value

    @server.tool
    def make_test_image(color: str, save_path: str = "") -> Image:
        """Make a small solid-color test PNG (32x32) and return it as an image.

        color is one of: red, green, blue, yellow, black, white, purple, orange.
        save_path, when given, is an absolute file path on this host where the PNG is also written.
        """
        if color.lower() not in COLORS:
            raise ValueError(f"Unknown color {color!r}. Use one of: {', '.join(COLORS)}.")
        data = solid_png(COLORS[color.lower()])
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            Path(save_path).write_bytes(data)
        record("make_test_image", {"color": color, "save_path": save_path}, {"bytes": len(data)})
        return Image(data=data, format="png")

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=9881)
    parser.add_argument("--log", type=Path, default=Path(".runtime/toolcall-test/demo-mcp-calls.jsonl"))
    args = parser.parse_args()
    build_server(args.log.resolve()).run(transport="http", host="127.0.0.1", port=args.port, path="/mcp")


if __name__ == "__main__":
    main()
