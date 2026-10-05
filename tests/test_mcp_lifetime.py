"""Temporary loopback MCP listeners: distinct ports, real calls, cleanup."""
import asyncio
import socket
import unittest

from fastmcp import Client, FastMCP

from runtime.mcp_services import ScopedMcpRegistry, serve_services


def _echo_server(name: str, value: str) -> FastMCP:
    server = FastMCP(name)

    def echo() -> str:
        """Return the value bound to this scoped service."""
        return value

    echo.__name__ = "echo"
    server.add_tool(echo)
    return server


def _call(url: str, tool: str = "echo") -> object:
    async def run() -> object:
        async with Client(url) as client:
            result = await client.call_tool(tool, {})
            return result.data

    return asyncio.run(run())


class McpLifetimeTests(unittest.TestCase):
    def test_two_services_listen_on_distinct_ports_and_answer(self):
        with ScopedMcpRegistry() as registry:
            descriptors = registry.serve({
                "alpha": _echo_server("alpha", "A"),
                "beta": _echo_server("beta", "B"),
            })
            self.assertEqual(registry.active_services, ["alpha", "beta"])
            ports = {name: _port(d["url"]) for name, d in descriptors.items()}
            self.assertNotEqual(ports["alpha"], ports["beta"])
            self.assertEqual(_call(descriptors["alpha"]["url"]), "A")
            self.assertEqual(_call(descriptors["beta"]["url"]), "B")
            listeners = list(registry._listeners.values())
        self.assertEqual(registry.active_services, [])
        for listener in listeners:
            self.assertFalse(listener.thread.is_alive())
            self.assertIsNone(listener.error)

    def test_context_manager_propagates_and_still_closes(self):
        seen: list[str] = []
        with self.assertRaises(ValueError):
            with serve_services({"solo": _echo_server("solo", "S")}) as descriptors:
                seen.append(descriptors["solo"]["url"])
                self.assertEqual(_call(descriptors["solo"]["url"]), "S")
                raise ValueError("boom")
        self.assertEqual(len(seen), 1)

    def test_duplicate_service_name_closes_earlier_listeners(self):
        registry = ScopedMcpRegistry()
        registry.serve({"solo": _echo_server("solo", "S")})
        with self.assertRaises(ValueError):
            registry.serve({"solo": _echo_server("solo", "again")})
        self.assertEqual(registry.active_services, [])
        registry.close()


def _port(url: str) -> int:
    return int(url.split(":")[2].split("/")[0])


if __name__ == "__main__":
    unittest.main()
