"""Exercise actual read grants, MCP transport, caches, and revision guards."""
import asyncio
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event, Timer
from types import SimpleNamespace
import unittest

from fastmcp import Client

from pm.host.mcp_listeners import serve_services
from pm.llm.config import EmbeddingConfig
from pm.memory.agent_tools import MemoryReadSession, MemoryToolConfig, MemoryView
from pm.memory.embeddings import EmbeddingService
from pm.memory.snapshot import GhostCancelled, GhostConflict
from pm.memory.store import MemoryStore
from pm.memory.store_tools import build_mcp, build_tools
from pm.records.base import RecordScope
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import ClusterType, FeatureType, FrameEnum
from pm.records.graph_records import ConceptNode, GraphEdge, GraphNode
from pm.records.memory_records import Feature, MemoryClusterRecord, new_feature


class MemoryAgentToolTests(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"))
        self.cancel = Event()

    def say(self, text):
        return new_feature(self.store, text, FeatureType.Dialogue, source="Jesse")

    def config(self, **kwargs):
        view = MemoryView(frozenset(self.store.revisions), frozenset({(-1, "in_world")}))
        return MemoryToolConfig(self.store, view, cancel=self.cancel, **kwargs)

    def graph(self):
        source = self.say("Jesse accepts any available timber for the small shelter.")
        concept = self.store.store_knoxel(ConceptNode(content="", description="Construction material"))
        nodes = [self.store.store_knoxel(GraphNode(content="", name=name, labels=["Material"],
                                                  source_memory_id=source.id, concept_id=concept.id))
                 for name in ("timber", "shelter")]
        edge = self.store.store_knoxel(GraphEdge(content="", source_id=nodes[0].id, target_id=nodes[1].id,
            label="acceptable_for", fact_text="Any timber is acceptable for the shelter.", source_memory_id=source.id,
            invalid_at=datetime(2026, 10, 13)))
        return source, concept, nodes, edge

    def test_shared_factory_disabled_and_enabled_over_real_mcp(self):
        source = self.say("A copper lantern marks the return route.")
        environment = SimpleNamespace(visible_knoxels=self.store.snapshot().serialize_knoxels(), visible_gwt={})
        self.assertEqual(set(build_tools(environment)), {"ghost_get", "ghost_search", "ghost_gwt"})
        with MemoryReadSession(self.config()) as memory:
            before = self.store.state_dict()
            async def exercise(descriptors):
                async with Client(descriptors["scoped"]["url"]) as client:
                    names = {tool.name for tool in await client.list_tools()}
                    self.assertIn("memory_search", names)
                    self.assertNotIn("memory_neighbors", names)
                    result = await client.call_tool("memory_search", {"query": "copper lantern"})
                    self.assertEqual(result.data["matches"][0]["record_id"], source.id)
                    page = await client.call_tool("memory_get", {"record_id": source.id})
                    self.assertIn(source.content, json.dumps(page.data))
            with serve_services({"scoped": build_mcp(environment, memory=memory)}) as descriptors:
                asyncio.run(exercise(descriptors))
            self.assertEqual(self.store.state_dict(), before)

    def test_graph_fields_sources_direction_and_pages(self):
        source, concept, nodes, edge = self.graph()
        with MemoryReadSession(self.config(lexical=False, graph=True)) as memory:
            result = asyncio.run(memory.search("timber", limit=4))
            ids = {row["record_id"] for row in result["matches"]}
            self.assertIn(edge.id, ids)
            self.assertIn(nodes[0].id, ids)
            self.assertNotIn(source.id, ids)
            relation = memory.neighbors(nodes[0].id, "outgoing")["relations"][0]
            self.assertEqual(relation["edge"]["record_id"], edge.id)
            self.assertEqual(relation["edge"]["invalid_at"], edge.invalid_at.isoformat())
            self.assertIn(source.id, relation["source_ids"])
            self.assertEqual(memory.neighbors(nodes[0].id, "incoming")["relations"], [])
            inspected = memory.sources(edge.id, limit=1)
            self.assertIsNotNone(inspected["next_offset"])
            self.assertEqual(memory.get(source.id, length=30)["next_offset"], 30)

    def test_hidden_source_and_endpoint_cannot_leak_through_any_tool(self):
        source, concept, nodes, edge = self.graph()
        nested = self.store.store_knoxel(MemoryClusterRecord(content="Secret timber summary", level=2,
            cluster_type=ClusterType.Topical, included_event_ids=str(source.id)))
        outer = self.store.store_knoxel(MemoryClusterRecord(content="Secret nested timber", level=3,
            cluster_type=ClusterType.Topical, included_cluster_ids=str(nested.id)))
        config = self.config(graph=True)
        config = replace(config, view=replace(config.view, record_ids=config.view.record_ids - {source.id}))
        with MemoryReadSession(config) as memory:
            self.assertNotIn(edge.id, memory.records)
            self.assertNotIn(nodes[0].id, memory.records)
            self.assertNotIn(nested.id, memory.records)
            self.assertNotIn(outer.id, memory.records)
            for key in (source.id, edge.id, outer.id, 999999):
                with self.assertRaisesRegex(ValueError, "unavailable"):
                    memory.get(key)
            self.assertEqual(asyncio.run(memory.search("Secret"))["matches"], [])
        config = self.config(graph=True)
        config = replace(config, view=replace(config.view, record_ids=config.view.record_ids - {nodes[1].id}))
        with MemoryReadSession(config) as memory:
            self.assertEqual(memory.neighbors(nodes[0].id)["relations"], [])
            self.assertNotIn("acceptable_for", json.dumps(memory.view()))

    def test_scope_frontier_categories_entities_and_time_apply_before_reads(self):
        visible = self.say("The allowed timber report.")
        self.store.store_knoxel(Feature(content="Hidden meta report", feature_type=FeatureType.Dialogue,
                                      scope=RecordScope(frame=FrameEnum.Meta)))
        config = self.config()
        config = replace(config, view=replace(config.view, frontier=(visible.tick_id, visible.sub_tick_id),
            categories=frozenset({"utterance"}), entity_ids=frozenset({visible.entity_id}),
            start=visible.timestamp_world_begin))
        with MemoryReadSession(config) as memory:
            self.assertEqual(set(memory.records), {visible.id})

    def test_graph_required_ids_do_not_use_optional_zero_sentinel(self):
        source, concept, nodes, edge = self.graph()
        invalid = self.store.store_knoxel(GraphEdge(content="Hidden endpoint claim", source_id=0,
            target_id=nodes[1].id, source_memory_id=source.id, label="missing", fact_text="An undefined endpoint claim."))
        with MemoryReadSession(self.config(graph=True)) as memory:
            self.assertNotIn(invalid.id, memory.records)
            self.assertEqual(asyncio.run(memory.search("undefined"))["matches"], [])

    def test_direct_lexical_search_retains_short_specific_names(self):
        oak = self.say("Use oak for the shelter.")
        tea = self.say("I prefer mint tea.")
        with MemoryReadSession(self.config()) as memory:
            self.assertEqual([row["record_id"] for row in asyncio.run(memory.search("oak"))["matches"]], [oak.id])
            self.assertEqual([row["record_id"] for row in asyncio.run(memory.search("tea"))["matches"]], [tea.id])

    def test_snapshot_refresh_and_guard_only_inspected_dependencies(self):
        source = self.say("Any wood is acceptable.")
        config = self.config()
        config = replace(config, refresh_view=lambda: self.config().view)
        with MemoryReadSession(config) as memory:
            identity = memory.snapshot_id
            memory.get(source.id)
            self.say("An unrelated new record.")
            with memory.guard():
                pass
            batch = self.store.begin_batch("correction")
            record = batch.view().get(source.id)
            record.content = "Use oak for this task."
            batch.replace(record)
            self.store.commit(batch)
            self.assertIn("Any wood", json.dumps(memory.get(source.id)))
            with self.assertRaises(GhostConflict):
                with memory.guard():
                    self.fail("A stale source reached the commit.")
            memory.refresh()
            self.assertNotEqual(identity, memory.snapshot_id)
            self.assertEqual(memory.accessed, {})
            self.assertIn("Use oak", json.dumps(memory.get(source.id)))
            with memory.guard():
                pass

    def test_reading_a_summary_guards_its_original_source(self):
        source = self.say("Any timber is fine.")
        summary = self.store.store_knoxel(MemoryClusterRecord(content="Timber preference", level=2,
            cluster_type=ClusterType.Topical, included_event_ids=str(source.id)))
        with MemoryReadSession(self.config()) as memory:
            memory.get(summary.id)
            self.assertIn(source.id, memory.accessed)

    def test_cancel_and_close_revoke_all_tools_and_guards(self):
        source = self.say("Any wood is acceptable.")
        memory = MemoryReadSession(self.config(graph=True))
        self.cancel.set()
        for fn in (memory.view, lambda: memory.get(source.id), lambda: memory.sources(source.id),
                   lambda: memory.neighbors(source.id), lambda: asyncio.run(memory.search("wood"))):
            with self.assertRaises(GhostCancelled):
                fn()
        self.cancel.clear()
        memory.close()
        with self.assertRaises(GhostCancelled):
            memory.view()

    def test_independent_grants_in_separate_client_process_and_listener_cleanup(self):
        first = self.say("Visible copper lantern.")
        hidden = self.say("Private granite marker.")
        config = self.config()
        config = replace(config, view=replace(config.view, record_ids=frozenset({first.id})))
        with MemoryReadSession(config) as memory:
            code = "import asyncio,json,sys; from fastmcp import Client\nasync def run():\n async with Client(sys.argv[1]) as c:\n  r=await c.call_tool('memory_search',{'query':'copper'}); print(json.dumps(r.data))\nasyncio.run(run())"
            with serve_services({"memory": memory.server()}) as descriptors:
                result = subprocess.run([sys.executable, "-c", code, descriptors["memory"]["url"]],
                                        capture_output=True, text=True, check=True, timeout=30)
                reply = json.loads(result.stdout)
                self.assertEqual([row["record_id"] for row in reply["matches"]], [first.id])
                self.assertNotIn(hidden.id, memory.accessed)
            async def closed():
                async with Client(descriptors["memory"]["url"], timeout=1):
                    pass
            with self.assertRaises(Exception):
                asyncio.run(closed())

    def test_semantic_only_uses_shared_cache_and_reports_missing_service(self):
        source = self.say("Any timber is acceptable.")
        with TemporaryDirectory() as directory:
            service = EmbeddingService(EmbeddingConfig(base_url="http://127.0.0.1:1/v1", model="cache-test",
                api_key="unused", dimensions=2, legacy_cache_dir=Path(directory) / "legacy",
                cache_dir=Path(directory) / "cache"), live_test=True)
            service.cache.put(source.content, [1.0, 0.0])
            service.cache.put("building materials", [1.0, 0.0])
            with MemoryReadSession(self.config(semantic=True, lexical=False, embeddings=service)) as memory:
                result = asyncio.run(memory.search("building materials"))
                self.assertEqual(result["matches"][0]["record_id"], source.id)
                self.assertEqual(result["trace"]["effective_method"], "semantic")
                self.assertEqual(result["trace"]["coverage"]["indexed"], 1)
                self.assertEqual(service.client.request_count, 0)
            with MemoryReadSession(self.config(semantic=True, lexical=False)) as memory:
                result = asyncio.run(memory.search("timber"))
                self.assertEqual(result["trace"]["effective_method"], "lexical")
                self.assertEqual(result["trace"]["semantic"]["reason"], "embedding_service_not_configured")

    def test_view_limits_and_methods_reject_invalid_calls_without_side_effects(self):
        source = self.say("A timber report.")
        with MemoryReadSession(self.config()) as memory:
            for call in (lambda: memory.get(source.id, length=5000), lambda: memory.sources(source.id, limit=65),
                         lambda: asyncio.run(memory.search("timber", method="semantic")),
                         lambda: asyncio.run(memory.search("", limit=1)), memory.refresh):
                with self.assertRaises(ValueError):
                    call()

    def test_incomplete_semantic_coverage_keeps_every_record_visible(self):
        sources = [self.say(f"Archived material report {index}.") for index in range(80)]
        with TemporaryDirectory() as directory:
            service = EmbeddingService(EmbeddingConfig(base_url="http://127.0.0.1:1/v1", model="cache-test",
                api_key="unused", dimensions=2, legacy_cache_dir=Path(directory) / "legacy",
                cache_dir=Path(directory) / "cache"), live_test=True)
            for index, source in enumerate(sources):
                if index != 20:
                    service.cache.put(source.content, [1.0, 0.0])
            service.cache.put("building supplies", [1.0, 0.0])
            with MemoryReadSession(self.config(semantic=True, embeddings=service)) as memory:
                result = asyncio.run(memory.search("building supplies"))
                self.assertEqual(result["trace"]["coverage"]["missing"], 1)
                self.assertIn(sources[20].id, memory.records)
                self.assertIn("Archived material", json.dumps(memory.get(sources[20].id)))
                self.assertEqual(service.client.request_count, 0)

    def test_cancel_during_provider_retry_returns_no_search_result(self):
        self.say("A material report without a cached embedding.")
        with TemporaryDirectory() as directory:
            service = EmbeddingService(EmbeddingConfig(base_url="http://127.0.0.1:1/v1", model="cache-test",
                api_key="unused", dimensions=2, legacy_cache_dir=Path(directory) / "legacy",
                cache_dir=Path(directory) / "cache"))
            with MemoryReadSession(self.config(semantic=True, embeddings=service)) as memory:
                timer = Timer(0.3, self.cancel.set)
                timer.start()
                try:
                    with self.assertRaises(asyncio.CancelledError):
                        asyncio.run(memory.search("material"))
                    self.assertGreater(service.client.request_count, 0)
                    self.assertEqual(memory.accessed, {})
                finally:
                    timer.cancel()
                    timer.join()

    def test_actual_unavailable_provider_records_lexical_fallback(self):
        source = self.say("The timber report.")
        with TemporaryDirectory() as directory:
            service = EmbeddingService(EmbeddingConfig(base_url="http://127.0.0.1:1/v1", model="cache-test",
                api_key="unused", dimensions=2, legacy_cache_dir=Path(directory) / "legacy",
                cache_dir=Path(directory) / "cache"), live_test=True)
            with MemoryReadSession(self.config(semantic=True, embeddings=service)) as memory:
                result = asyncio.run(memory.search("timber"))
                self.assertEqual(result["matches"][0]["record_id"], source.id)
                self.assertEqual(result["trace"]["semantic"]["status"], "unavailable")
                self.assertEqual(result["trace"]["semantic"]["fallback"], "lexical")
                self.assertEqual(result["trace"]["effective_method"], "lexical")

    def test_close_stops_pending_search_without_cancelling_its_parent(self):
        self.say("An uncached timber report.")
        with TemporaryDirectory() as directory:
            service = EmbeddingService(EmbeddingConfig(base_url="http://127.0.0.1:1/v1", model="cache-test",
                api_key="unused", dimensions=2, legacy_cache_dir=Path(directory) / "legacy",
                cache_dir=Path(directory) / "cache"))
            with MemoryReadSession(self.config(semantic=True, embeddings=service)) as memory:
                timer = Timer(0.3, memory.close)
                timer.start()
                try:
                    with self.assertRaises(GhostCancelled):
                        asyncio.run(memory.search("timber"))
                    self.assertFalse(self.cancel.is_set())
                    self.assertGreater(service.client.request_count, 0)
                finally:
                    timer.cancel()
                    timer.join()


if __name__ == "__main__":
    unittest.main()
