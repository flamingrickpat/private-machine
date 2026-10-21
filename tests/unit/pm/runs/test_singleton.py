"""Exercise admission with real kernels, threads, checkpoints, and file locks.

Captured Minecraft requests test the control boundary with local held roles.
These tests do not establish Minecraft execution or companion quality.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier, Event
import time
import unittest

import yaml

from pm.host.durable_files import ProcessLock
from pm.memory.store import MemoryStore
from pm.records.cycle_state import PersonaConfig
from pm.runs.run_receipt import ManagedRunError
from pm.runs.supervisor import ManagedRunSupervisor
from pm.runs.workflow_spec import WorkflowSpec
from pm.paths import REPO_ROOT


class SingletonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.entered, self.cleanup_started, self.release_cleanup = Event(), Event(), Event()
        self.release_cleanup.set()
        role = self.root / 'held_role.py'
        role.write_text(
            "def run(context):\n"
            "    context.runtime.tools['hold'](context.runtime.stop_event)\n"
            "    context.runtime.environment.write_text('outbox/report.yaml', 'summary: Held role stopped.\\n')\n"
            "    return {'status': 'done', 'summary': 'Held role stopped.'}\n", encoding='utf-8')
        manifest = self.root / 'held.workflow.md'
        manifest.write_text('---\n' + yaml.safe_dump({
            'name': 'held', 'driver': {'kind': 'python'}, 'human_resolver': {'mode': 'forbid'},
            'roles': {'work': {'skill': str(role), 'tools': ['hold'],
                'result_contract': {'schema': {'status': {'enum': ['done']}, 'summary': 'string'}}}},
            'phases': [{'name': 'work', 'kind': 'role', 'role': 'work',
                'on_status': {'done': 'stop'}, 'on_invalid': {'action': 'stop'}}],
        }) + '---\n', encoding='utf-8')
        spec = WorkflowSpec(manifest, tools_factory=lambda run, env: {'hold': self.hold},
            resumable=True, singleton=True, singleton_key='minecraft_body')
        self.events, self.errors = [], []
        self.ghost = MemoryStore(PersonaConfig(user_name='Jesse', companion_name='Spamuel'),
                                 checkpoint=self.root / 'ghost.pickle')
        self.supervisor = ManagedRunSupervisor(self.ghost, runtime_root=self.root / 'runtime',
            task_root=self.root / 'tasks', workflows={'minecraft': spec, 'minecraft_project': spec},
            clock=time.monotonic, event_sink=self.events.append, on_error=self.errors.append)

    def tearDown(self):
        self.release_cleanup.set()
        self.supervisor.shutdown()
        self.tmp.cleanup()

    def hold(self, cancel):
        self.entered.set()
        if not cancel.wait(20):
            raise TimeoutError('The test did not cancel its owned worker.')
        self.cleanup_started.set()
        if not self.release_cleanup.wait(10):
            raise TimeoutError('The test did not release worker cleanup.')

    def start(self, action, workflow='minecraft_project'):
        return self.supervisor.start(action_id=action, workflow_name=workflow,
                                     parent_correlation=action, instructions='Wait for my next action.')

    def test_captured_overlapping_requests_return_the_owner_without_new_workers(self):
        capture = json.loads((REPO_ROOT / 'test_data/captures/alpha/minecraft-overlapping-starts.json').read_text(encoding='utf-8'))
        first = self.supervisor.start(**capture['starts'][0])
        self.assertTrue(self.entered.wait(5))
        for request in capture['starts'][1:]:
            rejected = self.supervisor.start(**request)
            self.assertEqual(rejected['state'], 'failed')
            self.assertEqual(rejected['blocked_by_run_id'], first['run_id'])
            self.assertEqual(rejected['artifact_path'], '')
            self.assertNotIn(rejected['run_id'], self.supervisor._threads)
            count = len(self.events)
            self.assertEqual(self.supervisor.start(**request), rejected)
            self.assertEqual(len(self.events), count)
        self.assertEqual(len(self.supervisor._threads), 1)
        self.assertEqual(self.errors, [])
        event = self.events[-1]
        self.assertEqual(event['event'], 'start_rejected')
        self.assertEqual(event['reason'], 'singleton_busy')
        self.assertEqual(event['blocked_by_run_id'], first['run_id'])
        self.assertIn(first['run_id'], event['detail'])
        self.assertFalse(self.supervisor._cancels[first['run_id']].is_set())

    def test_shared_key_blocks_the_other_minecraft_workflow(self):
        first = self.start('first', 'minecraft')
        second = self.start('second', 'minecraft_project')
        self.assertEqual(second['blocked_by_run_id'], first['run_id'])
        updated = self.supervisor.update(first['run_id'], 'Stay with the original task.', request_id='update')
        self.assertEqual(updated['revision'], 1)
        self.assertEqual(self.errors, [])

    def test_boolean_singleton_uses_the_workflow_name_without_a_shared_key(self):
        spec = replace(self.supervisor.workflows['minecraft'], singleton_key='')
        self.supervisor.workflows = {'minecraft': spec, 'minecraft_project': spec}
        first = self.start('first', 'minecraft')
        duplicate = self.start('duplicate', 'minecraft')
        independent = self.start('other-name', 'minecraft_project')
        self.assertEqual(duplicate['blocked_by_run_id'], first['run_id'])
        self.assertEqual(independent['state'], 'accepted')

    def test_default_workflows_still_run_in_parallel(self):
        spec = replace(self.supervisor.workflows['minecraft'], singleton=False, singleton_key='')
        self.supervisor.workflows['minecraft'] = spec
        self.assertEqual(self.start('first', 'minecraft')['state'], 'accepted')
        self.assertEqual(self.start('second', 'minecraft')['state'], 'accepted')
        self.assertEqual(len(self.supervisor._threads), 2)

    def test_concurrent_starts_admit_exactly_one_worker(self):
        barrier = Barrier(8)
        def start(index):
            barrier.wait(timeout=5)
            return self.start(f'concurrent-{index}')
        with ThreadPoolExecutor(max_workers=8) as pool:
            rows = list(pool.map(start, range(8)))
        accepted = [row for row in rows if row['state'] == 'accepted']
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(self.supervisor._threads), 1)
        self.assertEqual({row['blocked_by_run_id'] for row in rows if row['state'] == 'failed'}, {accepted[0]['run_id']})

    def test_cancelled_run_keeps_its_slot_until_cleanup_finishes(self):
        self.release_cleanup.clear()
        first = self.start('first')
        self.assertTrue(self.entered.wait(5))
        cancelled = self.supervisor.terminate(first['run_id'], request_id='stop-first')
        self.assertEqual(cancelled['state'], 'cancelled')
        self.assertTrue(self.cleanup_started.wait(5))
        rejected = self.start('during-cleanup', 'minecraft')
        self.assertEqual(rejected['blocked_by_run_id'], first['run_id'])
        self.release_cleanup.set()
        self.supervisor.join(first['run_id'], timeout=5)
        self.assertFalse(self.supervisor._threads[first['run_id']].is_alive())
        self.assertEqual(self.start('after-cleanup', 'minecraft')['state'], 'accepted')

    def test_resume_cannot_bypass_another_owner(self):
        first = self.start('first')
        self.assertTrue(self.entered.wait(5))
        self.supervisor.terminate(first['run_id'])
        self.supervisor.join(first['run_id'], timeout=5)
        other = self.start('other', 'minecraft')
        with self.assertRaisesRegex(ManagedRunError, other['run_id']):
            self.supervisor.resume(first['run_id'], request_id='resume')
        self.assertEqual(self.supervisor.get(first['run_id'])['execution_epoch'], 1)
        event = next(row for row in self.events if row['event'] == 'resume_rejected')
        self.assertEqual(event['blocked_by_run_id'], other['run_id'])

    def test_alias_rejection_reaches_cognition_as_a_workflow_stimulus(self):
        from pm.bus.envelopes import new_envelope
        from pm.records.enums import ActionType, SourceKind
        from pm.records.memory_records import ActionSchemaRecord
        from tests.support import Recorder
        from workflows.companion import build_app

        configured = self.supervisor.workflows
        self.supervisor.shutdown()
        app = build_app(REPO_ROOT / 'test_data/personas/spamuel/canon.yaml',
            ghost=self.ghost, cognition='real', runtime_root=self.root / 'app',
            config_path=str(REPO_ROOT / 'config.example.yaml'),
            record_lifecycle=False, world_poll_seconds=None)
        self.supervisor = app.supervisor
        self.supervisor.workflows = configured
        events = Recorder(app.bus, 'workflow.event')
        first = self.start('original')
        batch = self.ghost.begin_batch('competing-action')
        action = batch.create(ActionSchemaRecord(action_type=ActionType.ToolCall,
                                                content='Start another Minecraft project.'))
        self.ghost.commit(batch)
        app.self_management.handle_batch([new_envelope('main_cog', 'data', 'action.request', 'new-user',
            {'knoxel_id': action.id, 'action_id': 'competing-action', 'arguments': {}}, target='minecraft')])
        feedback = next(row for row in events.events if row.payload['event'] == 'start_rejected')
        self.assertEqual(feedback.correlation_id, 'new-user')
        key, text, kind = app.main_cog._stimulus_from(feedback)
        self.assertIn(first['run_id'], text)
        self.assertIn('already in use', text)
        self.assertEqual(kind, 'workflow.event')
        self.assertEqual(self.ghost.get_knoxel_by_id(key).provenance.source_kind, SourceKind.Workflow)
        self.assertIsNone(app.error)
        self.assertEqual(len(self.supervisor._threads), 1)

    def test_late_operating_system_contention_fails_only_that_task(self):
        path = self.root / 'external-owner.lock'
        held = ProcessLock(path).acquire()
        def tools(run, env):
            ProcessLock(path).acquire()
            raise AssertionError('The occupied resource admitted a second owner.')
        self.supervisor.workflows['minecraft'] = replace(self.supervisor.workflows['minecraft'], tools_factory=tools)
        try:
            failed = self.start('external-busy', 'minecraft')
            self.supervisor.join(failed['run_id'], timeout=5)
            self.assertEqual(self.supervisor.get(failed['run_id'])['state'], 'failed')
            self.assertEqual(self.events[-1]['reason'], 'resource_busy')
            self.assertIn('Resource is owned', self.events[-1]['detail'])
            self.assertEqual(self.errors, [])
        finally:
            held.close()
        self.assertEqual(self.start('next-task')['state'], 'accepted')
