"""Verify singleton feedback with the real body, full canon, and real cognition.

Explicit control actions exercise admission and cancellation. One real main
cycle consumes the busy event. Its choices remain evidence for human review.
This gate does not establish general conversation or gameplay acceptance.
"""
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from threading import Timer
import time
import traceback
import unittest
from uuid import uuid4

import yaml

from dev.tools.alpha_snapshot import capture
from dev.work_items.a18b.runner import case_config
from dev.work_items.conversation_review import SEED, start_minecraft
from dev.work_items.shared.evidence import save_json
from pm.bus.envelopes import new_envelope
from pm.memory.store import MemoryStore
from pm.paths import REPO_ROOT
from pm.records.enums import ActionType
from pm.records.memory_records import ActionSchemaRecord
from pm.runs.workflow_runner import WorkflowStatus
from tests.support import Recorder


@unittest.skipUnless(os.environ.get('PM_SINGLETON_LIVE') == '1', 'Set PM_SINGLETON_LIVE=1 for real services and inference.')
class LiveWorkflowSingletonTests(unittest.TestCase):
    def test_busy_feedback_and_explicit_release(self):
        name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + uuid4().hex[:8]
        out = REPO_ROOT / '.runtime/a18b-current/singleton' / ('live-' + name)
        out.mkdir(parents=True)
        receipt = {'ok': False, 'scope': 'Real singleton admission and one cognitive cycle; not full A18B acceptance',
                   'human_review': 'pending', 'started_at': datetime.now(timezone.utc).isoformat()}
        try:
            with ExitStack() as cleanup:
                previous = {key: os.environ.get(key) for key in ['MINECRAFT_MCP_URL', 'PM_PROMPT_LOG_DIR']}
                def restore():
                    for key, value in previous.items():
                        if value is None:
                            os.environ.pop(key, None)
                        else:
                            os.environ[key] = value
                cleanup.callback(restore)
                (out / 'source').mkdir()
                save_json(out / 'origin.json', capture(REPO_ROOT, out / 'source'))
                config = case_config(REPO_ROOT / 'config.alpha.openrouter.yaml', out)
                stack = start_minecraft(out, False)
                cleanup.callback(stack.stop)
                os.environ['MINECRAFT_MCP_URL'] = stack.mcp_url
                os.environ['PM_PROMPT_LOG_DIR'] = str(out / 'llm-logs')
                from workflows.companion import build_app
                from workflows.minecraft_body.lease import BODY_LEASES
                from workflows.minecraft_body.mcp_client import build_minecraft_tools
                ghost = MemoryStore.from_yaml(SEED, checkpoint=out / 'ghost.pickle')
                receipt['seed_records'] = len(ghost.all_knoxels)
                self.assertEqual(receipt['seed_records'], 11202)
                app = build_app(SEED, ghost=ghost, runtime_root=out / 'runtime', cognition='real',
                    config_path=str(config), record_lifecycle=False, world_poll_seconds=None)
                def stop_app():
                    app.shutdown()
                    app.join()
                cleanup.callback(stop_app)
                timer = Timer(600, app.shutdown)
                timer.daemon = True
                timer.start()
                cleanup.callback(timer.cancel)
                events = Recorder(app.bus, 'workflow.event')
                actions = Recorder(app.bus, 'action.request')

                def control(identity, target, arguments):
                    batch = ghost.begin_batch(identity)
                    action = batch.create(ActionSchemaRecord(action_type=ActionType.ToolCall,
                        content='Explicit singleton boundary gate operation.', metadata={'gate_control': True}))
                    ghost.commit(batch)
                    app.self_management.handle_batch([new_envelope('singleton_gate', 'data', 'action.request', identity,
                        {'knoxel_id': action.id, 'action_id': identity, 'arguments': arguments}, target=target)])

                def wait_for(predicate, seconds=120):
                    deadline = time.monotonic() + seconds
                    while not predicate():
                        app.raise_if_error()
                        if time.monotonic() >= deadline or app.shutdown_event.is_set():
                            raise TimeoutError('The real singleton gate exceeded its observation deadline.')
                        time.sleep(0.1)

                control('held-read', 'minecraft', {'instructions': 'Observe the world and wait for my next instruction.',
                    'mode': 'boundary_check', 'wait_for_update': True, 'include_image': False})
                first = next(row for row in app.supervisor.receipts().values() if row['action_id'] == 'held-read')
                root = Path(first['artifact_path'])
                wait_for(lambda: (root / 'outbox/world.yaml').exists())
                self.assertEqual(BODY_LEASES.owner(stack.mcp_url), first['run_id'])
                save_json(out / 'held-world.json', yaml.safe_load((root / 'outbox/world.yaml').read_text(encoding='utf-8')))
                control('competing-project', 'minecraft_project', {'goal': 'Start a second project with the same body.'})
                busy = next(row for row in events.events if row.payload['event'] == 'start_rejected')
                self.assertEqual(busy.payload['blocked_by_run_id'], first['run_id'])
                self.assertEqual(len(app.supervisor._threads), 1)
                self.assertIsNone(app.error)
                save_json(out / 'busy-event.json', json.loads(busy.to_json()))
                receipt['owner_run_id'] = first['run_id']

                # Execute one real cognitive cycle. Its selected actions remain
                # unexecuted so the gate does not impose a model control choice.
                before = len(actions.events)
                app.main_cog.handle_batch([busy])
                app.raise_if_error()
                self.assertFalse(app.shutdown_event.is_set())
                self.assertEqual(app.main_cog.last_result.status, WorkflowStatus.Completed)
                key, text, kind = app.main_cog._event_stimuli[busy.id]
                self.assertIn(first['run_id'], text)
                self.assertEqual(kind, 'workflow.event')
                save_json(out / 'cognitive-result.json', {'stimulus_id': key, 'text': text,
                    'selected_actions': [json.loads(row.to_json()) for row in actions.events[before:]],
                    'kernel': app.main_cog.last_kernel_summary, 'human_review': 'pending'})

                control('explicit-stop', 'self_management', {'operation': 'terminate', 'run_id': first['run_id']})
                app.supervisor.join(first['run_id'], timeout=120)
                self.assertFalse(app.supervisor._threads[first['run_id']].is_alive())
                self.assertIsNone(BODY_LEASES.owner(stack.mcp_url))
                control('fresh-read', 'minecraft', {'instructions': 'Read the actual world.',
                    'mode': 'boundary_check', 'action_tool': 'minecraft_observe',
                    'action_arguments': {'include_image': False}, 'include_image': False})
                second = next(row for row in app.supervisor.receipts().values() if row['action_id'] == 'fresh-read')
                wait_for(lambda: app.supervisor.get(second['run_id'])['state'] in {'completed', 'failed'})
                self.assertEqual(app.supervisor.get(second['run_id'])['state'], 'completed')
                final = build_minecraft_tools(stack.mcp_url)['minecraft_call']('minecraft_observe', {'include_image': False})
                self.assertTrue(final['ok'])
                save_json(out / 'final-world.json', final)
                save_json(out / 'managed-receipts.json', app.supervisor.receipts())
                receipt['replacement_run_id'] = second['run_id']
            receipt['ok'] = True
        except BaseException as error:
            receipt['error'] = f'{type(error).__name__}: {error}'
            receipt['traceback'] = traceback.format_exc()
            raise
        finally:
            receipt['ended_at'] = datetime.now(timezone.utc).isoformat()
            save_json(out / 'receipt.json', receipt)
