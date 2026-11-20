"""Exercise the baseline's rejected agent report through a real model session."""
from pathlib import Path
from threading import Event
import unittest
from uuid import uuid4

from pm_workflows.protocol import AgentResult

from pm.llm.activation import ModelActivation
from pm.llm.config import load_config
from pm.llm.prompt_log import log_scope
from pm.llm.structured import repair_agent_result
from pm.llm.workflow_sessions import WorkflowModels
from pm.utils.operating_system import OperatingSystem
from workflows.main_cognition.schemas import AgentCodeletAnswer


class CodeletReportRepairTests(unittest.TestCase):
    def test_repair_preserves_cited_evidence_on_configured_endpoint(self) -> None:
        config = Path(OperatingSystem.environment.get("PM_MODEL_CONFIG", "config.example.yaml"))
        models = WorkflowModels(load_config(config), ModelActivation(), live_test=True)
        root = Path(".runtime/codelet-repair") / uuid4().hex
        stop = Event()
        report = AgentResult(exit_code=0, stdout="I inspected record 42. Wait for Jesse.", result_json=None)
        with log_scope(root), models.session("Default", False, stop, receipt_path=root / "session.jsonl") as model:
            answer = repair_agent_result(report, AgentCodeletAnswer, model, cancel=stop,
                instruction="Give the recommendation and cite the single inspected record in evidence_ids.")
        self.assertEqual(answer.evidence_ids, [42])
        self.assertTrue(answer.recommendation.strip())
