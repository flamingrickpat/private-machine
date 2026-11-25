"""Run the real codelets phase on an ordinary favor and on reciprocated adult desire.

Use `python -m unittest tests.live.test_codelet_grounding` from the repository root.
The chat and embedding endpoints come from config.yaml or the file named by PM_MODEL_CONFIG.
"""
from pathlib import Path
from threading import Event
import unittest
from uuid import uuid4

from pm_workflows.python_role import RoleContext

from pm.llm.activation import ModelActivation
from pm.llm.config import load_config, resolve_connection
from pm.llm.prompt_log import log_scope
from pm.llm.session import ModelSession
from pm.llm.stage_model import LazyStageModel
from pm.memory.bash_view import MemoryBashTarget
from pm.memory.embeddings import EmbeddingService
from pm.memory.store import MemoryStore
from pm.records.cycle_state import PersonaConfig
from pm.utils import yaml_util
from pm.utils.operating_system import OperatingSystem
from workflows.main_cognition.codelet_agent import CodeletAgent
from workflows.main_cognition.codelet_registry import CodeletDefinitions, load_definitions
from workflows.main_cognition.csm_view import Embedder, enter_retrieval
from workflows.main_cognition.cycle_work import CycleRequest
from workflows.main_cognition.phases import codelets, intake
from workflows.main_cognition.runtime import CognitionRuntime
from workflows.main_cognition.schemas import CodeletsResult, RecordView, Stimulus


ROOT = Path(__file__).resolve().parents[2]
CARD = "Mara, 33, repairs synthesizers. Their friend Jules, 34, plays bass. They are talking by text."


class CodeletGroundingTests(unittest.TestCase):
    def setUp(self) -> None:
        config = load_config(Path(OperatingSystem.environment.get("PM_MODEL_CONFIG", str(ROOT / "config.yaml"))))
        self.output: Path = ROOT / ".runtime/codelet-grounding" / uuid4().hex
        self.output.mkdir(parents=True)
        print(f"Codelet grounding receipts: {self.output}", flush=True)
        self.model: LazyStageModel = LazyStageModel(resolve_connection(config), ModelActivation(), live_test=True)
        self.embeddings: EmbeddingService = EmbeddingService(config.embedding, live_test=True)
        catalog = load_definitions()
        definition = catalog.by_name()["EroticDesire"]
        only_desire = CodeletDefinitions(version=catalog.definitions.version, codelets=[definition])
        self.catalog: Path = self.output / "codelets.yaml"
        catalog_yaml = yaml_util.safe_dump(only_desire.model_dump(mode="json"), sort_keys=False)  # pm-lint: allow PM007 -- registry configuration file
        self.catalog.write_text(catalog_yaml, encoding="utf-8")

    def examine(self, scene: str, card: str, message: str) -> CodeletsResult:
        """Force the same shipped codelet to run once, including when the selector rates it zero."""
        directory = self.output / scene
        directory.mkdir()
        persona = PersonaConfig(user_name="Jules", companion_name="Mara", character_card_story=card)
        store = MemoryStore(persona)
        environment = MemoryBashTarget(store).open(scene)
        stop = Event()
        runtime = CognitionRuntime(environment=environment, tools={}, stop_event=stop,  # pm-lint: allow PM005 -- empty tool grant required by pm-workflows
                                   model_session=ModelSession(self.model, stop), embedding_service=self.embeddings,
                                   codelet_agent=CodeletAgent(store, environment, self.embeddings, stop))
        context = RoleContext(run_id=scene, task_id="codelets", attempt=1, workspace=directory, task_dir=directory,
                              base_dir=ROOT, kernel_data=directory, role="main cognition", phase="codelets",
                              prompt="", task_text=message, current_item=None, feedback=None, answer=None, runtime=runtime)
        record = RecordView(record_id=25, kind="Dialogue", source=persona.user_name, content=message)
        stimulus = Stimulus(knoxel_id=record.record_id, source="user_input", kind="user.input", entity_id=2, text=message)
        request = CycleRequest(run_id=scene, companion_name=persona.companion_name, user_name=persona.user_name,
                               character_card=card, self_entity_id=1, user_entity_id=2, stimuli=[stimulus],
                               codelet_definitions=str(self.catalog))
        request.records[record.record_id] = record
        work = intake.start(request)
        work.triage.codelet_budget = 1
        with log_scope(directory / "prompts"):
            enter_retrieval(work, Embedder(context, work.records))
            result = codelets.run(context, work).codelets
        receipt = result.model_dump_json(indent=2)  # pm-lint: allow PM007 -- persist the actual typed phase result
        (directory / "result.json").write_text(receipt, encoding="utf-8")
        self.assertEqual([run.codelet for run in result.runs], ["EroticDesire"])
        return result

    def test_friendly_favor_stays_quiet_and_reciprocated_desire_registers(self) -> None:
        friendly = self.examine("friendly", CARD, "Your synth sounded great at rehearsal. I brought an extra cable.")
        desired = self.examine("desired", CARD + " Mara privately desires Jules and just told Jules so.",
                               "Your voice turns me on too. I want you and I like knowing you want me.")
        self.assertLess(friendly.ratings["EroticDesire"], desired.ratings["EroticDesire"])
        quiet = friendly.outputs[0].variants[0]
        self.assertEqual(len(quiet.percepts), 1, quiet.recommendation)
        self.assertEqual(quiet.percepts[0].salience, 0, quiet.percepts[0].text)
        self.assertEqual(quiet.percepts[0].valence, 0, quiet.percepts[0].text)
        active = desired.outputs[0].variants[0]
        self.assertGreater(active.salience, 0, active.recommendation)
        self.assertGreater(active.valence, 0, active.recommendation)
