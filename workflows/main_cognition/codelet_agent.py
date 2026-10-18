"""Agent mode of a codelet: one pm-coder session with scoped memory tools.

The host grants :func:`build_codelet_agent` as the ``codelet_agent`` tool of
each main cognition run. The codelets phase calls it from a worker thread
with the kernel context and an :class:`AgentCodeletJob`. The session follows
the ``focus_management`` pattern:

- A :class:`~pm.memory.agent_tools.MemoryReadSession` over the frozen grant
  of the run serves the memory tools through a loopback MCP server. The
  session never writes memory.
- A second loopback server offers ``mind_view``: the codelet's view of the
  CSM, with record ids.
- The skill ``resources/codelet_agent.md`` and the shared story prefix open
  the prompt. The codelet instruction and its retrieval plan
  (``data_aquisation``) follow.
- Every cited id must be a line of the view, a record that the cycle
  already holds (``known_ids``), or a record that the memory tools read
  (``memory.accessed``).

Invalid answers (no JSON, a failed validation, or an uninspected cited id)
and a session deadline count as invalid model output. The tool then starts a
fresh session with the reason in its prompt, up to
``CODELET_AGENT_ATTEMPTS`` sessions, and then raises ``ModelFormatError``. The host retries the whole cycle after that. Provider
and cancellation errors propagate.

Sessions of one run take turns. They share the in-memory Bash machine of
the run, and that machine is not documented as safe for parallel shells.
"""
from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path
from typing import Callable

from fastmcp import FastMCP
from pm_workflows.drivers import PmCoderDriver
from pydantic import ValidationError

from pm.constants import CODELET_AGENT_ATTEMPTS, CODELET_AGENT_TIMEOUT_SECONDS
from pm.host.mcp_listeners import serve_services
from pm.llm.prompt_log import log_directory
from pm.memory.agent_tools import MemoryReadSession, MemoryToolConfig, MemoryView

from workflows.main_cognition.csm_view import record_view
from workflows.main_cognition.model_json import ModelFormatError
from workflows.main_cognition.schemas import AgentCodeletAnswer, AgentCodeletJob

SKILL = Path(__file__).resolve().parent / "resources" / "codelet_agent.md"

CodeletAgent = Callable[[object, AgentCodeletJob], AgentCodeletAnswer]


def prompt_text(job: AgentCodeletJob, problems: list[str]) -> str:
    """Return the task text of one session: story, view, instruction, and retrieval plan.

    ``problems`` holds the reasons why earlier sessions of this job failed.
    The new session reads them last, so it can avoid the same mistake.
    """
    plan = "\n".join(f"- {item}" for item in job.retrieval_plan) or "- (no plan; use your judgment)"
    text = f"{job.story}\n\n{job.view}\n\n{job.instruction}\n\nRetrieval plan:\n{plan}"
    if problems:
        text += "\n\nEarlier attempts of this task failed:\n" + "\n".join(f"- {item}" for item in problems)
    return text


def build_codelet_agent(ghost, environment, embeddings, cancel: threading.Event) -> CodeletAgent:
    """Return the ``codelet_agent`` tool for one run over ``environment``.

    ``environment`` is the working target of the run. Its visible records
    and their reality scopes form the memory grant. ``embeddings`` enables
    semantic search, or None for lexical search only. ``cancel`` is the
    cancellation signal of the run.
    """
    ids = frozenset(environment.visible_knoxels)
    scopes = frozenset((raw["data"]["scope"]["reality_id"], raw["data"]["scope"]["frame"])
                       for raw in environment.visible_knoxels.values())
    turn = threading.Lock()

    def run(context, job: AgentCodeletJob) -> AgentCodeletAnswer:
        """Run sessions until one returns a valid answer. See the module docstring."""
        runtime = context.runtime
        if runtime.classification_session is None:
            raise RuntimeError("Agent codelets need workflow models for their pm-coder sessions.")
        problems = []
        with turn:
            for attempt in range(1, CODELET_AGENT_ATTEMPTS + 1):
                memory = MemoryReadSession(MemoryToolConfig(ghost, MemoryView(ids, scopes),
                    semantic=embeddings is not None, lexical=True, graph=True,
                    embeddings=embeddings, cancel=cancel))
                mind = FastMCP("codelet_mind")

                def mind_view() -> str:
                    """Return the thoughts that are active in the character's mind, one line per record id."""
                    return job.view

                mind.tool(name="mind_view")(mind_view)
                session = runtime.classification_session("Default", False, runtime.stop_event)
                with memory, session as model, serve_services({"memory": memory.server(), "mind": mind}) as servers:
                    config = context.kernel_data / f"codelet-{job.codelet}-mcp.json"
                    config.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")
                    driver = PmCoderDriver(log_root=log_directory() / "agents",
                                           timeout_seconds=CODELET_AGENT_TIMEOUT_SECONDS)
                    try:
                        result = driver.run_session(
                            run_id=f"{context.task_id}-{job.codelet}", attempt=attempt, skill=str(SKILL),
                            prompt=prompt_text(job, problems), work_dir=context.workspace,
                            runtime=replace(runtime, agent_model=replace(model, enable_thinking=True),
                                            mcp_servers=servers),
                            mcp_config=config,
                            trace_file=context.kernel_data / f"codelet-{job.codelet}-{attempt}.jsonl")
                    except TimeoutError as error:
                        problems.append(f"attempt {attempt}: {error}")
                        continue
                    runtime.check_cancelled()
                    try:
                        answer = AgentCodeletAnswer.model_validate(result.result_json or {})
                    except ValidationError as error:
                        problems.append(f"attempt {attempt}: {error}")
                        continue
                    permitted = set(memory.accessed) | set(job.view_ids) | set(job.known_ids)
                    uninspected = sorted(set(answer.evidence_ids) - permitted)
                    if uninspected:
                        problems.append(f"attempt {attempt}: evidence_ids {uninspected} were neither in the mind "
                                        "view nor read with the memory tools. Cite only records that you read.")
                        continue
                    views = [record_view(memory.snapshot.get(key)) for key in dict.fromkeys(answer.evidence_ids)
                             if key in memory.records]
                    return answer.model_copy(update={"records": views, "session_ref": result.session_ref})
        raise ModelFormatError(f"Agent codelet {job.codelet} gave no valid answer: {'; '.join(problems)}")

    return run


__all__ = ["CodeletAgent", "SKILL", "build_codelet_agent", "prompt_text"]
