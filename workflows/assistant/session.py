"""A pm-coder turn with only the character's explicitly configured MCP tools."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import pm_coder
from pydantic_ai import Agent
from pydantic_ai.toolsets import CombinedToolset
from pm_workflows.runtime import AgentModel
from pm.llm.multimodal import ImageAttachment
from pm.runs.role_runtime import PMWorkflowRuntime

if TYPE_CHECKING:
    from workflows.tool_call.steps import NativeToolPrompt


PRIVATE_INSTRUCTIONS = (
    "You are the companion in the supplied character and workspace. This is a private action. "
    "Use the configured MCP tools for your current purpose. Finish naturally with what you learned "
    "or did, including any uncertainty. Your conclusion returns to your thoughts. "
    "Distinguish partner reports, imagination, structured observations and interpreted images."
)


async def read_private_conclusion(agent: Agent[None, str], settings: pm_coder.Settings, store: pm_coder.SessionStore,
                                  text: str, images: tuple[ImageAttachment, ...], runtime: PMWorkflowRuntime) -> str:
    """Keep native per-request recovery active until the workflow is cancelled or the turn ends."""
    async with pm_coder.connected_agent(agent, store):
        turn = asyncio.create_task(pm_coder.run_turn(agent, settings, store, text, images))
        try:
            while not turn.done():
                runtime.check_cancelled()
                await asyncio.wait({turn}, timeout=0.2)
            return turn.result().response.strip()
        finally:
            turn.cancel()
            await asyncio.gather(turn, return_exceptions=True)


async def private_session(model: AgentModel, config: Path, log_root: Path, session_name: str,
                          prompt: NativeToolPrompt, runtime: PMWorkflowRuntime, timeout_seconds: int) -> str:
    config = config.resolve()
    log_root = log_root.resolve()
    settings = pm_coder.build_settings(cwd=config.parent, model=model.model, api_key=model.api_key,
        base_url=model.base_url, context_window=model.context_window, mcp_config=config,
        enable_write=False, workspace_discovery=False, enable_thinking=model.enable_thinking)
    store = pm_coder.SessionStore.open(config.parent, session_name, log_root=log_root)
    store.context_window = settings.context_window
    toolsets = pm_coder.load_timed_mcp_toolsets(config)
    native_model = pm_coder.make_model(settings, "agent", session=store)
    agent: Agent[None, str] = Agent(model=native_model, instructions=PRIVATE_INSTRUCTIONS,
        toolsets=[pm_coder.SnapshotToolset(CombinedToolset(toolsets), session=store)],
        model_settings=pm_coder.OpenAIChatModelSettings(parallel_tool_calls=False, extra_body=pm_coder.thinking_body(settings)))
    from pm.runs.draft_session import close_native_model
    try:
        async with asyncio.timeout(timeout_seconds):
            conclusion = await read_private_conclusion(agent, settings, store, prompt.prompt, prompt.images, runtime)
            if conclusion:
                return conclusion
            summary: Agent[None, str] = Agent(model=native_model,
                instructions="Summarize the private session from the recorded evidence in one sentence.")
            while not conclusion:
                runtime.check_cancelled()
                conclusion = await read_private_conclusion(summary, settings, store, "What happened?", (), runtime)
    finally:
        await close_native_model(agent)
    return conclusion
