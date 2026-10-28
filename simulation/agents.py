"""Director and conversational actors owned entirely by the simulation harness."""
from __future__ import annotations

import json
from pathlib import Path
import threading
from uuid import uuid4
from pydantic import BaseModel, ValidationError

from pm.llm.activation import ModelActivation
from pm.llm.config import load_config, resolve_connection
from pm.llm.stage_model import LazyStageModel
from pm.host.durable_files import atomic_json
from simulation.plans import ScenarioPlan, ExtractedFacts
from pm.llm.prompt_log import log_scope


class Agents:
    """Use the configured default model and retain every full input and output.

    Calls have no simulation spending or turn limit. Cancellation pauses a
    run through the existing model cancellation path. Logic failures propagate.
    """

    def __init__(self, config: str, evidence: Path, cancel: threading.Event | None = None):
        self.model = LazyStageModel(resolve_connection(load_config(config)), ModelActivation())
        self.evidence = evidence
        self.cancel = cancel

    def call(self, role: str, instruction: str, payload: dict, *, schema: type[BaseModel] | None = None,
             prepare=None) -> dict:
        """Run one focused JSON task. Invalid output retains its failed receipt."""
        from pm.llm.structured import extract_json, JsonOutputError
        required = " Required JSON schema: " + json.dumps(schema.model_json_schema()) if schema is not None else ""
        turns = [{"role": "system", "content": instruction + " Return one JSON object." + required},
                 {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        while True:
            call_id = uuid4().hex
            request = {"role": role, "messages": turns}
            path = self.evidence / "agents" / f"{call_id}.json"
            atomic_json(path, {"id": call_id, "status": "running", "request": request})
            output = None
            try:
                from pm.llm.prompt_log import invocation_scope
                with log_scope(self.evidence / "model-logs"), invocation_scope(role, call_id):
                    measured = self.model.measure_context(turns, cancel=self.cancel)
                    if measured["input_tokens"] > measured["input_budget_tokens"]:
                        raise ValueError("The scenario task exceeds the serving context. Its evidence was retained.")
                    output = self.model.complete(turns, cancel=self.cancel)
                result = extract_json(output)
                if prepare is not None:
                    result = prepare(result)
                if schema is not None:
                    result = schema.model_validate(result).model_dump(mode="json")
            except (JsonOutputError, ValidationError) as error:
                atomic_json(path, {"id": call_id, "status": "rejected", "request": request,
                                   "output": output, "error": str(error)})
                turns = [*turns, {"role": "assistant", "content": output},
                         {"role": "user", "content": "Correct only this response to satisfy the schema: " + str(error)}]
                measured = self.model.measure_context(turns, cancel=self.cancel)
                if measured["input_tokens"] > measured["input_budget_tokens"]:
                    raise ValueError("Scenario repair filled the model context. Edit the retained plan or input.") from error
                continue
            except BaseException as error:
                atomic_json(path, {"id": call_id, "status": "failed", "request": request,
                                   "output": locals().get("output"), "error": repr(error)})
                raise
            atomic_json(path, {"id": call_id, "status": "completed", "request": request, "output": output, "result": result})
            return result

    def actor(self, context: dict) -> dict:
        """Generate and check the scripted entity's next line before delivery.

        Rejected drafts remain evidence. Repair retains every rejection until
        acceptance or an explicit context failure. PM supplies its own replies.
        """
        from simulation.plans import ActorTurn, ActorDraftReview
        speaker = context["stage"]["speaker"]
        rejected = []
        while True:
            draft = self.call("participant_actor", f"Write only the next spoken message from {speaker}. "
                f"Your identity is {speaker}. The PM companion supplies its own speech. "
                "Follow the current stage instruction and explicitly deliver its mandatory facts. "
                "Keep the actor's scenario commitments even if the companion argues. "
                "Do not introduce future events or mention the harness. Return {content:string}.",
                {**context, "rejected_drafts": rejected}, schema=ActorTurn)
            review = self.call("actor_draft_check", "Check this scenario actor draft before delivery. "
                "Return {accepted:boolean,reason:string}. Check speaker identity, current stage facts, "
                "and the actor's required commitments. Reject a draft that speaks as another character, "
                "omits a required fact, or introduces a future event. Check compliance only; do not rate style. "
                "A request to greet someone must remain a request from the specified speaker.",
                {**context, "draft": draft}, schema=ActorDraftReview)
            if review["accepted"]:
                return draft
            rejected.append({"draft": draft, "reason": review["reason"]})

    def plan(self, brief: str, start_time: str) -> ScenarioPlan:
        """Extract mandatory facts, then build a separate editable chronological plan."""
        facts = self.call("scenario_facts", "Extract every mandatory fact, timed transition, and commitment from this scenario. "
            "Return {facts:[{id,text,at}], chronology:[{at,event}]}. Use string IDs and explicit ISO UTC offsets. "
            "Make going to sleep and waking up separate facts when specified. "
            "Interpret clock hours in the start time's zone; preserve that offset when crossing midnight. "
            "Set at:null only for facts without a stated time. "
            "Keep overnight changes and commitments that persist despite setbacks.",
            {"scenario": brief, "start_time": start_time}, schema=ExtractedFacts)
        result = self.call("scenario_plan", "Build a chronological simulation plan from these facts. "
            "Return {title,brief,start_time,facts:[{id,text}],steps:[{id,at,instruction,fact_ids,completion,target,speaker}]}. "
            "Retain every supplied fact ID. target is primary. speaker is Jesse. "
            "Use string IDs. completion is a text condition, never a boolean. "
            "A stage must use the exact at value of its timed facts. "
            "Completion requires spoken delivery to PM, not an unspoken thought or an invented physical observation. "
            "Stages can contain several turns. Complete the entire scenario; allow early termination only for failure. "
            "Do not add spending, turn, model-call, or elapsed-time limits.",
            {"brief": brief, "start_time": start_time, "extracted": facts}, schema=ScenarioPlan,
            prepare=lambda value: {**value, "facts": facts["facts"], "brief": brief, "start_time": start_time})
        return ScenarioPlan.model_validate(result)
