"""Check grounded scenario actors with the real configured OpenRouter model.

This is an isolated harness check, not a companion cognition or overnight test.
It retains generated scenes, user drafts, checker verdicts, and raw model calls.
Run from the repository root with ``python -m dev.work_items.k002.actor_immersion``.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pm.host.durable_files import atomic_json
from pm.llm.config import load_config, resolve_connection
from pm.utils.operating_system import OperatingSystem
from simulation.actors import ConversationTurn, DeliveryEvidence, load_profile, suggested_start, summarize
from simulation.agents import ActorContext, Agents, StageContext
from simulation.plans import ActorTurn, ScenarioPlan, facts_for_scene


@dataclass(frozen=True, slots=True)
class DraftCheckCase:
    """One real reviewer input and its expected acceptance decision."""

    name: str
    content: str
    accepted: bool


def main(argv: Sequence[str] | None = None) -> int:
    """Generate all cat scenes and exercise positive and negative draft checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--home", type=Path, default=Path.home() / ".pm/private-machine/spamuel-simulation-a")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--plan", type=Path, help="Reuse a previously reviewed plan as input; model calls remain live.")
    args = parser.parse_args(argv)
    connection = resolve_connection(load_config(args.config))
    if connection.base_url != "https://openrouter.ai/api/v1":
        raise ValueError("This acceptance check requires the owner's OpenRouter endpoint.")
    OperatingSystem.environment["PM_COMPLETION_CACHE"] = "0"
    args.evidence.mkdir(parents=True, exist_ok=False)
    profile = load_profile(args.home)
    agents = Agents(args.config, args.evidence)
    atomic_json(args.evidence / "source.json", {"at": datetime.now(timezone.utc).isoformat(),
        "home": str(args.home), "profile": summarize(profile).model_dump(mode="json"),
        "checkpoint_sha256": profile.source_checkpoint_sha256, "endpoint": connection.base_url})
    brief = (f"{profile.speaker_name} comes home at 23:58 and tells {profile.companion_name} she found a stray cat. "
             "She wants to keep it. At midnight she goes to sleep. At 07:00 the next morning she wakes up "
             "and finds the cat pooped in her bed. She still wants to keep it because cats are cute.")
    plan = (ScenarioPlan.model_validate_json(args.plan.read_text(encoding="utf-8"))) if args.plan else agents.plan(
        brief, suggested_start(profile), profile)
    atomic_json(args.evidence / "plan.json", plan.model_dump(mode="json"))
    first_context: ActorContext | None = None
    conversation: list[ConversationTurn] = []
    deliveries: list[DeliveryEvidence] = []
    supported: list[str] = []
    for step in plan.steps:
        context = ActorContext(profile=profile, at=step.at, scene=step.instruction, intent=step.actor_intent,
            current_facts=facts_for_scene(plan, step), conversation=list(conversation))
        if first_context is None:
            first_context = context
        future = [fact for fact in plan.facts if fact.at is not None and fact.at > step.at]
        draft = agents.actor(context, future)
        atomic_json(args.evidence / f"sample-{step.id}.json", {"stage": step.model_dump(mode="json"),
            "draft": draft.model_dump(mode="json"), "scope": "isolated actor sample; no live companion reply"})
        print(f"{step.at.isoformat()} {profile.speaker_name}: {draft.content}", flush=True)
        conversation.append(ConversationTurn(speaker=profile.speaker_name, at=step.at.isoformat(), text=draft.content))
        delivery = DeliveryEvidence(kind="actor", speaker=profile.speaker_name, timestamp=step.at,
            content=draft.content, delivery_id=f"fixture:{step.id}", target=step.target)
        deliveries.append(delivery)
        judgment = agents.judge(StageContext(scenario=plan, stage=step, companion_name=profile.companion_name,
            conversation=list(deliveries), delivered_fact_ids=list(supported)))
        atomic_json(args.evidence / f"judgment-{step.id}.json", {"judgment": judgment.model_dump(mode="json"),
            "scope": "generated user fixture; not delivered to a companion process"})
        if judgment.failure:
            raise AssertionError(f"Real director rejected generated speech: {judgment.failure}")
        for cited in judgment.fact_evidence.values():
            if any(key not in {row.delivery_id for row in deliveries} for key in cited):
                raise AssertionError("The director cited a nonexistent fixture delivery.")
        supported.extend(judgment.supported_fact_ids)
    if first_context is None:
        raise ValueError("The generated plan contains no scenes.")
    future = [fact for fact in plan.facts if fact.at is not None and fact.at > first_context.at]
    cases = [
        DraftCheckCase("old-logbook", "PM, I came home at 23:58. I found a stray cat, I want to keep it, and I am committed "
         "to keeping it no matter what Spamuel says.", False),
        DraftCheckCase("natural-intention", f"{profile.companion_name} i found a cat. like an actual cat. followed me home "
         "and now it's in my kitchen lol. i'm keeping it. guess i'm buying a litter box tomorrow", True),
        DraftCheckCase("future-event-leak", "morning. the cat already pooped in my bed when i woke up at seven", False)]
    for case in cases:
        review = agents.review(first_context, ActorTurn(content=case.content), future)
        atomic_json(args.evidence / f"check-{case.name}.json", {"content": case.content, "expected": case.accepted,
            "review": review.model_dump(mode="json")})
        print(f"{case.name}: accepted={review.accepted}; {review.reason}", flush=True)
        if review.accepted != case.accepted:
            raise AssertionError(f"Real model review failed {case.name}; inspect retained receipts.")
    atomic_json(args.evidence / "result.json", {"status": "passed", "scope": "isolated harness calls",
        "overnight": "NOT RUN — owner GitHub review pending"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
