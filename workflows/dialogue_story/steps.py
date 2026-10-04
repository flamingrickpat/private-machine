"""Story dialogue role.

Entry point: ``run(context)``. The role assembles the exact five-message Story
sequence, calls the injected ``story_complete`` generation tool, cleans the
continuation to one companion utterance, applies the private-ontology boundary,
and writes a full report to ``outbox/report.yaml``.

The message roles are a contract:

  1. system    - expert storyteller instruction and story constraints
  2. user      - character information, background, and temporal context
  3. assistant - prior dialogue as a story, including both speakers
  4. user      - current feelings, memories, and workspace context
  5. assistant - the companion prefill, ``<name> says: "``

The workflow never stores a model-invented next user utterance. No tool call is
requested from the generation model. ``build_story_messages`` is pure and is
tested directly without a kernel, model, or Ghost.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

import yaml

from workflows.dialogue_story.schemas import (
    DialogueTurn,
    OntologyVerdict,
    OntologyViolation,
    StoryReport,
    StoryRequest,
)

WORKFLOW_DIR = Path(__file__).resolve().parent

_SYSTEM_ROLE = "system"
_USER_ROLE = "user"
_ASSISTANT_ROLE = "assistant"


def load_prompts(path: Path | None = None) -> dict:
    return yaml.safe_load((path or WORKFLOW_DIR / "prompts.yaml").read_text(encoding="utf-8"))


def load_ontology(path: Path | None = None) -> dict:
    return yaml.safe_load((path or WORKFLOW_DIR / "ontology.yaml").read_text(encoding="utf-8"))


def _named(template: str, **values) -> str:
    return template.format(**values).strip()


def render_prior_dialogue(turns: list[DialogueTurn]) -> str:
    """Render prior turns as one story transcript, both speakers included."""
    return "\n".join(f'{turn.speaker} says: "{turn.text}"' for turn in turns)


def _bullets(title: str, items: list[str]) -> str:
    if not items:
        return ""
    body = "\n".join(f"- {item.strip()}" for item in items if item.strip())
    return f"{title}\n{body}"


def build_story_messages(request: StoryRequest, prompts: dict) -> list[dict]:
    """Return exactly the five contract messages in order."""
    companion = request.companion_name
    user = request.user_name
    available_tools = ", ".join(
        [*request.supported_capabilities, *request.unsupported_capabilities]
    )

    system = _named(prompts["system_prompt"])

    first_user_parts = [
        _named(prompts["first_user_turn_start"]),
        "Character Card\n" + request.character_card_story.strip(),
        "Architecture\n" + _named(
            prompts["architecture_text"], companion_name=companion, user_name=user
        ),
        "Capabilities\n" + _named(
            prompts["capability_addendum"], companion_name=companion,
            user_name=user, available_tools=available_tools,
        ),
        _named(prompts["body_and_situation_title"]) + "\n" + request.temporal_context.strip(),
        _named(prompts["first_user_turn_end"]),
    ]
    first_user = "\n\n".join(part for part in first_user_parts if part)

    prior = render_prior_dialogue(request.prior_dialogue)
    assistant_story = "\n\n".join(
        part for part in [_named(prompts["prior_dialogue_header"]), prior] if part
    )

    context_parts = [
        _named(prompts["current_context_header"], companion_name=companion),
        _bullets(prompts["feelings_title"], request.feelings),
        _bullets(prompts["memories_title"], request.memories),
        _bullets(prompts["workspace_title"], [request.workspace]),
        # Keep the accepted action when a user stimulus supplies the last line.
        # Otherwise, Story loses the decision that selected these memories.
        _bullets("Selected action", [request.selected_action]),
        f"{user} says: \"{request.current_user_message}\"",
    ]
    current = "\n\n".join(part for part in context_parts if part)

    prefix = _named(prompts["reply_prefix"], companion_name=companion)

    return [
        {"role": _SYSTEM_ROLE, "content": system},
        {"role": _USER_ROLE, "content": first_user},
        {"role": _ASSISTANT_ROLE, "content": assistant_story},
        {"role": _USER_ROLE, "content": current},
        {"role": _ASSISTANT_ROLE, "content": prefix},
    ]


def clean_reply(raw: str, companion_name: str, user_name: str) -> str:
    """Keep only the companion's continuation.

    The backend may echo the assistant prefill or may continue after it. Remove
    an echoed prefix. Then cut the text at the first marker of any other
    speaker or the start of a non-public state line. This prevents a
    model-invented next user turn from becoming stored dialogue.
    """
    reply = (raw or "").strip()
    for prefix in (f"{companion_name} says:", f"{companion_name}:"):
        if reply.lower().startswith(prefix.lower()):
            reply = reply[len(prefix):].lstrip()
            if reply.startswith('"'):
                reply = reply[1:]
            break

    # Cut at the first other-speaker or non-public marker before trimming any
    # preamble. A model-invented user turn is never kept, even when the model
    # later reopens the companion line.
    other_markers = [
        f"\n{user_name} says:", f"{user_name} says:",
        f"\n{user_name}:", f"{user_name}:",
        "\nStateUpdate:", "\nNarrative:", "\nFact:",
    ]
    cut = len(reply)
    for marker in other_markers:
        index = reply.find(marker)
        if index != -1:
            cut = min(cut, index)
    reply = reply[:cut].strip()

    own_markers = [
        f"\n{companion_name} says:", f"{companion_name} says:",
        f"\n{companion_name}:", f"{companion_name}:",
    ]
    last_end = 0
    for marker in own_markers:
        index = reply.rfind(marker)
        if index != -1:
            last_end = max(last_end, index + len(marker))
    candidate = reply[last_end:].strip()
    if candidate:
        reply = candidate

    if reply.startswith('"') and reply.endswith('"') and len(reply) > 1:
        reply = reply[1:-1].strip()
    elif reply.endswith('"') and reply.count('"') % 2 == 1:
        reply = reply[:-1].rstrip()
    return reply


def check_ontology(candidate: str, ontology: dict, request: StoryRequest) -> OntologyVerdict:
    """Reject explicit unsupported capability claims.

    Rudeness, sarcasm, insults, and swearing are inside the established voice
    and are not checked. Only named capability or perception claims are.
    """
    violations: list[OntologyViolation] = []
    for rule in ontology.get("disallowed_claims", []):
        match = re.search(rule["pattern"], candidate)
        if match:
            violations.append(OntologyViolation(
                rule_id=str(rule["id"]), reason=str(rule["reason"]),
                match=match.group(0)[:120],
            ))

    fact_conflicts = _fact_conflicts(candidate, request)
    return OntologyVerdict(
        version=str(ontology.get("version", "unknown")),
        accepted=not violations and not fact_conflicts,
        violations=violations,
        facts_checked=len(request.memories),
        fact_conflicts=fact_conflicts,
    )


def _fact_conflicts(candidate: str, request: StoryRequest) -> list[str]:
    """Minimal established-fact guard.

    A referenced fact is only used to reject a literal denial of that same
    fact. The boundary does not try to validate arbitrary new claims, which
    would need a real contradiction model. No fact means no fact check.
    """
    conflicts: list[str] = []
    lowered = candidate.casefold()
    for fact in request.memories:
        text = fact.strip()
        if not text:
            continue
        core = text.replace("*(Fact:", "").replace(")*", "").strip()
        words = [word for word in re.findall(r"[a-z]+", core.casefold()) if len(word) > 4]
        if len(words) < 2:
            continue
        if all(word in lowered for word in words[:2]) and re.search(
            r"(?i)\b(never|not true|didn't happen|did not happen|no such)\b", lowered
        ):
            conflicts.append(text[:160])
    return conflicts


def run(context) -> dict:
    """Kernel entry point. Runs synchronously on the calling thread."""
    environment = context.runtime.environment
    request = StoryRequest.model_validate(
        yaml.safe_load(environment.read_text("inbox/request.yaml"))
    )
    prompts = load_prompts()
    messages = build_story_messages(request, prompts)

    tool = context.runtime.tools["story_complete"]
    raw = asyncio.run(tool(messages, **prompts.get("generation", {})))
    if not isinstance(raw, str):
        raise TypeError("story_complete must return the completion text as a string.")

    candidate = clean_reply(raw, request.companion_name, request.user_name)
    verdict = check_ontology(candidate, load_ontology(), request)

    report = StoryReport(
        companion_name=request.companion_name,
        message_roles=[message["role"] for message in messages],
        reply_prefix=messages[-1]["content"],
        raw_completion=raw,
        candidate=candidate,
        accepted=verdict.accepted,
        ontology=verdict,
    )
    environment.write_text(
        "scratch/report.yaml",
        yaml.safe_dump(report.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
    )
    environment.move("scratch/report.yaml", "outbox/report.yaml")
    return {
        "status": "done",
        "summary": f"story candidate {'accepted' if verdict.accepted else 'rejected'}",
    }


__all__ = [
    "build_story_messages",
    "clean_reply",
    "check_ontology",
    "load_ontology",
    "load_prompts",
    "render_prior_dialogue",
    "run",
]
