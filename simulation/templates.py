"""Versioned simulation templates for the browser, agents, and command line.

Templates describe a scenario before execution. An optional exact plan bypasses
generation. Daily planning keeps month-sized briefs inside individual model
contexts and produces the same editable plan that the engine already consumes.
See ``simulation.__init__`` for run and evidence ownership.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from pm.host.durable_files import atomic_json
from pm.llm.structured import JsonOutputError
from simulation.agents import Agents
from simulation.analysis import AnalysisSettings, analyze_run
from simulation.actors import ActorProfile, load_profile, suggested_start
from simulation.engine import RunEngine
from simulation.personas import DEFAULT_RUNS
from simulation.plans import DirectorSettings, Fact, ScenarioPlan, Step


class TemplateParticipant(BaseModel):
    """One source character home. Run creation forks it without changing it."""

    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    home: str = Field(min_length=1)


class TemplateConnection(BaseModel):
    """A directed public speech connection between participants."""

    model_config = ConfigDict(extra="forbid")
    source: str
    target: str
    name: str = "Other companion"


class SimulationTemplate(BaseModel):
    """Portable owner-editable input, distinct from mutable execution state."""

    model_config = ConfigDict(extra="forbid")
    schema_id: Literal["pm.simulation-template.v1"] = "pm.simulation-template.v1"
    title: str = "New simulation"
    brief: str = Field(min_length=1)
    start_time: AwareDatetime | None = None
    participants: list[TemplateParticipant] = Field(min_length=1)
    edges: list[TemplateConnection] = Field(default_factory=list[TemplateConnection])
    director: DirectorSettings = Field(default_factory=DirectorSettings)
    analysis: AnalysisSettings = Field(default_factory=AnalysisSettings)
    plan: ScenarioPlan | None = None

    @model_validator(mode="after")
    def known_participants(self) -> Self:
        """Reject ambiguous identities before a model or participant starts."""
        ids = {participant.id for participant in self.participants}
        if len(ids) != len(self.participants):
            raise ValueError("Template participant IDs must be unique.")
        for edge in self.edges:
            if edge.source not in ids or edge.target not in ids or edge.source == edge.target:
                raise ValueError("Connections must join distinct template participants.")
        if self.plan is not None and any(step.target not in ids for step in self.plan.steps):
            raise ValueError("Every planned scene must target a template participant.")
        return self


class DayRequest(BaseModel):
    """Planner context, including future plot that never reaches actor input."""

    title: str
    brief: str
    day_number: int
    director: DirectorSettings
    scene_times: list[datetime]
    target: str
    speaker: str
    companion: str
    previous_scenes: list[Step]


def generate_plan(template: SimulationTemplate, agents: Agents, profile: ActorProfile) -> ScenarioPlan:
    """Generate one reviewed-shape daily plan at a time and retain progress.

    Times are owner-selected metadata. The planner supplies current scenes and
    private human motives, never fixed public dialogue or companion responses.
    The final plan remains editable before a run starts.
    """
    start = template.start_time or suggested_start(profile)
    if start < profile.continuity_at:
        raise ValueError("The template starts before the selected character's saved history.")
    settings = template.director
    first_day = start.replace(hour=settings.first_hour, minute=0, second=0, microsecond=0)
    if first_day < start:
        first_day += timedelta(days=1)
    facts: list[Fact] = []
    steps: list[Step] = []
    for day in range(settings.duration_days):
        day_start = first_day + timedelta(days=day)
        span_hours = settings.last_hour - settings.first_hour
        times: list[datetime] = []
        for scene in range(settings.scenes_per_day):
            offset = span_hours * scene / max(1, settings.scenes_per_day - 1)
            times.append(day_start + timedelta(hours=offset))
        payload = DayRequest(title=template.title, brief=template.brief, day_number=day + 1,
            director=settings, scene_times=times, target=template.participants[0].id,
            speaker=profile.speaker_name, companion=profile.companion_name, previous_scenes=steps)

        def validate_day(value: ScenarioPlan) -> ScenarioPlan:
            if len(value.steps) != len(times) or [step.at for step in value.steps] != times:
                raise JsonOutputError("Use exactly the supplied scene_times, one scene per time, in order.")
            if any(step.speaker != payload.speaker or step.target != payload.target for step in value.steps):
                raise JsonOutputError("Use the supplied target and speaker for every scene.")
            return value

        daily = agents.call("plan_day",
            "Plan only day_number of an eventful text-based AI companion simulation. The brief and director "
            "define the whole arc. Respect scheduled_events and previous_scenes. Create exactly one step at "
            "each supplied scene_time. Mix mundane chat, consequences, and developing story threads. "
            "The user and companion are separate: no physical co-presence or physical companion actions. "
            "Romance, flirting, disagreement, remorse and joy can develop if the characters choose. "
            "Never require the companion to consent, love, forgive, agree, or produce a particular response. "
            "A user can express desire and invite closeness while respecting refusal. Give the user private "
            "motives and current circumstances, never prescribed lines or knowledge of later events. "
            "Facts describe only what the USER reports or chooses in this day's scenes. Split later outcomes "
            "from initial choices. Use event or commitment kinds, unique IDs, assigned fact_ids and semantic "
            "completion conditions. Not every exchange needs drama. Avoid checklist dialogue. "
            "start_time is the first scene_time. All timestamps retain the supplied UTC offset.",
            payload, schema=ScenarioPlan, prepare=validate_day)
        prefix = f"d{day + 1}-"
        for fact in daily.facts:
            facts.append(fact.model_copy(update={"id": prefix + fact.id}))
        for step in daily.steps:
            steps.append(step.model_copy(update={"id": prefix + step.id,
                "fact_ids": [prefix + key for key in step.fact_ids],
                "minimum_turns": settings.minimum_turns_per_scene}))
        atomic_json(agents.evidence / "planning-progress.json", {"completed_days": day + 1,
            "total_days": settings.duration_days, "steps": [step.model_dump(mode="json") for step in steps]})
    return ScenarioPlan(title=template.title, brief=template.brief, start_time=start,
                        facts=facts, steps=steps, director=settings)


def create_run(template: SimulationTemplate, root: Path, config: str) -> RunEngine:
    """Create a run from an exact plan or real daily planner calls."""
    template = template.model_copy(update={"participants": [
        TemplateParticipant(id=item.id, home=str(Path(item.home).expanduser().resolve()))
        for item in template.participants]})
    profile = load_profile(Path(template.participants[0].home))
    plan = template.plan
    if plan is None:
        plan = generate_plan(template, Agents(config, root.parent / "planning" / root.name), profile)
    participants = [{"id": item.id, "home": item.home} for item in template.participants]
    edges = [{"source": item.source, "target": item.target, "name": item.name} for item in template.edges]
    engine = RunEngine.create(root, plan, participants, edges, config)
    atomic_json(root / "template.json", template.model_copy(update={"plan": plan}).model_dump(mode="json"))
    atomic_json(root / "analysis-settings.json", template.analysis.model_dump(mode="json"))
    return engine


def main() -> int:
    """Validate, create, or analyze a simulation without a browser."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("schema")
    validate = commands.add_parser("validate")
    validate.add_argument("template", type=Path)
    create = commands.add_parser("create")
    create.add_argument("template", type=Path)
    create.add_argument("--config", default="config.yaml")
    create.add_argument("--data", type=Path, default=DEFAULT_RUNS)
    create.add_argument("--run-id", required=True)
    create.add_argument("--start", action="store_true")
    analyze = commands.add_parser("analyze")
    analyze.add_argument("root", type=Path)
    analyze.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    if args.command == "schema":
        print(json.dumps(SimulationTemplate.model_json_schema(), indent=2))
        return 0
    if args.command == "analyze":
        print(analyze_run(args.root, args.config))
        return 0
    template = SimulationTemplate.model_validate_json(args.template.read_text(encoding="utf-8"))
    if args.command == "validate":
        print(f"Valid template: {template.title}")
        return 0
    from pm.utils.operating_system import OperatingSystem
    OperatingSystem.install_child_cleanup()
    OperatingSystem.environment["PM_COMPLETION_CACHE"] = "0"
    # Resolve source homes relative to the template, not the caller's shell.
    for participant in template.participants:
        participant.home = str((args.template.resolve().parent / Path(participant.home).expanduser()).resolve())
    if not args.run_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Run IDs use letters, numbers, underscores, or hyphens.")
    engine = create_run(template, args.data / "runs" / args.run_id, args.config)
    print(f"Created {engine.root}", flush=True)
    if args.start:
        engine.start()
        assert engine.thread is not None
        engine.thread.join()
        print(engine.view()["status"], flush=True)
        return 0 if engine.view()["status"] == "completed" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
