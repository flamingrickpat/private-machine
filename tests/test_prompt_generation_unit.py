from __future__ import annotations

from datetime import datetime, timedelta
import importlib
from types import SimpleNamespace

from pm.data_structures import (
    ClusterType,
    DeclarativeFactKnoxel,
    Feature,
    FeatureType,
    MemoryClusterKnoxel,
    Narrative,
    NarrativeTypes,
)
from pm.ghosts.agent_context import AgentContextComposer, AgentContextConfig, AgentContextWeights
from pm.ghosts.procedures.action import ActionSelectionProc
from pm.ghosts.procedures.reply import ReplyGenerationProc
from pm.ghosts.schemas import BehaviorOutput


class _PromptLlm:
    def __init__(self, model_ctx: int = 4096):
        self.model_ctx = int(model_ctx)
        self.last_messages = []

    def get_embedding(self, text: str):
        t = str(text or "").lower()
        if any(k in t for k in ["migration", "schema", "rollout", "backfill", "order"]):
            return [1.0, 0.0, 0.0]
        if any(k in t for k in ["weather", "coffee", "commute", "sunny"]):
            return [0.0, 1.0, 0.0]
        return [0.2, 0.2, 0.2]

    def get_max_tokens(self, _preset):
        return self.model_ctx

    def completion_tool(self, preset, inp, tools, comp_settings=None):
        self.last_messages = list(inp)
        schema_name = getattr(tools[0], "__name__", "") if tools else ""
        if schema_name == "BehaviorOutput":
            return None, [
                BehaviorOutput(
                    action_description="Provide concrete migration order",
                    speech="Backup first, then schema migration, then backfill.",
                    internal_thought="Use direct structured instructions.",
                    tool_call=None,
                )
            ]
        return None, []

    def completion_text(self, preset, inp, comp_settings=None, discard_thinks=True):
        self.last_messages = list(inp)
        return "Stub reply."


class _PromptGhost:
    def __init__(
        self,
        *,
        companion_name: str = "Companion",
        user_name: str = "User",
        model_ctx: int = 4096,
    ):
        self.current_tick_id = 80
        self.llm = _PromptLlm(model_ctx=model_ctx)
        self.config = SimpleNamespace(
            companion_name=companion_name,
            user_name=user_name,
            universal_character_card=(
                f"{companion_name} is concise, technically precise, and honest about limitations."
            ),
            agent_context_weights={
                "workspace": 0.25,
                "latest": 0.30,
                "timeline": 0.30,
                "static": 0.15,
            },
            agent_context_min_section_tokens=48,
            agent_context_latest_messages=10,
            agent_context_workspace_items=10,
            agent_context_target_ratio=0.70,
            agent_context_safety_margin_tokens=96,
            supported_capabilities=[
                "I can communicate with you via text in this chat.",
                "I can reason about my internal state and report simulated emotions.",
            ],
            unsupported_capabilities=[
                "I cannot create or host a literal virtual reality world you can physically join.",
            ],
            capability_notes=[
                "Never claim capabilities beyond the current implementation.",
            ],
        )

        self.all_knoxels = {}
        self.all_features = []
        self.all_declarative_facts = []
        self.all_narratives = []
        self.all_episodic_memories = []
        self.all_intentions = [SimpleNamespace(content="Keep replies concrete and ordered.")]
        self.current_state = SimpleNamespace(
            latent_mental_state=SimpleNamespace(
                state_core=SimpleNamespace(valence=0.2, arousal=0.35),
                state_emotions="focused",
                state_needs="stable",
                state_cognition="analytical",
            ),
            timestamp=datetime(2026, 2, 18, 19, 5),
        )
        self.primary_stimulus = SimpleNamespace(content="Need exact migration rollout order.")
        self.conscious_broadcast = None
        self.conscious_candidates = []
        self.current_coalition = []
        self.ego_directive = "Prioritize practical clarity."
        self.thought_blueprint_directive = "Convert context into direct concrete steps."
        self.thought_blueprint_action_hints = ["Reply", "LoopBack"]
        self.qualia_action_bias = 0.0
        self.qualia_action_directive = ""
        self.generation_token_cap = 280
        self.memory_recon_last = {"tick": 78, "ok": True, "skipped": False, "reason": "periodic"}
        self.reply_blueprint_last = {
            "name": "answer_directive",
            "description": "direct concise reply",
            "generation_prefix": "Keep the answer direct and ordered.",
        }
        self.reply_blueprint_history = []
        self.selected_action_schema = BehaviorOutput(
            action_description="Provide exact migration order",
            speech="Backup, migrate schema, then backfill.",
            internal_thought="Stay concise and technical.",
            tool_call=None,
        )
        self.simulated_reply = ""
        self.simulation_bundle_last_model = SimpleNamespace(
            winner_lens="world",
            outcomes=[
                SimpleNamespace(
                    lens="world",
                    summary="A direct ordered reply should increase user trust.",
                    utility=0.82,
                    polarity="positive",
                    repellers=["ambiguity"],
                )
            ],
        )

    def add_knoxel(self, knoxel, generate_embedding: bool = True):
        if getattr(knoxel, "id", -1) == -1:
            knoxel.id = len(self.all_knoxels) + 1
        if getattr(knoxel, "tick_id", -1) == -1:
            knoxel.tick_id = self.current_tick_id
        if generate_embedding and not list(getattr(knoxel, "embedding", []) or []):
            knoxel.embedding = self.llm.get_embedding(str(getattr(knoxel, "content", "") or ""))
        self.all_knoxels[knoxel.id] = knoxel
        if isinstance(knoxel, Feature):
            self.all_features.append(knoxel)
        if isinstance(knoxel, DeclarativeFactKnoxel):
            self.all_declarative_facts.append(knoxel)
        if isinstance(knoxel, Narrative):
            self.all_narratives.append(knoxel)
        if isinstance(knoxel, MemoryClusterKnoxel):
            self.all_episodic_memories.append(knoxel)
        return knoxel.id

    def get_knoxel_by_id(self, kid):
        return self.all_knoxels.get(kid)

    def get_narrative(self, narrative_type, target_name):
        for n in reversed(self.all_narratives):
            if getattr(n, "narrative_type", None) == narrative_type and str(getattr(n, "target_name", "") or "") == str(target_name or ""):
                return n
        return None


def _seed_prompt_data(ghost: _PromptGhost) -> None:
    base = datetime(2026, 2, 10, 8, 0)

    scripted = [
        "We should discuss migration strategy soon.",
        "Sure, I can help you with that.",
        "Yesterday was sunny and warm.",
        "I had coffee before the commute.",
        "Please give exact migration order for schema rollout.",
        "Use strict ordering to avoid data corruption.",
        "Unrelated weather check: cloudy now.",
        "I need concrete backfill sequence now.",
        "How do we avoid locking issues during migration?",
        "Order should be backup, schema update, then backfill.",
        "Small talk about weather again.",
        "Need implementation details for rollback safety.",
    ]

    for i, text in enumerate(scripted):
        t0 = base + timedelta(hours=i * 4)
        feat = Feature(
            content=text,
            feature_type=FeatureType.Dialogue,
            source="User" if i % 2 == 0 else ghost.config.companion_name,
            causal=True,
            tick_id=i + 1,
            timestamp_world_begin=t0,
            timestamp_world_end=t0 + timedelta(minutes=1),
            incentive_salience=0.85 if "migration" in text.lower() or "backfill" in text.lower() else 0.10,
            embedding=ghost.llm.get_embedding(text),
        )
        ghost.add_knoxel(feat, generate_embedding=False)

    broadcast = Feature(
        content="Need exact migration rollout order and backfill sequence.",
        feature_type=FeatureType.Thought,
        source="Consciousness",
        causal=True,
        tick_id=79,
        timestamp_world_begin=base + timedelta(days=2),
        timestamp_world_end=base + timedelta(days=2, minutes=1),
        incentive_salience=0.95,
        embedding=ghost.llm.get_embedding("migration rollout order"),
    )
    ghost.add_knoxel(broadcast, generate_embedding=False)
    ghost.conscious_broadcast = broadcast

    cand_relevant = Feature(
        content="The user repeatedly asks for strict migration order and rollback-safe steps.",
        feature_type=FeatureType.MemoryRecall,
        source="PAM",
        causal=True,
        tick_id=78,
        timestamp_world_begin=base + timedelta(days=1, hours=20),
        timestamp_world_end=base + timedelta(days=1, hours=20, minutes=1),
        incentive_salience=0.9,
        embedding=ghost.llm.get_embedding("strict migration order rollback"),
    )
    cand_irrelevant = Feature(
        content="Weather and commute chatter with low task relevance.",
        feature_type=FeatureType.MemoryRecall,
        source="PAM",
        causal=True,
        tick_id=40,
        timestamp_world_begin=base + timedelta(hours=10),
        timestamp_world_end=base + timedelta(hours=10, minutes=1),
        incentive_salience=0.1,
        embedding=ghost.llm.get_embedding("weather commute chatter"),
    )
    ghost.add_knoxel(cand_relevant, generate_embedding=False)
    ghost.add_knoxel(cand_irrelevant, generate_embedding=False)
    ghost.conscious_candidates = [cand_relevant, cand_irrelevant]
    ghost.current_coalition = [(cand_relevant.id, 0.95), (cand_irrelevant.id, 0.10)]

    ghost.add_knoxel(
        DeclarativeFactKnoxel(
            content="User prefers concrete ordered migration steps.",
            reason="repeated direct requests",
            category=["preference", "workflow"],
            importance=0.92,
            time_dependent=0.08,
            embedding=ghost.llm.get_embedding("concrete ordered migration steps"),
            tick_id=75,
        ),
        generate_embedding=False,
    )
    ghost.add_knoxel(
        DeclarativeFactKnoxel(
            content="Current session had coffee small talk.",
            reason="ephemeral topic",
            category=["ephemeral"],
            importance=0.20,
            time_dependent=0.95,
            embedding=ghost.llm.get_embedding("coffee small talk"),
            tick_id=76,
        ),
        generate_embedding=False,
    )

    ghost.add_knoxel(
        Narrative(
            content="Companion typically resolves technical tasks with direct ordered steps.",
            narrative_type=NarrativeTypes.BehaviorActionSelection,
            target_name=ghost.config.companion_name,
            embedding=ghost.llm.get_embedding("direct ordered technical steps"),
            tick_id=70,
        ),
        generate_embedding=False,
    )
    ghost.add_knoxel(
        Narrative(
            content="Companion sees self as grounded and precise under pressure.",
            narrative_type=NarrativeTypes.SelfImage,
            target_name=ghost.config.companion_name,
            embedding=ghost.llm.get_embedding("grounded precise under pressure"),
            tick_id=69,
        ),
        generate_embedding=False,
    )
    ghost.add_knoxel(
        Narrative(
            content="Companion values transparent collaboration with the user.",
            narrative_type=NarrativeTypes.Relations,
            target_name=ghost.config.companion_name,
            embedding=ghost.llm.get_embedding("transparent collaboration"),
            tick_id=68,
        ),
        generate_embedding=False,
    )
    ghost.add_knoxel(
        Narrative(
            content="User prefers compact technical instructions over broad theory.",
            narrative_type=NarrativeTypes.PsychologicalAnalysis,
            target_name=ghost.config.user_name,
            embedding=ghost.llm.get_embedding("compact technical instructions"),
            tick_id=67,
        ),
        generate_embedding=False,
    )

    ghost.add_knoxel(
        MemoryClusterKnoxel(
            content="Week summary: repeated migration-order discussions, user rejected vague answers.",
            level=5,
            cluster_type=ClusterType.Temporal,
            tick_id=66,
            timestamp_world_begin=base,
            timestamp_world_end=base + timedelta(days=2),
            embedding=ghost.llm.get_embedding("migration order summary"),
        ),
        generate_embedding=False,
    )


def test_agent_context_composer_balances_sections_and_prefers_relevant_content():
    ghost = _PromptGhost(model_ctx=2400)
    _seed_prompt_data(ghost)

    packet = AgentContextComposer.build(
        ghost,
        focus_text="Need exact migration rollout order with safe backfill sequence.",
        config=AgentContextConfig(
            max_tokens=560,
            weights=AgentContextWeights(workspace=0.25, latest=0.30, timeline=0.30, static=0.15),
            latest_messages=10,
            workspace_items=8,
            min_section_tokens=48,
        ),
        purpose="unit_test_prompt_generation",
    )

    assert "[WORKSPACE]" in packet.text
    assert "[LATEST]" in packet.text
    assert "[TIMELINE]" in packet.text
    assert "[STATIC]" in packet.text
    assert "migration rollout order" in packet.text.lower()
    assert "User prefers concrete ordered migration steps.".lower() in packet.text.lower()
    assert "Current session had coffee small talk.".lower() not in packet.text.lower()
    assert set(packet.debug.get("sections_present", [])) >= {"workspace", "latest", "timeline", "static"}
    assert int(packet.debug.get("final_tokens", 0) or 0) <= 560


def test_action_selection_prompt_is_well_formed_and_uses_balanced_history():
    ghost = _PromptGhost(companion_name="Astra", user_name="Riley", model_ctx=2800)
    _seed_prompt_data(ghost)

    out = ActionSelectionProc.run(ghost)
    assert out is not None

    msgs = list(ghost.llm.last_messages)
    assert msgs
    assert msgs[0][0] == "system"
    system_prompt = msgs[0][1]
    user_prompt = [text for role, text in msgs if role == "user"][-1]

    assert "Action Selection module for Astra" in system_prompt
    assert "Hard architecture capability boundary:" in system_prompt
    assert "Conscious Content:" in user_prompt
    assert "Simulation Prediction:" in user_prompt
    assert "Relevant lived history:" in user_prompt
    assert "[WORKSPACE]" in user_prompt
    assert "[LATEST]" in user_prompt
    assert "[TIMELINE]" in user_prompt
    assert "[STATIC]" in user_prompt
    assert "Architecture Capability Profile:" in user_prompt
    assert {"workspace", "latest", "timeline", "static"}.issubset(
        set((getattr(ghost, "agent_context_last", {}) or {}).get("sections_present", []))
    )


def test_reply_story_prompt_has_expected_system_user_assistant_shape():
    ghost = _PromptGhost(companion_name="Astra", user_name="Riley", model_ctx=3200)
    _seed_prompt_data(ghost)

    behavior = ghost.selected_action_schema
    context = ReplyGenerationProc._build_reply_context(ghost)
    system_prompt, user_prompt, msgs = ReplyGenerationProc._build_prompts(ghost, behavior, context)

    assert msgs[0][0] == "system"
    assert msgs[1][0] == "user"
    assert msgs[2][0] == "assistant"
    assert system_prompt.startswith("You are an expert story writer.")
    assert "Astra" in system_prompt
    assert "Riley" in system_prompt
    assert "**Astra's Character:**" in user_prompt
    assert "**Behavior Prior:**" in user_prompt
    assert "**Self Narrative Prior:**" in user_prompt
    assert "**Relation Prior:**" in user_prompt
    assert "**User Model Prior:**" in user_prompt
    assert "**Relevant Facts:**" in user_prompt
    assert "**Architecture Capability Profile:**" in user_prompt
    assert "[WORKSPACE]" in msgs[2][1]
    assert "[LATEST]" in msgs[2][1]
    assert "[TIMELINE]" in msgs[2][1]
    assert "[STATIC]" in msgs[2][1]
    assert msgs[2][1].strip().endswith('Astra says: "')


def test_agent_context_composer_handles_new_chat_without_history():
    ghost = _PromptGhost(model_ctx=1600)
    ghost.conscious_broadcast = None
    ghost.conscious_candidates = []
    ghost.current_coalition = []
    ghost.all_features = []
    ghost.all_declarative_facts = []
    ghost.all_narratives = []
    ghost.all_episodic_memories = []

    packet = AgentContextComposer.build(
        ghost,
        focus_text="Fresh chat start with no prior history.",
        config=AgentContextConfig(
            max_tokens=240,
            weights=AgentContextWeights(workspace=0.25, latest=0.30, timeline=0.30, static=0.15),
            latest_messages=6,
            workspace_items=6,
            min_section_tokens=32,
        ),
        purpose="new_chat_bootstrap",
    )

    assert packet.text
    assert "[STATIC]" in packet.text
    assert "[WORKSPACE]" not in packet.text
    assert "[LATEST]" not in packet.text
    assert "[TIMELINE]" not in packet.text
    assert set(packet.debug.get("sections_present", [])) == {"static"}


def test_character_card_detailed_architecture_exists_and_formats_names():
    import pm.character_card as character_card
    import pm.config_loader as cfg

    original = {
        "companion_name": cfg.companion_name,
        "user_name": cfg.user_name,
        "available_tools": list(cfg.available_tools),
        "character_card_story": cfg.character_card_story,
    }
    try:
        cfg.companion_name = "Astra"
        cfg.user_name = "Riley"
        cfg.available_tools = ["weather_lookup", "calendar_sync"]
        cfg.character_card_story = (
            "{companion_name} unit-test story card for {user_name}.\n\n"
            "{architecture_description_story}"
        )
        character_card = importlib.reload(character_card)

        assert hasattr(character_card, "architecture_description_story_detailed")
        detailed = str(character_card.architecture_description_story_detailed or "")
        assert detailed.strip()
        assert "Astra" in detailed
        assert "Riley" in detailed
        assert "weather_lookup" in detailed
        assert "Astra unit-test story card for Riley." in character_card.character_card_story
        assert "{companion_name}" not in character_card.character_card_story
        assert "{user_name}" not in character_card.character_card_story
    finally:
        cfg.companion_name = original["companion_name"]
        cfg.user_name = original["user_name"]
        cfg.available_tools = list(original["available_tools"])
        cfg.character_card_story = original["character_card_story"]
        importlib.reload(character_card)
