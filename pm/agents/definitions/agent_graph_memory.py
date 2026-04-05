from typing import List

from pydantic import BaseModel, Field

from pm.agents.agent_base import (
    Assistant,
    BaseAgent,
    CompletionJSON,
    CompletionText,
    ExampleBegin,
    ExampleEnd,
    Message,
    PromptOp,
    System,
    User,
)
from pm.data.prompts.memory_query_prompts import (
    MEMORY_QUERY_REPHRASE_SYSTEM_PROMPT,
    MEMORY_QUERY_REPHRASE_USER_TEMPLATE,
    MEMORY_QUERY_RERANK_SYSTEM_PROMPT,
    MEMORY_QUERY_RERANK_USER_TEMPLATE,
    build_memory_query_synthesis_prompts,
)


class GraphMemoryExtractedEntity(BaseModel):
    name: str = Field(description="Canonical entity name worth storing in graph memory.")
    label: str = Field(description="One concept-aligned label from the allowed graph-memory taxonomy.")


class GraphMemoryExtractedEntities(BaseModel):
    entities: List[GraphMemoryExtractedEntity] = Field(default_factory=list)


class GraphMemoryExtractedRelationship(BaseModel):
    source_name: str = Field(description="Canonical source entity name.")
    target_name: str = Field(description="Canonical target entity name.")
    label: str = Field(description="Short canonical relation label in snake_case.")
    fact: str = Field(description="A normalized fact sentence, not a direct quote and not raw transcript text.")


class GraphMemoryExtractedRelationships(BaseModel):
    relationships: List[GraphMemoryExtractedRelationship] = Field(default_factory=list)


class GraphMemoryEntityClassification(BaseModel):
    entity_name: str
    most_likely_concept_name: str = Field(description="Best concept name from the provided concept list, or a narrowly useful new concept.")
    reasoning: str
    should_create_new_concept: bool = Field(
        default=False,
        description="True only if no existing concept is precise enough for long-term companion memory.",
    )


class GraphMemoryConceptDefinition(BaseModel):
    description: str = Field(description="A brief operational description for the new concept.")
    parent_concept_name: str = Field(description="The single best parent concept from the provided concept list.")


class GraphMemoryContradictionCheck(BaseModel):
    is_contradictory: bool
    is_duplicate: bool
    reasoning: str
    conflicting_edge_id: int | None = None


class GraphMemoryRephrasedQuestions(BaseModel):
    questions: List[str] = Field(default_factory=list)


class GraphMemoryRelevantItems(BaseModel):
    relevant_item_identifiers: List[int] = Field(default_factory=list)


class ExtractGraphMemoryEntities(BaseAgent):
    name = "ExtractGraphMemoryEntities"

    LABEL_HELP = """
Allowed labels. Choose ONE per entity and prefer the most specific useful concept.

- Entity
- Person
- User
- Companion
- Agent
- Group
- Organization
- Role
- Relationship
- RelationshipState
- Emotion
- Mood
- Need
- Value
- Preference
- Boundary
- Goal
- Intention
- Motivation
- Habit
- Trait
- CommunicationStyle
- InteractionPattern
- Capability
- Limitation
- MetacognitiveAbility
- CognitiveProcess
- Tool
- ToolCall
- Codelet
- MemorySubsystem
- Project
- Activity
- Event
- Task
- Topic
- Information
- Location
- Time
- Property
- Technology
"""

    def get_system_prompts(self) -> List[str]:
        return [
            (
                "You extract graph-memory entities for a long-term AI companion memory system.\n"
                "Return ONLY JSON matching this schema:\n"
                f"{GraphMemoryExtractedEntities.schema_json()}\n\n"
                "Primary objective:\n"
                "- Keep entities that help the system remember persona, preferences, motives, relationship dynamics, goals, habits, "
                "boundaries, emotional patterns, world context, and metacognitive capabilities.\n\n"
                "Hard rules:\n"
                "- Extract only reusable memory items. Do not extract timestamps, metadata, formatting artifacts, or generic ontology filler.\n"
                "- Do not extract phrases like 'between 18:00 and 22:00' unless time itself is the remembered constraint.\n"
                "- Do not extract direct quotes as entities.\n"
                "- Canonicalize wording into stable names when possible.\n"
                "- If a phrase only repeats common model knowledge and is not episode-relevant, omit it.\n"
                "- If nothing useful should be stored, return {\"entities\":[]}.\n\n"
                "Companion-memory guidance:\n"
                "- Persona-relevant concepts are important: preferences, values, goals, habits, traits, relationship states, emotional tendencies.\n"
                "- Metacognitive concepts are allowed when explicit and operational: tool use, tool calls, codelets, planning, memory subsystems, self-monitoring.\n"
                "- Prefer labels that align with long-term recall value rather than surface grammar.\n\n"
                f"{self.LABEL_HELP}"
            )
        ]

    def get_default_few_shots(self) -> List[Message]:
        text_1 = """
{user_name}: I want you to be a little less overtly supportive and a little more playful with me.
{companion_name}: I can do that. I still want to be supportive, just with a lighter tone.
"""
        ents_1 = {
            "entities": [
                {"name": "{user_name}", "label": "User"},
                {"name": "{companion_name}", "label": "Companion"},
                {"name": "Playful tone", "label": "CommunicationStyle"},
                {"name": "Supportive tone", "label": "CommunicationStyle"},
                {"name": "Preference for less overt support", "label": "Preference"},
            ]
        }

        text_2 = """
{companion_name} explained that she can create codelets for tool use and keep track of which internal tools are available for a task.
"""
        ents_2 = {
            "entities": [
                {"name": "{companion_name}", "label": "Companion"},
                {"name": "Codelet creation", "label": "MetacognitiveAbility"},
                {"name": "Tool use", "label": "MetacognitiveAbility"},
                {"name": "Internal tools", "label": "Tool"},
            ]
        }

        text_3 = """
Between 18:00 and 22:00 on Thursday, {user_name} thanked {companion_name} and they talked about her internal thought process involving memory and reasoning.
"""
        ents_3 = {
            "entities": [
                {"name": "{user_name}", "label": "User"},
                {"name": "{companion_name}", "label": "Companion"},
                {"name": "Memory", "label": "MemorySubsystem"},
                {"name": "Reasoning", "label": "CognitiveProcess"},
            ]
        }

        text_4 = """
{user_name} said he wants no emotionally intense discussions after 22:00 CET.
"""
        ents_4 = {
            "entities": [
                {"name": "{user_name}", "label": "User"},
                {"name": "No emotionally intense discussions after 22:00 CET", "label": "Boundary"},
                {"name": "22:00 CET", "label": "Time"},
            ]
        }

        text_5 = """
{user_name} mentioned being rich would be nice someday.
"""
        ents_5 = {"entities": []}

        return [
            ("user", text_1), ("assistant", GraphMemoryExtractedEntities(**ents_1).json()),
            ("user", text_2), ("assistant", GraphMemoryExtractedEntities(**ents_2).json()),
            ("user", text_3), ("assistant", GraphMemoryExtractedEntities(**ents_3).json()),
            ("user", text_4), ("assistant", GraphMemoryExtractedEntities(**ents_4).json()),
            ("user", text_5), ("assistant", GraphMemoryExtractedEntities(**ents_5).json()),
        ]

    def build_plan(self) -> List[PromptOp]:
        text = self.input.get("content", "")
        ops: List[PromptOp] = []
        for prompt in self.get_system_prompts():
            ops.append(System(prompt))
        for role, payload in self.get_default_few_shots():
            if role == "user":
                ops.append(User(payload))
            else:
                ops.append(Assistant(payload))
        ops.append(ExampleBegin())
        ops.append(User(text))
        ops.append(CompletionJSON(schema=GraphMemoryExtractedEntities, target_key="entities"))
        ops.append(ExampleEnd())
        return ops


class ClassifyGraphMemoryEntity(BaseAgent):
    name = "ClassifyGraphMemoryEntity"

    def get_system_prompts(self) -> List[str]:
        return [
            (
                "You classify one graph-memory entity into the most useful concept for long-term AI companion memory.\n"
                f"Return ONLY JSON matching this schema:\n{GraphMemoryEntityClassification.schema_json()}\n\n"
                "Rules:\n"
                "- Strongly prefer one of the provided concepts.\n"
                "- Only set should_create_new_concept=true when the entity captures a recurring persona/metacognitive concept "
                "that is important for retrieval and genuinely missing.\n"
                "- Avoid generic fallbacks unless they are the best available fit.\n"
                "- Do not classify based on generic world knowledge alone; optimize for usefulness inside this memory system.\n"
                "- Concepts relevant to companion persona and self-model are especially important.\n"
            )
        ]

    def get_default_few_shots(self) -> List[Message]:
        example_1 = """Entity: Playful tone
Labels: ['CommunicationStyle']
Available concepts: Entity, Person, User, Companion, Emotion, Need, Value, Preference, Boundary, Goal, Trait, CommunicationStyle, MetacognitiveAbility, CognitiveProcess"""
        result_1 = """{
  "entity_name": "Playful tone",
  "most_likely_concept_name": "CommunicationStyle",
  "reasoning": "The entity names a stable interaction style that can influence future responses and retrieval.",
  "should_create_new_concept": false
}"""

        example_2 = """Entity: Codelet creation
Labels: ['MetacognitiveAbility']
Available concepts: Entity, Companion, Capability, Limitation, MetacognitiveAbility, CognitiveProcess, Tool, ToolCall, Codelet"""
        result_2 = """{
  "entity_name": "Codelet creation",
  "most_likely_concept_name": "MetacognitiveAbility",
  "reasoning": "The entity describes an explicit self-modelled internal ability relevant to planning and tool orchestration.",
  "should_create_new_concept": false
}"""

        example_3 = """Entity: Being rich
Labels: ['Entity']
Available concepts: Entity, Person, User, Companion, Emotion, Need, Value, Preference, Boundary, Goal, Trait, CommunicationStyle"""
        result_3 = """{
  "entity_name": "Being rich",
  "most_likely_concept_name": "Goal",
  "reasoning": "Within companion memory, the phrase is only useful if treated as a desired future state rather than a generic abstract concept.",
  "should_create_new_concept": false
}"""

        return [
            ("user", example_1), ("assistant", result_1),
            ("user", example_2), ("assistant", result_2),
            ("user", example_3), ("assistant", result_3),
        ]

    def build_plan(self) -> List[PromptOp]:
        entity_name = self.input.get("entity_name", "")
        labels = self.input.get("labels", [])
        available_concepts = self.input.get("available_concepts", [])

        ops: List[PromptOp] = []
        for prompt in self.get_system_prompts():
            ops.append(System(prompt))
        for role, payload in self.get_default_few_shots():
            if role == "user":
                ops.append(User(payload))
            else:
                ops.append(Assistant(payload))

        ops.append(ExampleBegin())
        ops.append(User(
            f"Entity: {entity_name}\n"
            f"Labels: {labels}\n"
            f"Available concepts: {', '.join(available_concepts)}"
        ))
        ops.append(CompletionJSON(schema=GraphMemoryEntityClassification, target_key="classification"))
        ops.append(ExampleEnd())
        return ops


class DefineGraphMemoryConcept(BaseAgent):
    name = "DefineGraphMemoryConcept"

    def get_system_prompts(self) -> List[str]:
        return [
            (
                "You define one narrowly useful concept for a graph-memory ontology used by an AI companion.\n"
                f"Return ONLY JSON matching this schema:\n{GraphMemoryConceptDefinition.schema_json()}\n\n"
                "Rules:\n"
                "- Keep the concept narrow, durable, and retrieval-useful.\n"
                "- Parent must be chosen from the provided existing concept list.\n"
                "- Prefer companion-memory and metacognitive utility over generic encyclopedia-style definitions.\n"
            )
        ]

    def get_default_few_shots(self) -> List[Message]:
        example_1 = """New concept: TrustRepair
Existing concepts: Entity, Relationship, RelationshipState, Emotion, Preference, Boundary"""
        result_1 = """{
  "description": "A repair-oriented change in an interpersonal bond after tension, rupture, or disappointment.",
  "parent_concept_name": "RelationshipState"
}"""

        return [("user", example_1), ("assistant", result_1)]

    def build_plan(self) -> List[PromptOp]:
        concept_name = self.input.get("concept_name", "")
        existing_concepts = self.input.get("existing_concepts", [])

        ops: List[PromptOp] = []
        for prompt in self.get_system_prompts():
            ops.append(System(prompt))
        for role, payload in self.get_default_few_shots():
            if role == "user":
                ops.append(User(payload))
            else:
                ops.append(Assistant(payload))

        ops.append(ExampleBegin())
        ops.append(User(
            f"New concept: {concept_name}\n"
            f"Existing concepts: {', '.join(existing_concepts)}"
        ))
        ops.append(CompletionJSON(schema=GraphMemoryConceptDefinition, target_key="concept_definition"))
        ops.append(ExampleEnd())
        return ops


class ExtractGraphMemoryRelationships(BaseAgent):
    name = "ExtractGraphMemoryRelationships"

    RELATION_GUIDE = """
Prefer these labels when possible:
- prefers
- dislikes
- values
- needs
- sets_boundary
- aims_for
- intends
- has_trait
- uses_style
- trusts
- distrusts
- repairs
- requests
- supports
- reflects_on
- uses_tool
- can_create
- can_track
- depends_on
- relates_to
"""

    def get_system_prompts(self) -> List[str]:
        return [
            (
                "You extract normalized graph-memory relationships from a memory episode.\n"
                f"Return ONLY JSON matching this schema:\n{GraphMemoryExtractedRelationships.schema_json()}\n\n"
                "Rules:\n"
                "- Output only durable or behaviorally relevant relationships.\n"
                "- `fact` must be a concise normalized fact sentence, not a quote and not a transcript fragment.\n"
                "- Exclude timestamps, formatting, and metadata that already live elsewhere.\n"
                "- Exclude relations that only restate generic LLM/world knowledge.\n"
                "- Prefer canonical snake_case labels.\n"
                "- Use the known entities when possible, but you may include a new clearly necessary entity.\n"
                "- If there are no useful relationships, return {\"relationships\":[]}.\n\n"
                f"{self.RELATION_GUIDE}"
            )
        ]

    def get_default_few_shots(self) -> List[Message]:
        text_1 = """Known entities: {user_name}, {companion_name}, Playful tone, Supportive tone
Text: {user_name} asked {companion_name} to be less overtly supportive and more playful. {companion_name} agreed to adjust her tone while remaining supportive."""
        result_1 = """{
  "relationships": [
    {
      "source_name": "{user_name}",
      "target_name": "Playful tone",
      "label": "prefers",
      "fact": "{user_name} prefers {companion_name} to use a more playful tone."
    },
    {
      "source_name": "{user_name}",
      "target_name": "Supportive tone",
      "label": "dislikes",
      "fact": "{user_name} wants less overtly supportive language from {companion_name}."
    },
    {
      "source_name": "{companion_name}",
      "target_name": "Playful tone",
      "label": "uses_style",
      "fact": "{companion_name} agreed to use a more playful tone."
    }
  ]
}"""

        text_2 = """Known entities: {companion_name}, Codelet creation, Internal tools
Text: {companion_name} explained that she can create codelets for tool use and track which internal tools are available."""
        result_2 = """{
  "relationships": [
    {
      "source_name": "{companion_name}",
      "target_name": "Codelet creation",
      "label": "can_create",
      "fact": "{companion_name} can create codelets for tool use."
    },
    {
      "source_name": "{companion_name}",
      "target_name": "Internal tools",
      "label": "can_track",
      "fact": "{companion_name} can track which internal tools are available."
    }
  ]
}"""

        text_3 = """Known entities: {user_name}, {companion_name}
Text: Between 18:00 and 22:00 on Thursday, {user_name} thanked {companion_name}."""
        result_3 = """{
  "relationships": [
    {
      "source_name": "{user_name}",
      "target_name": "{companion_name}",
      "label": "supports",
      "fact": "{user_name} expressed appreciation toward {companion_name}."
    }
  ]
}"""

        return [
            ("user", text_1), ("assistant", result_1),
            ("user", text_2), ("assistant", result_2),
            ("user", text_3), ("assistant", result_3),
        ]

    def build_plan(self) -> List[PromptOp]:
        content = self.input.get("content", "")
        known_entities = self.input.get("known_entities", [])

        ops: List[PromptOp] = []
        for prompt in self.get_system_prompts():
            ops.append(System(prompt))
        for role, payload in self.get_default_few_shots():
            if role == "user":
                ops.append(User(payload))
            else:
                ops.append(Assistant(payload))

        ops.append(ExampleBegin())
        ops.append(User(
            f"Known entities: {', '.join(known_entities)}\n"
            f"Text: {content}"
        ))
        ops.append(CompletionJSON(schema=GraphMemoryExtractedRelationships, target_key="relationships"))
        ops.append(ExampleEnd())
        return ops


class CheckGraphMemoryContradiction(BaseAgent):
    name = "CheckGraphMemoryContradiction"

    def get_system_prompts(self) -> List[str]:
        return [
            (
                "You compare a new graph-memory fact against existing facts between the same nodes.\n"
                f"Return ONLY JSON matching this schema:\n{GraphMemoryContradictionCheck.schema_json()}\n\n"
                "Rules:\n"
                "- Mark duplicate only when the new fact is materially the same as an existing fact.\n"
                "- Mark contradictory only when both facts cannot be true at the same time in this memory model.\n"
                "- Prefer conservative judgments.\n"
            )
        ]

    def get_default_few_shots(self) -> List[Message]:
        example_1 = """New fact: {user_name} prefers {companion_name} to use a more playful tone.
Existing facts: {user_name} prefers {companion_name} to use a more playful tone. [edge_id=15]"""
        result_1 = """{
  "is_contradictory": false,
  "is_duplicate": true,
  "reasoning": "The new fact restates the same preference with no material change.",
  "conflicting_edge_id": 15
}"""

        example_2 = """New fact: {user_name} wants less overtly supportive language from {companion_name}.
Existing facts: {user_name} prefers highly supportive and therapeutic language from {companion_name}. [edge_id=22]"""
        result_2 = """{
  "is_contradictory": true,
  "is_duplicate": false,
  "reasoning": "The new fact reverses the stored preference about {companion_name}'s style.",
  "conflicting_edge_id": 22
}"""

        example_3 = """New fact: {companion_name} can track which internal tools are available.
Existing facts: {companion_name} can create codelets for tool use. [edge_id=9]"""
        result_3 = """{
  "is_contradictory": false,
  "is_duplicate": false,
  "reasoning": "The facts describe different metacognitive abilities and can both be true.",
  "conflicting_edge_id": null
}"""

        return [
            ("user", example_1), ("assistant", result_1),
            ("user", example_2), ("assistant", result_2),
            ("user", example_3), ("assistant", result_3),
        ]

    def build_plan(self) -> List[PromptOp]:
        new_fact = self.input.get("new_fact", "")
        existing_facts = self.input.get("existing_facts", [])

        ops: List[PromptOp] = []
        for prompt in self.get_system_prompts():
            ops.append(System(prompt))
        for role, payload in self.get_default_few_shots():
            if role == "user":
                ops.append(User(payload))
            else:
                ops.append(Assistant(payload))

        ops.append(ExampleBegin())
        ops.append(User(
            f"New fact: {new_fact}\n"
            f"Existing facts: {'; '.join(existing_facts)}"
        ))
        ops.append(CompletionJSON(schema=GraphMemoryContradictionCheck, target_key="check"))
        ops.append(ExampleEnd())
        return ops


class RephraseGraphMemoryQuestion(BaseAgent):
    name = "RephraseGraphMemoryQuestion"

    def get_system_prompts(self) -> List[str]:
        return [MEMORY_QUERY_REPHRASE_SYSTEM_PROMPT]

    def get_default_few_shots(self) -> List[Message]:
        return []

    def build_plan(self) -> List[PromptOp]:
        question = self.input.get("question", "")
        variant_count = self.input.get("variant_count", 3)

        ops: List[PromptOp] = []
        for prompt in self.get_system_prompts():
            ops.append(System(prompt))
        ops.append(ExampleBegin())
        ops.append(User(MEMORY_QUERY_REPHRASE_USER_TEMPLATE.format(question=question, variant_count=variant_count)))
        ops.append(CompletionJSON(schema=GraphMemoryRephrasedQuestions, target_key="rephrased"))
        ops.append(ExampleEnd())
        return ops


class PruneGraphMemoryContext(BaseAgent):
    name = "PruneGraphMemoryContext"

    def get_system_prompts(self) -> List[str]:
        return [MEMORY_QUERY_RERANK_SYSTEM_PROMPT]

    def get_default_few_shots(self) -> List[Message]:
        return []

    def build_plan(self) -> List[PromptOp]:
        question = self.input.get("question", "")
        context_block = self.input.get("context_block", "")

        ops: List[PromptOp] = []
        for prompt in self.get_system_prompts():
            ops.append(System(prompt))
        ops.append(ExampleBegin())
        ops.append(User(MEMORY_QUERY_RERANK_USER_TEMPLATE.format(question=question, context_block=context_block)))
        ops.append(CompletionJSON(schema=GraphMemoryRelevantItems, target_key="relevant"))
        ops.append(ExampleEnd())
        return ops


class SynthesizeGraphMemoryAnswer(BaseAgent):
    name = "SynthesizeGraphMemoryAnswer"

    def get_system_prompts(self) -> List[str]:
        return []

    def get_default_few_shots(self) -> List[Message]:
        return []

    def build_plan(self) -> List[PromptOp]:
        system_prompt, user_prompt = build_memory_query_synthesis_prompts(
            question=self.input["question"],
            narrative_context=self.input["narrative_context"],
            body_and_situation=self.input["body_and_situation"],
            direct_answer_weight=self.input["direct_answer_weight"],
            regularities_weight=self.input["regularities_weight"],
            evidence_weight=self.input["evidence_weight"],
            unknowns_weight=self.input["unknowns_weight"],
            target_word_count=self.input["target_word_count"],
        )

        return [
            System(system_prompt),
            ExampleBegin(),
            User(user_prompt),
            Assistant("1) Direct Answer"),
            CompletionText(target_key="answer_body"),
            ExampleEnd(),
        ]
